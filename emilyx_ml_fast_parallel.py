#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMILY-X fast parallel terrestrial LTE dataset generator
========================================================

Purpose
-------
Generate many independent terrestrial LTE transmitter realizations for machine
learning without launching the full v3.2 simulator once per scenario.

This file is intentionally self-contained: it contains the scenario sampler,
genuine-srsRAN IQ template loader, RF spectral model, terrestrial antenna model,
P.452/SRTM or FSPL propagation, receiver/noise model, local EMILY-like detector,
parallel batch runner, and ML-safe output writer.

External inputs are still required:
  * srsran_idle.cf32
  * srsran_light.cf32
  * srsran_medium.cf32
  * srsran_loaded.cf32
  * Python packages: numpy, scipy, pandas, pyproj
  * For --propagation p452: astropy + pycraf and an SRTM cache/download path

Why it is much faster than repeatedly running v3.2
---------------------------------------------------
1. Genuine IQ -> RF spectral templates are built ONCE and cached, then reused.
2. The 200--2500 MHz full spectrogram is not constructed.  Only the same local
   +/-40 MHz windows used by the EMILY preview are simulated.
3. P.452 is evaluated at the emission center frequencies, then the small
   frequency dependence across each local window is approximated by the FSPL
   frequency slope.  This is a deliberate fast-dataset approximation.
4. No per-scenario PNG/NPZ products are written by default.
5. Independent scenarios run in a ProcessPoolExecutor.
6. Each worker loads the RF template bank once and reuses it for many scenarios.

The default "localization" campaign varies tower location, carrier center,
conducted power, antenna geometry, traffic schedule, shadowing and random noise.
Deep RF-chain variation is represented through a small precomputed bank of RF
profiles (default 1 = current v3.2 terrestrial RF chain).  Use --rf-profiles 4
or similar if you want several PA/ACLR/harmonic variants without paying the
full IQ/RF processing cost for every scenario.

Outputs
-------
<output-dir>/
    dataset_metadata.json
    dataset_index.csv
    feature_schema.json
    ml_inputs/emily_records.jsonl
    ground_truth/scenario_labels.csv
    scenarios.jsonl
    cache/rf_template_bank_<hash>.npz
    cache/rf_template_bank_<hash>.json
    results/scenario_XXXXXX.json       # compact resumable per-scenario result
    qc/...                             # only if --qc-every > 0

The ML input file deliberately excludes transmitter truth.  scenario_id and
split are grouping fields, not model features.  The scenario split is assigned
at the transmitter-realization level to prevent train/test leakage.

Examples
--------
Fast five-scenario check using already cached SRTM tiles:

  python emilyx_ml_fast_parallel.py --n-datasets 5 --workers 2 \
      --output-dir emilyx_fast_test5 --no-srtm-download

100 scenarios:

  python emilyx_ml_fast_parallel.py --n-datasets 100 --workers 4 \
      --output-dir emilyx_fast_100 --no-srtm-download

Diagnostic FSPL-only speed test:

  python emilyx_ml_fast_parallel.py --n-datasets 100 --workers 8 \
      --propagation fspl --output-dir emilyx_fast_fspl_100

Generate several RF-chain variants once, then reuse them across all scenarios:

  python emilyx_ml_fast_parallel.py --n-datasets 1000 --workers 4 \
      --rf-profiles 4 --output-dir emilyx_fast_1000

Notes
-----
* The first invocation is slower because the RF template cache must be built.
  Later runs with unchanged IQ files / RF-profile count reuse it.
* For P.452, SRTM/terrain calculations remain the dominant per-scenario cost.
* On Windows, 2--4 workers is usually a good first choice.  More workers may
  become memory/disk limited and may contend for SRTM files.
"""

from __future__ import annotations

# Avoid each process spawning its own BLAS thread pool.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures as cf
import csv
import hashlib
import json
import math
import sys
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pyproj import Geod
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, median_filter, binary_dilation, label, find_objects

try:
    from astropy import units as u
    from pycraf import conversions as cnv, pathprof
    from pycraf.antenna import imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz
    HAVE_PYCRAF = True
except Exception:
    HAVE_PYCRAF = False

VERSION = "1.0"
GEOD = Geod(ellps="WGS84")

# -----------------------------------------------------------------------------
# Fixed receiver/backend assumptions (same current Nevada example)
# -----------------------------------------------------------------------------
FREQ_MIN_HZ = 200e6
FREQ_MAX_HZ = 2500e6
CHANNEL_SPACING_HZ = 70e3
RBW_HZ = 140e3
TIME_RESOLUTION_S = 1.0
LOCAL_TIMEZONE = "America/Los_Angeles"
LOCAL_HALFSPAN_HZ = 40e6

RX_ANT_GAIN_DBI = 0.0
RX_HEIGHT_AGL_M = 2.0
RX_NOISE_FIGURE_DB = 6.0
REFERENCE_TEMP_K = 290.0
NOISE_JITTER_DB = 0.20
BANDPASS_RIPPLE_DB = 0.35
GAIN_DRIFT_DB = 0.20

# EMILY local detector settings from v3.x.
THRESHOLD = 8.0
COARSE_FACTOR = 50
SMOOTH_COARSE_BINS = 31
NOISE_COARSE_BINS = 31
DILATE_TIME = 1
DILATE_FREQ = 2
GROUP_GAP_TIME_PIX = 3
GROUP_GAP_FREQ_PIX = 20
MIN_PIXELS = 50

# LTE / IQ assumptions for current terrestrial source.
LTE_CHANNEL_BW_HZ = 10e6
LTE_N_PRB = 50
IQ_SAMPLE_RATE_HZ = 11.52e6
MAX_CAPTURE_S = 0.50
TEMPLATE_WINDOW_MS = 100.0
TEMPLATE_HOP_MS = 100.0
MAX_TEMPLATES_PER_STATE = 5
STATE_NAMES = ("idle", "light", "medium", "loaded")
STATE_TO_INDEX = {s: i for i, s in enumerate(STATE_NAMES)}

# RF model defaults.
TX_OVERSAMPLE_FACTOR = 4
TX_DIGITAL_FILTER_PASS_HZ = 4.55e6
TX_DIGITAL_FILTER_STOP_HZ = 5.30e6
TX_DIGITAL_FILTER_TAPS = 257
TX_RF_FILTER_PASS_HZ = 4.50e6
TX_RF_FILTER_STOP_HZ = 5.50e6
TX_RF_FILTER_MAX_NEAR_ATTEN_DB = 30.0
TX_RF_FILTER_FAR_STOP_START_HZ = 18.0e6
TX_RF_FILTER_FAR_TRANSITION_HZ = 3.0e6
TX_RF_FILTER_FAR_STOP_ATTEN_DB = 70.0

# Bank stores RBW power versus carrier-relative frequency every 10 kHz.
# This is dense enough for cheap interpolation onto 70-kHz backend channels.
BANK_OFFSET_STEP_HZ = 10e3
BANK_OFFSETS_HZ = np.arange(-LOCAL_HALFSPAN_HZ, LOCAL_HALFSPAN_HZ + 0.5*BANK_OFFSET_STEP_HZ,
                            BANK_OFFSET_STEP_HZ, dtype=np.float64)

P452_PROFILE_STEP_M = 100.0
P452_POLARIZATION = 0
EARTH_RADIUS_M = 6_371_008.8

RECEIVERS = (
    {
        "name": "RX1", "lat_deg": 39.524900, "lon_deg": -114.373325,
        "height_agl_m": 2.0, "antenna_gain_dbi": 0.0,
        "noise_figure_db": 6.0, "excess_path_loss_db": 0.0,
    },
    {
        "name": "RX2", "lat_deg": 39.595518, "lon_deg": -114.535418,
        "height_agl_m": 2.0, "antenna_gain_dbi": 0.0,
        "noise_figure_db": 6.0, "excess_path_loss_db": 0.0,
    },
)

# -----------------------------------------------------------------------------
# Embedded campaign configuration
# -----------------------------------------------------------------------------
DEFAULT_CONFIG: dict[str, Any] = {
    "duration_s": 600,
    "simulation_start_utc": "2026-09-02T19:00:00Z",
    "p452_time_percent": 50.0,
    "location": {
        "reference_lat_deg": 39.560209,
        "reference_lon_deg": -114.4543715,
        "min_distance_km": 5.0,
        "max_distance_km": 80.0,
        "sampling": "uniform_distance",
    },
    "transmitter": {
        "height_agl_m": [20.0, 80.0],
        "conducted_power_dbm": [35.0, 46.0],
        "max_gain_dbi": [12.0, 20.0],
        "horizontal_hpbw_deg": [45.0, 90.0],
        "sector_azimuth_deg": [0.0, 360.0],
        "electrical_downtilt_deg": [0.0, 10.0],
        # These are synthetic RF translations of the same 10-MHz LTE baseband.
        "carrier_mhz_choices": [700.0, 800.0, 900.0, 1800.0, 1900.0, 2100.0],
    },
    "rf_profile_ranges": {
        "pa_input_backoff_db": [5.0, 11.0],
        "pa_rapp_p": [2.0, 4.0],
        "target_aclr_db": [45.0, 55.0],
        "second_harmonic_dbc": [-75.0, -50.0],
        "third_harmonic_dbc": [-80.0, -55.0],
        "second_harmonic_antenna_extra_loss_db": [5.0, 20.0],
        "third_harmonic_antenna_extra_loss_db": [10.0, 25.0],
    },
    "baseline_rf_profile": {
        "pa_input_backoff_db": 8.0,
        "pa_rapp_p": 3.0,
        "target_aclr_db": 48.0,
        "second_harmonic_dbc": -60.0,
        "third_harmonic_dbc": -65.0,
        "second_harmonic_antenna_extra_loss_db": 10.0,
        "third_harmonic_antenna_extra_loss_db": 15.0,
        "carrier_lo_leakage_dbc": -55.0,
        "clock_spur_dbc": -70.0,
    },
    "propagation_nuisance": {
        "shadowing_sigma_db": [0.3, 2.0],
        "shadowing_correlation_s": [10.0, 60.0],
    },
    "traffic": {
        "min_segment_s": 30,
        "max_segment_s": 120,
        "state_probabilities": {
            "idle": 0.20, "light": 0.25, "medium": 0.30, "loaded": 0.25,
        },
    },
    "split": {"train": 0.70, "validation": 0.15, "test": 0.15},
}

# Exact fields deliberately exposed to the student's ML model.
EMILY_ML_FIELDS = [
    "data_type", "timezone",
    "t_start", "t_mid", "t_end",
    "t_start_local", "t_mid_local", "t_end_local",
    "t_start_offset_sec", "t_mid_offset_sec", "t_end_offset_sec",
    "duration_sec", "ongoing", "time_structure",
    "f_low_hz", "f_high_hz", "f_center_hz", "bw_hz", "band_tag",
    "intensity_kind", "intensity_value",
    "Directional", "RA", "Dec", "Az", "El",
    "lat", "lon", "altitude",
    "file_freq_resolution_hz", "file_time_resolution_sec",
    "n_pixels", "n_time_pixels", "n_freq_pixels",
]


# -----------------------------------------------------------------------------
# Small data objects
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Tx:
    lat_deg: float
    lon_deg: float
    height_agl_m: float
    conducted_power_dbm: float
    max_gain_dbi: float
    horizontal_hpbw_deg: float
    sector_azimuth_deg: float
    electrical_downtilt_deg: float


@dataclass(frozen=True)
class Rx:
    name: str
    lat_deg: float
    lon_deg: float
    height_agl_m: float
    antenna_gain_dbi: float
    noise_figure_db: float
    excess_path_loss_db: float


RXS = tuple(Rx(**r) for r in RECEIVERS)


# -----------------------------------------------------------------------------
# JSON / CSV / hashing helpers
# -----------------------------------------------------------------------------
def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_utc(value: str) -> datetime:
    dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def finite(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [finite(x) for x in v]
    if isinstance(v, np.ndarray):
        return finite(v.tolist())
    return v


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(finite(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256_file(path: Path, chunk=8*1024*1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(obj: Any) -> str:
    raw = json.dumps(finite(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            out = {}
            for k in keys:
                x = row.get(k)
                if isinstance(x, (dict, list)):
                    out[k] = json.dumps(finite(x), separators=(",", ":"))
                else:
                    out[k] = x
            w.writerow(out)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(finite(row), allow_nan=False, separators=(",", ":")) + "\n")


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def uniform_range(rng: np.random.Generator, pair) -> float:
    a, b = float(pair[0]), float(pair[1])
    return float(rng.uniform(min(a, b), max(a, b)))


def assign_splits(n: int, rng: np.random.Generator, cfg: dict[str, float]) -> list[str]:
    a = np.array([cfg.get("train", .7), cfg.get("validation", .15), cfg.get("test", .15)], dtype=float)
    a /= a.sum()
    counts = np.floor(a * n).astype(int)
    while counts.sum() < n:
        counts[np.argmax(a*n - counts)] += 1
    labels = ["train"]*counts[0] + ["validation"]*counts[1] + ["test"]*counts[2]
    rng.shuffle(labels)
    return labels


def sample_position(rng: np.random.Generator, cfg: dict[str, Any]) -> dict[str, float]:
    lat0, lon0 = float(cfg["reference_lat_deg"]), float(cfg["reference_lon_deg"])
    r0, r1 = float(cfg["min_distance_km"]), float(cfg["max_distance_km"])
    bearing = float(rng.uniform(0, 360))
    if cfg.get("sampling") == "uniform_area":
        r = float(np.sqrt(rng.uniform(r0*r0, r1*r1)))
    else:
        r = float(rng.uniform(r0, r1))
    lon, lat, _ = GEOD.fwd(lon0, lat0, bearing, r*1000.0)
    return {"lat_deg": float(lat), "lon_deg": float(lon), "distance_from_reference_km": r,
            "bearing_from_reference_deg": bearing}


def random_traffic_schedule(rng: np.random.Generator, cfg: dict[str, Any], duration_s: int):
    probs = cfg["state_probabilities"]
    states = list(probs)
    p = np.array([float(probs[s]) for s in states], dtype=float)
    p /= p.sum()
    out = []
    t = 0
    prev = None
    while t < duration_s:
        seg = int(rng.integers(int(cfg["min_segment_s"]), int(cfg["max_segment_s"])+1))
        t1 = min(duration_s, t+seg)
        state = str(rng.choice(states, p=p))
        for _ in range(4):
            if state != prev:
                break
            state = str(rng.choice(states, p=p))
        out.append({"start_s": float(t), "end_s": float(t1), "state": state})
        prev = state
        t = t1
    return out


def make_rf_profiles(count: int, seed: int, cfg: dict[str, Any]) -> list[dict[str, float]]:
    count = max(1, int(count))
    base = dict(cfg["baseline_rf_profile"])
    profiles = [base]
    rng = np.random.default_rng(seed + 9917)
    ranges = cfg["rf_profile_ranges"]
    for _ in range(1, count):
        p = dict(base)
        for key, pair in ranges.items():
            p[key] = uniform_range(rng, pair)
        profiles.append(p)
    for i, p in enumerate(profiles):
        p["rf_profile_id"] = i
    return profiles


def sample_scenarios(n: int, seed: int, cfg: dict[str, Any], rf_profiles: list[dict[str, float]],
                     propagation: str) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    splits = assign_splits(n, rng, cfg["split"])
    out = []
    for i in range(n):
        pos = sample_position(rng, cfg["location"])
        txc = cfg["transmitter"]
        pn = cfg["propagation_nuisance"]
        rf_id = int(rng.integers(0, len(rf_profiles)))
        rf = rf_profiles[rf_id]
        carrier = float(rng.choice(np.asarray(txc["carrier_mhz_choices"], dtype=float))) * 1e6
        scenario_seed = int(rng.integers(1, 2**31-1))
        sc = {
            "scenario_id": f"scenario_{i:06d}", "scenario_index": i, "split": splits[i],
            "batch_seed": int(seed), "simulation_seed": scenario_seed,
            "simulation_start_utc": str(cfg["simulation_start_utc"]),
            "duration_s": int(cfg["duration_s"]), "propagation": propagation,
            "p452_time_percent": float(cfg["p452_time_percent"]),
            "tx_lat_deg": pos["lat_deg"], "tx_lon_deg": pos["lon_deg"],
            "tx_distance_from_reference_km": pos["distance_from_reference_km"],
            "tx_bearing_from_reference_deg": pos["bearing_from_reference_deg"],
            "tx_height_agl_m": uniform_range(rng, txc["height_agl_m"]),
            "tx_conducted_power_dbm": uniform_range(rng, txc["conducted_power_dbm"]),
            "tx_max_gain_dbi": uniform_range(rng, txc["max_gain_dbi"]),
            "tx_horizontal_hpbw_deg": uniform_range(rng, txc["horizontal_hpbw_deg"]),
            "tx_sector_azimuth_deg": uniform_range(rng, txc["sector_azimuth_deg"]),
            "tx_electrical_downtilt_deg": uniform_range(rng, txc["electrical_downtilt_deg"]),
            "carrier_center_hz": carrier, "lte_channel_bw_hz": LTE_CHANNEL_BW_HZ,
            "lte_n_prb": LTE_N_PRB, "lte_dl_earfcn": -1, "sample_rate_hz": IQ_SAMPLE_RATE_HZ,
            "rf_profile_id": rf_id,
            "shadowing_sigma_db": uniform_range(rng, pn["shadowing_sigma_db"]),
            "shadowing_correlation_s": uniform_range(rng, pn["shadowing_correlation_s"]),
            "traffic_schedule": random_traffic_schedule(rng, cfg["traffic"], int(cfg["duration_s"])),
        }
        for k, v in rf.items():
            if k != "rf_profile_id":
                sc[k] = v
        out.append(finite(sc))
    return out


# -----------------------------------------------------------------------------
# RF template cache
# -----------------------------------------------------------------------------
def normalized_psd(iq: np.ndarray, sample_rate_hz: float, nperseg=32768):
    nperseg = min(nperseg, len(iq))
    noverlap = nperseg // 2
    f, p = signal.welch(iq, fs=sample_rate_hz, window="hann", nperseg=nperseg,
                        noverlap=noverlap, detrend=False, return_onesided=False, scaling="density")
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(np.maximum(p, 0.0))
    area = np.trapezoid(p, f)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Invalid PSD area")
    return f.astype(np.float64), (p/area).astype(np.float64)


def design_reconstruction_filter(fs: float):
    cutoff = .5*(TX_DIGITAL_FILTER_PASS_HZ + TX_DIGITAL_FILTER_STOP_HZ)
    return signal.firwin(TX_DIGITAL_FILTER_TAPS, cutoff=cutoff, fs=fs, window=("kaiser", 7.5))


def apply_rapp_pa(iq: np.ndarray, backoff_db: float, p: float):
    x = iq.astype(np.complex128)
    x /= np.sqrt(np.mean(np.abs(x)**2))
    a_sat = 10**(backoff_db/20)
    amp = np.abs(x)
    y = x / (1 + (amp/a_sat)**(2*p))**(1/(2*p))
    y /= np.sqrt(np.mean(np.abs(y)**2))
    return y.astype(np.complex64)


def aclr_db(f: np.ndarray, p: np.ndarray, measure_bw=9e6):
    half = measure_bw/2
    main = np.abs(f) <= half
    lower = (f >= -LTE_CHANNEL_BW_HZ-half) & (f <= -LTE_CHANNEL_BW_HZ+half)
    upper = (f >= LTE_CHANNEL_BW_HZ-half) & (f <= LTE_CHANNEL_BW_HZ+half)
    def integ(m):
        return float(np.trapezoid(p[m], f[m])) if np.count_nonzero(m) > 1 else 0.0
    return 10*np.log10(max(integ(main), 1e-30)/max(integ(lower), integ(upper), 1e-30))


def rf_filter_gain(offset_hz: np.ndarray, near_stop_atten_db: float):
    f = np.abs(np.asarray(offset_hz, dtype=float))
    near_stop_atten_db = float(np.clip(near_stop_atten_db, 0, TX_RF_FILTER_MAX_NEAR_ATTEN_DB))
    a = np.zeros_like(f)
    m = (f > TX_RF_FILTER_PASS_HZ) & (f < TX_RF_FILTER_STOP_HZ)
    if np.any(m):
        x = (f[m]-TX_RF_FILTER_PASS_HZ)/(TX_RF_FILTER_STOP_HZ-TX_RF_FILTER_PASS_HZ)
        a[m] = near_stop_atten_db*(.5-.5*np.cos(np.pi*x))
    m = (f >= TX_RF_FILTER_STOP_HZ) & (f < TX_RF_FILTER_FAR_STOP_START_HZ)
    a[m] = near_stop_atten_db
    far_end = TX_RF_FILTER_FAR_STOP_START_HZ + TX_RF_FILTER_FAR_TRANSITION_HZ
    m = (f >= TX_RF_FILTER_FAR_STOP_START_HZ) & (f < far_end)
    if np.any(m):
        x = (f[m]-TX_RF_FILTER_FAR_STOP_START_HZ)/(far_end-TX_RF_FILTER_FAR_STOP_START_HZ)
        s = .5-.5*np.cos(np.pi*x)
        a[m] = near_stop_atten_db + (TX_RF_FILTER_FAR_STOP_ATTEN_DB-near_stop_atten_db)*s
    a[f >= far_end] = TX_RF_FILTER_FAR_STOP_ATTEN_DB
    return 10**(-a/10)


def choose_near_stop(f: np.ndarray, raw: np.ndarray, target: float):
    raw_a = aclr_db(f, raw)
    if raw_a >= target:
        return 0.0, raw_a
    lo, hi = 0.0, TX_RF_FILTER_MAX_NEAR_ATTEN_DB
    test = raw*rf_filter_gain(f, hi); test /= np.trapezoid(test, f)
    if aclr_db(f, test) < target:
        return hi, raw_a
    for _ in range(32):
        mid = .5*(lo+hi)
        test = raw*rf_filter_gain(f, mid); test /= np.trapezoid(test, f)
        if aclr_db(f, test) >= target:
            hi = mid
        else:
            lo = mid
    return hi, raw_a


def gaussian_unit_psd(fwhm_hz: float, n=4097):
    sigma = fwhm_hz/(2*np.sqrt(2*np.log(2)))
    span = max(8*sigma, 2*CHANNEL_SPACING_HZ)
    f = np.linspace(-span, span, n)
    p = np.exp(-.5*(f/sigma)**2)
    p /= np.trapezoid(p, f)
    return f, p


def rbw_fraction(centers: np.ndarray, f: np.ndarray, p: np.ndarray):
    cdf = cumulative_trapezoid(p, f, initial=0.0)
    if cdf[-1] <= 0:
        return np.zeros_like(centers)
    cdf /= cdf[-1]
    lo = centers - RBW_HZ/2
    hi = centers + RBW_HZ/2
    return np.maximum(np.interp(hi, f, cdf, left=0, right=1) -
                      np.interp(lo, f, cdf, left=0, right=1), 0.0)


def load_iq_templates(iq_dir: Path):
    states = {}
    max_samples = int(round(MAX_CAPTURE_S*IQ_SAMPLE_RATE_HZ))
    win = int(round(TEMPLATE_WINDOW_MS*1e-3*IQ_SAMPLE_RATE_HZ))
    hop = int(round(TEMPLATE_HOP_MS*1e-3*IQ_SAMPLE_RATE_HZ))
    for state in STATE_NAMES:
        path = iq_dir/f"srsran_{state}.cf32"
        if not path.exists():
            raise FileNotFoundError(path)
        x = np.fromfile(path, dtype=np.complex64, count=max_samples)
        if len(x) < win:
            raise ValueError(f"{path} is shorter than {TEMPLATE_WINDOW_MS} ms")
        x = x.astype(np.complex64, copy=False)
        capture = x - np.mean(x)
        capture_rms = float(np.sqrt(np.mean(np.abs(capture)**2)))
        starts = list(range(0, len(x)-win+1, hop))
        if len(starts) > MAX_TEMPLATES_PER_STATE:
            choose = np.linspace(0, len(starts)-1, MAX_TEMPLATES_PER_STATE).round().astype(int)
            starts = [starts[i] for i in np.unique(choose)]
        templates = []
        for ti, st in enumerate(starts):
            w = np.array(x[st:st+win], dtype=np.complex64, copy=True)
            w -= np.mean(w)
            rms = float(np.sqrt(np.mean(np.abs(w)**2)))
            w /= rms
            templates.append({"template_index": ti, "raw_rms": rms, "iq": w})
        states[state] = {"capture_rms": capture_rms, "templates": templates, "path": str(path)}
    ref = states["loaded"]["capture_rms"]
    for state in STATE_NAMES:
        states[state]["state_relative_power_db"] = 20*np.log10(states[state]["capture_rms"]/ref)
        for t in states[state]["templates"]:
            t["relative_power_db"] = float(20*np.log10(t["raw_rms"]/ref))
    return states


def build_one_template_spectra(iq: np.ndarray, rel_db: float, rf: dict[str, float]):
    # Returns 3 arrays: fundamental/spurs window, 2nd harmonic, 3rd harmonic.
    up = TX_OVERSAMPLE_FACTOR
    x = signal.resample_poly(iq, up, 1).astype(np.complex64)
    fs = IQ_SAMPLE_RATE_HZ*up
    taps = design_reconstruction_filter(fs)
    xf = signal.lfilter(taps, [1.0], x).astype(np.complex64)
    if len(xf) > 2*len(taps):
        xf = xf[len(taps):]
    xf /= np.sqrt(np.mean(np.abs(xf)**2))

    y = apply_rapp_pa(xf, float(rf["pa_input_backoff_db"]), float(rf["pa_rapp_p"]))
    f1, p1raw = normalized_psd(y, fs)
    near, raw_aclr = choose_near_stop(f1, p1raw, float(rf["target_aclr_db"]))
    p1 = p1raw*rf_filter_gain(f1, near)
    p1 /= np.trapezoid(p1, f1)
    frac1 = rbw_fraction(BANK_OFFSETS_HZ, f1, p1)

    # 0-dBm loaded reference.  Apply the genuine per-template power offset here.
    scale = 10**(rel_db/10)
    fundamental_mw = scale * frac1  # total reference fundamental = 1 mW before truncation

    # Narrow spurs inside the fundamental local window.
    for offset, dbc, fwhm in [
        (0.0, float(rf.get("carrier_lo_leakage_dbc", -55.0)), 8e3),
        (-30.72e6, float(rf.get("clock_spur_dbc", -70.0)), 30e3),
        (+30.72e6, float(rf.get("clock_spur_dbc", -70.0)), 30e3),
    ]:
        fs_, ps_ = gaussian_unit_psd(fwhm)
        frac = rbw_fraction(BANK_OFFSETS_HZ-offset, fs_, ps_)
        fundamental_mw += scale * 10**(dbc/10) * frac

    harmonics = []
    for order, dbc in [(2, float(rf["second_harmonic_dbc"])),
                       (3, float(rf["third_harmonic_dbc"]))]:
        h = xf.astype(np.complex128)**order
        h -= np.mean(h)
        h /= np.sqrt(np.mean(np.abs(h)**2))
        fh, ph = normalized_psd(h.astype(np.complex64), fs)
        frac = rbw_fraction(BANK_OFFSETS_HZ, fh, ph)
        harmonics.append(scale * 10**(dbc/10) * frac)

    diag = {"raw_post_pa_aclr_db": float(raw_aclr), "measured_aclr_db": float(aclr_db(f1, p1)),
            "near_stop_atten_db": float(near)}
    arrs = [fundamental_mw, harmonics[0], harmonics[1]]
    return np.stack(arrs).astype(np.float32), diag


def cache_key(iq_dir: Path, profiles: list[dict[str, float]]) -> tuple[str, dict[str, Any]]:
    iq_meta = []
    for state in STATE_NAMES:
        p = iq_dir/f"srsran_{state}.cf32"
        if not p.exists():
            raise FileNotFoundError(p)
        iq_meta.append({"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)})
    payload = {
        "version": VERSION, "iq": iq_meta, "profiles": profiles,
        "sample_rate": IQ_SAMPLE_RATE_HZ, "window_ms": TEMPLATE_WINDOW_MS,
        "bank_step_hz": BANK_OFFSET_STEP_HZ, "halfspan_hz": LOCAL_HALFSPAN_HZ,
        "rbw_hz": RBW_HZ,
    }
    return stable_hash(payload)[:16], payload


def build_or_load_rf_bank(iq_dir: Path, cache_dir: Path, profiles: list[dict[str, float]]):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key, meta = cache_key(iq_dir, profiles)
    npz_path = cache_dir/f"rf_template_bank_{key}.npz"
    json_path = cache_dir/f"rf_template_bank_{key}.json"
    if npz_path.exists() and json_path.exists():
        print(f"RF template bank cache: {npz_path.name} (reused)")
        return npz_path, json_path

    print(f"Building RF template bank once ({len(profiles)} RF profile(s))...")
    states = load_iq_templates(iq_dir)
    nprof, nstate, ntemp, nwin, nbin = len(profiles), 4, MAX_TEMPLATES_PER_STATE, 3, len(BANK_OFFSETS_HZ)
    spectra = np.zeros((nprof, nstate, ntemp, nwin, nbin), dtype=np.float32)
    valid = np.zeros((nstate, ntemp), dtype=np.uint8)
    rel_db = np.full((nstate, ntemp), np.nan, dtype=np.float32)
    diagnostics: dict[str, Any] = {}

    for si, state in enumerate(STATE_NAMES):
        for t in states[state]["templates"]:
            ti = int(t["template_index"])
            valid[si, ti] = 1
            rel_db[si, ti] = float(t["relative_power_db"])

    total = len(profiles)*sum(len(states[s]["templates"]) for s in STATE_NAMES)
    done = 0
    for pi, rf in enumerate(profiles):
        for si, state in enumerate(STATE_NAMES):
            for t in states[state]["templates"]:
                ti = int(t["template_index"])
                arr, diag = build_one_template_spectra(t["iq"], float(t["relative_power_db"]), rf)
                spectra[pi, si, ti] = arr
                diagnostics[f"p{pi}_{state}_t{ti}"] = diag
                done += 1
                print(f"  RF bank {done:3d}/{total}: profile={pi} {state} T{ti}", flush=True)

    np.savez(npz_path, offsets_hz=BANK_OFFSETS_HZ.astype(np.float64), spectra_ref_mw=spectra,
             template_valid=valid, template_relative_power_db=rel_db)
    meta.update({
        "created_utc": utc_now(), "profiles": profiles,
        "state_capture_rms": {s: states[s]["capture_rms"] for s in STATE_NAMES},
        "state_relative_power_db": {s: states[s]["state_relative_power_db"] for s in STATE_NAMES},
        "diagnostics": diagnostics,
    })
    write_json(json_path, meta)
    print(f"Saved RF bank: {npz_path}")
    return npz_path, json_path


# -----------------------------------------------------------------------------
# Geometry / antenna / propagation
# -----------------------------------------------------------------------------
def link_geometry(tx: Tx, rx: Rx):
    az_tx, az_rx, d = GEOD.inv(tx.lon_deg, tx.lat_deg, rx.lon_deg, rx.lat_deg)
    elev = math.degrees(math.atan2(rx.height_agl_m - tx.height_agl_m, d))
    return {"distance_m": float(d), "distance_km": float(d/1e3),
            "bearing_tx_to_rx_deg": float(az_tx % 360), "bearing_rx_to_tx_deg": float(az_rx % 360),
            "elevation_tx_to_rx_deg": float(elev)}


def wrap180(x):
    return (np.asarray(x)+180)%360-180


def tx_gain_db(tx: Tx, bearing: float, elevation: float, order: int, rf: dict[str, float]):
    rel_az = float(wrap180(bearing-tx.sector_azimuth_deg))
    extra = 0.0
    if order == 2:
        extra = float(rf["second_harmonic_antenna_extra_loss_db"])
    elif order == 3:
        extra = float(rf["third_harmonic_antenna_extra_loss_db"])
    gmax = tx.max_gain_dbi-extra
    theta_3db = 31000*10**(-0.1*tx.max_gain_dbi)/tx.horizontal_hpbw_deg
    if HAVE_PYCRAF:
        kp, kh, kv = (0.7, 0.7, 0.3)*cnv.dimless
        g = imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz(
            rel_az*u.deg, elevation*u.deg, gmax*cnv.dB,
            tx.horizontal_hpbw_deg*u.deg, theta_3db*u.deg, kp, kh, kv,
            tilt_m=0*u.deg, tilt_e=tx.electrical_downtilt_deg*u.deg)
        return float(np.asarray(g.to(cnv.dB).value).squeeze())
    rel_el = elevation+tx.electrical_downtilt_deg
    atten = 12*(rel_az/tx.horizontal_hpbw_deg)**2 + 12*(rel_el/theta_3db)**2
    return float(gmax-min(30.0, atten))


def fspl_db(distance_m: float, freq_hz):
    f = np.asarray(freq_hz, dtype=float)
    c = 299_792_458.0
    return 20*np.log10(4*np.pi*distance_m*f/c)


def standard_atmosphere(h_m: float):
    h = float(np.clip(h_m, -500, 11000))
    return 288.15-.0065*h, 1013.25*(1-2.25577e-5*h)**5.25588


def terrain_profile(tx: Tx, rx: Rx, srtm_dir: Path, download_missing: bool):
    if not HAVE_PYCRAF:
        raise RuntimeError("P.452 mode requires astropy + pycraf")
    srtm_dir.mkdir(parents=True, exist_ok=True)
    with pathprof.SrtmConf.set(srtm_dir=str(srtm_dir.resolve()),
                               download="missing" if download_missing else "never",
                               server="viewpano", interp="linear"):
        lons, lats, distance, distances, heights, bearing, back_bearing, back_bearings = \
            pathprof.srtm_height_profile(tx.lon_deg*u.deg, tx.lat_deg*u.deg,
                                         rx.lon_deg*u.deg, rx.lat_deg*u.deg,
                                         P452_PROFILE_STEP_M*u.m)
    hm = np.asarray(heights.to(u.m).value, dtype=float)
    if not np.all(np.isfinite(hm)) or np.nanmax(np.abs(hm)) < 10:
        raise RuntimeError(f"Invalid SRTM profile for {rx.name}")
    total = float(distance.to(u.m).value)
    curv = total**2/(2*EARTH_RADIUS_M)
    direct_el = math.degrees(math.atan2((hm[-1]+rx.height_agl_m)-(hm[0]+tx.height_agl_m)-curv, total))
    tm, pm = standard_atmosphere(float(hm[len(hm)//2]))
    return {"distance_m": total, "distance_km": total/1e3,
            "distances": distances, "heights": heights, "bearing": bearing, "back_bearing": back_bearing,
            "tx_ground_elevation_m": float(hm[0]), "rx_ground_elevation_m": float(hm[-1]),
            "midpoint_ground_elevation_m": float(hm[len(hm)//2]),
            "temperature_k": tm, "pressure_hpa": pm, "direct_elevation_deg": direct_el}


def propagation_centers(tx: Tx, rx: Rx, centers_hz: list[float], mode: str, srtm_dir: Path,
                        download_missing: bool, p452_percent: float):
    base = link_geometry(tx, rx)
    centers = np.asarray(centers_hz, dtype=float)
    if mode == "fspl":
        losses = fspl_db(base["distance_m"], centers)
        return base, {float(f): float(l) for f, l in zip(centers, losses)}, None

    terr = terrain_profile(tx, rx, srtm_dir, download_missing)
    geom = dict(base)
    geom["distance_m"] = terr["distance_m"]
    geom["distance_km"] = terr["distance_km"]
    geom["elevation_tx_to_rx_deg"] = terr["direct_elevation_deg"]
    result = pathprof.losses_complete(
        centers*1e-9*u.GHz, terr["temperature_k"]*u.K, terr["pressure_hpa"]*u.hPa,
        tx.lon_deg*u.deg, tx.lat_deg*u.deg, rx.lon_deg*u.deg, rx.lat_deg*u.deg,
        tx.height_agl_m*u.m, rx.height_agl_m*u.m, P452_PROFILE_STEP_M*u.m,
        float(p452_percent)*u.percent, G_t=0*cnv.dBi, G_r=0*cnv.dBi,
        omega=0*u.percent, zone_t=pathprof.CLUTTER.UNKNOWN, zone_r=pathprof.CLUTTER.UNKNOWN,
        polarization=P452_POLARIZATION, version=16,
        hprof_dists=terr["distances"], hprof_heights=terr["heights"],
        hprof_bearing=terr["bearing"], hprof_backbearing=terr["back_bearing"])
    losses = np.asarray(result["L_b"].to(cnv.dB).value, dtype=float).reshape(-1)
    if not np.all(np.isfinite(losses)):
        raise RuntimeError(f"P.452 non-finite loss for {rx.name}")
    return geom, {float(f): float(l) for f, l in zip(centers, losses)}, terr


# -----------------------------------------------------------------------------
# Time / traffic / receiver nuisance
# -----------------------------------------------------------------------------
def make_time_axes(start_utc: str, nt: int):
    start = parse_utc(start_utc)
    rel = np.arange(nt, dtype=float)*TIME_RESOLUTION_S
    local_tz = ZoneInfo(LOCAL_TIMEZONE)
    utc = np.array([isoformat_z(start+timedelta(seconds=float(s))) for s in rel])
    local = np.array([(start+timedelta(seconds=float(s))).astimezone(local_tz).isoformat() for s in rel])
    return rel, utc, local


def schedule_arrays(schedule: list[dict[str, Any]], nt: int, rng: np.random.Generator,
                    valid_templates: np.ndarray):
    states = np.full(nt, "idle", dtype="U16")
    for seg in schedule:
        i0 = max(0, int(round(float(seg["start_s"]))))
        i1 = min(nt, int(round(float(seg["end_s"]))))
        if i1 > i0:
            states[i0:i1] = str(seg["state"])
    templates = np.full(nt, -1, dtype=np.int16)
    for state in STATE_NAMES:
        si = STATE_TO_INDEX[state]
        tids = np.flatnonzero(valid_templates[si] > 0)
        idx = np.flatnonzero(states == state)
        if len(idx) == 0:
            continue
        if len(tids) == 0:
            raise RuntimeError(f"No RF templates for state {state}")
        seq = []
        while len(seq) < len(idx):
            seq.extend(rng.permutation(tids).tolist())
        templates[idx] = np.asarray(seq[:len(idx)], dtype=np.int16)
    if np.any(templates < 0):
        raise RuntimeError("Unassigned template")
    return states, templates


def slow_shadowing(nt: int, sigma_db: float, corr_s: float, rng: np.random.Generator):
    if sigma_db <= 0:
        return np.zeros(nt, dtype=np.float32)
    x = rng.normal(size=nt)
    y = gaussian_filter1d(x, sigma=max(1.0, corr_s/TIME_RESOLUTION_S), mode="reflect")
    y -= y.mean()
    if y.std() > 0:
        y *= sigma_db/y.std()
    return y.astype(np.float32)


def thermal_noise_dbm(rbw: float, nf_db: float):
    p_w = 1.380649e-23*REFERENCE_TEMP_K*rbw
    return float(10*np.log10(p_w/1e-3)+nf_db)


def backend_freq_window(center_hz: float):
    lo = max(FREQ_MIN_HZ, center_hz-LOCAL_HALFSPAN_HZ)
    hi = min(FREQ_MAX_HZ, center_hz+LOCAL_HALFSPAN_HZ)
    i0 = int(math.ceil((lo-FREQ_MIN_HZ)/CHANNEL_SPACING_HZ))
    i1 = int(math.floor((hi-FREQ_MIN_HZ)/CHANNEL_SPACING_HZ))
    if i1 < i0:
        return np.empty(0, dtype=float)
    return FREQ_MIN_HZ + np.arange(i0, i1+1, dtype=float)*CHANNEL_SPACING_HZ


# -----------------------------------------------------------------------------
# EMILY detector
# -----------------------------------------------------------------------------
def dbm_to_mw(x):
    return 10**(np.asarray(x, dtype=float)/10)


def mw_to_dbm(x, floor_dbm=-300):
    return 10*np.log10(np.maximum(np.asarray(x, dtype=float), 10**(floor_dbm/10)))


def fast_flagger(spec_dbm: np.ndarray, freq_mhz: np.ndarray):
    x = np.asarray(spec_dbm, dtype=np.float32)
    nt, nf = x.shape
    nf_coarse = nf//COARSE_FACTOR
    ntrim = nf_coarse*COARSE_FACTOR
    if nf_coarse < 2:
        raise ValueError("Frequency window too narrow for EMILY coarse factor")
    xb = x[:, :ntrim].reshape(nt, nf_coarse, COARSE_FACTOR)
    xc = np.nanmedian(xb, axis=2).astype(np.float32)
    fc = freq_mhz[:ntrim].reshape(nf_coarse, COARSE_FACTOR).mean(axis=1)
    bgc = median_filter(xc, size=(1, SMOOTH_COARSE_BINS), mode="nearest")
    bg = np.empty_like(x)
    for it in range(nt):
        bg[it] = np.interp(freq_mhz, fc, bgc[it])
    resid = x-bg
    rb = resid[:, :ntrim].reshape(nt, nf_coarse, COARSE_FACTOR)
    arc = np.nanmedian(np.abs(rb), axis=2).astype(np.float32)
    sc = 1.4826*median_filter(arc, size=(1, NOISE_COARSE_BINS), mode="nearest")
    sigma = np.empty_like(x)
    for it in range(nt):
        sigma[it] = np.interp(freq_mhz, fc, sc[it])
    good = np.isfinite(sigma) & (sigma > 0)
    sigma[~good] = np.nanmedian(sigma[good]) if np.any(good) else 1.0
    mask = (resid/sigma) > THRESHOLD
    if DILATE_TIME > 0 or DILATE_FREQ > 0:
        mask = binary_dilation(mask, structure=np.ones((2*DILATE_TIME+1, 2*DILATE_FREQ+1), dtype=bool))
    return mask


def band_tag(flo, fhi):
    bands = [("VHF",30e6,300e6),("UHF",300e6,1e9),("L",1e9,2e9),("S",2e9,4e9),("C",4e9,8e9)]
    return "/".join(n for n,a,b in bands if fhi >= a and flo <= b) or None


def extract_events(mask, spec_dbm, freq_hz, time_utc, time_local, time_rel):
    grouped = binary_dilation(mask, structure=np.ones((2*GROUP_GAP_TIME_PIX+1, 2*GROUP_GAP_FREQ_PIX+1), dtype=bool))
    labels, _ = label(grouped)
    rows = []
    nt = mask.shape[0]
    for slc in find_objects(labels):
        if slc is None:
            continue
        ts, fs = slc
        submask = mask[ts, fs]
        if int(submask.sum()) < MIN_PIXELS:
            continue
        tt, ff = np.where(submask)
        ti = tt+ts.start; fi = ff+fs.start
        t0,t1 = int(ti.min()), int(ti.max()); f0,f1 = int(fi.min()), int(fi.max())
        vals = dbm_to_mw(spec_dbm[ti,fi])
        flo, fhi = int(round(freq_hz[f0])), int(round(freq_hz[f1]))
        mid = int(round(.5*(t0+t1)))
        left, right = t0==0, t1==nt-1
        rows.append({
            "t_start": None if left else str(time_utc[t0]), "t_mid": str(time_utc[mid]),
            "t_end": None if right else str(time_utc[t1]),
            "t_start_local": None if left else str(time_local[t0]), "t_mid_local": str(time_local[mid]),
            "t_end_local": None if right else str(time_local[t1]),
            "t_start_offset_sec": None if left else float(time_rel[t0]), "t_mid_offset_sec": float(time_rel[mid]),
            "t_end_offset_sec": None if right else float(time_rel[t1]),
            "duration_sec": None if (left or right) else float(time_rel[t1]-time_rel[t0]),
            "ongoing": bool(right), "time_structure": "continuous" if (left or right) else "bounded",
            "f_low_hz": flo, "f_high_hz": fhi, "f_center_hz": int(round(.5*(flo+fhi))),
            "bw_hz": int(fhi-flo), "band_tag": band_tag(flo,fhi), "intensity_kind": "mW",
            "intensity_value": float(np.nanmax(vals)), "n_pixels": int(submask.sum()),
            "n_time_pixels": int(len(np.unique(ti))), "n_freq_pixels": int(len(np.unique(fi))),
        })
    return rows


# -----------------------------------------------------------------------------
# Worker globals and one-scenario simulation
# -----------------------------------------------------------------------------
_W_BANK = None
_W_META = None
_W_SETTINGS = None


def worker_init(bank_path: str, bank_json: str, settings: dict[str, Any]):
    global _W_BANK, _W_META, _W_SETTINGS
    z = np.load(bank_path, allow_pickle=False)
    _W_BANK = {
        "offsets_hz": z["offsets_hz"], "spectra_ref_mw": z["spectra_ref_mw"],
        "template_valid": z["template_valid"], "template_relative_power_db": z["template_relative_power_db"],
    }
    _W_META = read_json(Path(bank_json))
    _W_SETTINGS = settings


def simulate_window(rx: Rx, tx: Tx, scenario: dict[str, Any], rf: dict[str, Any], order: int,
                    center_hz: float, center_loss_db: float, geom: dict[str, Any],
                    state_t: np.ndarray, template_t: np.ndarray, time_axes, rng: np.random.Generator,
                    return_spec: bool = False):
    freq = backend_freq_window(center_hz)
    if len(freq) < COARSE_FACTOR*2:
        return [], None, freq
    offsets = freq-center_hz
    nt, nf = len(state_t), len(freq)
    noise = thermal_noise_dbm(RBW_HZ, rx.noise_figure_db)

    # Receiver bandpass ripple uses absolute position in the full backend, so
    # local-window simulation remains consistent with a hypothetical full band.
    phase = rng.uniform(0, 2*np.pi)
    x = 8*np.pi*(freq-FREQ_MIN_HZ)/(FREQ_MAX_HZ-FREQ_MIN_HZ)
    bandpass = (BANDPASS_RIPPLE_DB*np.sin(x+phase) +
                .25*BANDPASS_RIPPLE_DB*np.sin(.27*x+.4*phase)).astype(np.float32)
    drift = gaussian_filter1d(rng.normal(size=nt), sigma=25.0)
    drift -= drift.mean()
    if drift.std() > 0:
        drift *= GAIN_DRIFT_DB/drift.std()
    drift = drift.astype(np.float32)
    shadow = slow_shadowing(nt, float(scenario["shadowing_sigma_db"]),
                            float(scenario["shadowing_correlation_s"]), rng)

    spec = np.empty((nt,nf), dtype=np.float32)
    for i0 in range(0,nt,64):
        i1=min(nt,i0+64)
        jit=rng.normal(0,NOISE_JITTER_DB,size=(i1-i0,nf)).astype(np.float32)
        spec[i0:i1] = noise+bandpass[None,:]+drift[i0:i1,None]+jit

    pi = int(scenario["rf_profile_id"])
    window_index = order-1
    gain = tx_gain_db(tx, geom["bearing_tx_to_rx_deg"], geom["elevation_tx_to_rx_deg"], order, rf)
    # Fast P.452 frequency interpolation: exact/basic loss at window center plus
    # the free-space 20log10(f/f_center) frequency slope across +/-40 MHz.
    path = center_loss_db + 20*np.log10(freq/center_hz)
    power_scale = 10**(float(scenario["tx_conducted_power_dbm"])/10)  # bank reference is 0 dBm = 1 mW
    ant_scale = 10**((gain + rx.antenna_gain_dbi - rx.excess_path_loss_db)/10)

    # Precompute received signal for each state/template only once.
    sig = {}
    for si,state in enumerate(STATE_NAMES):
        for ti in np.flatnonzero(_W_BANK["template_valid"][si] > 0):
            ref = _W_BANK["spectra_ref_mw"][pi,si,ti,window_index]
            tx_mw = np.interp(offsets, _W_BANK["offsets_hz"], ref, left=0.0, right=0.0)
            sig[(state,int(ti))] = tx_mw*power_scale*ant_scale*10**(-path/10)

    for state in STATE_NAMES:
        for ti in np.flatnonzero(_W_BANK["template_valid"][STATE_TO_INDEX[state]] > 0):
            idx=np.flatnonzero((state_t==state)&(template_t==ti))
            if len(idx)==0:
                continue
            smw = sig[(state,int(ti))][None,:] * 10**(shadow[idx,None]/10)
            spec[idx] = mw_to_dbm(dbm_to_mw(spec[idx])+smw).astype(np.float32)

    mask = fast_flagger(spec, freq/1e6)
    events = extract_events(mask,spec,freq,*time_axes)
    for e in events:
        e["preview_window_center_hz"] = float(center_hz)
        e["window_harmonic_order"] = int(order)
    return events, spec if return_spec else None, freq


def safe_record(e: dict[str, Any], scenario: dict[str, Any], rx: Rx, event_index: int):
    rec = {
        "scenario_id": scenario["scenario_id"], "split": scenario["split"],
        "receiver_name": rx.name, "event_index": int(event_index),
        "data_type": "spectrometer", "timezone": LOCAL_TIMEZONE,
        "Directional": False, "RA": 0, "Dec": 0, "Az": 0, "El": 0,
        "lat": rx.lat_deg, "lon": rx.lon_deg, "altitude": None,
        "file_freq_resolution_hz": CHANNEL_SPACING_HZ, "file_time_resolution_sec": TIME_RESOLUTION_S,
    }
    for k in EMILY_ML_FIELDS:
        if k in e:
            rec[k]=finite(e[k])
    # Harmonic order is observable only through frequency in real EMILY, so it is
    # intentionally NOT copied to the ML record.
    return rec


def maybe_qc_plot(path: Path, freq_hz: np.ndarray, spec: np.ndarray, title: str):
    if spec is None:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,5))
        ax.imshow(spec,aspect="auto",origin="upper",extent=[freq_hz[0]/1e6,freq_hz[-1]/1e6,spec.shape[0]/60,0])
        ax.set_xlabel("Frequency [MHz]"); ax.set_ylabel("Time [min]"); ax.set_title(title)
        fig.tight_layout(); path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=140); plt.close(fig)
    except Exception:
        pass


def run_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    try:
        settings = _W_SETTINGS
        cfg = settings["config"]
        rf = _W_META["profiles"][int(scenario["rf_profile_id"])]
        rng = np.random.default_rng(int(scenario["simulation_seed"]))
        tx = Tx(float(scenario["tx_lat_deg"]), float(scenario["tx_lon_deg"]),
                float(scenario["tx_height_agl_m"]), float(scenario["tx_conducted_power_dbm"]),
                float(scenario["tx_max_gain_dbi"]), float(scenario["tx_horizontal_hpbw_deg"]),
                float(scenario["tx_sector_azimuth_deg"]), float(scenario["tx_electrical_downtilt_deg"]))
        nt=int(scenario["duration_s"])
        t_rel,t_utc,t_local=make_time_axes(str(scenario["simulation_start_utc"]),nt)
        state_t,template_t=schedule_arrays(scenario["traffic_schedule"],nt,rng,_W_BANK["template_valid"])

        carrier=float(scenario["carrier_center_hz"])
        centers=[]
        orders=[]
        for order in (1,2,3):
            c=order*carrier
            if FREQ_MIN_HZ <= c <= FREQ_MAX_HZ:
                centers.append(c); orders.append(order)

        all_records=[]
        label_row={k:v for k,v in scenario.items() if k!="traffic_schedule"}
        label_row["traffic_schedule_json"]=scenario["traffic_schedule"]
        summary={"scenario_id":scenario["scenario_id"],"scenario_index":scenario["scenario_index"],
                 "split":scenario["split"],"status":"complete","n_emily_events":0}

        qc_this = int(settings.get("qc_every",0))>0 and int(scenario["scenario_index"])%int(settings["qc_every"])==0
        for rx in RXS:
            # One P.452 vector call per receiver for all in-band emission centers.
            geom, losses, terr = propagation_centers(tx,rx,centers,str(scenario["propagation"]),
                Path(settings["srtm_dir"]),bool(settings["srtm_download"]),float(scenario["p452_time_percent"]))
            prefix=rx.name.lower()
            label_row[f"{prefix}_distance_km"]=geom["distance_km"]
            label_row[f"{prefix}_bearing_tx_to_rx_deg"]=geom["bearing_tx_to_rx_deg"]
            label_row[f"{prefix}_elevation_tx_to_rx_deg"]=geom["elevation_tx_to_rx_deg"]
            if terr is not None:
                label_row[f"{prefix}_tx_ground_elevation_m"]=terr["tx_ground_elevation_m"]
                label_row[f"{prefix}_rx_ground_elevation_m"]=terr["rx_ground_elevation_m"]
            rx_events=[]
            for order,center in zip(orders,centers):
                loss=losses[float(center)]
                label_row[f"{prefix}_path_loss_h{order}_db"]=loss
                label_row[f"{prefix}_tx_gain_h{order}_db"]=tx_gain_db(tx,geom["bearing_tx_to_rx_deg"],geom["elevation_tx_to_rx_deg"],order,rf)
                ev,spec,freq=simulate_window(rx,tx,scenario,rf,order,center,loss,geom,state_t,template_t,
                                             (t_utc,t_local,t_rel),rng,
                                             return_spec=(qc_this and order==1))
                rx_events.extend(ev)
                if qc_this and order==1 and spec is not None:
                    qcp=Path(settings["qc_dir"])/f"{scenario['scenario_id']}_{rx.name}_fundamental.png"
                    maybe_qc_plot(qcp,freq,spec,f"{scenario['scenario_id']} {rx.name} fundamental")
            rx_events.sort(key=lambda e:(e.get("t_mid_offset_sec",0),e.get("f_center_hz",0)))
            for j,e in enumerate(rx_events):
                all_records.append(safe_record(e,scenario,rx,j))
            summary[f"n_events_{prefix}"]=len(rx_events)

        summary["n_emily_events"]=len(all_records)
        summary["has_detection"]=bool(all_records)
        label_row["n_emily_events"]=len(all_records)
        label_row["has_detection"]=bool(all_records)
        return finite({"summary":summary,"label":label_row,"records":all_records})
    except Exception:
        return {"summary":{"scenario_id":scenario.get("scenario_id"),"scenario_index":scenario.get("scenario_index"),
                           "split":scenario.get("split"),"status":"failed","error":traceback.format_exc()},
                "label":{k:v for k,v in scenario.items() if k!="traffic_schedule"},"records":[]}


# -----------------------------------------------------------------------------
# Batch assembly
# -----------------------------------------------------------------------------
def aggregate_results(results_dir: Path):
    result_files=sorted(results_dir.glob("scenario_*.json"))
    summaries=[]; labels=[]; records=[]
    for p in result_files:
        r=read_json(p)
        summaries.append(r["summary"]); labels.append(r["label"]); records.extend(r["records"])
    summaries.sort(key=lambda x:int(x.get("scenario_index",0)))
    labels.sort(key=lambda x:int(x.get("scenario_index",0)))
    records.sort(key=lambda x:(int(x["scenario_id"].split("_")[-1]),x["receiver_name"],int(x["event_index"])))
    return summaries,labels,records


def parse_args():
    p=argparse.ArgumentParser(description="Fast parallel EMILY-X terrestrial LTE ML dataset generator")
    p.add_argument("--n-datasets",type=int,required=False)
    p.add_argument("--workers",type=int,default=max(1,min(4,(os.cpu_count() or 2)//2)))
    p.add_argument("--output-dir",type=Path,default=Path("emilyx_ml_fast"))
    p.add_argument("--iq-dir",type=Path,default=Path("."))
    p.add_argument("--srtm-dir",type=Path,default=Path("srtm_data"))
    p.add_argument("--cache-dir",type=Path,default=None,
                   help="shared RF-template cache; default is <iq-dir>/.emilyx_rf_cache")
    p.add_argument("--seed",type=int,default=20260919)
    p.add_argument("--propagation",choices=["p452","fspl"],default="p452")
    p.add_argument("--rf-profiles",type=int,default=1,
                   help="number of RF-chain variants precomputed once; 1 is fastest/current baseline")
    p.add_argument("--config",type=Path,default=None,help="optional JSON override of embedded DEFAULT_CONFIG")
    p.add_argument("--write-default-config",type=Path,default=None)
    p.add_argument("--no-srtm-download",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--qc-every",type=int,default=0,
                   help="save a fundamental spectrogram PNG every N scenarios; 0 disables QC plots")
    return p.parse_args()


def deep_update(base: dict[str,Any], override: dict[str,Any]):
    out=json.loads(json.dumps(base))
    def rec(a,b):
        for k,v in b.items():
            if isinstance(v,dict) and isinstance(a.get(k),dict): rec(a[k],v)
            else: a[k]=v
    rec(out,override); return out


def main():
    args=parse_args()
    if args.write_default_config is not None:
        write_json(args.write_default_config,DEFAULT_CONFIG)
        print(f"Wrote {args.write_default_config}")
        if args.n_datasets is None:
            return 0
    if args.n_datasets is None or args.n_datasets <= 0:
        raise ValueError("--n-datasets must be > 0")
    if args.propagation=="p452" and not HAVE_PYCRAF and not args.dry_run:
        raise RuntimeError("--propagation p452 requires astropy + pycraf in this Python environment")

    cfg=deep_update(DEFAULT_CONFIG,read_json(args.config) if args.config else {})
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=True)
    results_dir=root/"results"; results_dir.mkdir(exist_ok=True)
    qc_dir=root/"qc"; qc_dir.mkdir(exist_ok=True)
    iq_dir=args.iq_dir.resolve(); srtm_dir=args.srtm_dir.resolve()
    cache=(args.cache_dir.resolve() if args.cache_dir is not None else iq_dir/".emilyx_rf_cache")

    profiles=make_rf_profiles(args.rf_profiles,args.seed,cfg)
    scenarios=sample_scenarios(args.n_datasets,args.seed,cfg,profiles,args.propagation)
    write_jsonl(root/"scenarios.jsonl",scenarios)
    write_json(root/"feature_schema.json",{
        "grouping_fields_not_features":["scenario_id","split","receiver_name","event_index"],
        "model_visible_event_fields":EMILY_ML_FIELDS,
        "label_file":"ground_truth/scenario_labels.csv",
        "warning":"Do not add scenario/ground-truth fields to model inputs. Split is by scenario."
    })

    if args.dry_run:
        write_csv(root/"ground_truth"/"scenario_labels.csv",[{k:v for k,v in s.items() if k!="traffic_schedule"} for s in scenarios])
        print(f"Dry run: sampled {len(scenarios)} scenarios -> {root}")
        return 0

    bank_path,bank_json=build_or_load_rf_bank(iq_dir,cache,profiles)
    write_json(root/"dataset_metadata.json",{
        "created_utc":utc_now(),"generator":"emilyx_ml_fast_parallel.py","version":VERSION,
        "n_datasets":args.n_datasets,"workers":args.workers,"seed":args.seed,
        "propagation":args.propagation,
        "p452_fast_approximation":"P.452 evaluated at emission centers; within each +/-40 MHz window only FSPL frequency slope is applied" if args.propagation=="p452" else None,
        "rf_profiles":profiles,"rf_bank":str(bank_path),"config":cfg,
        "iq_files":[{"name":f"srsran_{s}.cf32","sha256":sha256_file(iq_dir/f"srsran_{s}.cf32")} for s in STATE_NAMES],
        "receivers":list(RECEIVERS),
    })

    if not args.resume:
        existing=list(results_dir.glob("scenario_*.json"))
        if existing:
            raise FileExistsError(f"{results_dir} already has results; use --resume or a new --output-dir")

    pending=[]
    for s in scenarios:
        rp=results_dir/f"{s['scenario_id']}.json"
        if args.resume and rp.exists():
            continue
        pending.append(s)
    print(f"Scenarios: {len(scenarios)} total, {len(pending)} to run")
    print(f"Workers: {args.workers}; propagation={args.propagation}; RF profiles={len(profiles)}")

    settings={"config":cfg,"srtm_dir":str(srtm_dir),"srtm_download":not args.no_srtm_download,
              "qc_every":int(args.qc_every),"qc_dir":str(qc_dir)}

    if pending:
        with cf.ProcessPoolExecutor(max_workers=max(1,args.workers),initializer=worker_init,
                                    initargs=(str(bank_path),str(bank_json),settings)) as ex:
            futs={ex.submit(run_scenario,s):s for s in pending}
            done=0
            for fut in cf.as_completed(futs):
                s=futs[fut]; done+=1
                try: r=fut.result()
                except Exception:
                    r={"summary":{"scenario_id":s["scenario_id"],"scenario_index":s["scenario_index"],
                                  "split":s["split"],"status":"failed","error":traceback.format_exc()},
                       "label":{k:v for k,v in s.items() if k!="traffic_schedule"},"records":[]}
                write_json(results_dir/f"{s['scenario_id']}.json",r)
                st=r["summary"].get("status")
                nev=r["summary"].get("n_emily_events",0)
                print(f"[{done}/{len(pending)}] {s['scenario_id']} {st} events={nev}",flush=True)

    summaries,labels,records=aggregate_results(results_dir)
    write_csv(root/"dataset_index.csv",summaries)
    write_csv(root/"ground_truth"/"scenario_labels.csv",labels)
    write_jsonl(root/"ml_inputs"/"emily_records.jsonl",records)
    ok=sum(x.get("status")=="complete" for x in summaries)
    fail=len(summaries)-ok
    zero=sum(x.get("status")=="complete" and not x.get("has_detection") for x in summaries)
    print(f"Done: {ok} complete, {fail} failed, {zero} complete scenarios with zero EMILY detections")
    print(f"ML records: {root/'ml_inputs'/'emily_records.jsonl'}")
    print(f"Labels:     {root/'ground_truth'/'scenario_labels.csv'}")
    print(f"Index:      {root/'dataset_index.csv'}")
    return 0 if fail==0 else 2


if __name__=="__main__":
    raise SystemExit(main())
