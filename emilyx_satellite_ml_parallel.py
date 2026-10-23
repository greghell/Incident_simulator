#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMILY-X fast parallel satellite dataset generator
=================================================

Purpose
-------
Generate large machine-learning datasets of *one moving LEO satellite per
scenario* observed by an arbitrary number of fixed receivers.

This generator is deliberately optimized for thousands to millions of scenario
realizations.  It does not generate or save full 10-minute spectrogram cubes.
Instead it:

  1. loads genuine 5-MHz / 25-PRB srsRAN D2C IQ captures once;
  2. builds and caches a small bank of per-RBW LTE spectral templates;
  3. generates a new physically constrained circular LEO pass for every scenario;
  4. computes range/elevation/range-rate/Doppler and spot-beam gain to every receiver;
  5. evaluates individual 1-s spectra at selected times during common visibility;
  6. runs a 1-D, per-spectrum EMILY-like RFI detector independently at every receiver;
  7. retains only synchronized samples for which *all receivers detect RFI*;
  8. writes ML-safe EMILY records separately from satellite/orbit ground truth.

The synthetic-orbit mode is intended for high-volume ML training.  The existing
EMILY-X v3.2 TLE/SGP4 implementation remains the higher-fidelity validation path.
A useful workflow is to train on this fast generator and periodically compare a
random subset against the full v3.2 TLE implementation.

Trajectory model
----------------
Each scenario gets a different circular LEO trajectory.  Randomized quantities
include:

  * orbital altitude;
  * local ground-track heading;
  * closest-approach distance and bearing from the receiver-network centroid;
  * scenario epoch;
  * beam/EIRP and LTE load/template realization.

The orbit is propagated in an inertial frame with two-body circular motion and
rotated into ECEF using Earth rotation.  This is much faster than creating and
propagating a synthetic TLE for every realization while still producing a
smooth, physically consistent LEO trajectory, velocity and Doppler.

Per-time-sample EMILY records
-----------------------------
The old local preview grouped connected pixels across time into multi-second
incidents.  This generator intentionally does NOT do that.  One spectrum is one
measurement interval.  For every synchronized sample_id there is exactly one
retained EMILY record per receiver.  Thus with N receivers, each sample_id has N
rows in ml_inputs/emily_records.csv.gz.

The model-visible rows contain receiver/time/frequency/intensity information only.
Satellite truth is never copied into the ML-input file.

Outputs
-------
<output-dir>/
    dataset_metadata.json
    feature_schema.json
    dataset_index.csv
    ground_truth/scenario_labels.csv
    ground_truth/sample_labels.csv.gz
    ml_inputs/emily_records.csv.gz
    cache/d2c_template_bank_<hash>.npz
    cache/d2c_template_bank_<hash>.json

scenario_labels.csv:
    one row per accepted satellite/pass realization.

sample_labels.csv.gz:
    one row per synchronized time sample, containing true instantaneous satellite
    lat/lon/altitude, ECEF position/velocity, load state and orbit parameters.

emily_records.csv.gz:
    N receiver rows per synchronized sample_id.  These are the ML inputs.

Scaling
-------
For a first validation run, use all common detections or 32 samples/pass.  For a
million satellite realizations, do not retain hundreds of seconds per pass unless
that volume is truly needed.  For example, 1,000,000 scenarios x 32 synchronized
samples x 2 receivers is already 64 million EMILY rows.

Examples
--------
Two current EMILY-X receivers, 100 scenarios, 4 workers, 32 synchronized samples
per accepted pass:

  python emilyx_satellite_ml_parallel.py ^
    --n-datasets 100 ^
    --workers 4 ^
    --samples-per-scenario 32 ^
    --receiver RX1,39.5249,-114.373325 ^
    --receiver RX2,39.595518,-114.535418 ^
    --iq-dir . ^
    --output-dir emilyx_satellite_100

Add more receivers simply by repeating --receiver.

Receiver syntax:
    NAME,LAT,LON
    NAME,LAT,LON,HEIGHT_M
"""

from __future__ import annotations

# Keep each worker single-threaded internally.  Scenario-level multiprocessing is
# the intended parallelism.
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import concurrent.futures as cf
import csv
import gzip
import hashlib
import json
import math
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from pyproj import Geod, Transformer
from scipy import signal
from scipy.ndimage import median_filter, binary_dilation, label


VERSION = "0.1"
C_LIGHT = 299_792_458.0
MU_EARTH = 3.986004418e14       # m^3 s^-2
OMEGA_EARTH = 7.2921150e-5      # rad s^-1
EARTH_RADIUS_M = 6_371_008.8
GEOD = Geod(ellps="WGS84")
GEODETIC_TO_ECEF = Transformer.from_crs("EPSG:4979", "EPSG:4978", always_xy=True)
ECEF_TO_GEODETIC = Transformer.from_crs("EPSG:4978", "EPSG:4979", always_xy=True)

# D2C LTE / receiver backend assumptions inherited from v3.2.
LTE_CENTER_HZ = 1992.5e6
LTE_BW_HZ = 5e6
LTE_N_PRB = 25
IQ_SAMPLE_RATE_HZ = 5.76e6
RBW_HZ = 140e3
CHANNEL_SPACING_HZ = 70e3
TIME_RESOLUTION_S = 1.0
STATE_NAMES = ("idle", "light", "medium", "loaded")

# We only need a local window for per-spectrum RFI measurement.
LOCAL_HALFSPAN_HZ = 20e6
FREQ_OFFSETS_HZ = np.arange(
    -LOCAL_HALFSPAN_HZ,
    LOCAL_HALFSPAN_HZ + 0.5 * CHANNEL_SPACING_HZ,
    CHANNEL_SPACING_HZ,
    dtype=np.float64,
)

# Receiver/noise assumptions.
DEFAULT_RX_HEIGHT_M = 2.0
DEFAULT_RX_GAIN_DBI = 0.0
DEFAULT_RX_NOISE_FIGURE_DB = 6.0
REFERENCE_TEMP_K = 290.0
NOISE_JITTER_DB = 0.20
BANDPASS_RIPPLE_DB = 0.35

# Per-spectrum detector defaults.  These are intentionally separate from the old
# 2-D connected-component detector because time grouping is no longer desired.
DETECT_THRESHOLD_SIGMA = 8.0
DETECT_BG_WIDTH_CH = 201       # ~14.1 MHz; safely wider than a 5-MHz LTE carrier
DETECT_DILATE_FREQ_CH = 2
DETECT_MIN_FREQ_PIXELS = 8

# Genuine IQ template settings.
MAX_CAPTURE_S = 0.50
TEMPLATE_WINDOW_MS = 100.0
TEMPLATE_HOP_MS = 100.0
MAX_TEMPLATES_PER_STATE = 5

# Beam model defaults inherited from the v3.2 toy D2C example.
DEFAULT_PEAK_EIRP_DBM = 75.0
DEFAULT_BEAM_HPBW_DEG = 8.0
DEFAULT_SIDELOBE_FLOOR_DB = -30.0
DEFAULT_MAX_SCAN_LOSS_DB = 6.0


@dataclass(frozen=True)
class Rx:
    name: str
    lat_deg: float
    lon_deg: float
    height_agl_m: float = DEFAULT_RX_HEIGHT_M
    antenna_gain_dbi: float = DEFAULT_RX_GAIN_DBI
    noise_figure_db: float = DEFAULT_RX_NOISE_FIGURE_DB
    excess_path_loss_db: float = 0.0


@dataclass(frozen=True)
class OrbitScenario:
    altitude_km: float
    ground_heading_deg: float
    closest_approach_km: float
    closest_approach_bearing_deg: float
    closest_lat_deg: float
    closest_lon_deg: float
    peak_eirp_dbm: float
    beam_hpbw_deg: float
    sidelobe_floor_db: float
    max_scan_loss_db: float


# Worker globals initialized once per process.
G_BANK: dict[str, Any] | None = None
G_RECEIVERS: tuple[Rx, ...] = ()
G_CONFIG: dict[str, Any] = {}
G_BASE_EPOCH: datetime | None = None
G_LOCAL_TZ: ZoneInfo | None = None


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------

def finite(v: Any) -> Any:
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if math.isfinite(x) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return finite(v.tolist())
    if isinstance(v, dict):
        return {str(k): finite(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [finite(x) for x in v]
    return v


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def dbm_to_mw(x):
    return 10.0 ** (np.asarray(x, dtype=np.float64) / 10.0)


def mw_to_dbm(x, floor_dbm=-300.0):
    floor_mw = 10.0 ** (floor_dbm / 10.0)
    return 10.0 * np.log10(np.maximum(np.asarray(x, dtype=np.float64), floor_mw))


def thermal_noise_floor_dbm(rbw_hz: float, nf_db: float) -> float:
    k = 1.380649e-23
    watts = k * REFERENCE_TEMP_K * rbw_hz
    return float(10.0 * np.log10(watts / 1e-3) + nf_db)


def free_space_loss_db(range_m, freq_hz):
    r = np.maximum(np.asarray(range_m, dtype=np.float64), 1.0)
    f = np.maximum(np.asarray(freq_hz, dtype=np.float64), 1.0)
    return 20.0 * np.log10(4.0 * np.pi * r * f / C_LIGHT)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def band_tag(f_low_hz: float, f_high_hz: float) -> str | None:
    bands = [
        ("VHF", 30e6, 300e6), ("UHF", 300e6, 1e9), ("L", 1e9, 2e9),
        ("S", 2e9, 4e9), ("C", 4e9, 8e9), ("X", 8e9, 12e9),
    ]
    names = [n for n, lo, hi in bands if f_high_hz >= lo and f_low_hz <= hi]
    return "/".join(names) if names else None


def parse_receiver(spec: str, idx: int) -> Rx:
    p = [x.strip() for x in spec.split(",")]
    if len(p) not in (3, 4):
        raise ValueError("--receiver must be NAME,LAT,LON or NAME,LAT,LON,HEIGHT_M")
    name = p[0] or f"RX{idx}"
    lat = float(p[1]); lon = float(p[2])
    h = float(p[3]) if len(p) == 4 else DEFAULT_RX_HEIGHT_M
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid receiver coordinates: {spec}")
    return Rx(name=name, lat_deg=lat, lon_deg=lon, height_agl_m=h)


def ecef_from_geodetic(lat_deg: float, lon_deg: float, alt_m: float = 0.0) -> np.ndarray:
    x, y, z = GEODETIC_TO_ECEF.transform(lon_deg, lat_deg, alt_m)
    return np.array([x, y, z], dtype=np.float64)


def local_axes(lat_deg: float, lon_deg: float):
    lat = np.deg2rad(lat_deg); lon = np.deg2rad(lon_deg)
    east = np.array([-np.sin(lon), np.cos(lon), 0.0])
    north = np.array([-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)])
    up = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    return east, north, up


def receiver_centroid(receivers: tuple[Rx, ...]) -> tuple[float, float]:
    # ECEF-vector centroid is robust to longitude wrap and adequate for a local network.
    v = np.mean(np.stack([ecef_from_geodetic(r.lat_deg, r.lon_deg, 0.0) for r in receivers]), axis=0)
    lon, lat, _ = ECEF_TO_GEODETIC.transform(v[0], v[1], v[2])
    return float(lat), float(lon)


def angle_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    c = np.sum(a*b, axis=1) / np.maximum(den, 1e-30)
    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))


def rz_minus(theta: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Apply Rz(-theta) row-by-row."""
    c = np.cos(theta); s = np.sin(theta)
    x = c*xyz[:, 0] + s*xyz[:, 1]
    y = -s*xyz[:, 0] + c*xyz[:, 1]
    return np.column_stack([x, y, xyz[:, 2]])


# -----------------------------------------------------------------------------
# Genuine D2C IQ -> reusable spectral-template bank
# -----------------------------------------------------------------------------

def normalized_psd(iq: np.ndarray, fs: float, nperseg: int = 16384):
    nperseg = min(nperseg, len(iq))
    noverlap = nperseg // 2
    f, p = signal.welch(
        iq, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap,
        detrend=False, return_onesided=False, scaling="density",
    )
    f = np.fft.fftshift(f)
    p = np.fft.fftshift(np.maximum(p, 0.0))
    area = np.trapezoid(p, f)
    if not np.isfinite(area) or area <= 0:
        raise RuntimeError("Could not normalize IQ PSD")
    return f.astype(np.float64), (p / area).astype(np.float64)


def rbw_fraction(offset_centers_hz: np.ndarray, psd_f: np.ndarray, psd: np.ndarray) -> np.ndarray:
    # Numerical CDF without scipy cumulative_trapezoid dependency.
    df = np.diff(psd_f)
    trap = 0.5 * (psd[:-1] + psd[1:]) * df
    cdf = np.concatenate([[0.0], np.cumsum(trap)])
    cdf /= max(cdf[-1], 1e-30)
    lo = offset_centers_hz - RBW_HZ/2.0
    hi = offset_centers_hz + RBW_HZ/2.0
    a = np.interp(lo, psd_f, cdf, left=0.0, right=1.0)
    b = np.interp(hi, psd_f, cdf, left=0.0, right=1.0)
    return np.maximum(b-a, 0.0)


def build_template_bank(iq_dir: Path, cache_dir: Path) -> tuple[Path, dict[str, Any]]:
    paths = {s: iq_dir / f"srsran_d2c_{s}.cf32" for s in STATE_NAMES}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing genuine D2C IQ file(s):\n  " + "\n  ".join(missing))

    sig = {
        "version": VERSION,
        "files": {s: {"path": str(paths[s].resolve()), "sha256": sha256_file(paths[s])} for s in STATE_NAMES},
        "fs": IQ_SAMPLE_RATE_HZ,
        "window_ms": TEMPLATE_WINDOW_MS,
        "hop_ms": TEMPLATE_HOP_MS,
        "max_templates": MAX_TEMPLATES_PER_STATE,
        "rbw_hz": RBW_HZ,
        "offsets_hash": hashlib.sha256(FREQ_OFFSETS_HZ.tobytes()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz = cache_dir / f"d2c_template_bank_{key}.npz"
    meta_path = cache_dir / f"d2c_template_bank_{key}.json"
    if npz.exists() and meta_path.exists():
        return npz, json.loads(meta_path.read_text(encoding="utf-8"))

    full_rms = {}
    raw = {}
    max_n = int(round(MAX_CAPTURE_S * IQ_SAMPLE_RATE_HZ))
    for s, p in paths.items():
        x = np.fromfile(p, dtype=np.complex64, count=max_n)
        if len(x) < int(IQ_SAMPLE_RATE_HZ * 0.1):
            raise RuntimeError(f"IQ file too short: {p}")
        raw[s] = x
        full_rms[s] = float(np.sqrt(np.mean(np.abs(x.astype(np.complex128))**2)))
    ref = full_rms["loaded"]

    template_power_mw = []
    template_state = []
    template_index = []
    template_rel_db = []
    win_n = int(round(TEMPLATE_WINDOW_MS * 1e-3 * IQ_SAMPLE_RATE_HZ))
    hop_n = int(round(TEMPLATE_HOP_MS * 1e-3 * IQ_SAMPLE_RATE_HZ))

    for s in STATE_NAMES:
        x = raw[s]
        starts = list(range(0, max(1, len(x)-win_n+1), hop_n))[:MAX_TEMPLATES_PER_STATE]
        if not starts:
            starts = [0]
        for ti, i0 in enumerate(starts):
            w = x[i0:i0+win_n]
            if len(w) < win_n:
                continue
            rms = float(np.sqrt(np.mean(np.abs(w.astype(np.complex128))**2)))
            rel_db = 20.0*np.log10(max(rms,1e-30)/max(ref,1e-30))
            # Normalize before PSD estimation; absolute state/template power is restored separately.
            wn = w.astype(np.complex128)
            wn -= np.mean(wn)
            wn /= max(np.sqrt(np.mean(np.abs(wn)**2)), 1e-30)
            f, pxx = normalized_psd(wn.astype(np.complex64), IQ_SAMPLE_RATE_HZ)
            frac = rbw_fraction(FREQ_OFFSETS_HZ, f, pxx)
            template_power_mw.append(frac.astype(np.float32))
            template_state.append(s)
            template_index.append(ti)
            template_rel_db.append(rel_db)

    arr = np.stack(template_power_mw).astype(np.float32)
    np.savez_compressed(
        npz,
        fractions=arr,
        offsets_hz=FREQ_OFFSETS_HZ,
        state=np.asarray(template_state, dtype="U16"),
        template_index=np.asarray(template_index, dtype=np.int16),
        relative_power_db=np.asarray(template_rel_db, dtype=np.float32),
    )
    meta = {
        **sig,
        "cache_key": key,
        "template_count": int(len(arr)),
        "state_rms": full_rms,
        "note": "Genuine srsRAN D2C IQ PSD templates; fractions are power captured per 140-kHz RBW.",
    }
    write_json(meta_path, meta)
    return npz, meta


def load_bank(npz_path: Path) -> dict[str, Any]:
    d = np.load(npz_path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    out["by_state"] = {}
    for s in STATE_NAMES:
        out["by_state"][s] = np.flatnonzero(out["state"] == s)
        if not len(out["by_state"][s]):
            raise RuntimeError(f"Template bank has no state {s}")
    return out


# -----------------------------------------------------------------------------
# Fast physically consistent synthetic circular LEO pass
# -----------------------------------------------------------------------------

def sample_orbit(rng: np.random.Generator, receivers: tuple[Rx, ...], cfg: dict[str, Any]) -> OrbitScenario:
    lat0, lon0 = receiver_centroid(receivers)
    alt = float(rng.uniform(*cfg["altitude_km"]))
    heading = float(rng.uniform(0.0, 360.0))
    ca = float(rng.uniform(*cfg["closest_approach_km"]))
    ca_bearing = float(rng.uniform(0.0, 360.0))
    lonc, latc, _ = GEOD.fwd(lon0, lat0, ca_bearing, ca*1000.0)
    return OrbitScenario(
        altitude_km=alt,
        ground_heading_deg=heading,
        closest_approach_km=ca,
        closest_approach_bearing_deg=ca_bearing,
        closest_lat_deg=float(latc),
        closest_lon_deg=float(lonc),
        peak_eirp_dbm=float(rng.uniform(*cfg["peak_eirp_dbm"])),
        beam_hpbw_deg=float(rng.uniform(*cfg["beam_hpbw_deg"])),
        sidelobe_floor_db=float(cfg["sidelobe_floor_db"]),
        max_scan_loss_db=float(cfg["max_scan_loss_db"]),
    )


def propagate_circular_pass(orb: OrbitScenario, t_s: np.ndarray) -> dict[str, np.ndarray]:
    # Initial sub-satellite radial direction and local desired heading in ECEF.
    r0_ecef_ground = ecef_from_geodetic(orb.closest_lat_deg, orb.closest_lon_deg, 0.0)
    u0 = r0_ecef_ground / np.linalg.norm(r0_ecef_ground)
    east, north, _ = local_axes(orb.closest_lat_deg, orb.closest_lon_deg)
    hd = np.deg2rad(orb.ground_heading_deg)
    tangent = np.cos(hd)*north + np.sin(hd)*east
    tangent = tangent - np.dot(tangent, u0)*u0
    tangent /= np.linalg.norm(tangent)

    rmag = EARTH_RADIUS_M + orb.altitude_km*1000.0
    n = math.sqrt(MU_EARTH / rmag**3)
    phase = n * t_s
    cp = np.cos(phase)[:,None]; sp = np.sin(phase)[:,None]
    r_eci = rmag * (cp*u0[None,:] + sp*tangent[None,:])
    v_eci = rmag*n * (-sp*u0[None,:] + cp*tangent[None,:])

    theta = OMEGA_EARTH * t_s
    r_ecef = rz_minus(theta, r_eci)
    # dr_ecef/dt = Rz(-wt) * (v_eci - Omega x r_eci)
    omega_cross_r = np.column_stack([
        -OMEGA_EARTH*r_eci[:,1],
         OMEGA_EARTH*r_eci[:,0],
         np.zeros(len(t_s)),
    ])
    v_ecef = rz_minus(theta, v_eci - omega_cross_r)

    lon, lat, alt = ECEF_TO_GEODETIC.transform(r_ecef[:,0], r_ecef[:,1], r_ecef[:,2])
    return {
        "xyz_m": r_ecef,
        "vel_mps": v_ecef,
        "lat_deg": np.asarray(lat),
        "lon_deg": np.asarray(lon),
        "alt_km": np.asarray(alt)/1000.0,
        "orbital_angular_rate_rad_s": np.full(len(t_s), n),
    }


def link_geometry(traj: dict[str, np.ndarray], rx: Rx) -> dict[str, np.ndarray]:
    site = ecef_from_geodetic(rx.lat_deg, rx.lon_deg, rx.height_agl_m)
    delta = traj["xyz_m"] - site[None,:]
    rng_m = np.linalg.norm(delta, axis=1)
    los = delta / np.maximum(rng_m[:,None], 1.0)
    east, north, up = local_axes(rx.lat_deg, rx.lon_deg)
    e = delta@east; n = delta@north; ucomp = delta@up
    az = (np.rad2deg(np.arctan2(e,n))+360.0)%360.0
    el = np.rad2deg(np.arctan2(ucomp, np.hypot(e,n)))
    rr = np.sum(traj["vel_mps"]*los, axis=1)
    doppler = -LTE_CENTER_HZ*rr/C_LIGHT
    return {
        "range_m": rng_m,
        "range_km": rng_m/1000.0,
        "azimuth_deg": az,
        "elevation_deg": el,
        "range_rate_mps": rr,
        "doppler_hz": doppler,
    }


def beam_geometry(orb: OrbitScenario, traj: dict[str,np.ndarray], rx: Rx,
                  target_lat: float, target_lon: float) -> dict[str,np.ndarray]:
    sat = traj["xyz_m"]
    rx_xyz = ecef_from_geodetic(rx.lat_deg, rx.lon_deg, rx.height_agl_m)
    target_xyz = ecef_from_geodetic(target_lat, target_lon, 0.0)
    rv = rx_xyz[None,:] - sat
    tv = target_xyz[None,:] - sat
    nadir = -sat
    off = angle_rows(tv, rv)
    scan = angle_rows(tv, nadir)
    att = 12.0*(off/max(orb.beam_hpbw_deg,1e-6))**2
    rel = -np.minimum(att, abs(orb.sidelobe_floor_db))
    scan_loss = -np.minimum(orb.max_scan_loss_db, 3.0*(scan/60.0)**2)
    return {"offaxis_deg":off, "scan_angle_deg":scan, "relative_gain_db":rel+scan_loss}


# -----------------------------------------------------------------------------
# Traffic/template scheduling
# -----------------------------------------------------------------------------

def sample_state(rng: np.random.Generator, cfg: dict[str,Any]) -> str:
    probs = cfg["state_probabilities"]
    states = list(probs)
    p = np.asarray([probs[s] for s in states], dtype=float); p /= p.sum()
    return str(rng.choice(states, p=p))


def select_template_index(rng: np.random.Generator, bank: dict[str,Any], state: str) -> int:
    return int(rng.choice(bank["by_state"][state]))


# -----------------------------------------------------------------------------
# One-spectrum receiver model + one-spectrum RFI detector
# -----------------------------------------------------------------------------

def shift_template_fraction(bank: dict[str,Any], bank_i: int, doppler_hz: float) -> np.ndarray:
    src_off = bank["offsets_hz"]
    frac = bank["fractions"][bank_i]
    # Output offset f samples the unshifted template at f-doppler.
    return np.interp(FREQ_OFFSETS_HZ - doppler_hz, src_off, frac, left=0.0, right=0.0)


def synthesize_spectrum(rx: Rx, orb: OrbitScenario, link: dict[str,np.ndarray],
                        beam: dict[str,np.ndarray], t_idx: int,
                        bank: dict[str,Any], bank_i: int,
                        rng: np.random.Generator) -> tuple[np.ndarray, dict[str,float]]:
    freq = LTE_CENTER_HZ + FREQ_OFFSETS_HZ
    noise_dbm = thermal_noise_floor_dbm(RBW_HZ, rx.noise_figure_db)

    # Smooth receiver bandpass + independent RBW jitter for this spectrum.
    x = np.linspace(0.0, 4.0*np.pi, len(freq))
    phase = rng.uniform(0.0, 2*np.pi)
    bandpass = BANDPASS_RIPPLE_DB*np.sin(x+phase) + 0.25*BANDPASS_RIPPLE_DB*np.sin(0.31*x+0.3*phase)
    jitter = rng.normal(0.0, NOISE_JITTER_DB, size=len(freq))
    spec_dbm = noise_dbm + bandpass + jitter

    dop = float(link["doppler_hz"][t_idx])
    frac = shift_template_fraction(bank, bank_i, dop)
    rel_db = float(bank["relative_power_db"][bank_i])
    tx_total_dbm = orb.peak_eirp_dbm + rel_db
    tx_bin_mw = dbm_to_mw(tx_total_dbm) * frac

    fspl = free_space_loss_db(float(link["range_m"][t_idx]), freq)
    scale_db = float(beam["relative_gain_db"][t_idx]) + rx.antenna_gain_dbi - rx.excess_path_loss_db - fspl
    rx_sig_mw = tx_bin_mw * 10.0**(scale_db/10.0)
    spec_mw = dbm_to_mw(spec_dbm) + rx_sig_mw
    spec_dbm = mw_to_dbm(spec_mw).astype(np.float32)

    diag = {
        "noise_floor_dbm": noise_dbm,
        "rx_total_signal_dbm": float(mw_to_dbm(np.sum(rx_sig_mw))),
        "peak_signal_bin_dbm": float(mw_to_dbm(np.max(rx_sig_mw))),
        "doppler_hz": dop,
        "beam_gain_relative_db": float(beam["relative_gain_db"][t_idx]),
        "fspl_center_db": float(free_space_loss_db(float(link["range_m"][t_idx]), LTE_CENTER_HZ)),
    }
    return spec_dbm, diag


def detect_one_spectrum(spec_dbm: np.ndarray, freq_hz: np.ndarray,
                        threshold_sigma: float = DETECT_THRESHOLD_SIGMA) -> dict[str,Any] | None:
    """Return strongest contiguous RFI region in one spectrum, or None."""
    x = np.asarray(spec_dbm, dtype=np.float64)
    # Median background is deliberately much wider than the LTE carrier.
    bg = median_filter(x, size=DETECT_BG_WIDTH_CH, mode="nearest")
    resid = x - bg
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid-med)))
    sigma = max(1.4826*mad, 0.05)
    mask = resid > threshold_sigma*sigma
    if DETECT_DILATE_FREQ_CH:
        mask = binary_dilation(mask, structure=np.ones(2*DETECT_DILATE_FREQ_CH+1, dtype=bool))
    labs, nlab = label(mask)
    if nlab == 0:
        return None

    best = None
    for lab_id in range(1, nlab+1):
        idx = np.flatnonzero(labs == lab_id)
        if len(idx) < DETECT_MIN_FREQ_PIXELS:
            continue
        # Use integrated excess linear power to choose the primary component.
        excess_mw = np.maximum(dbm_to_mw(x[idx]) - dbm_to_mw(bg[idx]), 0.0)
        score = float(np.sum(excess_mw))
        if best is None or score > best[0]:
            best = (score, idx)
    if best is None:
        return None

    idx = best[1]
    f0 = float(freq_hz[idx.min()]); f1 = float(freq_hz[idx.max()])
    vals_mw = dbm_to_mw(x[idx])
    return {
        "f_low_hz": int(round(f0)),
        "f_high_hz": int(round(f1)),
        "f_center_hz": int(round(0.5*(f0+f1))),
        "bw_hz": int(round(f1-f0)),
        "band_tag": band_tag(f0,f1),
        "intensity_kind": "mW",
        "intensity_value": float(np.max(vals_mw)),
        "peak_power_mw": float(np.max(vals_mw)),
        "mean_power_mw": float(np.mean(vals_mw)),
        "sum_power_mw": float(np.sum(vals_mw)),
        "n_freq_pixels": int(len(idx)),
        "detector_sigma_db": float(sigma),
        "detector_peak_resid_db": float(np.max(resid[idx])),
    }


def emily_record(rx: Rx, det: dict[str,Any], scenario_id: str, split: str,
                 sample_id: str, time_index: int, t_mid: datetime,
                 integration_s: float, receiver_index: int) -> dict[str,Any]:
    t0 = t_mid - timedelta(seconds=integration_s/2)
    t1 = t_mid + timedelta(seconds=integration_s/2)
    local = G_LOCAL_TZ or ZoneInfo("UTC")
    return {
        # Grouping / bookkeeping: not ML features.
        "scenario_id": scenario_id,
        "sample_id": sample_id,
        "split": split,
        "time_index": int(time_index),
        "receiver_index": int(receiver_index),
        "receiver_name": rx.name,

        # EMILY-like model-visible fields.
        "data_type": "spectrometer",
        "timezone": str(local.key),
        "t_start": iso_z(t0),
        "t_mid": iso_z(t_mid),
        "t_end": iso_z(t1),
        "t_start_local": t0.astimezone(local).isoformat(),
        "t_mid_local": t_mid.astimezone(local).isoformat(),
        "t_end_local": t1.astimezone(local).isoformat(),
        "t_start_offset_sec": None,  # sample-relative timing is represented by time_index
        "t_mid_offset_sec": None,
        "t_end_offset_sec": None,
        "duration_sec": float(integration_s),
        "ongoing": False,
        "time_structure": "single_spectrum",
        "f_low_hz": det["f_low_hz"],
        "f_high_hz": det["f_high_hz"],
        "f_center_hz": det["f_center_hz"],
        "bw_hz": det["bw_hz"],
        "band_tag": det["band_tag"],
        "intensity_kind": "mW",
        "intensity_value": det["intensity_value"],
        "Directional": False,
        "RA": 0.0,
        "Dec": 0.0,
        "Az": 0.0,
        "El": 0.0,
        "lat": rx.lat_deg,
        "lon": rx.lon_deg,
        "altitude": None,
        "file_freq_resolution_hz": CHANNEL_SPACING_HZ,
        "file_time_resolution_sec": integration_s,
        "n_time_pixels": 1,
        "n_freq_pixels": det["n_freq_pixels"],
        "n_pixels": det["n_freq_pixels"],
        "method": "EMILY-X per-spectrum robust median/MAD detector",
    }


# -----------------------------------------------------------------------------
# Scenario generation / acceptance
# -----------------------------------------------------------------------------

def deterministic_split(index: int, seed: int, split_cfg: dict[str,float]) -> str:
    h = hashlib.blake2b(f"{seed}:{index}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "little") / 2**64
    tr = float(split_cfg["train"]); va = float(split_cfg["validation"]); te = float(split_cfg["test"])
    s = tr+va+te; tr/=s; va/=s
    if u < tr: return "train"
    if u < tr+va: return "validation"
    return "test"


def candidate_time_indices(common_visible: np.ndarray, requested: int, oversample: int,
                           rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(common_visible)
    if requested <= 0 or len(idx) <= requested*oversample:
        return idx
    n = min(len(idx), max(requested*oversample, requested))
    # Stratified selection across the pass preserves approach/closest/recede geometry.
    edges = np.linspace(0, len(idx), n+1, dtype=int)
    picks = []
    for a,b in zip(edges[:-1], edges[1:]):
        if b > a:
            picks.append(int(idx[rng.integers(a,b)]))
    return np.asarray(sorted(set(picks)), dtype=int)


def run_one_scenario(index: int) -> dict[str,Any]:
    assert G_BANK is not None and G_BASE_EPOCH is not None
    cfg = G_CONFIG
    scenario_id = f"scenario_{index:09d}"
    split = deterministic_split(index, int(cfg["seed"]), cfg["split"])
    # Independent reproducible seed per scenario.
    seed_bytes = hashlib.blake2b(f"{cfg['seed']}:{index}".encode(), digest_size=8).digest()
    scenario_seed = int.from_bytes(seed_bytes, "little") & 0x7FFFFFFF
    rng = np.random.default_rng(scenario_seed)

    duration = float(cfg["trajectory"]["duration_s"])
    dt = float(cfg["trajectory"]["time_resolution_s"])
    nt = int(round(duration/dt))
    # These are sample-center times relative to closest-approach epoch t=0.
    t_rel = (np.arange(nt, dtype=np.float64) + 0.5)*dt - duration/2.0

    centroid_lat, centroid_lon = receiver_centroid(G_RECEIVERS)
    target_xyz = ecef_from_geodetic(centroid_lat, centroid_lon, 0.0)

    max_attempts = int(cfg["acceptance"]["max_attempts"])
    min_el = float(cfg["acceptance"]["min_elevation_deg"])
    requested = int(cfg["output"]["samples_per_scenario"])
    min_common = int(cfg["acceptance"]["min_common_detected_samples"])
    oversample = int(cfg["acceptance"]["candidate_oversample"])

    last_reason = "not attempted"
    for attempt in range(1, max_attempts+1):
        orb = sample_orbit(rng, G_RECEIVERS, cfg["orbit"])
        traj = propagate_circular_pass(orb, t_rel)
        links = {rx.name: link_geometry(traj, rx) for rx in G_RECEIVERS}
        common_vis = np.ones(nt, dtype=bool)
        for rx in G_RECEIVERS:
            common_vis &= links[rx.name]["elevation_deg"] >= min_el
        if np.count_nonzero(common_vis) < min_common:
            last_reason = f"only {np.count_nonzero(common_vis)} common-visible samples"
            continue

        beams = {}
        sat = traj["xyz_m"]
        target_vec = target_xyz[None,:]-sat
        target_vis_ok = np.ones(nt, dtype=bool)
        # Beam center is always steered at receiver-network centroid.
        for rx in G_RECEIVERS:
            beams[rx.name] = beam_geometry(orb, traj, rx, centroid_lat, centroid_lon)

        cand = candidate_time_indices(common_vis, requested if requested>0 else min_common, oversample, rng)
        if len(cand) < min_common:
            last_reason = "too few candidate samples"
            continue

        # Random scenario epoch to prevent timestamp memorization.  Closest approach
        # occurs near epoch_center.
        days = float(rng.uniform(*cfg["epoch_random_days"]))
        epoch_center = G_BASE_EPOCH + timedelta(days=days)

        accepted_samples = []
        for ti in cand:
            state = sample_state(rng, cfg["traffic"])
            bank_i = select_template_index(rng, G_BANK, state)
            per_rx = []
            all_detect = True
            rx_truth = {}
            for irx, rx in enumerate(G_RECEIVERS):
                spec, diag = synthesize_spectrum(rx, orb, links[rx.name], beams[rx.name], ti, G_BANK, bank_i, rng)
                freq = LTE_CENTER_HZ + FREQ_OFFSETS_HZ
                det = detect_one_spectrum(spec, freq, float(cfg["detector"]["threshold_sigma"]))
                if det is None:
                    all_detect = False
                    break
                t_mid = epoch_center + timedelta(seconds=float(t_rel[ti]))
                sample_id = f"{scenario_id}:t{int(ti):04d}"
                rec = emily_record(rx, det, scenario_id, split, sample_id, int(ti), t_mid, dt, irx)
                # Diagnostics remain ground truth, not in ML records.
                rx_truth[rx.name] = {
                    "azimuth_deg": float(links[rx.name]["azimuth_deg"][ti]),
                    "elevation_deg": float(links[rx.name]["elevation_deg"][ti]),
                    "slant_range_km": float(links[rx.name]["range_km"][ti]),
                    "range_rate_mps": float(links[rx.name]["range_rate_mps"][ti]),
                    "doppler_hz": float(links[rx.name]["doppler_hz"][ti]),
                    "beam_offaxis_deg": float(beams[rx.name]["offaxis_deg"][ti]),
                    "beam_scan_angle_deg": float(beams[rx.name]["scan_angle_deg"][ti]),
                    "beam_gain_relative_db": float(beams[rx.name]["relative_gain_db"][ti]),
                    **diag,
                }
                per_rx.append(rec)
            if not all_detect:
                continue

            x,y,z = traj["xyz_m"][ti]; vx,vy,vz = traj["vel_mps"][ti]
            t_mid = epoch_center + timedelta(seconds=float(t_rel[ti]))
            sample_id = f"{scenario_id}:t{int(ti):04d}"
            sample_truth = {
                "scenario_id": scenario_id,
                "sample_id": sample_id,
                "split": split,
                "time_index": int(ti),
                "t_mid": iso_z(t_mid),
                "sat_lat_deg": float(traj["lat_deg"][ti]),
                "sat_lon_deg": float(traj["lon_deg"][ti]),
                "sat_alt_km": float(traj["alt_km"][ti]),
                "sat_ecef_x_m": float(x), "sat_ecef_y_m": float(y), "sat_ecef_z_m": float(z),
                "sat_ecef_vx_mps": float(vx), "sat_ecef_vy_mps": float(vy), "sat_ecef_vz_mps": float(vz),
                "lte_state": state,
                "template_bank_index": int(bank_i),
                "template_relative_power_db": float(G_BANK["relative_power_db"][bank_i]),
                "receiver_truth_json": json.dumps(finite(rx_truth), separators=(",",":")),
            }
            accepted_samples.append((sample_truth, per_rx))
            if requested > 0 and len(accepted_samples) >= requested:
                break

        needed = requested if requested > 0 else min_common
        if len(accepted_samples) < min_common or (requested>0 and len(accepted_samples)<requested):
            last_reason = f"only {len(accepted_samples)} simultaneous detections"
            continue

        scenario_truth = {
            "scenario_id": scenario_id,
            "scenario_index": int(index),
            "split": split,
            "simulation_seed": int(scenario_seed),
            "attempt": int(attempt),
            "n_receivers": len(G_RECEIVERS),
            "receiver_names": ";".join(rx.name for rx in G_RECEIVERS),
            "network_centroid_lat_deg": centroid_lat,
            "network_centroid_lon_deg": centroid_lon,
            "orbit_altitude_km": orb.altitude_km,
            "ground_heading_deg": orb.ground_heading_deg,
            "closest_approach_km": orb.closest_approach_km,
            "closest_approach_bearing_deg": orb.closest_approach_bearing_deg,
            "closest_lat_deg": orb.closest_lat_deg,
            "closest_lon_deg": orb.closest_lon_deg,
            "peak_eirp_dbm": orb.peak_eirp_dbm,
            "beam_hpbw_deg": orb.beam_hpbw_deg,
            "sidelobe_floor_db": orb.sidelobe_floor_db,
            "max_scan_loss_db": orb.max_scan_loss_db,
            "min_elevation_deg": min_el,
            "common_visible_samples": int(np.count_nonzero(common_vis)),
            "retained_common_detected_samples": int(len(accepted_samples)),
            "epoch_center_utc": iso_z(epoch_center),
            "trajectory_model": "synthetic circular two-body LEO + Earth rotation",
            "lte_center_hz": LTE_CENTER_HZ,
            "lte_bw_hz": LTE_BW_HZ,
            "lte_n_prb": LTE_N_PRB,
        }
        return {
            "ok": True,
            "scenario": finite(scenario_truth),
            "samples": [finite(x[0]) for x in accepted_samples],
            "records": [finite(r) for _, rows in accepted_samples for r in rows],
        }

    return {
        "ok": False,
        "scenario_id": scenario_id,
        "scenario_index": index,
        "split": split,
        "simulation_seed": scenario_seed,
        "error": f"No acceptable pass after {max_attempts} attempts: {last_reason}",
    }


# -----------------------------------------------------------------------------
# Worker initialization
# -----------------------------------------------------------------------------

def worker_init(bank_path: str, receivers_json: str, config_json: str, base_epoch: str, timezone_name: str):
    global G_BANK, G_RECEIVERS, G_CONFIG, G_BASE_EPOCH, G_LOCAL_TZ
    G_BANK = load_bank(Path(bank_path))
    G_RECEIVERS = tuple(Rx(**x) for x in json.loads(receivers_json))
    G_CONFIG = json.loads(config_json)
    G_BASE_EPOCH = datetime.fromisoformat(base_epoch.replace("Z","+00:00"))
    G_LOCAL_TZ = ZoneInfo(timezone_name)


# -----------------------------------------------------------------------------
# Streaming output writers
# -----------------------------------------------------------------------------

def csv_row_writer(path: Path, fieldnames: list[str], gzip_output: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if gzip_output:
        fh = gzip.open(path, "wt", newline="", encoding="utf-8", compresslevel=5)
    else:
        fh = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    return fh, writer


# -----------------------------------------------------------------------------
# CLI / main
# -----------------------------------------------------------------------------

def default_config(seed: int, samples_per_scenario: int) -> dict[str,Any]:
    return {
        "seed": int(seed),
        "split": {"train":0.70,"validation":0.15,"test":0.15},
        "epoch_random_days": [0.0, 365.0],
        "trajectory": {"duration_s": 360.0, "time_resolution_s": 1.0},
        "orbit": {
            "altitude_km": [400.0, 650.0],
            "closest_approach_km": [0.0, 40.0],
            "peak_eirp_dbm": [74.0, 80.0],
            "beam_hpbw_deg": [8.0, 16.0],
            "sidelobe_floor_db": -30.0,
            "max_scan_loss_db": 6.0,
        },
        "acceptance": {
            "min_elevation_deg": 20.0,
            "min_common_detected_samples": max(1, min(16, samples_per_scenario if samples_per_scenario>0 else 16)),
            "candidate_oversample": 3,
            "max_attempts": 30,
        },
        "output": {"samples_per_scenario": int(samples_per_scenario)},
        "traffic": {
            "state_probabilities": {"idle":0.15,"light":0.25,"medium":0.30,"loaded":0.30}
        },
        "detector": {"threshold_sigma": DETECT_THRESHOLD_SIGMA},
    }


def parse_args():
    p = argparse.ArgumentParser(description="Fast parallel EMILY-X single-LEO ML dataset generator")
    p.add_argument("--n-datasets", type=int, required=True,
                   help="number of accepted satellite/pass scenarios to request")
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--samples-per-scenario", type=int, default=32,
                   help="synchronized all-receiver time samples retained per pass; 0 = all detected candidates")
    p.add_argument("--receiver", action="append", default=[],
                   help="repeatable NAME,LAT,LON[,HEIGHT_M]")
    p.add_argument("--iq-dir", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("emilyx_satellite_dataset"))
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--base-epoch", default="2026-01-01T00:00:00Z")
    p.add_argument("--timezone", default="America/Los_Angeles")
    p.add_argument("--config-json", type=Path, default=None,
                   help="optional JSON deep override of default campaign configuration")
    return p.parse_args()


def deep_update(a: dict, b: dict) -> dict:
    out = dict(a)
    for k,v in b.items():
        if isinstance(v,dict) and isinstance(out.get(k),dict):
            out[k] = deep_update(out[k],v)
        else:
            out[k] = v
    return out


def main():
    args = parse_args()
    if args.n_datasets <= 0: raise ValueError("--n-datasets must be positive")
    if args.workers <= 0: raise ValueError("--workers must be positive")
    if args.samples_per_scenario < 0: raise ValueError("--samples-per-scenario must be >=0")

    if args.receiver:
        receivers = tuple(parse_receiver(x,i+1) for i,x in enumerate(args.receiver))
    else:
        # Current two EMILY-X receivers as convenient defaults.
        receivers = (
            Rx("RX1",39.524900,-114.373325),
            Rx("RX2",39.595518,-114.535418),
        )
    if len(receivers) < 1:
        raise ValueError("At least one receiver is required")

    cfg = default_config(args.seed,args.samples_per_scenario)
    if args.config_json is not None:
        cfg = deep_update(cfg, json.loads(args.config_json.read_text(encoding="utf-8")))

    out = args.output_dir
    (out/"cache").mkdir(parents=True,exist_ok=True)
    (out/"ml_inputs").mkdir(parents=True,exist_ok=True)
    (out/"ground_truth").mkdir(parents=True,exist_ok=True)

    bank_path, bank_meta = build_template_bank(args.iq_dir, out/"cache")

    scenario_fields = [
        "scenario_id","scenario_index","split","simulation_seed","attempt","n_receivers","receiver_names",
        "network_centroid_lat_deg","network_centroid_lon_deg","orbit_altitude_km","ground_heading_deg",
        "closest_approach_km","closest_approach_bearing_deg","closest_lat_deg","closest_lon_deg",
        "peak_eirp_dbm","beam_hpbw_deg","sidelobe_floor_db","max_scan_loss_db","min_elevation_deg",
        "common_visible_samples","retained_common_detected_samples","epoch_center_utc","trajectory_model",
        "lte_center_hz","lte_bw_hz","lte_n_prb"
    ]
    sample_fields = [
        "scenario_id","sample_id","split","time_index","t_mid","sat_lat_deg","sat_lon_deg","sat_alt_km",
        "sat_ecef_x_m","sat_ecef_y_m","sat_ecef_z_m","sat_ecef_vx_mps","sat_ecef_vy_mps","sat_ecef_vz_mps",
        "lte_state","template_bank_index","template_relative_power_db","receiver_truth_json"
    ]
    record_fields = [
        "scenario_id","sample_id","split","time_index","receiver_index","receiver_name",
        "data_type","timezone","t_start","t_mid","t_end","t_start_local","t_mid_local","t_end_local",
        "t_start_offset_sec","t_mid_offset_sec","t_end_offset_sec","duration_sec","ongoing","time_structure",
        "f_low_hz","f_high_hz","f_center_hz","bw_hz","band_tag","intensity_kind","intensity_value",
        "Directional","RA","Dec","Az","El","lat","lon","altitude","file_freq_resolution_hz",
        "file_time_resolution_sec","n_time_pixels","n_freq_pixels","n_pixels","method"
    ]
    index_fields = ["scenario_id","scenario_index","split","status","n_samples","n_records","error"]

    sc_fh, sc_w = csv_row_writer(out/"ground_truth"/"scenario_labels.csv", scenario_fields, False)
    sa_fh, sa_w = csv_row_writer(out/"ground_truth"/"sample_labels.csv.gz", sample_fields, True)
    re_fh, re_w = csv_row_writer(out/"ml_inputs"/"emily_records.csv.gz", record_fields, True)
    ix_fh, ix_w = csv_row_writer(out/"dataset_index.csv", index_fields, False)

    metadata = {
        "generator":"EMILY-X fast satellite ML generator",
        "version":VERSION,
        "requested_scenarios":args.n_datasets,
        "workers":args.workers,
        "receivers":[asdict(r) for r in receivers],
        "config":cfg,
        "template_bank":bank_meta,
        "notes":[
            "Each accepted sample_id has exactly one ML record per receiver.",
            "Satellite truth is stored only in ground_truth files.",
            "Train/validation/test split is deterministic at whole-scenario level.",
            "Synthetic circular-orbit mode is optimized for high-volume training; validate subsets with v3.2 TLE/SGP4.",
        ],
    }
    write_json(out/"dataset_metadata.json", metadata)
    write_json(out/"feature_schema.json", {
        "grouping_fields":["scenario_id","sample_id","split","time_index","receiver_index","receiver_name"],
        "model_input_file":"ml_inputs/emily_records.csv.gz",
        "sample_truth_file":"ground_truth/sample_labels.csv.gz",
        "scenario_truth_file":"ground_truth/scenario_labels.csv",
        "primary_possible_targets":["sat_lat_deg","sat_lon_deg","sat_alt_km"],
        "warning":"Never expose sample/scenario truth columns to the model input pipeline.",
    })

    receivers_json = json.dumps([asdict(r) for r in receivers])
    cfg_json = json.dumps(cfg)

    print("EMILY-X satellite ML generator")
    print("==============================")
    print(f"Scenarios requested: {args.n_datasets:,}")
    print(f"Workers:             {args.workers}")
    print(f"Receivers:           {len(receivers)} ({', '.join(r.name for r in receivers)})")
    print(f"Samples/scenario:    {args.samples_per_scenario if args.samples_per_scenario else 'all common detections'}")
    print(f"Template bank:       {bank_path.name} ({bank_meta['template_count']} templates)")
    print(f"Output:              {out.resolve()}")
    print()

    ok_count = 0
    fail_count = 0
    try:
        with cf.ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=worker_init,
            initargs=(str(bank_path), receivers_json, cfg_json, args.base_epoch, args.timezone),
        ) as ex:
            # executor.map preserves index order and does not require collecting all results.
            for i,res in enumerate(ex.map(run_one_scenario, range(args.n_datasets), chunksize=1), start=1):
                if res["ok"]:
                    sc_w.writerow(res["scenario"])
                    for row in res["samples"]: sa_w.writerow(row)
                    for row in res["records"]: re_w.writerow(row)
                    ix_w.writerow({
                        "scenario_id":res["scenario"]["scenario_id"],
                        "scenario_index":res["scenario"]["scenario_index"],
                        "split":res["scenario"]["split"],
                        "status":"ok",
                        "n_samples":len(res["samples"]),
                        "n_records":len(res["records"]),
                        "error":"",
                    })
                    ok_count += 1
                else:
                    ix_w.writerow({
                        "scenario_id":res["scenario_id"],"scenario_index":res["scenario_index"],
                        "split":res["split"],"status":"failed","n_samples":0,"n_records":0,"error":res["error"]
                    })
                    fail_count += 1
                if i % max(1,min(100,args.n_datasets//20 if args.n_datasets>=20 else 1)) == 0 or i==args.n_datasets:
                    print(f"[{i:,}/{args.n_datasets:,}] accepted={ok_count:,} failed={fail_count:,}")
                # Keep compressed streams reasonably current.
                if i % 100 == 0:
                    sc_fh.flush(); sa_fh.flush(); re_fh.flush(); ix_fh.flush()
    finally:
        sc_fh.close(); sa_fh.close(); re_fh.close(); ix_fh.close()

    print()
    print(f"Done. Accepted {ok_count:,}/{args.n_datasets:,}; failed {fail_count:,}.")
    print(f"ML inputs:    {out/'ml_inputs'/'emily_records.csv.gz'}")
    print(f"Sample truth: {out/'ground_truth'/'sample_labels.csv.gz'}")
    print(f"Scenario GT:  {out/'ground_truth'/'scenario_labels.csv'}")


if __name__ == "__main__":
    main()
