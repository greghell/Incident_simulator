#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EMILY-X LTE end-to-end forward simulator, v3.1
=============================================

This v3.1 version is designed around a *genuine LTE eNodeB baseband waveform* captured
from srsRAN 4G.  srsRAN is responsible for generating standards-compliant LTE
physical-layer samples (PSS/SSS, PBCH, reference signals, control channels,
PDSCH, scheduling, etc.).  GNU Radio can be inserted as a ZeroMQ broker to tee
those samples to a complex64 file while still passing them to srsUE.

The EMILY-X simulator then adds the parts that are outside the eNodeB digital
baseband itself:

    srsRAN LTE IQ
        -> reconstruction/channel filter
        -> memoryless PA non-linearity / spectral regrowth
        -> finite RF output-filter leakage
        -> 2nd and 3rd harmonics
        -> configurable narrow spurious products
        -> directional base-station antenna
        -> frequency-dependent propagation
        -> receiver noise/bandpass/gain drift
        -> 200--2500 MHz spectrometer (70 kHz spacing, 140 kHz RBW, 1 s)
        -> NPZ files compatible with the existing EMILY extractor
        -> local EMILY-like incident preview + plots + ground truth

Important modeling choices
--------------------------
1. The LTE carrier is 10 MHz / 50 Physical Resource Blocks (PRB), centered at 800 MHz.  For srsRAN this is
   configured as LTE Band 20 EARFCN 6240 (800.0 MHz downlink).  This is a
   standards-valid LTE carrier; it is a simulation choice and is not a claim
   that the Nevada tower is actually licensed for Band 20.
2. srsRAN 4G commonly uses 11.52 Msamples/s for 50 PRB in its RF abstraction.
   The exact capture sample rate is configurable below and is stored in output
   metadata.
3. We do NOT generate 600 s of raw IQ.  At 11.52 Msamples/s, a ten-minute
   complex64 capture would be ~55 GB.  Instead v3.1 splits each short genuine
   eNodeB capture into multiple 100-ms LTE windows, constructs one RF spectral
   template per window, and selects among those templates during the final
   1-s / 140-kHz measurement simulation.  This preserves genuine scheduler
   variation within each nominal traffic-load state without huge IQ files.
4. Out-of-band emission around the fundamental is produced from the captured
   waveform after a PA model and a finite RF output filter.  The script reports
   the resulting approximate Adjacent Channel Leakage Ratio (ACLR).  The target is configurable; 3GPP TS 36.104
   specifies a 45 dB minimum ACLR for normal paired-spectrum E-UTRA BS cases.
5. Harmonic powers and spurious-product levels are hardware-dependent.  The
   default values here are explicit scenario assumptions, not measurements of
   the Nevada tower.  They are deliberately kept in the configuration section.
6. The receiver latitude/longitude written to each NPZ and to every locally
   generated EMILY candidate incident is the *receiver* position.  Transmitter
   coordinates are kept separately as simulation ground truth.

Expected input files
--------------------
By default the script looks for:
    srsran_idle.cf32
    srsran_light.cf32
    srsran_medium.cf32
    srsran_loaded.cf32

These should contain interleaved GNU Radio gr_complex / NumPy complex64 samples
from the *downlink* of the same srsENB configuration, captured through the
supplied GNU Radio broker.  v3.1 measures both state-to-state power and
window-to-window scheduler variation.  Missing states are still substituted by
the nearest available state, as in v3.

For development only, --dev-fallback generates a simplified OFDM waveform when
no srsRAN capture is available.  This is intentionally opt-in and should not be
used for the production ML dataset.

To launch eNodeB script
-----------------------
open Ubuntu
cd /mnt/c/Users/gregh/Desktop/EMILY/EMILY-X/emilyx_lte_v3_bundle_with_autocapture/emilyx_lte_v3
chmod +x generate_lte_iq_states_fixed.sh
./generate_lte_iq_states_fixed.sh

To launch script
----------------
conda activate emilyx

cd C:\Users\gregh\Desktop\EMILY\EMILY-X\emilyx_lte_v3_bundle_with_autocapture\emilyx_lte_v3

python emilyx_lte_end_to_end_simulator_v3_1.py ^
  --duration 600 ^
  --output-dir output_emilyx_lte_v3_1_600s
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d, median_filter, binary_dilation, label, find_objects
from pyproj import Geod

# Optional pycraf dependency.  pycraf is preferred for the antenna pattern and
# free-space calculations.  The script keeps transparent fallbacks for testing.
try:
    from astropy import units as u
    from pycraf import conversions as cnv, pathprof
    from pycraf.antenna import (
        imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz,
    )
    HAVE_PYCRAF = True
except Exception:
    HAVE_PYCRAF = False


# =============================================================================
# USER CONFIGURATION
# =============================================================================

OUTPUT_DIR = Path("output_emilyx_lte_v3_1")
RANDOM_SEED = 20260902

# Ten minutes of spectra, one spectrum per second.
SIM_DURATION_S = 600
TIME_RESOLUTION_S = 1.0
SIM_START_UTC = "2026-09-02T19:00:00+00:00"
LOCAL_TIMEZONE = "America/Los_Angeles"

# Spectrum-monitor backend.
FREQ_MIN_HZ = 200e6
FREQ_MAX_HZ = 2500e6
CHANNEL_SPACING_HZ = 70e3
RBW_HZ = 140e3

# Genuine LTE source configuration.
LTE_CENTER_HZ = 800e6
LTE_CHANNEL_BW_HZ = 10e6
LTE_N_PRB = 50
LTE_EARFCN_DL = 6240          # Band 20 -> 800.0 MHz downlink
LTE_EARFCN_UL = 24240         # Corresponding Band-20 uplink -> 841.0 MHz
SRSRAN_IQ_SAMPLE_RATE_HZ = 11.52e6
# v3.1 uses the full short capture to build multiple genuine LTE PSD templates.
SRSRAN_MAX_CAPTURE_SECONDS = 0.50
LTE_TEMPLATE_WINDOW_MS = 100.0     # integer LTE subframe multiple; default gives 5 templates / 0.5-s capture
LTE_TEMPLATE_HOP_MS = 100.0        # set to 50 ms for overlapping windows (up to 9 templates / 0.5-s capture)
LTE_MAX_TEMPLATES_PER_STATE = 5

# Base-station radiated-power assumptions.
# Conducted power is total power of the fundamental LTE carrier before antenna gain.
TX_CONDUCTED_POWER_DBM = 43.0       # 20 W, configurable
TX_MAX_ANT_GAIN_DBI = 17.0
TX_HORIZ_HPBW_DEG = 65.0
TX_SECTOR_AZIMUTH_DEG = 10.0
TX_ELECTRICAL_DOWNTILT_DEG = 4.0
TX_HEIGHT_AGL_M = 40.0

# Frequency-dependent antenna efficiency assumptions for harmonic radiation.
# A real 800-MHz antenna normally radiates out-of-band harmonics less efficiently.
HARMONIC_ANT_EXTRA_LOSS_DB = {
    1: 0.0,
    2: 10.0,
    3: 15.0,
}

# Transmitter RF-chain model.
TX_OVERSAMPLE_FACTOR = 4
TX_DIGITAL_FILTER_PASS_HZ = 4.55e6
TX_DIGITAL_FILTER_STOP_HZ = 5.30e6
TX_DIGITAL_FILTER_TAPS = 257
TX_PA_INPUT_BACKOFF_DB = 8.0
TX_PA_RAPP_P = 3.0
TX_RF_FILTER_PASS_HZ = 4.50e6
TX_RF_FILTER_STOP_HZ = 5.50e6
# The post-PA RF channel filter is calibrated so that the adjacent-channel
# leakage is realistic for the scenario rather than unrealistically perfect.
# 3GPP TS 36.104 requires >=45 dB BS ACLR for paired-spectrum LTE; 48 dB is
# used here as a configurable example operating point, NOT a claim about the
# specific Nevada hardware.  Far-out leakage is then suppressed more strongly.
TARGET_ACLR_DB = 48.0
TX_RF_FILTER_MAX_NEAR_ATTEN_DB = 30.0
TX_RF_FILTER_FAR_STOP_START_HZ = 18.0e6
TX_RF_FILTER_FAR_STOP_ATTEN_DB = 70.0

# Harmonic total powers relative to the fundamental carrier.
# These are scenario assumptions.  Replace with measurements/vendor/FCC data
# when a specific transmitter hardware model is available.
SECOND_HARMONIC_DBC = -60.0
THIRD_HARMONIC_DBC = -65.0

# Example narrow spurious products.  These are intentionally explicit and
# configurable.  The +/-30.72 MHz offsets represent a plausible reference-clock
# leakage family, not a claim about this particular tower.
SPURIOUS_PRODUCTS = [
    {
        # A real direct-conversion transmitter can leave a residual carrier/LO
        # term at the nominal center frequency even though LTE leaves the DC
        # subcarrier unused.  The level is a scenario assumption.
        "name": "carrier_lo_leakage",
        "frequency_hz": LTE_CENTER_HZ,
        "level_dbc": -55.0,
        "fwhm_hz": 8e3,
        "antenna_extra_loss_db": 0.0,
    },
    {
        "name": "reference_clock_minus",
        "frequency_hz": LTE_CENTER_HZ - 30.72e6,
        "level_dbc": -70.0,
        "fwhm_hz": 30e3,
        "antenna_extra_loss_db": 0.0,
    },
    {
        "name": "reference_clock_plus",
        "frequency_hz": LTE_CENTER_HZ + 30.72e6,
        "level_dbc": -70.0,
        "fwhm_hz": 30e3,
        "antenna_extra_loss_db": 0.0,
    },
]

# Receiver assumptions.
RX_ANT_GAIN_DBI = 0.0
RX_HEIGHT_AGL_M = 2.0
RX_NOISE_FIGURE_DB = 6.0
REFERENCE_TEMP_K = 290.0
NOISE_JITTER_DB = 0.20
BANDPASS_RIPPLE_DB = 0.35
GAIN_DRIFT_DB = 0.20

# Propagation refinements.  Free space remains the validated baseline model;
# a future P.452/DEM module can replace this without changing the rest of the code.
SHADOWING_SIGMA_DB = 0.8
SHADOWING_CORRELATION_S = 30.0

# EMILY detector settings copied from the user's current extractor.
THRESHOLD = 8.0
COARSE_FACTOR = 50
SMOOTH_COARSE_BINS = 31
NOISE_COARSE_BINS = 31
DILATE_TIME = 1
DILATE_FREQ = 2
GROUP_GAP_TIME_PIX = 3
GROUP_GAP_FREQ_PIX = 20
MIN_PIXELS = 50

SAVE_COMPRESSED_NPZ = True
RUN_LOCAL_EMILY_PIPELINE = True

# Local preview windows.  The operational extractor can still inspect the full
# 200--2500 MHz NPZ.  Using windows here reduces memory for diagnostics.
LOCAL_EMILY_WINDOW_HALFSPAN_HZ = 40e6


@dataclass
class Transmitter:
    name: str
    lat_deg: float
    lon_deg: float
    height_agl_m: float
    conducted_power_dbm: float
    center_frequency_hz: float
    channel_bandwidth_hz: float
    max_gain_dbi: float
    horizontal_hpbw_deg: float
    sector_azimuth_deg: float
    electrical_downtilt_deg: float


@dataclass
class Receiver:
    name: str
    lat_deg: float
    lon_deg: float
    height_agl_m: float = RX_HEIGHT_AGL_M
    antenna_gain_dbi: float = RX_ANT_GAIN_DBI
    noise_figure_db: float = RX_NOISE_FIGURE_DB
    excess_path_loss_db: float = 0.0


@dataclass
class EmissionComponent:
    """One physically related RF product emitted by the same transmitter."""
    name: str
    kind: str
    center_hz: float
    total_power_dbm: float
    psd_offset_hz: np.ndarray
    psd_norm_per_hz: np.ndarray
    harmonic_order: int = 1
    antenna_extra_loss_db: float = 0.0


# Coordinates supplied by the user.
TX = Transmitter(
    name="LTE_cell_tower",
    lat_deg=39 + 9/60 + 38.00/3600,
    lon_deg=-(114 + 36/60 + 51.69/3600),
    height_agl_m=TX_HEIGHT_AGL_M,
    conducted_power_dbm=TX_CONDUCTED_POWER_DBM,
    center_frequency_hz=LTE_CENTER_HZ,
    channel_bandwidth_hz=LTE_CHANNEL_BW_HZ,
    max_gain_dbi=TX_MAX_ANT_GAIN_DBI,
    horizontal_hpbw_deg=TX_HORIZ_HPBW_DEG,
    sector_azimuth_deg=TX_SECTOR_AZIMUTH_DEG,
    electrical_downtilt_deg=TX_ELECTRICAL_DOWNTILT_DEG,
)

RECEIVERS = [
    Receiver(
        name="RX1",
        lat_deg=39 + 31/60 + 29.64/3600,
        lon_deg=-(114 + 22/60 + 23.97/3600),
    ),
    Receiver(
        name="RX2",
        lat_deg=39.595518,
        lon_deg=-114.535418,
    ),
]

GEOD = Geod(ellps="WGS84")


# =============================================================================
# BASIC HELPERS
# =============================================================================

def dbm_to_mw(x_dbm):
    return 10.0 ** (np.asarray(x_dbm, dtype=np.float64) / 10.0)


def mw_to_dbm(x_mw, floor_dbm=-300.0):
    x = np.asarray(x_mw, dtype=np.float64)
    floor_mw = 10.0 ** (floor_dbm / 10.0)
    return 10.0 * np.log10(np.maximum(x, floor_mw))


def wrap180(angle_deg):
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def linear_average_dbm(x_dbm, axis=0):
    return mw_to_dbm(np.nanmean(dbm_to_mw(x_dbm), axis=axis))


def isoformat_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def normalized_psd(iq: np.ndarray, sample_rate_hz: float, nperseg: int = 32768):
    """Return two-sided PSD normalized so its integral over frequency is one."""
    nperseg = min(nperseg, len(iq))
    if nperseg < 1024:
        raise ValueError("IQ capture is too short to estimate a useful PSD.")
    noverlap = nperseg // 2
    f, pxx = signal.welch(
        iq,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    f = np.fft.fftshift(f)
    pxx = np.fft.fftshift(np.maximum(pxx, 0.0))
    area = np.trapezoid(pxx, f)
    if not np.isfinite(area) or area <= 0:
        raise ValueError("Could not normalize waveform PSD.")
    return f.astype(np.float64), (pxx / area).astype(np.float64)


# =============================================================================
# GEOMETRY, ANTENNA, PROPAGATION
# =============================================================================

def link_geometry(tx: Transmitter, rx: Receiver):
    az_tx_to_rx, az_rx_to_tx, distance_m = GEOD.inv(
        tx.lon_deg, tx.lat_deg, rx.lon_deg, rx.lat_deg
    )
    elev_deg = math.degrees(
        math.atan2(rx.height_agl_m - tx.height_agl_m, distance_m)
    )
    return {
        "distance_m": float(distance_m),
        "distance_km": float(distance_m / 1e3),
        "bearing_tx_to_rx_deg": float(az_tx_to_rx % 360.0),
        "bearing_rx_to_tx_deg": float(az_rx_to_tx % 360.0),
        "elevation_tx_to_rx_deg": float(elev_deg),
    }


def tx_antenna_gain_db(tx: Transmitter, bearing_deg: float, elevation_deg: float,
                       harmonic_order: int = 1, extra_loss_db: float = 0.0):
    """ITU-R F.1336-like sector gain, plus out-of-band antenna-efficiency loss."""
    rel_az = float(wrap180(bearing_deg - tx.sector_azimuth_deg))
    base_max_gain = tx.max_gain_dbi - HARMONIC_ANT_EXTRA_LOSS_DB.get(harmonic_order, 0.0)
    base_max_gain -= extra_loss_db

    theta_3db = 31000.0 * 10.0 ** (-0.1 * tx.max_gain_dbi) / tx.horizontal_hpbw_deg

    if HAVE_PYCRAF:
        k_p, k_h, k_v = (0.7, 0.7, 0.3) * cnv.dimless
        gain = imt_advanced_sectoral_peak_sidelobe_pattern_400_to_6000_mhz(
            rel_az * u.deg,
            elevation_deg * u.deg,
            base_max_gain * cnv.dB,
            tx.horizontal_hpbw_deg * u.deg,
            theta_3db * u.deg,
            k_p, k_h, k_v,
            tilt_m=0.0 * u.deg,
            tilt_e=tx.electrical_downtilt_deg * u.deg,
        )
        return float(np.asarray(gain.to(cnv.dB).value).squeeze())

    rel_el = elevation_deg + tx.electrical_downtilt_deg
    attenuation = 12.0 * (rel_az / tx.horizontal_hpbw_deg) ** 2
    attenuation += 12.0 * (rel_el / theta_3db) ** 2
    return float(base_max_gain - min(30.0, attenuation))


def free_space_path_loss_db(distance_m: float, freq_hz):
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    if HAVE_PYCRAF:
        gain_db = cnv.free_space_loss(distance_m * u.m, freq_hz * u.Hz).to(cnv.dB).value
        return -np.asarray(gain_db, dtype=np.float64)
    c = 299_792_458.0
    return 20.0 * np.log10(4.0 * np.pi * distance_m * freq_hz / c)


def make_slow_shadowing(nt: int, sigma_db: float, correlation_s: float,
                        dt_s: float, rng: np.random.Generator):
    if sigma_db <= 0:
        return np.zeros(nt, dtype=np.float32)
    white = rng.normal(size=nt)
    sigma_samples = max(1.0, correlation_s / dt_s)
    smooth = gaussian_filter1d(white, sigma=sigma_samples, mode="reflect")
    smooth -= np.mean(smooth)
    if np.std(smooth) > 0:
        smooth *= sigma_db / np.std(smooth)
    return smooth.astype(np.float32)


# =============================================================================
# GENUINE srsRAN WAVEFORM INPUT + DEVELOPMENT FALLBACK
# =============================================================================

def load_srsran_iq(path: Path, sample_rate_hz: float, max_seconds: float):
    """Load GNU Radio / srsRAN complex64 IQ and keep only the requested duration."""
    if not path.exists():
        raise FileNotFoundError(path)
    max_samples = int(round(sample_rate_hz * max_seconds))
    iq = np.fromfile(path, dtype=np.complex64, count=max_samples)
    if len(iq) < int(0.020 * sample_rate_hz):
        raise ValueError(
            f"{path} contains only {len(iq)} samples; capture at least ~20 ms."
        )
    iq = iq.astype(np.complex64, copy=False)
    iq = iq - np.mean(iq)
    rms = np.sqrt(np.mean(np.abs(iq) ** 2))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError(f"{path} has invalid/zero signal power.")
    return (iq / rms).astype(np.complex64)


def random_qam_symbols(n: int, order: int, rng: np.random.Generator):
    m_side = int(round(math.sqrt(order)))
    levels = np.arange(-(m_side - 1), m_side, 2, dtype=np.float64)
    i = rng.integers(0, m_side, size=n)
    q = rng.integers(0, m_side, size=n)
    x = levels[i] + 1j * levels[q]
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    return x


def generate_dev_ofdm_fallback(rng: np.random.Generator, duration_s=0.08):
    """
    Development-only 50-PRB OFDM fallback.

    This is NOT used unless --dev-fallback is explicitly requested.  It exists
    so colleagues can test the post-eNodeB RF/propagation/EMILY code before the
    first srsRAN capture has been made.
    """
    fs = SRSRAN_IQ_SAMPLE_RATE_HZ
    nfft = 768                    # srsRAN commonly uses 11.52 Msps for 50 PRB
    if abs(fs / nfft - 15e3) > 1:
        raise RuntimeError("Development OFDM fallback numerology is inconsistent.")
    n_active = 600
    cp_first = 60
    cp_other = 54
    n_symbols = int(duration_s / 0.0005) * 7
    out = []
    for sym_idx in range(max(n_symbols, 14)):
        bins = np.zeros(nfft, dtype=np.complex128)
        qam = random_qam_symbols(n_active, 64, rng)
        half = n_active // 2
        bins[1:half+1] = qam[half:]
        bins[-half:] = qam[:half]
        x = np.fft.ifft(bins) * np.sqrt(nfft)
        cp = cp_first if (sym_idx % 7 == 0) else cp_other
        out.append(np.concatenate([x[-cp:], x]))
    iq = np.concatenate(out)
    iq -= np.mean(iq)
    iq /= np.sqrt(np.mean(np.abs(iq) ** 2))
    return iq.astype(np.complex64)


# =============================================================================
# TRANSMITTER FILTER / PA / OOB / HARMONICS / SPURS
# =============================================================================

def design_reconstruction_filter(sample_rate_hz: float):
    """Linear-phase FIR approximating a transmitter reconstruction/channel filter."""
    cutoff = 0.5 * (TX_DIGITAL_FILTER_PASS_HZ + TX_DIGITAL_FILTER_STOP_HZ)
    taps = signal.firwin(
        TX_DIGITAL_FILTER_TAPS,
        cutoff=cutoff,
        fs=sample_rate_hz,
        window=("kaiser", 7.5),
    )
    return taps.astype(np.float64)


def apply_rapp_pa(iq: np.ndarray, input_backoff_db: float, p: float):
    """
    Memoryless Rapp AM/AM model.

    The input waveform is first normalized to unit RMS.  A_sat is then chosen
    from the requested RMS input backoff.  The model generates realistic
    near-carrier spectral regrowth but, as a complex-envelope model, harmonics
    are generated separately from x^2 and x^3 below.
    """
    x = iq.astype(np.complex128)
    x /= np.sqrt(np.mean(np.abs(x) ** 2))
    a_sat = 10.0 ** (input_backoff_db / 20.0)
    amp = np.abs(x)
    denom = (1.0 + (amp / a_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))
    y = x / denom
    y /= np.sqrt(np.mean(np.abs(y) ** 2))
    return y.astype(np.complex64)


def rf_output_filter_power_gain(offset_hz: np.ndarray, near_stop_atten_db: float):
    """
    Smooth effective RF-output-filter *power* response.

    The first stop-band level controls adjacent-channel leakage and is chosen
    automatically from the measured post-PA ACLR.  Farther away from the LTE
    carrier, the response rolls toward a much stronger far-stop attenuation.

    This is deliberately an effective RF-chain model.  Once measured emission
    masks or FCC/vendor data for a particular transmitter are available, this
    function can be replaced by a tabulated measured response.
    """
    f = np.abs(np.asarray(offset_hz, dtype=np.float64))
    near_stop_atten_db = float(np.clip(near_stop_atten_db, 0.0, TX_RF_FILTER_MAX_NEAR_ATTEN_DB))

    atten_db = np.zeros_like(f)

    # Transition from the occupied LTE band into the adjacent-channel region.
    trans = (f > TX_RF_FILTER_PASS_HZ) & (f < TX_RF_FILTER_STOP_HZ)
    if np.any(trans):
        x = (f[trans] - TX_RF_FILTER_PASS_HZ) / (TX_RF_FILTER_STOP_HZ - TX_RF_FILTER_PASS_HZ)
        smooth = 0.5 - 0.5 * np.cos(np.pi * x)
        atten_db[trans] = near_stop_atten_db * smooth

    # Near stop band.  Keeping this finite makes filter leakage visible.
    near = (f >= TX_RF_FILTER_STOP_HZ) & (f < TX_RF_FILTER_FAR_STOP_START_HZ)
    atten_db[near] = near_stop_atten_db

    # Far-out attenuation.  This affects the fundamental's distant spectral
    # leakage; explicit harmonics and spurs are modeled as separate components.
    far_trans_end = TX_RF_FILTER_FAR_STOP_START_HZ + 3.0e6
    far_trans = (f >= TX_RF_FILTER_FAR_STOP_START_HZ) & (f < far_trans_end)
    if np.any(far_trans):
        x = (f[far_trans] - TX_RF_FILTER_FAR_STOP_START_HZ) / (far_trans_end - TX_RF_FILTER_FAR_STOP_START_HZ)
        smooth = 0.5 - 0.5 * np.cos(np.pi * x)
        atten_db[far_trans] = near_stop_atten_db + (TX_RF_FILTER_FAR_STOP_ATTEN_DB - near_stop_atten_db) * smooth
    atten_db[f >= far_trans_end] = TX_RF_FILTER_FAR_STOP_ATTEN_DB

    return 10.0 ** (-atten_db / 10.0)


def choose_near_stop_attenuation(freq_hz: np.ndarray, raw_psd_per_hz: np.ndarray,
                                  target_aclr_db: float):
    """Choose the smallest effective adjacent-channel filter attenuation
    that reaches ``target_aclr_db``.

    The raw PA spectrum is retained as the physical source of spectral regrowth;
    the calibration only represents the finite post-PA channel/output filter.
    """
    raw_aclr = aclr_db(freq_hz, raw_psd_per_hz)
    if raw_aclr >= target_aclr_db:
        return 0.0, raw_aclr

    lo, hi = 0.0, TX_RF_FILTER_MAX_NEAR_ATTEN_DB
    # If even the allowed filter is insufficient, use the maximum and report it.
    test = raw_psd_per_hz * rf_output_filter_power_gain(freq_hz, hi)
    test /= np.trapezoid(test, freq_hz)
    if aclr_db(freq_hz, test) < target_aclr_db:
        return hi, raw_aclr

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        test = raw_psd_per_hz * rf_output_filter_power_gain(freq_hz, mid)
        test /= np.trapezoid(test, freq_hz)
        if aclr_db(freq_hz, test) >= target_aclr_db:
            hi = mid
        else:
            lo = mid
    return hi, raw_aclr


def aclr_db(freq_hz: np.ndarray, psd_per_hz: np.ndarray, channel_bw_hz=9e6):
    """Approximate E-UTRA ACLR from the simulated emission PSD."""
    half = channel_bw_hz / 2.0
    main = np.abs(freq_hz) <= half
    lower = (freq_hz >= -LTE_CHANNEL_BW_HZ - half) & (freq_hz <= -LTE_CHANNEL_BW_HZ + half)
    upper = (freq_hz >= LTE_CHANNEL_BW_HZ - half) & (freq_hz <= LTE_CHANNEL_BW_HZ + half)

    def integ(mask):
        if np.count_nonzero(mask) < 2:
            return 0.0
        return float(np.trapezoid(psd_per_hz[mask], freq_hz[mask]))

    p_main = integ(main)
    p_adj = max(integ(lower), integ(upper), 1e-30)
    return 10.0 * np.log10(max(p_main, 1e-30) / p_adj)


def gaussian_unit_psd(fwhm_hz: float, span_factor=8.0, n=4097):
    sigma = fwhm_hz / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    span = max(span_factor * sigma, 2.0 * CHANNEL_SPACING_HZ)
    f = np.linspace(-span, span, n)
    p = np.exp(-0.5 * (f / sigma) ** 2)
    p /= np.trapezoid(p, f)
    return f.astype(np.float64), p.astype(np.float64)


def build_emission_components(iq: np.ndarray, sample_rate_hz: float):
    """
    Turn genuine srsRAN digital baseband samples into a radiated-emission model.

    Returns a list of EmissionComponent objects that all originate from the same
    physical transmitter and are therefore linked in the ground truth.
    """
    # Upsample so the model has enough bandwidth to represent adjacent-channel
    # spectral regrowth.  We process only a short capture, so this remains tractable.
    up = TX_OVERSAMPLE_FACTOR
    iq_os = signal.resample_poly(iq, up, 1).astype(np.complex64)
    fs_os = sample_rate_hz * up

    # Reconstruction / channel filter before the PA.
    taps = design_reconstruction_filter(fs_os)
    iq_filt = signal.lfilter(taps, [1.0], iq_os).astype(np.complex64)
    transient = len(taps)
    if len(iq_filt) > 2 * transient:
        iq_filt = iq_filt[transient:]
    iq_filt /= np.sqrt(np.mean(np.abs(iq_filt) ** 2))

    # PA spectral regrowth around the fundamental.
    iq_pa = apply_rapp_pa(iq_filt, TX_PA_INPUT_BACKOFF_DB, TX_PA_RAPP_P)
    f1, p1_raw = normalized_psd(iq_pa, fs_os)

    # Calibrate the finite post-PA channel/output filter to the scenario ACLR.
    # This avoids the unrealistically clean spectrum produced by an arbitrary
    # very-deep stop band while still preserving PA-generated spectral regrowth.
    near_stop_atten_db, raw_aclr = choose_near_stop_attenuation(
        f1, p1_raw, TARGET_ACLR_DB
    )
    p1 = p1_raw * rf_output_filter_power_gain(f1, near_stop_atten_db)
    p1 /= np.trapezoid(p1, f1)
    measured_aclr = aclr_db(f1, p1)

    components = [
        EmissionComponent(
            name="LTE_fundamental",
            kind="fundamental_plus_OOB",
            center_hz=LTE_CENTER_HZ,
            total_power_dbm=TX_CONDUCTED_POWER_DBM,
            psd_offset_hz=f1,
            psd_norm_per_hz=p1,
            harmonic_order=1,
            antenna_extra_loss_db=0.0,
        )
    ]

    # Passband polynomial terms generate harmonics in a real RF PA.  The complex
    # envelopes around n*f0 are proportional to x^n for a memoryless polynomial.
    # Their absolute levels are then calibrated with explicit dBc parameters.
    for order, dbc in [(2, SECOND_HARMONIC_DBC), (3, THIRD_HARMONIC_DBC)]:
        h = iq_filt.astype(np.complex128) ** order
        h -= np.mean(h)
        h /= np.sqrt(np.mean(np.abs(h) ** 2))
        fh, ph = normalized_psd(h.astype(np.complex64), fs_os)
        components.append(
            EmissionComponent(
                name=f"harmonic_{order}",
                kind="harmonic",
                center_hz=order * LTE_CENTER_HZ,
                total_power_dbm=TX_CONDUCTED_POWER_DBM + dbc,
                psd_offset_hz=fh,
                psd_norm_per_hz=ph,
                harmonic_order=order,
                antenna_extra_loss_db=0.0,
            )
        )

    for spur in SPURIOUS_PRODUCTS:
        fs, ps = gaussian_unit_psd(float(spur["fwhm_hz"]))
        components.append(
            EmissionComponent(
                name=str(spur["name"]),
                kind="spurious",
                center_hz=float(spur["frequency_hz"]),
                total_power_dbm=TX_CONDUCTED_POWER_DBM + float(spur["level_dbc"]),
                psd_offset_hz=fs,
                psd_norm_per_hz=ps,
                harmonic_order=1,
                antenna_extra_loss_db=float(spur.get("antenna_extra_loss_db", 0.0)),
            )
        )

    diagnostics = {
        "oversampled_rate_hz": fs_os,
        "raw_post_pa_aclr_db": raw_aclr,
        "effective_near_stop_atten_db": near_stop_atten_db,
        "measured_aclr_db": measured_aclr,
        "target_aclr_db": TARGET_ACLR_DB,
    }
    return components, diagnostics, iq_filt, iq_pa


def rbw_power_fraction(freq_centers_hz: np.ndarray, component: EmissionComponent):
    """Fraction of one component's total power measured by each 140-kHz RBW."""
    f = component.psd_offset_hz
    p = component.psd_norm_per_hz
    cdf = cumulative_trapezoid(p, f, initial=0.0)
    if cdf[-1] <= 0:
        return np.zeros_like(freq_centers_hz, dtype=np.float64)
    cdf /= cdf[-1]
    lo = freq_centers_hz - component.center_hz - RBW_HZ / 2.0
    hi = freq_centers_hz - component.center_hz + RBW_HZ / 2.0
    c_lo = np.interp(lo, f, cdf, left=0.0, right=1.0)
    c_hi = np.interp(hi, f, cdf, left=0.0, right=1.0)
    return np.maximum(c_hi - c_lo, 0.0)


# =============================================================================
# RECEIVER / SPECTROMETER MODEL
# =============================================================================

def thermal_noise_floor_dbm(rbw_hz: float, noise_figure_db: float):
    k_b = 1.380649e-23
    p_w = k_b * REFERENCE_TEMP_K * rbw_hz
    return float(10.0 * np.log10(p_w / 1e-3) + noise_figure_db)


def build_tx_rbw_spectrum(freq_hz: np.ndarray, components: list[EmissionComponent]):
    total_mw = np.zeros(len(freq_hz), dtype=np.float64)
    fractions = {}
    for comp in components:
        frac = rbw_power_fraction(freq_hz, comp)
        fractions[comp.name] = frac
        total_mw += dbm_to_mw(comp.total_power_dbm) * frac
    return mw_to_dbm(total_mw, floor_dbm=-220.0), fractions


def make_receiver_spectrogram(rx: Receiver, tx: Transmitter,
                              freq_hz: np.ndarray,
                              components: list[EmissionComponent],
                              component_fractions: dict[str, np.ndarray],
                              nt: int,
                              rng: np.random.Generator):
    """Propagate every emission product independently, then add receiver noise."""
    nf = len(freq_hz)
    geom = link_geometry(tx, rx)
    noise_floor_dbm = thermal_noise_floor_dbm(RBW_HZ, rx.noise_figure_db)

    # Static smooth receiver bandpass ripple.
    x = np.linspace(0.0, 8.0 * np.pi, nf)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    bandpass_db = (
        BANDPASS_RIPPLE_DB * np.sin(x + phase)
        + 0.25 * BANDPASS_RIPPLE_DB * np.sin(0.27 * x + 0.4 * phase)
    ).astype(np.float32)

    # Slow receiver gain drift.
    gain_drift = gaussian_filter1d(rng.normal(size=nt), sigma=25.0)
    gain_drift -= np.mean(gain_drift)
    if np.std(gain_drift) > 0:
        gain_drift *= GAIN_DRIFT_DB / np.std(gain_drift)
    gain_drift = gain_drift.astype(np.float32)

    # Noise-only measurement cube.
    spec_dbm = np.empty((nt, nf), dtype=np.float32)
    chunk = 50
    for i0 in range(0, nt, chunk):
        i1 = min(nt, i0 + chunk)
        jitter = rng.normal(0.0, NOISE_JITTER_DB, size=(i1 - i0, nf)).astype(np.float32)
        spec_dbm[i0:i1] = noise_floor_dbm + bandpass_db[None, :] + gain_drift[i0:i1, None] + jitter

    shadowing_db = make_slow_shadowing(
        nt, SHADOWING_SIGMA_DB, SHADOWING_CORRELATION_S, TIME_RESOLUTION_S, rng
    )

    component_truth = []

    for comp in components:
        frac = component_fractions[comp.name]
        # Ignore numerical tails that cannot matter to the final measurement.
        active = frac > 1e-16
        if not np.any(active):
            continue
        idx = np.flatnonzero(active)
        f_active = freq_hz[idx]
        frac_active = frac[idx]

        tx_gain_db = tx_antenna_gain_db(
            tx,
            geom["bearing_tx_to_rx_deg"],
            geom["elevation_tx_to_rx_deg"],
            harmonic_order=comp.harmonic_order,
            extra_loss_db=comp.antenna_extra_loss_db,
        )
        path_loss_db = free_space_path_loss_db(geom["distance_m"], f_active)

        # Component power in each analyzer RBW.
        tx_bin_dbm = comp.total_power_dbm + 10.0 * np.log10(np.maximum(frac_active, 1e-30))
        rx_bin_dbm = (
            tx_bin_dbm[None, :]
            + tx_gain_db
            + rx.antenna_gain_dbi
            - path_loss_db[None, :]
            - rx.excess_path_loss_db
            + shadowing_db[:, None]
        )

        # Physical powers add in linear units.
        spec_dbm[:, idx] = mw_to_dbm(
            dbm_to_mw(spec_dbm[:, idx]) + dbm_to_mw(rx_bin_dbm)
        ).astype(np.float32)

        center_loss_db = float(np.asarray(
            free_space_path_loss_db(geom["distance_m"], comp.center_hz)
        ).squeeze())
        rx_total_dbm_t = (
            comp.total_power_dbm + tx_gain_db + rx.antenna_gain_dbi
            - center_loss_db - rx.excess_path_loss_db + shadowing_db
        )
        component_truth.append({
            "component": comp.name,
            "kind": comp.kind,
            "center_hz": comp.center_hz,
            "tx_total_power_dbm": comp.total_power_dbm,
            "tx_antenna_gain_db": tx_gain_db,
            "path_loss_center_db": center_loss_db,
            "median_rx_total_component_dbm": float(np.median(rx_total_dbm_t)),
            "rx_total_component_dbm_t": rx_total_dbm_t.astype(np.float32),
        })

    truth = {
        **geom,
        "noise_floor_per_rbw_dbm": noise_floor_dbm,
        "shadowing_db_t": shadowing_db,
        "component_truth": component_truth,
    }
    return spec_dbm, truth


# =============================================================================
# TIME + NPZ OUTPUT
# =============================================================================

def make_time_axes(nt: int):
    start = datetime.fromisoformat(SIM_START_UTC)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    start = start.astimezone(timezone.utc)
    local_tz = ZoneInfo(LOCAL_TIMEZONE)
    time_s_rel = np.arange(nt, dtype=np.float64) * TIME_RESOLUTION_S
    utc = np.array([isoformat_z(start + timedelta(seconds=float(s))) for s in time_s_rel])
    local = np.array([
        (start + timedelta(seconds=float(s))).astimezone(local_tz).isoformat()
        for s in time_s_rel
    ])
    return time_s_rel, utc, local


def serialize_emission_metadata(components: list[EmissionComponent]):
    rows = []
    for c in components:
        rows.append({
            "name": c.name,
            "kind": c.kind,
            "center_hz": c.center_hz,
            "total_power_dbm": c.total_power_dbm,
            "harmonic_order": c.harmonic_order,
            "antenna_extra_loss_db": c.antenna_extra_loss_db,
        })
    return json.dumps(rows)


def save_receiver_npz(path: Path, rx: Receiver, spec_dbm: np.ndarray,
                      freq_hz: np.ndarray, time_s_rel: np.ndarray,
                      time_utc_iso: np.ndarray, time_local_iso: np.ndarray,
                      truth: dict, components: list[EmissionComponent],
                      waveform_source: str):
    payload = dict(
        dataset_id=f"EMILYX_LTE_V2_{rx.name}_{SIM_START_UTC}",
        freq_mhz=(freq_hz / 1e6).astype(np.float64),
        time_s_rel=time_s_rel,
        time_utc_iso=time_utc_iso,
        time_local_iso=time_local_iso,
        avg_specgram_dbm=spec_dbm.astype(np.float32),

        # IMPORTANT: these are the RECEIVER coordinates consumed by EMILY.
        receiver_name=rx.name,
        receiver_lat=np.float64(rx.lat_deg),
        receiver_lon=np.float64(rx.lon_deg),
        receiver_height_agl_m=np.float64(rx.height_agl_m),

        # Separate transmitter ground truth.
        transmitter_name=TX.name,
        transmitter_lat=np.float64(TX.lat_deg),
        transmitter_lon=np.float64(TX.lon_deg),
        transmitter_center_hz=np.float64(TX.center_frequency_hz),

        channel_spacing_hz=np.float64(CHANNEL_SPACING_HZ),
        rbw_hz=np.float64(RBW_HZ),
        lte_channel_bandwidth_hz=np.float64(LTE_CHANNEL_BW_HZ),
        lte_n_prb=np.int32(LTE_N_PRB),
        lte_dl_earfcn=np.int32(LTE_EARFCN_DL),
        srsran_iq_sample_rate_hz=np.float64(SRSRAN_IQ_SAMPLE_RATE_HZ),
        waveform_source=waveform_source,
        emission_components_json=serialize_emission_metadata(components),
        propagation_model="free-space + optional excess loss + slow shadowing",
        path_distance_km=np.float64(truth["distance_km"]),
    )
    if SAVE_COMPRESSED_NPZ:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


# =============================================================================
# LOCAL EMILY DETECTOR PREVIEW
# =============================================================================

def fast_rfi_flagger_short_time_fast(
    spec_dbm, freq_mhz, threshold=THRESHOLD, coarse_factor=COARSE_FACTOR,
    smooth_coarse_bins=SMOOTH_COARSE_BINS, noise_coarse_bins=NOISE_COARSE_BINS,
    dilate_time=DILATE_TIME, dilate_freq=DILATE_FREQ,
):
    x = np.asarray(spec_dbm, dtype=np.float32)
    nt, nf = x.shape
    nf_coarse = nf // coarse_factor
    n_trim = nf_coarse * coarse_factor
    if nf_coarse < 2:
        raise ValueError("coarse_factor is too large for the selected preview window")
    x_trim = x[:, :n_trim]
    x_blocks = x_trim.reshape(nt, nf_coarse, coarse_factor)
    x_coarse = np.nanmedian(x_blocks, axis=2).astype(np.float32)
    freq_coarse = freq_mhz[:n_trim].reshape(nf_coarse, coarse_factor).mean(axis=1)
    bg_coarse = median_filter(x_coarse, size=(1, smooth_coarse_bins), mode="nearest")
    bg = np.empty_like(x)
    for it in range(nt):
        bg[it] = np.interp(freq_mhz, freq_coarse, bg_coarse[it])
    resid = x - bg
    resid_blocks = resid[:, :n_trim].reshape(nt, nf_coarse, coarse_factor)
    abs_resid_coarse = np.nanmedian(np.abs(resid_blocks), axis=2).astype(np.float32)
    sigma_coarse = 1.4826 * median_filter(
        abs_resid_coarse, size=(1, noise_coarse_bins), mode="nearest"
    )
    sigma = np.empty_like(x)
    for it in range(nt):
        sigma[it] = np.interp(freq_mhz, freq_coarse, sigma_coarse[it])
    good = np.isfinite(sigma) & (sigma > 0)
    sigma[~good] = np.nanmedian(sigma[good]) if np.any(good) else 1.0
    snr = resid / sigma
    mask = snr > threshold
    if dilate_time > 0 or dilate_freq > 0:
        structure = np.ones((2*dilate_time+1, 2*dilate_freq+1), dtype=bool)
        mask = binary_dilation(mask, structure=structure)
    return mask, {"bg": bg, "resid": resid, "snr": snr, "sigma": sigma}


def get_band_tag(f_low_hz, f_high_hz):
    bands = [
        ("VHF", 30e6, 300e6), ("UHF", 300e6, 1e9), ("L", 1e9, 2e9),
        ("S", 2e9, 4e9), ("C", 4e9, 8e9), ("X", 8e9, 12e9),
    ]
    return "/".join(name for name, lo, hi in bands if f_high_hz >= lo and f_low_hz <= hi) or None


def extract_rfi_events(mask, spec_dbm, time_utc_iso, time_local_iso,
                       time_s_rel, freq_mhz):
    structure = np.ones(
        (2 * GROUP_GAP_TIME_PIX + 1, 2 * GROUP_GAP_FREQ_PIX + 1), dtype=bool
    )
    grouped = binary_dilation(mask, structure=structure)
    labels, _ = label(grouped)
    slices = find_objects(labels)
    rows = []
    nt = mask.shape[0]
    for slc in slices:
        if slc is None:
            continue
        t_slice, f_slice = slc
        submask = mask[t_slice, f_slice]
        if int(np.sum(submask)) < MIN_PIXELS:
            continue
        tt, ff = np.where(submask)
        t_idx = tt + t_slice.start
        f_idx = ff + f_slice.start
        t0, t1 = int(np.min(t_idx)), int(np.max(t_idx))
        f0, f1 = int(np.min(f_idx)), int(np.max(f_idx))
        vals_mw = dbm_to_mw(spec_dbm[t_idx, f_idx])
        f_low_hz = int(round(float(freq_mhz[f0] * 1e6)))
        f_high_hz = int(round(float(freq_mhz[f1] * 1e6)))
        mid = int(round(0.5 * (t0 + t1)))
        left = (t0 == 0)
        right = (t1 == nt - 1)
        rows.append({
            "t_start": None if left else str(time_utc_iso[t0]),
            "t_mid": str(time_utc_iso[mid]),
            "t_end": None if right else str(time_utc_iso[t1]),
            "t_start_local": None if left else str(time_local_iso[t0]),
            "t_mid_local": str(time_local_iso[mid]),
            "t_end_local": None if right else str(time_local_iso[t1]),
            "t_start_offset_sec": None if left else float(time_s_rel[t0]),
            "t_mid_offset_sec": float(time_s_rel[mid]),
            "t_end_offset_sec": None if right else float(time_s_rel[t1]),
            "duration_sec": None if (left or right) else float(time_s_rel[t1] - time_s_rel[t0]),
            "ongoing": bool(right),
            "left_censored": bool(left),
            "right_censored": bool(right),
            "f_low_hz": f_low_hz,
            "f_high_hz": f_high_hz,
            "f_center_hz": int(round(0.5 * (f_low_hz + f_high_hz))),
            "bw_hz": int(round(f_high_hz - f_low_hz)),
            "band_tag": get_band_tag(f_low_hz, f_high_hz),
            "intensity_kind": "mW",
            "intensity_value": float(np.nanmax(vals_mw)),
            "peak_power_mw": float(np.nanmax(vals_mw)),
            "mean_power_mw": float(np.nanmean(vals_mw)),
            "sum_power_mw": float(np.nansum(vals_mw)),
            "n_pixels": int(np.sum(submask)),
            "n_time_pixels": int(len(np.unique(t_idx))),
            "n_freq_pixels": int(len(np.unique(f_idx))),
        })
    return pd.DataFrame(rows)


def run_local_emily_windows(spec_dbm, freq_hz, time_utc_iso, time_local_iso, time_s_rel):
    all_events = []
    masks = {}
    centers = [LTE_CENTER_HZ, 2*LTE_CENTER_HZ, 3*LTE_CENTER_HZ]
    for center in centers:
        if center < FREQ_MIN_HZ or center > FREQ_MAX_HZ:
            continue
        sel = np.abs(freq_hz - center) <= LOCAL_EMILY_WINDOW_HALFSPAN_HZ
        if np.count_nonzero(sel) < COARSE_FACTOR * 2:
            continue
        freq_mhz = freq_hz[sel] / 1e6
        sub = spec_dbm[:, sel]
        mask, _ = fast_rfi_flagger_short_time_fast(sub, freq_mhz)
        events = extract_rfi_events(mask, sub, time_utc_iso, time_local_iso, time_s_rel, freq_mhz)
        if not events.empty:
            events["preview_window_center_hz"] = center
            all_events.append(events)
        masks[f"{center/1e6:.0f} MHz"] = (freq_mhz, mask)
    if all_events:
        out = pd.concat(all_events, ignore_index=True)
        out.insert(0, "event_id", np.arange(1, len(out)+1))
    else:
        out = pd.DataFrame()
    return out, masks


def events_to_candidate_incidents(events: pd.DataFrame, rx: Receiver,
                                  freq_hz: np.ndarray,
                                  time_utc_iso: np.ndarray,
                                  time_local_iso: np.ndarray,
                                  time_s_rel: np.ndarray,
                                  npz_name: str,
                                  components: list[EmissionComponent],
                                  truth: dict,
                                  waveform_source: str):
    if events.empty:
        return []
    obs_start = str(time_utc_iso[0])
    obs_mid = str(time_utc_iso[len(time_utc_iso)//2])
    obs_end = str(time_utc_iso[-1])
    obs_start_local = str(time_local_iso[0])
    obs_mid_local = str(time_local_iso[len(time_local_iso)//2])
    obs_end_local = str(time_local_iso[-1])
    records = []
    for _, row in events.iterrows():
        left = bool(row["left_censored"])
        right = bool(row["right_censored"])
        records.append({
            "org_id": None,
            "site_id": None,
            "inst_id": None,
            "data_type": "spectrometer",
            "timezone": LOCAL_TIMEZONE,
            "privacy_tier": "internal",
            "t_start": obs_start if left or pd.isna(row["t_start"]) else row["t_start"],
            "t_mid": row["t_mid"] if not pd.isna(row["t_mid"]) else obs_mid,
            "t_end": obs_end if right or pd.isna(row["t_end"]) else row["t_end"],
            "t_start_local": obs_start_local if left or pd.isna(row["t_start_local"]) else row["t_start_local"],
            "t_mid_local": row["t_mid_local"] if not pd.isna(row["t_mid_local"]) else obs_mid_local,
            "t_end_local": obs_end_local if right or pd.isna(row["t_end_local"]) else row["t_end_local"],
            "t_start_offset_sec": 0.0 if left else row["t_start_offset_sec"],
            "t_mid_offset_sec": row["t_mid_offset_sec"],
            "t_end_offset_sec": float(time_s_rel[-1]) if right else row["t_end_offset_sec"],
            "duration_sec": float(time_s_rel[-1]) if (left and right) else row["duration_sec"],
            "ongoing": bool(row["ongoing"]),
            "time_structure": "continuous" if (left or right) else "bounded",
            "f_low_hz": int(row["f_low_hz"]),
            "f_high_hz": int(row["f_high_hz"]),
            "f_center_hz": int(row["f_center_hz"]),
            "bw_hz": int(row["bw_hz"]),
            "band_tag": row["band_tag"],
            "intensity_kind": "mW",
            "intensity_value": float(row["intensity_value"]),
            "Directional": False,
            "RA": 0, "Dec": 0, "Az": 0, "El": 0,

            # IMPORTANT: EMILY observation location = receiver, not transmitter.
            "lat": rx.lat_deg,
            "lon": rx.lon_deg,
            "altitude": None,

            "original_file": npz_name,
            "file_timestamp": obs_start,
            "file_freq_low_hz": int(round(float(freq_hz.min()))),
            "file_freq_high_hz": int(round(float(freq_hz.max()))),
            "file_freq_resolution_hz": float(np.median(np.diff(freq_hz))),
            "file_time_resolution_sec": float(np.median(np.diff(time_s_rel))),
            "observation_start": obs_start,
            "observation_mid": obs_mid,
            "observation_end": obs_end,
            "observation_duration_sec": float(time_s_rel[-1] - time_s_rel[0]),
            "n_time_samples": int(len(time_s_rel)),
            "n_freq_channels": int(len(freq_hz)),
            "n_pixels": int(row["n_pixels"]),
            "n_time_pixels": int(row["n_time_pixels"]),
            "n_freq_pixels": int(row["n_freq_pixels"]),
            "method": "EMILY local preview: median/MAD + morphology + connected components",
            "metadata": {
                "synthetic": True,
                "simulation": "EMILY-X LTE v2",
                "waveform_source": waveform_source,
                "true_transmitter": TX.name,
                "true_transmitter_lat": TX.lat_deg,
                "true_transmitter_lon": TX.lon_deg,
                "true_lte_center_hz": LTE_CENTER_HZ,
                "true_lte_bandwidth_hz": LTE_CHANNEL_BW_HZ,
                "true_lte_n_prb": LTE_N_PRB,
                "true_lte_dl_earfcn": LTE_EARFCN_DL,
                "emission_components": json.loads(serialize_emission_metadata(components)),
                "propagation_model": "free-space + optional excess loss + slow shadowing",
                "distance_km": truth["distance_km"],
            },
        })
    return records


# =============================================================================
# PLOTS
# =============================================================================

def plot_geometry(out: Path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter([TX.lon_deg], [TX.lat_deg], marker="*", s=180, label="LTE transmitter")
    for rx in RECEIVERS:
        ax.scatter([rx.lon_deg], [rx.lat_deg], s=65, label=rx.name)
        ax.plot([TX.lon_deg, rx.lon_deg], [TX.lat_deg, rx.lat_deg], linewidth=1)
    ax.set_xlabel("Longitude [deg]")
    ax.set_ylabel("Latitude [deg]")
    ax.set_title("EMILY-X LTE simulation geometry")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_waveform(iq: np.ndarray, fs: float, out: Path):
    nshow = min(len(iq), int(0.0015 * fs))
    f, p = normalized_psd(iq, fs, nperseg=min(32768, len(iq)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    axes[0].plot(np.arange(nshow) / fs * 1e3, np.real(iq[:nshow]), linewidth=0.7)
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("I amplitude")
    axes[0].set_title("Captured srsRAN eNodeB complex-baseband waveform")
    axes[0].grid(True, alpha=0.25)
    p_db = 10*np.log10(np.maximum(p, 1e-30))
    p_db -= np.nanmax(p_db)
    axes[1].plot(f/1e6, p_db, linewidth=0.8)
    axes[1].set_xlabel("Offset from LTE carrier [MHz]")
    axes[1].set_ylabel("Relative PSD [dB]")
    axes[1].set_title("Genuine LTE eNodeB baseband spectrum")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_tx_wideband(freq_hz: np.ndarray, tx_rbw_dbm: np.ndarray, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(freq_hz/1e6, tx_rbw_dbm, linewidth=0.8)
    for f in [LTE_CENTER_HZ, 2*LTE_CENTER_HZ, 3*LTE_CENTER_HZ]:
        if FREQ_MIN_HZ <= f <= FREQ_MAX_HZ:
            ax.axvline(f/1e6, linestyle="--", linewidth=0.7)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Conducted power / 140-kHz RBW [dBm]")
    ax.set_title("Transmitter spectrum: LTE + OOB leakage + harmonics + spurious products")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_fundamental_zoom(freq_hz: np.ndarray, tx_rbw_dbm: np.ndarray, out: Path):
    sel = np.abs(freq_hz - LTE_CENTER_HZ) <= 35e6
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(freq_hz[sel]/1e6, tx_rbw_dbm[sel], linewidth=0.9)
    ax.axvspan((LTE_CENTER_HZ-LTE_CHANNEL_BW_HZ/2)/1e6,
               (LTE_CENTER_HZ+LTE_CHANNEL_BW_HZ/2)/1e6, alpha=0.08)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Conducted power / 140-kHz RBW [dBm]")
    ax.set_title("800-MHz LTE carrier: in-band signal and out-of-band/filter leakage")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_antenna_patterns(out: Path):
    az = np.linspace(-180, 180, 721)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for order, label_text in [(1, "800 MHz fundamental"), (2, "1600 MHz 2nd harmonic"), (3, "2400 MHz 3rd harmonic")]:
        gains = [tx_antenna_gain_db(TX, TX.sector_azimuth_deg+a, 0.0, harmonic_order=order) for a in az]
        ax.plot(az, gains, label=label_text)
    ax.set_xlabel("Azimuth offset from sector boresight [deg]")
    ax.set_ylabel("Gain [dBi]")
    ax.set_title("Directional sector pattern with harmonic antenna-efficiency penalties")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_average_spectra(freq_hz: np.ndarray, tx_dbm: np.ndarray, rx_avg: dict, out: Path):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(freq_hz/1e6, tx_dbm, linewidth=0.8, label="Tx conducted")
    for name, y in rx_avg.items():
        ax.plot(freq_hz/1e6, y, linewidth=0.7, label=name)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Average power / 140-kHz RBW [dBm]")
    ax.set_title("200--2500 MHz average spectrum: transmitter versus receivers")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_emission_windows(time_s_rel, freq_hz, tx_rbw_dbm, rx_specs: dict, out: Path):
    centers = [LTE_CENTER_HZ, 2*LTE_CENTER_HZ, 3*LTE_CENTER_HZ]
    valid = [c for c in centers if FREQ_MIN_HZ <= c <= FREQ_MAX_HZ]
    fig, axes = plt.subplots(len(valid), 3, figsize=(15, 4.2*len(valid)), squeeze=False)
    for row, center in enumerate(valid):
        sel = np.abs(freq_hz-center) <= 25e6
        fmhz = freq_hz[sel]/1e6
        extent = [fmhz[0], fmhz[-1], time_s_rel[-1]/60.0, time_s_rel[0]/60.0]
        tx2d = np.repeat(tx_rbw_dbm[sel][None, :], len(time_s_rel), axis=0)
        txrel = tx2d - np.nanmax(tx2d)
        axes[row,0].imshow(txrel, aspect="auto", extent=extent)
        axes[row,0].set_title(f"Tx around {center/1e6:.0f} MHz (relative dB)")
        all_rx = np.concatenate([v[:,sel].ravel() for v in rx_specs.values()])
        vmin = np.nanpercentile(all_rx, 1)
        vmax = np.nanpercentile(all_rx, 99.8)
        for col, (name, arr) in enumerate(rx_specs.items(), start=1):
            axes[row,col].imshow(arr[:,sel], aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
            axes[row,col].set_title(f"{name} around {center/1e6:.0f} MHz")
        for col in range(3):
            axes[row,col].set_xlabel("Frequency [MHz]")
            axes[row,col].set_ylabel("Time [min]")
    fig.suptitle("Propagation of the fundamental and harmonics")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_component_received_power(time_s_rel, truth_by_rx: dict, out: Path):
    fig, axes = plt.subplots(len(RECEIVERS), 1, figsize=(11, 7), sharex=True)
    if len(RECEIVERS) == 1:
        axes = [axes]
    for ax, rx in zip(axes, RECEIVERS):
        truth = truth_by_rx[rx.name]
        for comp in truth["component_truth"]:
            ax.plot(time_s_rel/60.0, comp["rx_total_component_dbm_t"], label=comp["component"])
        ax.set_ylabel("Total component power [dBm]")
        ax.set_title(rx.name)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Time [min]")
    fig.suptitle("Received power of physically related emission products")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

# =============================================================================
# V3: TERRAIN-AWARE PROPAGATION + TIME-VARYING GENUINE LTE LOAD
# =============================================================================

# P.452 / SRTM defaults.  These are explicit scenario settings and are written
# to simulation_config.json so they can be changed/reproduced later.
P452_PROFILE_STEP_M = 100.0
P452_TIME_PERCENT = 50.0          # median/basic path condition, configurable via CLI
P452_FREQ_ANCHOR_STEP_HZ = 2.0e6 # P.452 is smooth over our narrow emission windows
P452_POLARIZATION = 0             # 0 horizontal, 1 vertical
EARTH_RADIUS_M = 6_371_008.8

# Default load timeline.  Fractions sum to one.  Missing states are replaced by
# the nearest available load state, so v3 can still run with only a loaded IQ
# file while making it obvious that the source is then static.
DEFAULT_LOAD_SEQUENCE = [
    ("idle",   0.10),
    ("light",  0.15),
    ("loaded", 0.15),
    ("medium", 0.15),
    ("idle",   0.10),
    ("loaded", 0.15),
    ("light",  0.10),
    ("medium", 0.10),
]
LOAD_RANK = {"idle": 0, "light": 1, "medium": 2, "loaded": 3}
GLOBAL_FREQ_HZ_FOR_PLOTS = None


def _quantity_value(x):
    """Return ndarray values from an astropy/pycraf Quantity-like object."""
    return np.asarray(getattr(x, "value", x))


def load_srsran_iq_state(path: Path, sample_rate_hz: float,
                         max_capture_seconds: float,
                         template_window_ms: float,
                         template_hop_ms: float,
                         max_templates: int):
    """Load one genuine LTE state and split it into normalized IQ templates.

    The complete short capture is used to measure the state's aggregate RMS.
    Individual windows retain their own pre-normalization RMS, which captures
    genuine scheduler/resource-allocation fluctuations within one nominal load
    state.  Every returned IQ template is normalized only *after* its raw RMS is
    measured so the RF model can apply the physical power offset explicitly.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    max_samples = int(round(sample_rate_hz * max_capture_seconds))
    x = np.fromfile(path, dtype=np.complex64, count=max_samples)
    if len(x) < int(0.020 * sample_rate_hz):
        raise ValueError(f"{path} contains only {len(x)} samples; capture at least ~20 ms.")

    x = x.astype(np.complex64, copy=False)
    x_mean = np.mean(x)
    capture_rms = float(np.sqrt(np.mean(np.abs(x - x_mean) ** 2)))
    if not np.isfinite(capture_rms) or capture_rms <= 0:
        raise ValueError(f"{path} has invalid/zero signal power.")

    win = int(round(template_window_ms * 1e-3 * sample_rate_hz))
    hop = int(round(template_hop_ms * 1e-3 * sample_rate_hz))
    if win < int(round(0.020 * sample_rate_hz)):
        raise ValueError("--template-window-ms must be at least 20 ms")
    if hop <= 0:
        raise ValueError("--template-hop-ms must be > 0")
    if len(x) < win:
        raise ValueError(
            f"{path.name} has {len(x)/sample_rate_hz:.3f} s, shorter than the "
            f"requested {template_window_ms/1000:.3f}-s template window"
        )

    starts = list(range(0, len(x) - win + 1, hop))
    if max_templates > 0 and len(starts) > max_templates:
        # Spread a limited template budget across the complete capture rather
        # than taking only its beginning.
        choose = np.linspace(0, len(starts)-1, max_templates).round().astype(int)
        starts = [starts[i] for i in np.unique(choose)]

    templates = []
    preview_samples = int(round(min(0.020, template_window_ms*1e-3) * sample_rate_hz))
    for template_index, start in enumerate(starts):
        w = np.array(x[start:start+win], dtype=np.complex64, copy=True)
        w -= np.mean(w)
        raw_rms = float(np.sqrt(np.mean(np.abs(w) ** 2)))
        if not np.isfinite(raw_rms) or raw_rms <= 0:
            continue
        w /= raw_rms
        templates.append({
            "template_index": int(template_index),
            "start_s": float(start / sample_rate_hz),
            "end_s": float((start + win) / sample_rate_hz),
            "raw_rms": raw_rms,
            "iq": w,
            "iq_excerpt": w[:preview_samples].copy(),
        })

    if not templates:
        raise ValueError(f"No valid IQ templates could be extracted from {path}")

    return {
        "source": path.name,
        "raw_rms": capture_rms,
        "capture_seconds_used": float(len(x) / sample_rate_hz),
        "templates": templates,
    }


def scaled_components_for_load(components: list[EmissionComponent], relative_power_db: float):
    """Apply measured state-to-state power scaling to traffic-dependent products.

    Narrow LO/reference-clock spurs remain at their full-scale absolute scenario
    levels.  Fundamental/OOB and harmonics track the measured baseband RMS.
    """
    out = []
    for c in components:
        delta = 0.0 if c.kind == "spurious" else relative_power_db
        out.append(EmissionComponent(
            name=c.name,
            kind=c.kind,
            center_hz=c.center_hz,
            total_power_dbm=c.total_power_dbm + delta,
            psd_offset_hz=c.psd_offset_hz,
            psd_norm_per_hz=c.psd_norm_per_hz,
            harmonic_order=c.harmonic_order,
            antenna_extra_loss_db=c.antenna_extra_loss_db,
        ))
    return out


def nearest_available_state(requested: str, available: list[str]):
    if requested in available:
        return requested
    target = LOAD_RANK.get(requested, 3)
    return min(available, key=lambda s: (abs(LOAD_RANK.get(s, 3) - target), LOAD_RANK.get(s, 3)))


def make_default_load_schedule(nt: int, available_states: list[str]):
    if not available_states:
        raise ValueError("No LTE waveform states are available.")
    schedule = np.empty(nt, dtype="U16")
    segments = []
    start = 0
    cumulative = 0.0
    for i, (requested, frac) in enumerate(DEFAULT_LOAD_SEQUENCE):
        cumulative += frac
        end = nt if i == len(DEFAULT_LOAD_SEQUENCE)-1 else int(round(cumulative * nt))
        end = min(max(end, start), nt)
        actual = nearest_available_state(requested, available_states)
        schedule[start:end] = actual
        if end > start:
            segments.append({
                "start_s": float(start * TIME_RESOLUTION_S),
                "end_s": float(end * TIME_RESOLUTION_S),
                "requested_state": requested,
                "state": actual,
            })
        start = end
    if start < nt:
        schedule[start:] = nearest_available_state("loaded", available_states)
    return schedule, segments


def load_schedule_from_json(path: Path, nt: int, available_states: list[str]):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("schedule JSON must be a non-empty list of segments")
    schedule = np.empty(nt, dtype="U16")
    schedule[:] = nearest_available_state("idle", available_states)
    segments = []
    for row in data:
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        requested = str(row["state"])
        actual = nearest_available_state(requested, available_states)
        i0 = max(0, int(round(start_s / TIME_RESOLUTION_S)))
        i1 = min(nt, int(round(end_s / TIME_RESOLUTION_S)))
        if i1 <= i0:
            continue
        schedule[i0:i1] = actual
        segments.append({
            "start_s": float(i0 * TIME_RESOLUTION_S),
            "end_s": float(i1 * TIME_RESOLUTION_S),
            "requested_state": requested,
            "state": actual,
        })
    return schedule, segments


def make_template_schedule(state_schedule: np.ndarray, state_models: dict,
                           rng: np.random.Generator):
    """Assign one genuine IQ-derived spectral template to every simulated second.

    For each load state, templates are consumed in shuffled cycles.  Every
    template is used once before reshuffling, avoiding both a fixed 5-s periodic
    pattern and accidental long runs of a single template.  The schedule is
    generated once and shared by all receivers so transmitter variation is
    physically common to every observation site.
    """
    out = np.full(len(state_schedule), -1, dtype=np.int16)
    for state_name, model in state_models.items():
        tidx = np.flatnonzero(state_schedule == state_name)
        n_templates = len(model["templates"])
        if len(tidx) == 0 or n_templates == 0:
            continue
        assigned = []
        while len(assigned) < len(tidx):
            assigned.extend(rng.permutation(n_templates).tolist())
        out[tidx] = np.asarray(assigned[:len(tidx)], dtype=np.int16)
    if np.any(out < 0):
        raise RuntimeError("Template schedule contains unassigned time samples")
    return out


def standard_atmosphere_from_height(height_m: float):
    """Simple ISA troposphere estimate for P.452 midpoint T/P inputs."""
    h = float(np.clip(height_m, -500.0, 11000.0))
    t_k = 288.15 - 0.0065 * h
    p_hpa = 1013.25 * (1.0 - 2.25577e-5 * h) ** 5.25588
    return float(t_k), float(p_hpa)


def terrain_profile_for_receiver(rx: Receiver, srtm_dir: Path, download_missing: bool):
    if not HAVE_PYCRAF:
        raise RuntimeError("pycraf is required for P.452 propagation but could not be imported")
    srtm_dir.mkdir(parents=True, exist_ok=True)
    download_mode = "missing" if download_missing else "never"
    with pathprof.SrtmConf.set(
        srtm_dir=str(srtm_dir.resolve()),
        download=download_mode,
        server="viewpano",
        interp="linear",
    ):
        (
            lons, lats, distance,
            distances, heights,
            bearing, back_bearing, back_bearings,
        ) = pathprof.srtm_height_profile(
            TX.lon_deg * u.deg, TX.lat_deg * u.deg,
            rx.lon_deg * u.deg, rx.lat_deg * u.deg,
            P452_PROFILE_STEP_M * u.m,
        )

    h_m = np.asarray(heights.to(u.m).value, dtype=np.float64)
    if not np.all(np.isfinite(h_m)):
        raise RuntimeError(f"SRTM profile for {rx.name} contains non-finite terrain heights")
    # Nevada endpoints are far above sea level.  A zero-valued profile usually
    # means the requested SRTM tile was unavailable rather than actual terrain.
    if np.nanmax(np.abs(h_m)) < 10.0:
        raise RuntimeError(
            f"SRTM profile for {rx.name} is essentially all zeros. "
            "Check internet access/SRTM tiles or rerun with --propagation fspl for diagnostics only."
        )

    d_m = np.asarray(distances.to(u.m).value, dtype=np.float64)
    tx_ground_m = float(h_m[0])
    rx_ground_m = float(h_m[-1])
    tx_abs_m = tx_ground_m + TX.height_agl_m
    rx_abs_m = rx_ground_m + rx.height_agl_m
    total_d_m = float(distance.to(u.m).value)
    curvature_drop_m = total_d_m**2 / (2.0 * EARTH_RADIUS_M)
    direct_elevation_deg = math.degrees(math.atan2(
        rx_abs_m - tx_abs_m - curvature_drop_m,
        total_d_m,
    ))

    midpoint_height_m = float(h_m[len(h_m)//2])
    temperature_k, pressure_hpa = standard_atmosphere_from_height(midpoint_height_m)

    return {
        "lons": lons,
        "lats": lats,
        "distance": distance,
        "distances": distances,
        "heights": heights,
        "bearing": bearing,
        "back_bearing": back_bearing,
        "back_bearings": back_bearings,
        "distance_m": total_d_m,
        "distance_km": total_d_m / 1e3,
        "tx_ground_elevation_m": tx_ground_m,
        "rx_ground_elevation_m": rx_ground_m,
        "midpoint_ground_elevation_m": midpoint_height_m,
        "temperature_k": temperature_k,
        "pressure_hpa": pressure_hpa,
        "direct_elevation_tx_to_rx_deg": direct_elevation_deg,
    }


def build_active_union(freq_hz: np.ndarray, state_models: dict):
    active = np.zeros(len(freq_hz), dtype=bool)
    for model in state_models.values():
        for template in model["templates"]:
            for frac in template["fractions"].values():
                active |= (frac > 1e-16)
    return active


def prepare_propagation(rx: Receiver, freq_hz: np.ndarray, active_mask: np.ndarray,
                        propagation_mode: str, srtm_dir: Path,
                        download_srtm: bool, p452_time_percent: float):
    """Prepare one reusable path-loss vector for all LTE load states."""
    base_geom = link_geometry(TX, rx)

    if propagation_mode == "fspl":
        path_loss = np.full(len(freq_hz), np.nan, dtype=np.float64)
        path_loss[active_mask] = free_space_path_loss_db(base_geom["distance_m"], freq_hz[active_mask])
        center_losses = {
            str(int(round(f))): float(free_space_path_loss_db(base_geom["distance_m"], f))
            for f in [LTE_CENTER_HZ, 2*LTE_CENTER_HZ, 3*LTE_CENTER_HZ,
                      LTE_CENTER_HZ-30.72e6, LTE_CENTER_HZ+30.72e6]
            if FREQ_MIN_HZ <= f <= FREQ_MAX_HZ
        }
        return {
            "mode": "free-space",
            "path_loss_db": path_loss,
            "center_losses_db": center_losses,
            "geometry": base_geom,
            "terrain": None,
            "p452": None,
        }

    if not HAVE_PYCRAF:
        raise RuntimeError("--propagation p452 requires pycraf")

    terrain = terrain_profile_for_receiver(rx, srtm_dir, download_srtm)
    geom = dict(base_geom)
    geom["distance_m"] = terrain["distance_m"]
    geom["distance_km"] = terrain["distance_km"]
    geom["elevation_tx_to_rx_deg"] = terrain["direct_elevation_tx_to_rx_deg"]

    f_active = freq_hz[active_mask]
    if len(f_active) == 0:
        raise RuntimeError("No active emission frequencies found for propagation")

    anchors = np.round(f_active / P452_FREQ_ANCHOR_STEP_HZ) * P452_FREQ_ANCHOR_STEP_HZ
    extra_centers = np.array([
        LTE_CENTER_HZ,
        2*LTE_CENTER_HZ,
        3*LTE_CENTER_HZ,
        LTE_CENTER_HZ - 30.72e6,
        LTE_CENTER_HZ + 30.72e6,
    ], dtype=np.float64)
    extra_centers = extra_centers[(extra_centers >= FREQ_MIN_HZ) & (extra_centers <= FREQ_MAX_HZ)]
    anchors = np.unique(np.concatenate([anchors, extra_centers]))
    anchors.sort()

    results = pathprof.losses_complete(
        anchors * 1e-9 * u.GHz,
        terrain["temperature_k"] * u.K,
        terrain["pressure_hpa"] * u.hPa,
        TX.lon_deg * u.deg, TX.lat_deg * u.deg,
        rx.lon_deg * u.deg, rx.lat_deg * u.deg,
        TX.height_agl_m * u.m,
        rx.height_agl_m * u.m,
        P452_PROFILE_STEP_M * u.m,
        float(p452_time_percent) * u.percent,
        G_t=0.0 * cnv.dBi,
        G_r=0.0 * cnv.dBi,
        omega=0.0 * u.percent,
        zone_t=pathprof.CLUTTER.UNKNOWN,
        zone_r=pathprof.CLUTTER.UNKNOWN,
        polarization=P452_POLARIZATION,
        version=16,
        hprof_dists=terrain["distances"],
        hprof_heights=terrain["heights"],
        hprof_bearing=terrain["bearing"],
        hprof_backbearing=terrain["back_bearing"],
    )
    anchor_losses = np.asarray(results["L_b"].to(cnv.dB).value, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(anchor_losses)):
        raise RuntimeError(f"P.452 returned non-finite path loss for {rx.name}")

    path_loss = np.full(len(freq_hz), np.nan, dtype=np.float64)
    path_loss[active_mask] = np.interp(f_active, anchors, anchor_losses)
    center_losses = {str(int(round(f))): float(np.interp(f, anchors, anchor_losses)) for f in extra_centers}

    path_type = _quantity_value(results.get("path_type", np.zeros_like(anchor_losses))).reshape(-1)
    eps_pt = _quantity_value(results.get("eps_pt", np.full_like(anchor_losses, np.nan))).reshape(-1)
    eps_pr = _quantity_value(results.get("eps_pr", np.full_like(anchor_losses, np.nan))).reshape(-1)

    return {
        "mode": "ITU-R P.452-16 + SRTM",
        "path_loss_db": path_loss,
        "center_losses_db": center_losses,
        "geometry": geom,
        "terrain": terrain,
        "p452": {
            "time_percent": float(p452_time_percent),
            "frequency_anchor_hz": anchors,
            "L_b_db": anchor_losses,
            "path_type": path_type,
            "eps_pt_deg": eps_pt,
            "eps_pr_deg": eps_pr,
            "temperature_k": terrain["temperature_k"],
            "pressure_hpa": terrain["pressure_hpa"],
        },
    }


def make_receiver_spectrogram_v3(rx: Receiver, tx: Transmitter,
                                 freq_hz: np.ndarray,
                                 state_models: dict,
                                 state_schedule: np.ndarray,
                                 template_schedule: np.ndarray,
                                 propagation: dict,
                                 nt: int,
                                 rng: np.random.Generator):
    """Time-varying LTE load + within-state LTE template variation + propagation."""
    nf = len(freq_hz)
    geom = propagation["geometry"]
    noise_floor_dbm = thermal_noise_floor_dbm(RBW_HZ, rx.noise_figure_db)

    x = np.linspace(0.0, 8.0 * np.pi, nf)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    bandpass_db = (
        BANDPASS_RIPPLE_DB * np.sin(x + phase)
        + 0.25 * BANDPASS_RIPPLE_DB * np.sin(0.27 * x + 0.4 * phase)
    ).astype(np.float32)

    gain_drift = gaussian_filter1d(rng.normal(size=nt), sigma=25.0)
    gain_drift -= np.mean(gain_drift)
    if np.std(gain_drift) > 0:
        gain_drift *= GAIN_DRIFT_DB / np.std(gain_drift)
    gain_drift = gain_drift.astype(np.float32)

    spec_dbm = np.empty((nt, nf), dtype=np.float32)
    chunk = 50
    for i0 in range(0, nt, chunk):
        i1 = min(nt, i0 + chunk)
        jitter = rng.normal(0.0, NOISE_JITTER_DB, size=(i1-i0, nf)).astype(np.float32)
        spec_dbm[i0:i1] = noise_floor_dbm + bandpass_db[None, :] + gain_drift[i0:i1, None] + jitter

    shadowing_db = make_slow_shadowing(
        nt, SHADOWING_SIGMA_DB, SHADOWING_CORRELATION_S, TIME_RESOLUTION_S, rng
    )

    template_signal_mw = {}
    template_component_center_dbm = {}
    component_names = []

    for state_name, model in state_models.items():
        for template in model["templates"]:
            ti = int(template["template_index"])
            key = (state_name, ti)
            total_signal_mw = np.zeros(nf, dtype=np.float64)
            center_by_component = {}
            for comp in template["components"]:
                if comp.name not in component_names:
                    component_names.append(comp.name)
                frac = template["fractions"][comp.name]
                active = frac > 1e-16
                if not np.any(active):
                    continue
                idx = np.flatnonzero(active)
                frac_active = frac[idx]
                path_loss_db = propagation["path_loss_db"][idx]
                if np.any(~np.isfinite(path_loss_db)):
                    raise RuntimeError(
                        f"Missing path loss values for {rx.name}/{state_name}/template{ti}/{comp.name}"
                    )

                tx_gain_db = tx_antenna_gain_db(
                    tx,
                    geom["bearing_tx_to_rx_deg"],
                    geom["elevation_tx_to_rx_deg"],
                    harmonic_order=comp.harmonic_order,
                    extra_loss_db=comp.antenna_extra_loss_db,
                )
                tx_bin_dbm = comp.total_power_dbm + 10.0*np.log10(np.maximum(frac_active, 1e-30))
                rx_bin_dbm = (
                    tx_bin_dbm + tx_gain_db + rx.antenna_gain_dbi
                    - path_loss_db - rx.excess_path_loss_db
                )
                total_signal_mw[idx] += dbm_to_mw(rx_bin_dbm)

                center_key = str(int(round(comp.center_hz)))
                if center_key in propagation["center_losses_db"]:
                    center_loss_db = propagation["center_losses_db"][center_key]
                else:
                    finite = np.isfinite(propagation["path_loss_db"])
                    center_loss_db = float(np.interp(
                        comp.center_hz, freq_hz[finite], propagation["path_loss_db"][finite]
                    ))
                center_by_component[comp.name] = {
                    "center_hz": comp.center_hz,
                    "kind": comp.kind,
                    "tx_total_power_dbm": comp.total_power_dbm,
                    "tx_antenna_gain_db": tx_gain_db,
                    "path_loss_center_db": center_loss_db,
                    "rx_center_no_shadow_dbm": (
                        comp.total_power_dbm + tx_gain_db + rx.antenna_gain_dbi
                        - center_loss_db - rx.excess_path_loss_db
                    ),
                }
            template_signal_mw[key] = total_signal_mw
            template_component_center_dbm[key] = center_by_component

    # Add exactly the transmitter template scheduled at each second.  Propagation
    # shadowing remains common to all products from the same physical path.
    for state_name, model in state_models.items():
        for template in model["templates"]:
            ti = int(template["template_index"])
            tidx = np.flatnonzero((state_schedule == state_name) & (template_schedule == ti))
            if len(tidx) == 0:
                continue
            scale = 10.0 ** (shadowing_db[tidx].astype(np.float64) / 10.0)
            signal_mw = scale[:, None] * template_signal_mw[(state_name, ti)][None, :]
            spec_dbm[tidx] = mw_to_dbm(dbm_to_mw(spec_dbm[tidx]) + signal_mw).astype(np.float32)

    component_truth = []
    for comp_name in component_names:
        rx_t = np.full(nt, np.nan, dtype=np.float32)
        tx_t = np.full(nt, np.nan, dtype=np.float32)
        truth_template = None
        for state_name, model in state_models.items():
            for template in model["templates"]:
                ti = int(template["template_index"])
                tidx = np.flatnonzero((state_schedule == state_name) & (template_schedule == ti))
                c = template_component_center_dbm[(state_name, ti)].get(comp_name)
                if c is None or len(tidx) == 0:
                    continue
                truth_template = c
                rx_t[tidx] = c["rx_center_no_shadow_dbm"] + shadowing_db[tidx]
                tx_t[tidx] = c["tx_total_power_dbm"]
        if truth_template is None:
            continue
        component_truth.append({
            "component": comp_name,
            "kind": truth_template["kind"],
            "center_hz": truth_template["center_hz"],
            "tx_antenna_gain_db": truth_template["tx_antenna_gain_db"],
            "path_loss_center_db": truth_template["path_loss_center_db"],
            "median_tx_total_power_dbm": float(np.nanmedian(tx_t)),
            "median_rx_total_component_dbm": float(np.nanmedian(rx_t)),
            "tx_total_power_dbm_t": tx_t,
            "rx_total_component_dbm_t": rx_t,
        })

    truth = {
        **geom,
        "noise_floor_per_rbw_dbm": noise_floor_dbm,
        "shadowing_db_t": shadowing_db,
        "component_truth": component_truth,
        "propagation_model": propagation["mode"],
        "center_losses_db": propagation["center_losses_db"],
    }
    if propagation["terrain"] is not None:
        truth.update({
            "tx_ground_elevation_m": propagation["terrain"]["tx_ground_elevation_m"],
            "rx_ground_elevation_m": propagation["terrain"]["rx_ground_elevation_m"],
            "midpoint_ground_elevation_m": propagation["terrain"]["midpoint_ground_elevation_m"],
        })
    return spec_dbm, truth


def build_tx_average_spectrum(state_models: dict, state_schedule: np.ndarray,
                              template_schedule: np.ndarray):
    total = None
    nt = len(state_schedule)
    for state_name, model in state_models.items():
        for template in model["templates"]:
            ti = int(template["template_index"])
            weight = float(np.count_nonzero(
                (state_schedule == state_name) & (template_schedule == ti)
            )) / max(nt, 1)
            if weight <= 0:
                continue
            mw = dbm_to_mw(template["tx_rbw_dbm"])
            total = weight*mw if total is None else total + weight*mw
    return mw_to_dbm(total, floor_dbm=-220.0)


def save_receiver_npz_v3(path: Path, rx: Receiver, spec_dbm: np.ndarray,
                         freq_hz: np.ndarray, time_s_rel: np.ndarray,
                         time_utc_iso: np.ndarray, time_local_iso: np.ndarray,
                         truth: dict, state_models: dict,
                         state_schedule: np.ndarray, template_schedule: np.ndarray,
                         load_segments: list, waveform_source: str):
    state_meta = {}
    for name, model in state_models.items():
        state_meta[name] = {
            "source": model["source"],
            "raw_rms": model["raw_rms"],
            "relative_power_db": model["relative_power_db"],
            "template_count": len(model["templates"]),
            "templates": [
                {
                    "template_index": int(t["template_index"]),
                    "start_s": t["start_s"],
                    "end_s": t["end_s"],
                    "raw_rms": t["raw_rms"],
                    "relative_power_db": t["relative_power_db"],
                }
                for t in model["templates"]
            ],
        }
    payload = dict(
        dataset_id=f"EMILYX_LTE_V31_{rx.name}_{SIM_START_UTC}",
        freq_mhz=(freq_hz/1e6).astype(np.float64),
        time_s_rel=time_s_rel,
        time_utc_iso=time_utc_iso,
        time_local_iso=time_local_iso,
        avg_specgram_dbm=spec_dbm.astype(np.float32),
        lte_state_t=state_schedule.astype("U16"),
        lte_template_index_t=template_schedule.astype(np.int16),
        lte_load_segments_json=json.dumps(load_segments),
        lte_states_json=json.dumps(state_meta),
        receiver_name=rx.name,
        receiver_lat=np.float64(rx.lat_deg),
        receiver_lon=np.float64(rx.lon_deg),
        receiver_height_agl_m=np.float64(rx.height_agl_m),
        transmitter_name=TX.name,
        transmitter_lat=np.float64(TX.lat_deg),
        transmitter_lon=np.float64(TX.lon_deg),
        transmitter_center_hz=np.float64(TX.center_frequency_hz),
        channel_spacing_hz=np.float64(CHANNEL_SPACING_HZ),
        rbw_hz=np.float64(RBW_HZ),
        lte_channel_bandwidth_hz=np.float64(LTE_CHANNEL_BW_HZ),
        lte_n_prb=np.int32(LTE_N_PRB),
        lte_dl_earfcn=np.int32(LTE_EARFCN_DL),
        srsran_iq_sample_rate_hz=np.float64(SRSRAN_IQ_SAMPLE_RATE_HZ),
        waveform_source=waveform_source,
        propagation_model=truth["propagation_model"],
        path_distance_km=np.float64(truth["distance_km"]),
    )
    if "tx_ground_elevation_m" in truth:
        payload.update(
            tx_ground_elevation_m=np.float64(truth["tx_ground_elevation_m"]),
            rx_ground_elevation_m=np.float64(truth["rx_ground_elevation_m"]),
            midpoint_ground_elevation_m=np.float64(truth["midpoint_ground_elevation_m"]),
        )
    if SAVE_COMPRESSED_NPZ:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def patch_candidate_incidents_v3(records, truth, state_models, load_segments, propagation_mode):
    state_meta = {
        name: {
            "source": m["source"],
            "raw_rms": m["raw_rms"],
            "relative_power_db": m["relative_power_db"],
            "template_count": len(m["templates"]),
        }
        for name, m in state_models.items()
    }
    for rec in records:
        rec["metadata"]["simulation"] = "EMILY-X LTE v3.1"
        rec["metadata"]["propagation_model"] = propagation_mode
        rec["metadata"]["lte_states"] = state_meta
        rec["metadata"]["lte_load_segments"] = load_segments
        if "tx_ground_elevation_m" in truth:
            rec["metadata"]["tx_ground_elevation_m"] = truth["tx_ground_elevation_m"]
            rec["metadata"]["rx_ground_elevation_m"] = truth["rx_ground_elevation_m"]
    return records


def plot_lte_states(state_models: dict, fs: float, out: Path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    for name, model in state_models.items():
        representative = model["templates"][0]
        iq = representative["iq_excerpt"]
        nshow = min(len(iq), int(0.00075*fs))
        axes[0].plot(np.arange(nshow)/fs*1e3, np.real(iq[:nshow]),
                     linewidth=0.55, alpha=0.8, label=f"{name} (template 0)")

        # Build the state-average baseband PSD from all genuine windows.  Each
        # window carries its measured absolute power offset relative to loaded.
        psd_sum = None
        f_ref = None
        for template in model["templates"]:
            # Use the retained normalized excerpt only for the time plot; the RF
            # model's final 140-kHz spectrum is the authoritative template PSD.
            # Restrict to the fundamental carrier window for this diagnostic.
            tx = template["tx_rbw_dbm"]
            sel = np.abs(GLOBAL_FREQ_HZ_FOR_PLOTS - LTE_CENTER_HZ) <= 6.0e6
            lin = dbm_to_mw(tx[sel])
            psd_sum = lin if psd_sum is None else psd_sum + lin
            f_ref = (GLOBAL_FREQ_HZ_FOR_PLOTS[sel] - LTE_CENTER_HZ) / 1e6
        state_mean = mw_to_dbm(psd_sum / len(model["templates"]), floor_dbm=-220.0)
        state_mean -= np.nanmax(state_mean)
        state_mean += model["relative_power_db"]
        axes[1].plot(f_ref, state_mean, linewidth=0.9,
                     label=f"{name} mean ({len(model['templates'])} templates)")

    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Normalized I amplitude")
    axes[0].set_title("Genuine srsRAN captures: representative LTE windows")
    axes[0].legend()
    axes[1].set_xlabel("Offset from LTE carrier [MHz]")
    axes[1].set_ylabel("Relative 140-kHz spectrum [dB]")
    axes[1].set_title("State-average LTE spectra built from multiple genuine windows")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_load_schedule(time_s_rel, state_schedule, state_models, out: Path):
    codes = np.array([LOAD_RANK.get(s, 3) for s in state_schedule], dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.step(time_s_rel/60.0, codes, where="post")
    ax.set_yticks([0,1,2,3])
    ax.set_yticklabels(["idle","light","medium","loaded"])
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("LTE state")
    ax.set_title("Time-varying genuine LTE traffic/load schedule")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_emission_windows_v3(time_s_rel, freq_hz, state_models, state_schedule,
                             template_schedule, rx_specs: dict, out: Path):
    centers = [LTE_CENTER_HZ, 2*LTE_CENTER_HZ, 3*LTE_CENTER_HZ]
    valid = [c for c in centers if FREQ_MIN_HZ <= c <= FREQ_MAX_HZ]
    fig, axes = plt.subplots(len(valid), 3, figsize=(15, 4.2*len(valid)), squeeze=False)
    for row, center in enumerate(valid):
        sel = np.abs(freq_hz-center) <= 25e6
        fmhz = freq_hz[sel]/1e6
        extent = [fmhz[0], fmhz[-1], time_s_rel[-1]/60.0, time_s_rel[0]/60.0]
        tx2d = np.empty((len(time_s_rel), np.count_nonzero(sel)), dtype=np.float32)
        for state_name, model in state_models.items():
            for template in model["templates"]:
                ti = int(template["template_index"])
                tidx = np.flatnonzero((state_schedule == state_name) & (template_schedule == ti))
                if len(tidx):
                    tx2d[tidx] = template["tx_rbw_dbm"][sel]
        txrel = tx2d - np.nanmax(tx2d)
        axes[row,0].imshow(txrel, aspect="auto", extent=extent)
        axes[row,0].set_title(f"Tx around {center/1e6:.0f} MHz (relative dB)")
        all_rx = np.concatenate([v[:,sel].ravel() for v in rx_specs.values()])
        vmin = np.nanpercentile(all_rx, 1)
        vmax = np.nanpercentile(all_rx, 99.8)
        for col, (name, arr) in enumerate(rx_specs.items(), start=1):
            axes[row,col].imshow(arr[:,sel], aspect="auto", extent=extent, vmin=vmin, vmax=vmax)
            axes[row,col].set_title(f"{name} around {center/1e6:.0f} MHz")
        for col in range(3):
            axes[row,col].set_xlabel("Frequency [MHz]")
            axes[row,col].set_ylabel("Time [min]")
    fig.suptitle("Dynamic LTE load and within-state scheduler variation")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_template_schedule(time_s_rel, state_schedule, template_schedule, out: Path):
    """Diagnostic showing which genuine 100-ms LTE window drives each output second."""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for state_name in ["idle", "light", "medium", "loaded"]:
        tidx = np.flatnonzero(state_schedule == state_name)
        if len(tidx):
            ax.scatter(time_s_rel[tidx]/60.0, template_schedule[tidx], s=16, label=state_name)
    ax.set_xlabel("Time [min]")
    ax.set_ylabel("IQ template index")
    ax.set_title("Genuine LTE IQ template selected at each simulated second")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_terrain_profiles(propagation_by_rx: dict, out: Path):
    valid = [(name,p) for name,p in propagation_by_rx.items() if p["terrain"] is not None]
    if not valid:
        return
    fig, axes = plt.subplots(len(valid), 1, figsize=(11, 4.0*len(valid)), squeeze=False)
    for row, (name, prop) in enumerate(valid):
        ax = axes[row,0]
        terr = prop["terrain"]
        d_km = terr["distances"].to(u.km).value
        h_m = terr["heights"].to(u.m).value
        ax.plot(d_km, h_m, linewidth=1.1, label="SRTM terrain")
        tx_abs = h_m[0] + TX.height_agl_m
        rx_obj = next(r for r in RECEIVERS if r.name == name)
        rx_abs = h_m[-1] + rx_obj.height_agl_m
        ax.plot([d_km[0], d_km[-1]], [tx_abs, rx_abs], linestyle="--", linewidth=0.9,
                label="endpoint straight line (curvature not drawn)")
        ax.scatter([d_km[0], d_km[-1]], [tx_abs, rx_abs], s=25)
        ax.set_xlabel("Distance from transmitter [km]")
        ax.set_ylabel("Elevation AMSL [m]")
        ax.set_title(f"{name}: SRTM terrain profile")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_p452_path_loss(propagation_by_rx: dict, out: Path):
    valid = [(name,p) for name,p in propagation_by_rx.items() if p["p452"] is not None]
    if not valid:
        return
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for name, prop in valid:
        p = prop["p452"]
        ax.plot(p["frequency_anchor_hz"]/1e6, p["L_b_db"], marker=".", linewidth=0.9, label=name)
    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("P.452 basic transmission loss L_b [dB]")
    ax.set_title("Terrain-aware ITU-R P.452-16 path loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# =============================================================================
# MAIN (V3)
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="EMILY-X genuine-LTE terrain/load simulator v3.1")
    p.add_argument("--iq-loaded", "--loaded-iq", dest="iq_loaded", type=Path,
                   default=Path("srsran_loaded.cf32"),
                   help="heavily loaded genuine srsRAN downlink capture")
    p.add_argument("--iq-idle", type=Path, default=Path("srsran_idle.cf32"),
                   help="idle-cell genuine srsRAN downlink capture (optional)")
    p.add_argument("--iq-light", type=Path, default=Path("srsran_light.cf32"),
                   help="light-load genuine srsRAN downlink capture (optional)")
    p.add_argument("--iq-medium", type=Path, default=Path("srsran_medium.cf32"),
                   help="medium-load genuine srsRAN downlink capture (optional)")
    p.add_argument("--sample-rate", type=float, default=SRSRAN_IQ_SAMPLE_RATE_HZ)
    p.add_argument("--template-window-ms", type=float, default=LTE_TEMPLATE_WINDOW_MS,
                   help="genuine LTE IQ window length used for each spectral template; default 100 ms")
    p.add_argument("--template-hop-ms", type=float, default=LTE_TEMPLATE_HOP_MS,
                   help="hop between IQ templates; default 100 ms (use 50 for overlap)")
    p.add_argument("--max-templates-per-state", type=int, default=LTE_MAX_TEMPLATES_PER_STATE,
                   help="maximum templates retained per LTE state; 0 means all")
    p.add_argument("--duration", type=int, default=SIM_DURATION_S,
                   help="simulation duration in seconds; default 600")
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--propagation", choices=["p452", "fspl"], default="p452",
                   help="terrain-aware P.452 (default) or diagnostic free-space baseline")
    p.add_argument("--srtm-dir", type=Path, default=Path("srtm_data"),
                   help="directory used by pycraf for SRTM .hgt tiles")
    p.add_argument("--no-srtm-download", action="store_true",
                   help="do not download missing SRTM tiles")
    p.add_argument("--p452-time-percent", type=float, default=P452_TIME_PERCENT,
                   help="P.452 time percentage, 0 < p <= 50; default 50")
    p.add_argument("--schedule-json", type=Path, default=None,
                   help="optional custom load schedule JSON list of {start_s,end_s,state}")
    p.add_argument("--dev-fallback", action="store_true",
                   help="development-only OFDM source if no genuine capture exists")
    p.add_argument("--skip-emily", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if not (0.0 < args.p452_time_percent <= 50.0):
        raise ValueError("--p452-time-percent must be >0 and <=50")

    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    plot_dir = outdir / "plots"
    plot_dir.mkdir(exist_ok=True)

    print("EMILY-X LTE v3.1")
    print("================")
    print(f"pycraf available: {HAVE_PYCRAF}")
    print(f"LTE carrier: {LTE_CENTER_HZ/1e6:.3f} MHz, {LTE_CHANNEL_BW_HZ/1e6:.1f} MHz, {LTE_N_PRB} PRB")
    print(f"Receiver backend: {CHANNEL_SPACING_HZ/1e3:.0f} kHz spacing, {RBW_HZ/1e3:.0f} kHz RBW, 1 s")
    print(f"Propagation: {args.propagation}")

    rng = np.random.default_rng(RANDOM_SEED)
    freq_hz = np.arange(FREQ_MIN_HZ, FREQ_MAX_HZ + 0.5*CHANNEL_SPACING_HZ,
                        CHANNEL_SPACING_HZ, dtype=np.float64)
    freq_hz = freq_hz[freq_hz <= FREQ_MAX_HZ + 1e-6]
    nt = int(round(args.duration / TIME_RESOLUTION_S))
    time_s_rel, time_utc_iso, time_local_iso = make_time_axes(nt)

    requested_paths = {
        "idle": args.iq_idle,
        "light": args.iq_light,
        "medium": args.iq_medium,
        "loaded": args.iq_loaded,
    }
    raw_states = {}
    for name, path in requested_paths.items():
        if path is not None and path.exists():
            raw_states[name] = load_srsran_iq_state(
                path, args.sample_rate, SRSRAN_MAX_CAPTURE_SECONDS,
                args.template_window_ms, args.template_hop_ms,
                args.max_templates_per_state,
            )

    if not raw_states:
        if not args.dev_fallback:
            raise FileNotFoundError(
                "No genuine LTE IQ files found. At minimum provide srsran_loaded.cf32 "
                "or --iq-loaded PATH."
            )
        iq = generate_dev_ofdm_fallback(rng)
        raw_rms = float(np.sqrt(np.mean(np.abs(iq) ** 2)))
        raw_states["loaded"] = {
            "source": "development-only OFDM fallback",
            "raw_rms": raw_rms,
            "capture_seconds_used": float(len(iq)/args.sample_rate),
            "templates": [{
                "template_index": 0,
                "start_s": 0.0,
                "end_s": float(len(iq)/args.sample_rate),
                "raw_rms": raw_rms,
                "iq": (iq/raw_rms).astype(np.complex64),
                "iq_excerpt": (iq[:min(len(iq), int(0.020*args.sample_rate))]/raw_rms).astype(np.complex64),
            }],
        }
        warnings.warn("Using development-only OFDM fallback; not suitable for production dataset")

    reference_state = "loaded" if "loaded" in raw_states else max(raw_states, key=lambda s: raw_states[s]["raw_rms"])
    reference_rms = raw_states[reference_state]["raw_rms"]

    state_models = {}
    print("LTE waveform states and genuine IQ templates:")
    for name in sorted(raw_states, key=lambda s: LOAD_RANK.get(s, 99)):
        rs = raw_states[name]
        state_relative_power_db = 20.0*np.log10(rs["raw_rms"] / reference_rms)
        templates = []
        for raw_template in rs["templates"]:
            template_relative_power_db = 20.0*np.log10(raw_template["raw_rms"] / reference_rms)
            components, rf_diag, iq_filt, iq_pa = build_emission_components(
                raw_template["iq"], args.sample_rate
            )
            components = scaled_components_for_load(components, template_relative_power_db)
            tx_dbm, fractions = build_tx_rbw_spectrum(freq_hz, components)
            templates.append({
                "template_index": int(raw_template["template_index"]),
                "start_s": raw_template["start_s"],
                "end_s": raw_template["end_s"],
                "raw_rms": raw_template["raw_rms"],
                "relative_power_db": float(template_relative_power_db),
                "iq_excerpt": raw_template["iq_excerpt"],
                "components": components,
                "rf_diag": rf_diag,
                "fractions": fractions,
                "tx_rbw_dbm": tx_dbm.astype(np.float32),
            })
            # Do not retain oversampled PA/filter IQ for every template; it would
            # add >1 GB with no benefit to the final 1-s/RBW simulation.
            del iq_filt, iq_pa

        # Representative template is closest in raw power to the complete state's
        # RMS.  State-average spectra are kept for diagnostics/backward metadata.
        rep_idx = int(np.argmin([abs(t["raw_rms"] - rs["raw_rms"]) for t in templates]))
        rep = templates[rep_idx]
        mean_mw = np.mean(np.stack([dbm_to_mw(t["tx_rbw_dbm"]) for t in templates]), axis=0)
        state_models[name] = {
            "source": rs["source"],
            "raw_rms": rs["raw_rms"],
            "relative_power_db": float(state_relative_power_db),
            "capture_seconds_used": rs["capture_seconds_used"],
            "templates": templates,
            "representative_template_index": rep_idx,
            "components": rep["components"],
            "rf_diag": rep["rf_diag"],
            "fractions": rep["fractions"],
            "tx_rbw_dbm": mw_to_dbm(mean_mw, floor_dbm=-220.0).astype(np.float32),
            "iq_excerpt": rep["iq_excerpt"],
        }
        print(f"  {name:7s}: {rs['source']}  capture RMS={rs['raw_rms']:.6g}  "
              f"state power={state_relative_power_db:+.2f} dB  templates={len(templates)}")
        for t in templates:
            print(f"      T{t['template_index']}: {t['start_s']*1e3:5.0f}-{t['end_s']*1e3:5.0f} ms  "
                  f"RMS={t['raw_rms']:.6g}  power={t['relative_power_db']:+.2f} dB  "
                  f"ACLR={t['rf_diag']['measured_aclr_db']:.1f} dB")

    # Free the full normalized template IQ once the RF spectral models are built.
    raw_states.clear()
    available = list(state_models)
    if args.schedule_json is not None:
        state_schedule, load_segments = load_schedule_from_json(args.schedule_json, nt, available)
    else:
        state_schedule, load_segments = make_default_load_schedule(nt, available)

    template_schedule = make_template_schedule(state_schedule, state_models, rng)

    if len(set(state_schedule.tolist())) == 1:
        warnings.warn(
            "Only one LTE load state is available; v3 will use terrain-aware propagation "
            "but the transmitter spectrum will be static in time. Capture idle/light/medium "
            "states with the supplied multi-capture broker for full load dynamics."
        )

    print("Load schedule:")
    for seg in load_segments:
        note = "" if seg["state"] == seg["requested_state"] else f" (substituted for {seg['requested_state']})"
        print(f"  {seg['start_s']:6.0f}-{seg['end_s']:6.0f} s : {seg['state']}{note}")

    active_mask = build_active_union(freq_hz, state_models)

    propagation_by_rx = {}
    for rx in RECEIVERS:
        print(f"Preparing propagation for {rx.name}...")
        prop = prepare_propagation(
            rx, freq_hz, active_mask,
            propagation_mode=args.propagation,
            srtm_dir=args.srtm_dir,
            download_srtm=not args.no_srtm_download,
            p452_time_percent=args.p452_time_percent,
        )
        propagation_by_rx[rx.name] = prop
        if prop["terrain"] is not None:
            t = prop["terrain"]
            print(f"  SRTM: Tx ground={t['tx_ground_elevation_m']:.0f} m, "
                  f"Rx ground={t['rx_ground_elevation_m']:.0f} m, "
                  f"midpoint={t['midpoint_ground_elevation_m']:.0f} m")
            print(f"  P.452 L_b @ 800 MHz: {prop['center_losses_db'][str(int(LTE_CENTER_HZ))]:.1f} dB")
        else:
            print(f"  FSPL @ 800 MHz: {prop['center_losses_db'][str(int(LTE_CENTER_HZ))]:.1f} dB")

    tx_avg_dbm = build_tx_average_spectrum(state_models, state_schedule, template_schedule).astype(np.float32)
    waveform_source = "; ".join(f"{name}={m['source']}" for name,m in state_models.items())

    # Compact transmitter reference.  Store the actual per-second LTE state/template
    # schedule plus every 140-kHz spectral template, but only tiny IQ excerpts.
    tx_payload = {
        "freq_mhz": freq_hz/1e6,
        "lte_state_t": state_schedule,
        "lte_template_index_t": template_schedule,
        "load_segments_json": json.dumps(load_segments),
        "iq_sample_rate_hz": np.float64(args.sample_rate),
        "template_window_ms": np.float64(args.template_window_ms),
        "template_hop_ms": np.float64(args.template_hop_ms),
    }
    for name, m in state_models.items():
        tx_payload[f"tx_power_per_rbw_dbm_{name}_mean"] = m["tx_rbw_dbm"]
        tx_payload[f"raw_rms_{name}"] = np.float64(m["raw_rms"])
        tx_payload[f"relative_power_db_{name}"] = np.float64(m["relative_power_db"])
        for t in m["templates"]:
            ti = int(t["template_index"])
            tx_payload[f"tx_power_per_rbw_dbm_{name}_template_{ti}"] = t["tx_rbw_dbm"]
            tx_payload[f"raw_rms_{name}_template_{ti}"] = np.float64(t["raw_rms"])
            tx_payload[f"relative_power_db_{name}_template_{ti}"] = np.float64(t["relative_power_db"])
            tx_payload[f"iq_excerpt_{name}_template_{ti}"] = t["iq_excerpt"]
    np.savez_compressed(outdir / "tx_reference_emissions_v3_1.npz", **tx_payload)

    rx_specs = {}
    rx_avg = {}
    truth_by_rx = {}
    link_rows = []
    component_rows = []

    representative_state = "loaded" if "loaded" in state_models else available[-1]
    representative_components = state_models[representative_state]["components"]

    for rx in RECEIVERS:
        print(f"Simulating {rx.name}...")
        rx_rng = np.random.default_rng(rng.integers(0, 2**32-1))
        spec_dbm, truth = make_receiver_spectrogram_v3(
            rx, TX, freq_hz, state_models, state_schedule, template_schedule,
            propagation_by_rx[rx.name], nt, rx_rng,
        )
        rx_specs[rx.name] = spec_dbm
        rx_avg[rx.name] = linear_average_dbm(spec_dbm, axis=0).astype(np.float32)
        truth_by_rx[rx.name] = truth

        npz_path = outdir / f"{rx.name.lower()}_synthetic_{args.duration}s_v3_1.npz"
        save_receiver_npz_v3(
            npz_path, rx, spec_dbm, freq_hz, time_s_rel,
            time_utc_iso, time_local_iso, truth, state_models,
            state_schedule, template_schedule, load_segments, waveform_source,
        )

        row = {
            "receiver": rx.name,
            "receiver_lat": rx.lat_deg,
            "receiver_lon": rx.lon_deg,
            "distance_km": truth["distance_km"],
            "bearing_tx_to_rx_deg": truth["bearing_tx_to_rx_deg"],
            "elevation_tx_to_rx_deg": truth["elevation_tx_to_rx_deg"],
            "noise_floor_per_140khz_rbw_dbm": truth["noise_floor_per_rbw_dbm"],
            "propagation_model": truth["propagation_model"],
        }
        for f in [LTE_CENTER_HZ, 2*LTE_CENTER_HZ, 3*LTE_CENTER_HZ]:
            key = str(int(round(f)))
            if key in truth["center_losses_db"]:
                row[f"path_loss_{int(f/1e6)}mhz_db"] = truth["center_losses_db"][key]
        if "tx_ground_elevation_m" in truth:
            row["tx_ground_elevation_m"] = truth["tx_ground_elevation_m"]
            row["rx_ground_elevation_m"] = truth["rx_ground_elevation_m"]
        link_rows.append(row)

        for c in truth["component_truth"]:
            component_rows.append({
                "receiver": rx.name,
                **{k:v for k,v in c.items() if k not in ("rx_total_component_dbm_t","tx_total_power_dbm_t")},
            })

        if RUN_LOCAL_EMILY_PIPELINE and not args.skip_emily:
            events, masks = run_local_emily_windows(
                spec_dbm, freq_hz, time_utc_iso, time_local_iso, time_s_rel
            )
            events.to_csv(outdir / f"{rx.name.lower()}_rfi_events.csv", index=False)
            incidents = events_to_candidate_incidents(
                events, rx, freq_hz, time_utc_iso, time_local_iso,
                time_s_rel, npz_path.name, representative_components, truth, waveform_source
            )
            incidents = patch_candidate_incidents_v3(
                incidents, truth, state_models, load_segments, truth["propagation_model"]
            )
            with open(outdir / f"{rx.name.lower()}_emily_candidate_incidents.json", "w", encoding="utf-8") as f:
                json.dump(incidents, f, indent=2)
            print(f"  local EMILY preview: {len(events)} event(s)")

    pd.DataFrame(link_rows).to_csv(outdir / "ground_truth_link_budget.csv", index=False)
    pd.DataFrame(component_rows).to_csv(outdir / "ground_truth_emission_components.csv", index=False)
    pd.DataFrame(load_segments).to_csv(outdir / "ground_truth_lte_load_schedule.csv", index=False)

    config = {
        "simulation": "EMILY-X LTE v3.1",
        "waveform_source": waveform_source,
        "transmitter": asdict(TX),
        "receivers": [asdict(r) for r in RECEIVERS],
        "lte": {
            "center_hz": LTE_CENTER_HZ,
            "channel_bw_hz": LTE_CHANNEL_BW_HZ,
            "n_prb": LTE_N_PRB,
            "dl_earfcn": LTE_EARFCN_DL,
            "ul_earfcn": LTE_EARFCN_UL,
            "srsran_iq_sample_rate_hz": args.sample_rate,
            "states": {
                name: {
                    "source": m["source"],
                    "raw_rms": m["raw_rms"],
                    "relative_power_db": m["relative_power_db"],
                    "template_count": len(m["templates"]),
                    "representative_rf_diagnostics": m["rf_diag"],
                    "templates": [
                        {
                            "template_index": int(t["template_index"]),
                            "start_s": t["start_s"],
                            "end_s": t["end_s"],
                            "raw_rms": t["raw_rms"],
                            "relative_power_db": t["relative_power_db"],
                            "rf_diagnostics": t["rf_diag"],
                        } for t in m["templates"]
                    ],
                } for name,m in state_models.items()
            },
            "load_segments": load_segments,
            "template_window_ms": args.template_window_ms,
            "template_hop_ms": args.template_hop_ms,
            "max_templates_per_state": args.max_templates_per_state,
            "template_selection": "shuffled cycles without replacement",
        },
        "backend": {
            "freq_min_hz": FREQ_MIN_HZ,
            "freq_max_hz": FREQ_MAX_HZ,
            "channel_spacing_hz": CHANNEL_SPACING_HZ,
            "rbw_hz": RBW_HZ,
            "time_resolution_s": TIME_RESOLUTION_S,
            "duration_s": args.duration,
        },
        "propagation": {
            "mode": args.propagation,
            "p452_version": 16 if args.propagation == "p452" else None,
            "p452_time_percent": args.p452_time_percent if args.propagation == "p452" else None,
            "profile_step_m": P452_PROFILE_STEP_M if args.propagation == "p452" else None,
            "srtm_dir": str(args.srtm_dir),
            "srtm_download_missing": not args.no_srtm_download,
            "shadowing_sigma_db": SHADOWING_SIGMA_DB,
            "shadowing_correlation_s": SHADOWING_CORRELATION_S,
        },
        "rf_chain": {
            "oversample_factor": TX_OVERSAMPLE_FACTOR,
            "pa_input_backoff_db": TX_PA_INPUT_BACKOFF_DB,
            "pa_rapp_p": TX_PA_RAPP_P,
            "target_aclr_db": TARGET_ACLR_DB,
            "rf_filter_far_stop_atten_db": TX_RF_FILTER_FAR_STOP_ATTEN_DB,
            "second_harmonic_dbc": SECOND_HARMONIC_DBC,
            "third_harmonic_dbc": THIRD_HARMONIC_DBC,
            "spurious_products": SPURIOUS_PRODUCTS,
        },
        "pycraf_available": HAVE_PYCRAF,
        "random_seed": RANDOM_SEED,
    }
    with open(outdir / "simulation_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # Plots.
    global GLOBAL_FREQ_HZ_FOR_PLOTS
    GLOBAL_FREQ_HZ_FOR_PLOTS = freq_hz
    plot_geometry(plot_dir / "01_geometry_map.png")
    plot_lte_states(state_models, args.sample_rate, plot_dir / "02_lte_state_waveforms.png")
    plot_tx_wideband(freq_hz, tx_avg_dbm, plot_dir / "03_transmitter_wideband_emissions_average.png")
    plot_fundamental_zoom(freq_hz, tx_avg_dbm, plot_dir / "04_lte_oob_filter_leakage_average.png")
    plot_antenna_patterns(plot_dir / "05_antenna_patterns_fundamental_harmonics.png")
    plot_average_spectra(freq_hz, tx_avg_dbm, rx_avg, plot_dir / "06_average_spectra_tx_rx.png")
    plot_emission_windows_v3(time_s_rel, freq_hz, state_models, state_schedule, template_schedule, rx_specs,
                             plot_dir / "07_spectrograms_dynamic_lte_harmonics.png")
    plot_component_received_power(time_s_rel, truth_by_rx,
                                  plot_dir / "08_received_component_power.png")
    plot_terrain_profiles(propagation_by_rx, plot_dir / "09_srtm_terrain_profiles.png")
    plot_p452_path_loss(propagation_by_rx, plot_dir / "10_p452_path_loss.png")
    plot_load_schedule(time_s_rel, state_schedule, state_models,
                       plot_dir / "11_lte_load_schedule.png")
    plot_template_schedule(time_s_rel, state_schedule, template_schedule,
                           plot_dir / "12_lte_template_schedule.png")

    print("\nDone.")
    print(f"Outputs: {outdir.resolve()}")
    print("Receiver coordinates written to EMILY candidate data:")
    for rx in RECEIVERS:
        print(f"  {rx.name}: lat={rx.lat_deg:.6f}, lon={rx.lon_deg:.6f}")


if __name__ == "__main__":
    main()
