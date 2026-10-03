#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terrain-aware attenuation cube using pycraf ITU-R P.452-16 + SRTM.

The output cube has shape (ny, nx, n_receivers):
    attenuation_db[iy, ix, irx]
where each (iy, ix) is a candidate transmitter pixel and each irx is one
receiver.  The common candidate grid is a metric square centered on the
locally-Euclidean centroid of all receiver coordinates.

For speed, each receiver uses pycraf.pathprof.height_map_data() plus
atten_map_fast() rather than one complete P.452 calculation per pixel.
Propagation loss is reciprocal, so the fixed receiver is used as the pycraf
map center, with endpoint heights swapped consistently.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from pyproj import CRS, Transformer
from astropy import units as u
from pycraf import pathprof, conversions as cnv


P452_VERSION = 16


@dataclass
class Receiver:
    name: str
    lat_deg: float
    lon_deg: float
    height_agl_m: float = 2.0


def parse_receiver(spec: str, idx: int, default_h: float) -> Receiver:
    """Accept LAT,LON ; NAME,LAT,LON ; LAT,LON,H ; NAME,LAT,LON,H."""
    p = [x.strip() for x in spec.split(",")]
    if len(p) == 2:
        name, lat, lon, h = f"RX{idx}", float(p[0]), float(p[1]), default_h
    elif len(p) == 3:
        try:
            float(p[0]); first_numeric = True
        except ValueError:
            first_numeric = False
        if first_numeric:
            name, lat, lon, h = f"RX{idx}", float(p[0]), float(p[1]), float(p[2])
        else:
            name, lat, lon, h = p[0], float(p[1]), float(p[2]), default_h
    elif len(p) == 4:
        name, lat, lon, h = p[0], float(p[1]), float(p[2]), float(p[3])
    else:
        raise ValueError(f"Bad receiver {spec!r}")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180 or h < 0:
        raise ValueError(f"Invalid receiver {spec!r}")
    return Receiver(name, lat, lon, h)


def prompt_receivers(default_h: float) -> list[Receiver]:
    n = int(input("Number of receivers: "))
    if n < 1:
        raise ValueError("At least one receiver is required")
    out = []
    for i in range(1, n + 1):
        s = input(f"Receiver {i} [LAT,LON or NAME,LAT,LON[,HEIGHT_M]]: ")
        out.append(parse_receiver(s, i, default_h))
    return out


def aeqd(lat0: float, lon0: float) -> CRS:
    return CRS.from_proj4(
        f"+proj=aeqd +lat_0={lat0:.12f} +lon_0={lon0:.12f} "
        "+datum=WGS84 +units=m +no_defs"
    )


def receiver_centroid(receivers: list[Receiver]) -> tuple[float, float]:
    """Metric-plane centroid, transformed back to WGS84."""
    lat0 = float(np.mean([r.lat_deg for r in receivers]))
    lon0 = float(np.mean([r.lon_deg for r in receivers]))
    crs = aeqd(lat0, lon0)
    fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    x, y = fwd.transform(
        [r.lon_deg for r in receivers], [r.lat_deg for r in receivers]
    )
    lonc, latc = inv.transform(float(np.mean(x)), float(np.mean(y)))
    return float(latc), float(lonc)


def common_grid(latc: float, lonc: float, area_km: float, res_km: float):
    ratio = area_km / res_km
    n = int(round(ratio))
    if area_km <= 0 or res_km <= 0 or n < 1 or not np.isclose(ratio, n, atol=1e-9):
        raise ValueError("area_km / resolution_km must be a positive integer")

    # Pixel centers: 20 km / 0.1 km -> exactly 200 x 200 pixels.
    half_m = 500.0 * area_km
    res_m = 1000.0 * res_km
    axis_m = -half_m + res_m / 2 + np.arange(n) * res_m
    xx, yy = np.meshgrid(axis_m, axis_m)

    crs = aeqd(latc, lonc)
    inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = inv.transform(xx, yy)
    return axis_m / 1000, axis_m / 1000, np.asarray(lat), np.asarray(lon), crs


def orient_map(z, x, y):
    z = np.asarray(z)
    if z.shape == (len(y), len(x)):
        return z
    if z.shape == (len(x), len(y)):
        return z.T
    raise RuntimeError(f"Unexpected pycraf map shape {z.shape}")


def interp_map(z, x, y, target_lat, target_lon):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = orient_map(z, x, y)
    if np.any(np.diff(x) < 0):
        x, z = x[::-1], z[:, ::-1]
    if np.any(np.diff(y) < 0):
        y, z = y[::-1], z[::-1, :]
    fn = RegularGridInterpolator((y, x), z, bounds_error=False, fill_value=np.nan)
    q = np.column_stack((target_lat.ravel(), target_lon.ravel()))
    return fn(q).reshape(target_lat.shape)


def one_receiver_map(
    rx: Receiver,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    freq_mhz: float,
    tx_height_m: float,
    time_percent: float,
    temperature_k: float,
    pressure_hpa: float,
    polarization: int,
    map_res_deg: float,
    cache_dir: Path,
):
    # Receiver-centered pycraf map large enough to contain the common square.
    margin = 4 * map_res_deg
    half_lon = float(np.max(np.abs(target_lon - rx.lon_deg))) + margin
    half_lat = float(np.max(np.abs(target_lat - rx.lat_deg))) + margin

    t0 = time.perf_counter()
    hprof = pathprof.height_map_data(
        rx.lon_deg * u.deg,
        rx.lat_deg * u.deg,
        2 * half_lon * u.deg,
        2 * half_lat * u.deg,
        map_resolution=map_res_deg * u.deg,
        do_cos_delta=False,
        zone_t=pathprof.CLUTTER.UNKNOWN,
        zone_r=pathprof.CLUTTER.UNKNOWN,
        omega_percent=0 * u.percent,
        cache_path=str(cache_dir),
        clobber=False,
    )
    t1 = time.perf_counter()

    # Center terminal is the actual receiver; map pixels are candidate towers.
    # By reciprocity this gives the same L_b as tower -> receiver.
    result = pathprof.atten_map_fast(
        freq_mhz * 1e-3 * u.GHz,
        temperature_k * u.K,
        pressure_hpa * u.hPa,
        rx.height_agl_m * u.m,
        tx_height_m * u.m,
        time_percent * u.percent,
        hprof,
        polarization=polarization,
        version=P452_VERSION,
    )
    t2 = time.perf_counter()

    x = np.asarray(hprof["xcoords"], float)
    y = np.asarray(hprof["ycoords"], float)
    lb = interp_map(result["L_b"].to(cnv.dB).value, x, y, target_lat, target_lon)
    ptype = interp_map(np.asarray(result["path_type"], float), x, y, target_lat, target_lon)

    if np.any(~np.isfinite(lb)):
        raise RuntimeError(f"{rx.name}: common grid extends outside pycraf map")

    meta = {
        "receiver": asdict(rx),
        "native_nlon": int(len(x)),
        "native_nlat": int(len(y)),
        "terrain_preprocessing_s": float(t1 - t0),
        "attenuation_calculation_s": float(t2 - t1),
        "total_s": float(t2 - t0),
    }
    return lb.astype(np.float32), np.rint(ptype).astype(np.int8), meta


def make_plots(plot_dir: Path, x_km, y_km, cube, receivers, freq_mhz):
    plot_dir.mkdir(parents=True, exist_ok=True)
    extent = [x_km[0], x_km[-1], y_km[0], y_km[-1]]
    finite = cube[np.isfinite(cube)]
    vmin = float(np.percentile(finite, 1))
    vmax = float(np.percentile(finite, 99))

    for i, rx in enumerate(receivers):
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cube[:, :, i], origin="lower", extent=extent,
                       aspect="equal", vmin=vmin, vmax=vmax)
        ax.set_xlabel("East offset from receiver centroid [km]")
        ax.set_ylabel("North offset from receiver centroid [km]")
        ax.set_title(f"{rx.name}: P.452 L_b at {freq_mhz:g} MHz")
        cb = fig.colorbar(im, ax=ax); cb.set_label("Basic transmission loss L_b [dB]")
        fig.tight_layout(); fig.savefig(plot_dir / f"attenuation_{rx.name}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.nanmin(cube, axis=2), origin="lower", extent=extent, aspect="equal")
    ax.set_xlabel("East offset from receiver centroid [km]")
    ax.set_ylabel("North offset from receiver centroid [km]")
    ax.set_title(f"Minimum P.452 loss to any receiver at {freq_mhz:g} MHz")
    cb = fig.colorbar(im, ax=ax); cb.set_label("min L_b [dB]")
    fig.tight_layout(); fig.savefig(plot_dir / "attenuation_min_any_receiver.png", dpi=180)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="pycraf P.452/SRTM attenuation cube")
    p.add_argument("--receiver", action="append", default=[],
                   help="Repeatable: LAT,LON or NAME,LAT,LON[,HEIGHT_M]")
    p.add_argument("--area-km", type=float, default=20.0)
    p.add_argument("--resolution-km", type=float, default=0.1)
    p.add_argument("--frequency-mhz", type=float, default=800.0)

    # Needed physical assumptions; defaults match the terrestrial EMILY-X scale.
    p.add_argument("--tx-height-m", type=float, default=40.0)
    p.add_argument("--default-rx-height-m", type=float, default=2.0)
    p.add_argument("--time-percent", type=float, default=50.0)
    p.add_argument("--temperature-k", type=float, default=290.0)
    p.add_argument("--pressure-hpa", type=float, default=1013.0)
    p.add_argument("--polarization", choices=["h", "v", "horizontal", "vertical"],
                   default="horizontal")

    p.add_argument("--srtm-dir", type=Path, default=Path("srtm_data"))
    p.add_argument("--no-srtm-download", action="store_true")
    p.add_argument("--cache-dir", type=Path, default=Path("pycraf_attenuation_cache"))
    p.add_argument("--threads", type=int, default=0,
                   help="0 keeps pycraf default thread count")
    p.add_argument("--output", type=Path, default=Path("attenuation_cube.npz"))
    p.add_argument("--plot-dir", type=Path, default=Path("attenuation_plots"))
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    if not (a.frequency_mhz > 0 and a.tx_height_m >= 0 and 0 < a.time_percent <= 50):
        raise ValueError("Check frequency, heights, and P.452 time percentage")

    receivers = [parse_receiver(s, i + 1, a.default_rx_height_m)
                 for i, s in enumerate(a.receiver)]
    if not receivers:
        receivers = prompt_receivers(a.default_rx_height_m)

    if a.threads > 0:
        pathprof.set_num_threads(a.threads)

    a.srtm_dir.mkdir(parents=True, exist_ok=True)
    a.cache_dir.mkdir(parents=True, exist_ok=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)

    latc, lonc = receiver_centroid(receivers)
    x_km, y_km, lat_grid, lon_grid, crs = common_grid(
        latc, lonc, a.area_km, a.resolution_km
    )
    ny, nx = lat_grid.shape
    nr = len(receivers)
    map_res_deg = (a.resolution_km * 1000.0) / 111_320.0
    pol = 0 if a.polarization in ("h", "horizontal") else 1

    print(f"Centroid: {latc:.8f}, {lonc:.8f}")
    print(f"Grid: {ny} x {nx} pixels, {nr} receivers -> cube {ny} x {nx} x {nr}")
    print(f"Area: {a.area_km:g} km square; resolution: {a.resolution_km:g} km")
    print(f"Frequency: {a.frequency_mhz:g} MHz; Tx height: {a.tx_height_m:g} m AGL")

    cube = np.full((ny, nx, nr), np.nan, np.float32)
    ptype_cube = np.full((ny, nx, nr), -1, np.int8)
    run_meta = []
    dl = "never" if a.no_srtm_download else "missing"

    t0 = time.perf_counter()
    with pathprof.SrtmConf.set(
        srtm_dir=str(a.srtm_dir.resolve()), download=dl,
        server="viewpano", interp="linear"
    ):
        for i, rx in enumerate(receivers):
            print(f"[{i+1}/{nr}] {rx.name} ...", flush=True)
            lb, ptype, meta = one_receiver_map(
                rx, lat_grid, lon_grid, a.frequency_mhz, a.tx_height_m,
                a.time_percent, a.temperature_k, a.pressure_hpa, pol,
                map_res_deg, a.cache_dir
            )
            cube[:, :, i] = lb
            ptype_cube[:, :, i] = ptype
            run_meta.append(meta)
            print(f"    L_b {np.nanmin(lb):.2f}..{np.nanmax(lb):.2f} dB; {meta['total_s']:.1f} s")

    elapsed = time.perf_counter() - t0
    metadata = {
        "product": "EMILY-X pycraf attenuation cube",
        "cube_axis_order": ["y", "x", "receiver"],
        "loss_quantity": "ITU-R P.452-16 basic transmission loss L_b",
        "antenna_gain_included": False,
        "antenna_assumption": "omnidirectional / 0 dBi; propagation loss only",
        "frequency_mhz": a.frequency_mhz,
        "area_size_km": a.area_km,
        "resolution_km": a.resolution_km,
        "center_lat_deg": latc,
        "center_lon_deg": lonc,
        "receivers": [asdict(r) for r in receivers],
        "candidate_tx_height_agl_m": a.tx_height_m,
        "p452_version": P452_VERSION,
        "p452_time_percent": a.time_percent,
        "temperature_k": a.temperature_k,
        "pressure_hpa": a.pressure_hpa,
        "polarization": "horizontal" if pol == 0 else "vertical",
        "endpoint_clutter": "UNKNOWN",
        "omega_percent": 0.0,
        "elapsed_seconds": elapsed,
        "receiver_map_metadata": run_meta,
    }

    np.savez_compressed(
        a.output,
        attenuation_db=cube,
        path_type=ptype_cube,
        latitude_deg=lat_grid.astype(np.float64),
        longitude_deg=lon_grid.astype(np.float64),
        x_km=x_km.astype(np.float64),
        y_km=y_km.astype(np.float64),
        receiver_names=np.asarray([r.name for r in receivers], dtype="U64"),
        receiver_lat_deg=np.asarray([r.lat_deg for r in receivers], float),
        receiver_lon_deg=np.asarray([r.lon_deg for r in receivers], float),
        receiver_height_m=np.asarray([r.height_agl_m for r in receivers], float),
        center_lat_deg=np.float64(latc),
        center_lon_deg=np.float64(lonc),
        frequency_hz=np.float64(a.frequency_mhz * 1e6),
        tx_height_agl_m=np.float64(a.tx_height_m),
        area_size_km=np.float64(a.area_km),
        resolution_km=np.float64(a.resolution_km),
        p452_time_percent=np.float64(a.time_percent),
        p452_version=np.int16(P452_VERSION),
        local_crs_wkt=np.asarray(crs.to_wkt()),
        metadata_json=np.asarray(json.dumps(metadata, indent=2)),
    )

    if not a.no_plots:
        make_plots(a.plot_dir, x_km, y_km, cube, receivers, a.frequency_mhz)

    print(f"Saved {a.output.resolve()}")
    print(f"Cube shape: {cube.shape} (y, x, receiver)")
    print(f"Runtime: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
