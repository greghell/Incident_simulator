# EMILY-X LTE v3.2 — terrestrial cell or single LEO D2C satellite

v3.2 keeps the v3.1 genuine-srsRAN LTE template pipeline and adds a selectable
single-satellite Direct-to-Cell (D2C) transmitter model.

The two modes share the same downstream chain:

`genuine LTE IQ -> RF/OOB model -> propagation -> receiver spectrogram -> local EMILY preview -> NPZ/ground truth`

## Modes

### Terrestrial (legacy/default)

- 800 MHz center
- 10 MHz LTE / 50 PRB
- 11.52 Msps srsRAN IQ
- fixed sector antenna
- SRTM + ITU-R P.452 (default) or diagnostic FSPL

Existing files are unchanged:

- `srsran_idle.cf32`
- `srsran_light.cf32`
- `srsran_medium.cf32`
- `srsran_loaded.cf32`

Example:

```bat
python emilyx_lte_end_to_end_simulator_v3_2.py ^
  --tx-type terrestrial ^
  --duration 600 ^
  --output-dir output_terrestrial_600s
```

### LEO D2C

The toy D2C profile uses:

- 1990--1995 MHz downlink
- 1992.5 MHz center
- 5 MHz LTE / 25 PRB
- downlink EARFCN 8665
- 5.76 Msps srsRAN IQ
- one TLE-driven LEO satellite
- time-varying slant range, azimuth/elevation, visibility and Doppler
- Earth-fixed spot-beam targets with multiple beam handoffs during a visible pass
- time-varying free-space loss instead of terrestrial P.452

The LEO beam parameters are explicit toy-model assumptions in the simulator
(default peak beam-center EIRP 75 dBm, 8 deg HPBW, -30 dB sidelobe floor).
They are not claims about any specific operational D2C satellite.

## 1. Install the one additional Python dependency

In the Windows `emilyx` conda environment:

```bat
conda activate emilyx
pip install sgp4
```

Astropy is already used by the existing project and is used to transform SGP4
TEME positions/velocities into the Earth-fixed ITRS frame.

## 2. Generate the D2C LTE IQ states in WSL

The capture script now accepts a radio profile.  The existing terrestrial command
still works, while D2C is:

```bash
chmod +x generate_lte_iq_states_fixed.sh
./generate_lte_iq_states_fixed.sh d2c
```

This starts a 25-PRB srsRAN LTE cell at EARFCN 8665 using 5.76 Msps and captures:

- `srsran_d2c_idle.cf32`
- `srsran_d2c_light.cf32`
- `srsran_d2c_medium.cf32`
- `srsran_d2c_loaded.cf32`

The D2C light/medium offered loads default to 2 and 7 Mbit/s.  The loaded state
still uses four parallel TCP streams to drive the cell toward saturation.

The GNU Radio broker gained only one new option, `--prefix`; the capture/forwarding
logic is otherwise unchanged.

## 3. Supply a TLE

Create a text file containing either two TLE lines or a name plus two TLE lines:

```text
SATELLITE NAME
1 .....
2 .....
```

Use a TLE close in epoch to the simulated pass.  v3.2 warns when the simulation
midpoint is more than 14 days from the TLE epoch.

## 4. Recommended first LEO run

The easiest demonstration is to let EMILY-X search forward from a specified time,
find the next pass above 10 degrees over the midpoint of RX1/RX2, and center the
10-minute observation on the pass maximum:

```bat
python emilyx_lte_end_to_end_simulator_v3_2.py ^
  --tx-type leo ^
  --tle-file satellite.tle ^
  --start-utc 2026-09-11T20:00:00Z ^
  --center-on-next-pass ^
  --duration 600 ^
  --output-dir output_d2c_600s
```

Without `--center-on-next-pass`, `--start-utc` is the exact beginning of the
observation.  If that 10-minute interval contains no pass above the minimum
elevation, the output will correctly contain no visible satellite signal.

## Default beam behavior

During every interval in which the satellite is above the requested minimum
receive elevation over the RX1/RX2 reference area, the default toy schedule hops:

`east_cell -> site -> northwest_cell -> site -> south_cell`

The targets are fixed on Earth.  The satellite continuously steers toward the
current target while moving along the TLE trajectory.  RX1 and RX2 therefore see
time-varying off-axis angle and beam gain.

A custom schedule can be supplied with `--beam-schedule-json`.  The JSON may be:

```json
{
  "targets": {
    "custom": {"lat_deg": 39.55, "lon_deg": -114.45}
  },
  "segments": [
    {"start_s": 100, "end_s": 180, "target": "custom"},
    {"start_s": 180, "end_s": 260, "target": "site"}
  ]
}
```

Unspecified times use target `none`.

## Doppler

Default:

```text
--doppler-mode geometric
```

The received LTE spectrum is shifted by the instantaneous radial-velocity
Doppler calculated independently for RX1 and RX2.

Optional:

```text
--doppler-mode beam-center-compensated
```

This subtracts the geometric Doppler of the current beam center, approximating a
satellite that frequency-precompensates toward the served terrestrial cell.  The
remaining offset at each receiver is therefore a residual Doppler.

## LEO outputs

In addition to the normal v3.1 products, LEO mode writes:

- `ground_truth_leo_links.csv` — one row per receiver per simulated second,
  including satellite lat/lon/altitude, az/el, slant range, range rate,
  visibility, geometric/applied Doppler, beam target, off-axis angle, beam gain,
  FSPL, LTE load state and IQ template.
- `ground_truth_beam_schedule.csv`
- satellite/beam/Doppler arrays inside each receiver NPZ

LEO-specific plots are:

1. `01_leo_ground_track.png`
2. `05_leo_spot_beam_pattern.png`
3. `09_leo_elevation_range.png`
4. `10_leo_doppler_beam.png`
5. `13_beam_target_schedule.png`

Plots 02, 03, 04, 06, 07, 08, 11 and 12 remain common to both modes.

## Important modeling boundaries

This first satellite implementation is intentionally a toy demonstrator:

- the orbit/trajectory is physical, coming from the supplied TLE and SGP4;
- visibility, range and Doppler are calculated from that trajectory;
- the spot-beam schedule and beam pattern are scenario assumptions;
- LEO propagation currently uses time-varying free-space loss plus the receiver
  model; atmospheric, ionospheric, polarization and local-clutter effects are not
  yet included;
- D2C harmonics fall outside the current 200--2500 MHz receiver band and are not
  generated by default;
- the close-in carrier-leakage term remains an illustrative RF-chain assumption.

## Validation status

The v3.2 Python files and Bash capture script were syntax-checked.  The D2C radio
profile, 25-PRB development OFDM fallback, RF/OOB model, and the time-varying LEO
receiver calculation were exercised with synthetic geometry in the patch-build
environment.  Full TLE propagation could not be runtime-tested there because
`sgp4`/Astropy are not installed in that isolated environment; the TLE conversion
follows the standard SGP4 TEME -> Astropy ITRS workflow.  The existing terrestrial
P.452 path is retained from the already-tested v3.1 project.
