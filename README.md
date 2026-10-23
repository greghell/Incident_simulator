# EMILY-X LTE v3.2 — terrestrial, LEO D2C, and ML dataset generation

EMILY-X is an end-to-end RF-interference forward simulator designed to generate synthetic observations compatible with the EMILY RFI pipeline.

The common simulation chain is:

`genuine LTE IQ -> RF/OOB model -> propagation -> receiver -> spectrum/spectrogram -> EMILY records -> ground truth`

v3.2 supports two reference transmitter models:

- a terrestrial LTE cell tower;
- a single LEO Direct-to-Cell (D2C) satellite.

The repository also includes fast batch generators intended to create large machine-learning datasets from these physical models.

---

# 1. Reference simulator: `emilyx_lte_end_to_end_simulator_v3_2.py`

The v3.2 simulator is the full reference implementation. It is intended primarily for validating the physical model, inspecting plots and ground truth, and generating detailed individual scenarios.

## Terrestrial mode

The default terrestrial example uses:

- 800 MHz center;
- 10 MHz LTE / 50 PRB;
- 11.52 Msps genuine srsRAN IQ;
- fixed sector antenna;
- SRTM terrain;
- ITU-R P.452 propagation by default;
- optional diagnostic free-space propagation.

Required IQ files:

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

---

## LEO D2C mode

The reference D2C profile uses:

- 1990–1995 MHz downlink;
- 1992.5 MHz center;
- 5 MHz LTE / 25 PRB;
- downlink EARFCN 8665;
- 5.76 Msps genuine srsRAN IQ;
- one TLE-driven LEO satellite;
- time-varying slant range;
- azimuth/elevation;
- visibility;
- Doppler;
- time-varying free-space loss;
- steerable Earth-fixed spot beams.

The current LEO beam parameters are explicit scenario assumptions rather than claims about a specific operational satellite:

- peak beam-center EIRP: 75 dBm;
- HPBW: 8 degrees;
- sidelobe floor: -30 dB;
- maximum scan loss: 6 dB.

---

# 2. Python dependencies

Activate the existing environment:

```bat
conda activate emilyx
```

For TLE propagation, install:

```bat
pip install sgp4
```

Astropy is used to convert the SGP4 TEME trajectory into the Earth-fixed ITRS frame.

The terrestrial P.452 model additionally uses `pycraf` and SRTM terrain data.

---

# 3. Generating genuine D2C LTE IQ

The capture script supports both terrestrial and D2C profiles.

For D2C:

```bash
chmod +x generate_lte_iq_states_fixed.sh
./generate_lte_iq_states_fixed.sh d2c
```

This starts a 25-PRB srsRAN LTE cell at EARFCN 8665 using 5.76 Msps and generates:

- `srsran_d2c_idle.cf32`
- `srsran_d2c_light.cf32`
- `srsran_d2c_medium.cf32`
- `srsran_d2c_loaded.cf32`

The D2C light and medium offered loads default to 2 and 7 Mbit/s. The loaded state uses four parallel TCP streams to drive the cell toward saturation.

The IQ files are intentionally short. EMILY-X divides each genuine capture into multiple waveform templates and reuses them during longer simulations.

---

# 4. Running a TLE-driven D2C example

Create a file containing either two TLE lines or a satellite name followed by the two TLE lines:

```text
SATELLITE NAME
1 .....
2 .....
```

Use a TLE reasonably close in epoch to the simulated observation.

A useful first run is:

```bat
python emilyx_lte_end_to_end_simulator_v3_2.py ^
  --tx-type leo ^
  --tle-file satellite.tle ^
  --start-utc 2026-09-11T20:00:00Z ^
  --center-on-next-pass ^
  --duration 600 ^
  --output-dir output_d2c_600s
```

`--center-on-next-pass` searches forward from `--start-utc`, finds the next qualifying pass over the receiver area, and centers the observation on the pass maximum.

Without this option, `--start-utc` is the exact beginning of the simulated observation.

---

# 5. D2C beam behavior

During a visible pass, the default demonstration beam schedule moves among several Earth-fixed targets:

`east_cell -> site -> northwest_cell -> site -> south_cell`

The satellite continuously steers toward the active target while moving along its orbit.

Each receiver therefore experiences a time-varying:

- beam off-axis angle;
- relative antenna gain;
- slant range;
- elevation;
- Doppler shift.

A custom schedule can be supplied with:

```text
--beam-schedule-json
```

For example:

```json
{
  "targets": {
    "custom": {
      "lat_deg": 39.55,
      "lon_deg": -114.45
    }
  },
  "segments": [
    {
      "start_s": 100,
      "end_s": 180,
      "target": "custom"
    },
    {
      "start_s": 180,
      "end_s": 260,
      "target": "site"
    }
  ]
}
```

Unspecified times use target `none`.

---

# 6. Doppler

The default mode is:

```text
--doppler-mode geometric
```

The LTE spectrum is shifted using the instantaneous range rate calculated independently for each receiver.

An alternative is:

```text
--doppler-mode beam-center-compensated
```

This subtracts the Doppler predicted at the current beam center and therefore approximates a satellite performing frequency pre-compensation toward the served terrestrial cell.

The remaining receiver offset is then a residual Doppler.

---

# 7. Detailed LEO outputs

The full v3.2 simulator produces detailed ground truth including:

### `ground_truth_leo_links.csv`

One row per receiver per simulated second, including:

- satellite latitude;
- satellite longitude;
- altitude;
- azimuth;
- elevation;
- slant range;
- range rate;
- visibility;
- geometric Doppler;
- applied Doppler;
- beam target;
- beam off-axis angle;
- beam gain;
- free-space path loss;
- LTE load state;
- selected IQ template.

### `ground_truth_beam_schedule.csv`

Contains the Earth-fixed beam-target schedule.

Satellite geometry, beam state and Doppler arrays are also stored in the receiver NPZ files.

LEO-specific diagnostic plots include:

1. `01_leo_ground_track.png`
2. `05_leo_spot_beam_pattern.png`
3. `09_leo_elevation_range.png`
4. `10_leo_doppler_beam.png`
5. `13_beam_target_schedule.png`

The remaining transmitter, RF, receiver and spectrogram plots are shared with the terrestrial mode.

---

# 8. Machine-learning dataset generation

The reference simulator is useful for validation but is not intended to be executed hundreds of thousands or millions of times.

Separate batch generators therefore implement faster versions of the same forward-model concept.

The machine-learning workflow deliberately separates:

```text
ML-visible EMILY observations
```

from:

```text
simulation ground truth
```

True transmitter or satellite information must never be included in the features presented to the ML model.

Train/validation/test splitting is performed at the complete transmitter or satellite-realization level rather than at the individual incident level.

This prevents nearly identical measurements from the same physical realization from leaking between training and evaluation datasets.

---

# 9. Terrestrial ML generator

`emilyx_ml_fast_parallel.py` is the fast terrestrial dataset generator.

It is designed to generate large numbers of independent cell-tower realizations without launching the complete v3.2 simulator for each one.

The fast implementation:

- loads genuine LTE IQ once;
- builds RF spectral templates once;
- caches the templates;
- reuses them for all subsequent scenarios;
- simulates only the frequency windows needed by the detector;
- runs independent scenarios in parallel;
- supports P.452/SRTM or diagnostic FSPL;
- avoids per-scenario plots and full-band NPZ products unless requested.

A typical run is:

```bat
python emilyx_ml_fast_parallel.py ^
  --n-datasets 1000 ^
  --workers 4 ^
  --output-dir emilyx_terrestrial_ml_1000 ^
  --no-srtm-download
```

The terrestrial campaign can vary parameters such as:

- transmitter latitude/longitude;
- tower height;
- frequency;
- conducted power;
- antenna gain;
- sector azimuth;
- HPBW;
- downtilt;
- traffic state;
- IQ template;
- shadowing;
- receiver noise;
- RF-chain profile.

---

# 10. Satellite ML generator

Large satellite ML campaigns use:

```text
emilyx_satellite_ml_parallel.py
```

This generator is designed for thousands to millions of synthetic learning examples.

The central difference from the full v3.2 simulator is that each scenario contains a new **synthetic physical LEO trajectory** rather than requiring a new external TLE and full SGP4/Astropy processing for every realization.

The original v3.2 TLE/SGP4 implementation remains the reference model and can be used to validate subsets of the fast synthetic dataset.

---

## Satellite realization

Each independent scenario contains one satellite.

The trajectory changes between scenarios.

The generator can vary quantities including:

- orbital altitude;
- ground-track orientation;
- closest approach to the receiver network;
- closest-approach direction;
- pass epoch;
- peak EIRP;
- beam width;
- LTE traffic state;
- genuine IQ template.

From the trajectory, the simulator calculates independently for every receiver:

- satellite azimuth;
- elevation;
- slant range;
- range rate;
- Doppler;
- beam off-axis angle;
- beam gain;
- received spectrum.

Candidate satellite realizations that do not provide sufficient common coverage of the receiver network are rejected and resampled.

---

# 11. Arbitrary receiver networks

The satellite dataset generator supports an arbitrary number of receivers.

The initial example uses the existing two EMILY-X receivers:

```bat
--receiver RX1,39.5249,-114.373325 ^
--receiver RX2,39.595518,-114.535418
```

Additional receivers can simply be appended:

```bat
--receiver RX3,39.55,-114.42 ^
--receiver RX4,39.48,-114.50
```

There is no conceptual restriction to two receivers.

The same satellite trajectory is evaluated independently from every receiver location.

---

# 12. Per-spectrum EMILY records

For machine learning, the definition of an EMILY incident is changed from the original multi-second connected-component event.

In the dataset generator:

**one receiver spectrum at one time sample is one potential EMILY measurement.**

There is no grouping across time.

The processing becomes:

```text
one spectrum
    ->
spectral background estimation
    ->
RFI detection
    ->
frequency grouping within that spectrum
    ->
EMILY-like record
```

A retained record therefore has:

```text
time_structure = single_spectrum
n_time_pixels  = 1
```

and represents one receiver integration.

The initial backend continues to use approximately one-second measurements.

Observable quantities can include:

- receiver identity/location;
- observation timestamp;
- low frequency;
- high frequency;
- center frequency;
- measured bandwidth;
- measured intensity;
- frequency resolution;
- number of detected frequency pixels.

No true satellite position is included in the ML-visible record.

---

# 13. Synchronized multi-receiver learning samples

The fundamental ML object is a **synchronized time sample**.

For example, with two receivers:

```text
sample_000001 at time t

    RX1 EMILY spectrum record
    RX2 EMILY spectrum record

                ↓

          machine-learning model

                ↓

    satellite position at time t
```

With four receivers, the same `sample_id` contains four records.

A sample is retained only when the satellite is independently detected by **every configured receiver** at that time.

Thus:

```text
RX1 detected
AND RX2 detected
AND ...
AND RXN detected
```

must be true.

This provides a clean initial localization dataset without missing-receiver cases.

Partial detections can be added later as a more difficult learning problem.

---

# 14. Satellite ML labels

Ground truth is separated into two levels.

## Scenario-level ground truth

Quantities describing the complete satellite realization, for example:

- scenario ID;
- train/validation/test split;
- orbital altitude;
- trajectory orientation;
- closest approach;
- closest-approach location;
- beam parameters;
- peak EIRP;
- pass epoch;
- receiver network definition.

## Time-sample ground truth

Each synchronized `sample_id` stores the instantaneous satellite state, including:

- satellite latitude;
- longitude;
- altitude;
- ECEF position;
- ECEF velocity;
- LTE state;
- IQ template.

Additional receiver-specific truth can include:

- azimuth;
- elevation;
- slant range;
- range rate;
- Doppler;
- beam off-axis angle;
- beam gain;
- received signal level.

The first localization problem can therefore be formulated as:

```text
INPUT:
    synchronized EMILY records from all receivers at time t

TARGET:
    satellite latitude(t)
    satellite longitude(t)
    satellite altitude(t)
```

Velocity or complete trajectory estimation can be added later.

---

# 15. Running the satellite ML generator

Assuming the genuine D2C IQ files are present:

- `srsran_d2c_idle.cf32`
- `srsran_d2c_light.cf32`
- `srsran_d2c_medium.cf32`
- `srsran_d2c_loaded.cf32`

a small first campaign is:

```bat
python emilyx_satellite_ml_parallel.py ^
  --n-datasets 100 ^
  --workers 4 ^
  --samples-per-scenario 32 ^
  --receiver RX1,39.5249,-114.373325 ^
  --receiver RX2,39.595518,-114.535418 ^
  --iq-dir . ^
  --output-dir emilyx_satellite_100
```

Here:

```text
--n-datasets
```

is the number of independent satellite realizations.

```text
--samples-per-scenario
```

is the requested number of synchronized detected time samples retained from each accepted realization.

With:

```text
100 satellite realizations
32 synchronized samples per realization
2 receivers
```

the campaign can produce approximately:

```text
3,200 labeled satellite positions
6,400 ML-visible EMILY records
```

If every requested sample passes the common-detection requirement.

---

# 16. Scaling

It is useful to distinguish a **satellite realization** from a **training example**.

For example:

```text
100,000 satellite trajectories
× 32 synchronized samples per trajectory
= 3.2 million localization examples
```

With two receivers this corresponds to approximately:

```text
6.4 million individual EMILY records
```

Therefore, millions of training examples do not necessarily require millions of separate satellite trajectories.

Independent satellite realizations are processed in parallel using multiple worker processes.

The genuine LTE IQ and derived RF templates are reused rather than regenerated for every scenario.

---

# 17. Satellite ML output structure

A satellite ML campaign produces a compact dataset such as:

```text
emilyx_satellite_dataset/
│
├── dataset_metadata.json
├── dataset_index.csv
├── feature_schema.json
│
├── ml_inputs/
│   └── emily_records.csv.gz
│
├── ground_truth/
│   ├── scenario_labels.csv
│   └── sample_labels.csv.gz
│
└── cache/
    ├── d2c_template_bank_....npz
    └── d2c_template_bank_....json
```

### `ml_inputs/emily_records.csv.gz`

Contains only quantities available to the machine-learning model.

There is one row per receiver per retained time sample.

### `ground_truth/scenario_labels.csv`

Contains the physical parameters defining each independent satellite realization.

### `ground_truth/sample_labels.csv.gz`

Contains the instantaneous satellite state associated with each synchronized sample.

### `dataset_index.csv`

Summarizes scenario acceptance, number of retained samples and processing status.

### `feature_schema.json`

Documents which fields are model features, grouping identifiers and labels.

---

# 18. Dataset splitting

A complete satellite realization belongs entirely to one of:

- training;
- validation;
- test.

Individual seconds from the same trajectory are never split among these groups.

This is essential because neighboring samples from one pass are highly correlated. Randomly splitting individual time samples would produce severe train/test leakage and unrealistically optimistic performance.

---

# 19. Modeling boundaries

The two satellite implementations serve different purposes.

## Full v3.2 TLE model

Designed for physical validation and detailed inspection:

- real supplied TLE;
- SGP4 propagation;
- TEME-to-ITRS conversion;
- detailed satellite ground truth;
- time-varying beam geometry;
- full receiver spectrogram;
- diagnostic plots;
- original multi-time EMILY event preview.

## Fast satellite ML generator

Designed for dataset scale:

- new synthetic physical LEO trajectory for every scenario;
- arbitrary receiver count;
- parallel scenario generation;
- cached genuine LTE/RF templates;
- only relevant spectral windows calculated;
- one EMILY measurement per receiver spectrum;
- samples retained only when all receivers detect the satellite;
- compact ML-safe outputs;
- separate labels/ground truth.

The full v3.2 model should therefore be treated as the higher-detail reference implementation, while the fast generator is optimized for statistical dataset production.

---

# 20. Current physical limitations

The current LEO models intentionally remain simplified.

In particular:

- beam patterns are configurable scenario models;
- satellite EIRP is a scenario parameter;
- atmospheric attenuation is not yet modeled in detail;
- ionospheric propagation is not yet included;
- polarization mismatch is not yet included;
- local terrestrial clutter is not currently applied to the satellite path;
- D2C harmonics are disabled in the reference configuration;
- close-in carrier leakage remains an illustrative RF-chain assumption.

These effects can be added progressively if they prove important for ML performance.

---

# 21. Validation status

The terrestrial SRTM + ITU-R P.452 chain has been exercised in the earlier EMILY-X versions and is retained in v3.2.

The v3.2 D2C implementation has now also been run end-to-end with:

- genuine 5 MHz / 25 PRB srsRAN D2C IQ;
- a real TLE;
- SGP4 orbit propagation;
- receiver-specific elevation/range/Doppler;
- steerable beam geometry;
- received spectrogram generation;
- local EMILY detection;
- detailed satellite ground truth.

The resulting detections followed the simulated beam illumination and propagation geometry as expected.

The fast satellite ML generator has been tested through its parallel dataset-generation path with synthetic IQ and synchronized multi-receiver samples. Large-scale generation using the genuine D2C IQ dataset remains the next validation step.

---

# 22. Recommended development path

The current suggested progression is:

1. use full v3.2 runs to validate individual physical scenarios;
2. generate small ML batches and inspect every record/label;
3. verify that no simulation ground truth leaks into model-visible features;
4. verify synchronization across receivers;
5. validate a subset of fast synthetic trajectories against the TLE/SGP4 reference model;
6. train a baseline localization model;
7. increase the number and diversity of satellite realizations;
8. add more receivers;
9. introduce partial/missed detections and more difficult RF environments;
10. eventually extend the same framework to multiple simultaneous emitters and additional transmitter classes.
