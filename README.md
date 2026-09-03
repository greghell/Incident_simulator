# EMILY-X LTE v3.1 — multi-window genuine LTE templates

v3.1 keeps the working v3 architecture (genuine srsRAN LTE, RF-chain model,
SRTM + ITU-R P.452, receiver backend, EMILY products) and removes the remaining
"one PSD per load state" limitation.

## What changed

Each existing 0.5-s srsRAN capture is now split into multiple genuine LTE IQ
windows.  By default:

- window length: 100 ms
- hop: 100 ms
- maximum templates per state: 5
- states: idle, light, medium, loaded
- total default RF spectral templates: 20

For every window, v3.1 independently measures the raw complex RMS, normalizes the
IQ only after that measurement, runs the transmitter filter/PA/OOB/harmonic
model, and builds a 140-kHz-RBW spectral template.  The state-to-state and
window-to-window measured power differences are therefore retained.

During the final 1-s simulation, the load-state timeline still selects
idle/light/medium/loaded, but a second template timeline selects one of the
real 100-ms IQ-derived templates for that state.  Templates are used in shuffled
cycles without replacement, so the model avoids an artificial fixed 5-s
periodicity while remaining deterministic from the simulation random seed.

No new Linux/srsRAN capture is required.  Use the four files already generated:

    srsran_idle.cf32
    srsran_light.cf32
    srsran_medium.cf32
    srsran_loaded.cf32

## Recommended 60-s validation run

From the Windows Anaconda Prompt, in the directory containing the simulator and
four IQ files:

    conda activate emilyx
    python emilyx_lte_end_to_end_simulator_v3_1.py ^
      --duration 60 ^
      --output-dir output_emilyx_lte_v3_1_dynamic60

P.452 remains the default propagation model.

At startup, v3.1 prints every extracted template, for example:

    idle: ... templates=5
        T0: 0-100 ms ...
        T1: 100-200 ms ...
        ...

The exact RMS/power differences should come from the real captures.

## New/changed outputs

Receiver NPZ files now contain:

- `lte_state_t` — nominal LTE load state for each simulated second
- `lte_template_index_t` — genuine IQ template selected for each second
- `lte_states_json` — state and per-template RMS/power metadata

The transmitter reference is now:

    tx_reference_emissions_v3_1.npz

It includes each per-state/per-template 140-kHz spectrum and short IQ excerpts.

A new diagnostic plot is generated:

    plots/12_lte_template_schedule.png

Plot 07 now uses the actual per-second template, so variation can occur both
between load states and within a single state.

## Full 10-minute run

Once the 60-s validation looks correct:

    python emilyx_lte_end_to_end_simulator_v3_1.py ^
      --duration 600 ^
      --output-dir output_emilyx_lte_v3_1_600s

## Optional overlapping templates

The default five non-overlapping 100-ms windows are recommended first.  To use
100-ms windows every 50 ms (up to nine windows in a 0.5-s capture):

    python emilyx_lte_end_to_end_simulator_v3_1.py ^
      --duration 60 ^
      --template-window-ms 100 ^
      --template-hop-ms 50 ^
      --max-templates-per-state 9 ^
      --output-dir output_emilyx_lte_v3_1_overlap60

This approximately doubles the transmitter RF-model work relative to the
default five-template setup.  It is not necessary unless the five-window
version still looks too repetitive.

## Validation performed before delivery

The v3.1 script was syntax-checked and executed end-to-end with synthetic
complex64 captures using both one and multiple templates per state.  The tests
verified template-specific RMS scaling, shuffled template scheduling, FSPL
propagation, receiver simulation, NPZ creation, and all plots including the new
template-schedule plot.  The actual P.452 branch is inherited from the already
validated v3 code; the user's existing SRTM/P.452 setup is unchanged.
