# Floquet Examples

This folder contains example scripts for the periodic-drive Floquet module.

## Scripts

- `driven_transmon_quasienergy_sweep.py`
  - Sweeps a strong transmon drive frequency, tracks quasienergy branches, and prints multiphoton resonance candidates.
- `transmon_cavity_sideband_floquet_scan.py`
  - Scans an effective red-sideband drive in a transmon-cavity model and plots the tracked quasienergy branches together with the minimum branch gap.

These examples use the same `UniversalCQEDModel` or `DispersiveTransmonCavityModel` APIs as the rest of the package; only the solver and drive container change.