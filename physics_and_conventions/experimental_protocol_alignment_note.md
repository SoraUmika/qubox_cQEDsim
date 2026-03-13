# Experimental-Protocol Alignment Note

Date: March 12, 2026

## Current Status

The repository now keeps the reusable preparation and measurement primitives inside the library while moving protocol-style orchestration back out to `examples/`.

## Reusable Library Surface

- `cqed_sim.core`
  - `StatePreparationSpec` and the subsystem-state helpers
  - `prepare_state(...)` and `prepare_ground_state(...)`
- `cqed_sim.measurement`
  - `QubitMeasurementSpec` and `measure_qubit(...)`
  - `ReadoutResonator`, `PurcellFilter`, `AmplifierChain`, `ReadoutChain`
- `cqed_sim.sim`
  - `pure_dephasing_time_from_t1_t2(...)`
  - solver, extractor, and noise primitives used by the workflows

## Example-Side Workflow Surface

- `examples/protocol_style_simulation.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`
- `examples/sequential_sideband_reset.py`

## Intended Boundary

- reusable, backend-agnostic simulation building blocks stay in `cqed_sim`
- typical usage recipes, orchestration helpers, and canned experiment flows stay in `examples`

This keeps the installed package focused on simulation primitives while preserving the repo's workflow demonstrations and validation artifacts.
