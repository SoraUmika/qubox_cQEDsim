# 2026-04-09 17:32:22 Floquet Overlap-Reference-Time Dead Config

## Confirmed Issues

### 1. `FloquetConfig.overlap_reference_time` was publicly documented but unused by `run_floquet_sweep(...)`

- What:
  - `FloquetConfig` exposed `overlap_reference_time`, and the public API documentation listed it as part of the Floquet solver configuration.
  - `run_floquet_sweep(...)` ignored that config field and always used its own `reference_time` argument default of `0.0` unless the caller overrode it explicitly.
- Where:
  - `cqed_sim/floquet/core.py`
  - `cqed_sim/floquet/analysis.py`
  - `API_REFERENCE.md`
  - `documentations/api/floquet.md`
- Affected components:
  - branch tracking during Floquet sweeps
  - user-facing Floquet configuration semantics
  - public API documentation for the sweep path
- Why this is inconsistent:
  - The config object advertised a sweep-relevant control knob that had no effect on the sweep helper using that same config.
- Consequences:
  - callers could believe they were changing the overlap-evaluation time when they were not
  - branch-tracking behavior was less configurable than the public API implied
  - documentation overstated the operational effect of the config surface

## Suspected / Follow-up Questions

- The overlap-based branch matcher remains a heuristic near true degeneracies and may still need a stronger tracking strategy even after the config path is wired correctly.

## Status

Fixed on 2026-04-09.

## Fix Record

- Updated `cqed_sim/floquet/analysis.py` so `run_floquet_sweep(...)` defaults to `config.overlap_reference_time` when `reference_time` is omitted.
- Added regression coverage in `tests/test_58_floquet.py` for both the default-config path and explicit override behavior.
- Updated public API documentation in `API_REFERENCE.md`, `documentations/api/floquet.md`, and `cqed_sim/floquet/README.md`.