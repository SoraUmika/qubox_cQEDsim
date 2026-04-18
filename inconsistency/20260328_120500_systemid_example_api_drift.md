# System-ID Example API Drift

Created: 2026-03-28 12:05:00
Status: Fixed on 2026-04-10

## Confirmed Issues

- `examples/calibration_systemid_rl_pipeline.py` no longer matches the current calibration-target API.
  - `run_t1(...)` is now called with explicit delay samples, but the example still uses the older no-delay call shape.
- The same example uses hardware prior keys that do not match the current `HardwareConfig` dataclass.
  - In particular, `amp_scale` is not an accepted hardware field in the current implementation.

## Where It Appears

- `examples/calibration_systemid_rl_pipeline.py`
- Current reference implementations inspected during tutorial extension work:
  - `cqed_sim/calibration_targets/t1.py`
  - `cqed_sim/pulses/hardware.py`
  - `cqed_sim/rl_control/configs.py`

## Why This Is Inconsistent

- The example presents itself as a calibration -> system-ID -> RL pipeline, but it no longer reflects the public call signatures and configuration fields exposed by the current code.
- That makes the example an unsafe source for tutorial generation and can mislead users about which uncertainty fields can be propagated into the control stack.

## Consequences

- Users copying the example can hit immediate runtime errors.
- Tutorial or documentation work that reuses the example without source verification can inherit stale API assumptions.
- The mismatch obscures the intended calibration-evidence -> randomizer -> environment workflow.

## Suspected / Follow-up Questions

- The notebook assets under `tutorials/31_system_identification_and_domain_randomization/` were not updated in this fix pass and should be kept under review if they are treated as canonical runnable workflows.

## Status

- Fixed for `examples/calibration_systemid_rl_pipeline.py`.
- The example now uses explicit delay arrays for calibration targets, re-fits measured-like traces through `cqed_sim.system_id`, and builds `CalibrationEvidence` through `evidence_from_fit(...)` plus `merge_calibration_evidence(...)`.

## Fix Record

- Fixed on 2026-04-10 in:
  - `examples/calibration_systemid_rl_pipeline.py`
  - `cqed_sim/system_id/fitting.py`
  - `cqed_sim/system_id/__init__.py`
  - `tests/test_48_system_id.py`
  - `cqed_sim/system_id/README.md`
  - `documentations/api/system_id.md`
  - `documentations/tutorials/system_identification.md`
  - `API_REFERENCE.md`
