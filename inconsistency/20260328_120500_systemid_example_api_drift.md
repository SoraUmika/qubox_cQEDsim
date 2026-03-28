# System-ID Example API Drift

Created: 2026-03-28 12:05:00
Status: Open

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

## Suspected Issues

- The example may contain additional stale assumptions beyond `run_t1(...)` and hardware-prior field names because it was not fully repaired in this task.

## Unresolved Questions

- Whether the example should be repaired to the current stable reset/inspection workflow or replaced with a narrower script that avoids unsupported rollout paths.

## Notes For Follow-Up

- During this tutorial-extension task, the new workflow notebooks were implemented from validated source-level APIs instead of reusing this stale example.
- A later cleanup pass should either update this example to current APIs or remove claims that it demonstrates the full calibration-to-RL path.
