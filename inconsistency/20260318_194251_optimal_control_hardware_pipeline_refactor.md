# 2026-03-18 19:42:51 Optimal-Control Hardware Pipeline Refactor

## Confirmed Issues

### 1. Control schedule semantics are overloaded across incompatible layers

- What:
  - the current optimal-control implementation uses the same dense array as the
    optimization variable, the exported command waveform, and the physical
    waveform passed into Hamiltonian propagation.
- Where:
  - `cqed_sim/optimal_control/parameterizations.py`,
    `cqed_sim/optimal_control/grape.py`, and
    `cqed_sim/optimal_control/evaluation.py`.
- Affected components:
  - held-sample parameterization, hardware filtering, command-vs-physical
    diagnostics, simulator replay, and any future quantization or I/Q radial
    constraint support.
- Why this is inconsistent:
  - the module already advertises a generic `ControlProblem` and parameterization
    layer, but the implementation still assumes the parameter array is identical
    to the propagated Hamiltonian coefficients.
- Consequences:
  - realistic hardware constraints cannot be modeled cleanly without conflating
    optimization variables and physical controls.

## Suspected / Follow-up Questions

- Projection-based hard constraints may require a custom optimizer loop or an
  explicit projected-forward-model mode rather than direct reuse of the current
  SciPy `minimize(...)` wrapper.

## Status

- Current status: fixed on 2026-03-18
- Resolution summary:
  - introduced an explicit parameterization-to-command stage via generalized
    parameterizations plus `HeldSampleParameterization`,
  - added a composable hardware-map stage via `HardwareModel` with differentiable
    low-pass, boundary-window, and IQ-radius maps,
  - updated solver diagnostics, results, and simulator replay to expose command
    and physical waveforms separately,
  - preserved the legacy unconstrained path when the hardware layer is absent.

## Fix Record

- Fixed by:
  - `cqed_sim/optimal_control/parameterizations.py`
  - `cqed_sim/optimal_control/hardware.py`
  - `cqed_sim/optimal_control/grape.py`
  - `cqed_sim/optimal_control/evaluation.py`
  - `cqed_sim/optimal_control/result.py`
  - `tests/test_42_optimal_control_hardware_constraints.py`
- Remaining concerns:
  - projection-based hard constraints, quantization-aware gradients, and richer
    basis families remain deferred follow-up work.