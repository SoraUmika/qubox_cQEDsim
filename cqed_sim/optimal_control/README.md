# `cqed_sim.optimal_control`

The `optimal_control` module is the direct-control layer of `cqed_sim`. It now supports two complementary workflows built on the same model, simulator, and pulse-export stack:

- model-backed GRAPE on a piecewise-constant propagation grid,
- structured, hardware-aware parameter-space optimization over smooth pulse families.

The structured workflow is intended for the simulator-to-hardware path where the optimization variables should be low-dimensional and physically meaningful rather than one free amplitude per time sample.

## Relevance in `cqed_sim`

Use the structured layer when the problem is naturally:

- propose a smooth pulse family,
- pass it through a transfer or distortion model,
- simulate the resulting effective control,
- optimize a small parameter vector,
- and later reuse the same parameterization for hardware-closed-loop updates.

Use GRAPE when you need fine-grained slice-level refinement on top of the same physics stack.

Both paths remain compatible with:

- `ControlProblem`,
- `HardwareModel`,
- `ControlResult`,
- `evaluate_control_with_simulator(...)`,
- pulse export through `Pulse` and `SequenceCompiler`.

## Main Capabilities

- `solve_grape(problem, config)` and `GrapeSolver` for dense model-backed GRAPE.
- NumPy-based density-matrix GRAPE for `ControlSystem(collapse_operators=...)` problems and `DensityMatrixTransferObjective(...)` targets.
- `solve_structured_control(problem, config)` and `StructuredControlSolver` for parameter-space optimization over smooth pulse families.
- `CallableParameterization` and `ControlParameterSpec` for user-defined waveform maps without a new parameterization class.
- `StructuredPulseFamily` plus concrete families including:
  - `CallablePulseFamily`,
  - `GaussianDragPulseFamily`,
  - `FourierSeriesPulseFamily`.
- `StructuredControlChannel` and `StructuredPulseParameterization` to map a flat parameter vector onto one or more repository control terms.
- `build_structured_control_problem_from_model(...)` to stay on the standard model-builder path instead of creating a parallel study-only abstraction.
- `HardwareModel` and `HardwareMap` stages for explicit command-to-physical waveform transforms.
- `CustomControlObjective` for user-defined objectives that return both a scalar cost and a physical-waveform gradient.
- `optimize_gate_time_with_grape(...)` and `optimize_gate_time_with_structured_control(...)` for explicit outer-loop duration search.
- `build_grape_refinement_problem(...)` and `solve_structured_then_grape(...)` for structured-to-GRAPE warm-start refinement.
- `save_structured_control_artifacts(...)` to persist optimized parameters, waveform tables, spectra, and optimization-history plots for studies.
- `synthesize_readout_emptying_pulse(...)`, `build_readout_emptying_parameterization(...)`, and `export_readout_emptying_to_pulse(...)` for segmented readout-resonator emptying pulses with optional Kerr-aware correction.
- `verify_readout_emptying_pulse(...)` and `refine_readout_emptying_pulse(...)` for qualification-first evaluation of those pulses under measurement, Lindblad, robustness, and hardware-distortion replay.

## Structured Pulse Families

The central structured-control abstraction is a smooth complex envelope

$$
u(t; \theta)
$$

defined by a small set of named parameters.

The built-in families are:

- `CallablePulseFamily`, which wraps `evaluator(time_rel_s, duration_s, values)` and an optional analytic `jacobian_evaluator(...)`.

- `GaussianDragPulseFamily`

  $$
  u(t; \theta) = A \left[g(t; \sigma, c) + i\,\alpha\,\frac{dg}{d\tau}\right] e^{i\phi}
  $$

  where `A` is the amplitude, `sigma_fraction` controls the width, `center_fraction` controls the center, `phase_rad` sets the global complex phase, and `drag_alpha` adds a derivative quadrature correction.

- `FourierSeriesPulseFamily`

  $$
  u(t; \theta) = I(t; \theta_I) + i Q(t; \theta_Q)
  $$

  with real Fourier bases for the I and Q quadratures:

  $$
  I(t) = \sum_k a_k \cos\left(\frac{2\pi k t}{T}\right) + \sum_{k>0} b_k \sin\left(\frac{2\pi k t}{T}\right)
  $$

  and the same structure for `Q(t)`.

Families expose:

- named parameter specs,
- hard parameter bounds,
- default values,
- waveform evaluation on the propagation grid,
- Jacobians for gradient-based optimization.

## Explicit Hardware Pipeline

The structured workflow keeps the hardware stage explicit:

$$
\theta \rightarrow u_{\mathrm{cmd}}(t; \theta) \rightarrow u_{\mathrm{phys}}(t) = \mathcal{H}[u_{\mathrm{cmd}}] \rightarrow H(t; u_{\mathrm{phys}})
$$

This uses the existing `HardwareModel` stack. Useful built-in maps include:

- `FirstOrderLowPassHardwareMap`,
- `GainHardwareMap`,
- `DelayHardwareMap`,
- `BoundaryWindowHardwareMap`,
- `SmoothIQRadiusLimitHardwareMap`,
- `FIRHardwareMap`,
- `FrequencyResponseHardwareMap`.

The repository drive convention is preserved throughout. Exported rotating-frame envelopes still satisfy

$$
c(t) = I(t) + i Q(t).
$$

For model-backed control problems, the `Q` quadrature is built as `+i(raising - lowering)`, so this defines the raw complex envelope presented to the runtime pulse stack while preserving the optimizer Hamiltonian under replay. Absolute positive drive frequencies remain a separate boundary translation handled through the `cqed_sim.core` frequency helpers before assigning any raw `Pulse.carrier` values.

## Solvers and Objectives

Structured optimization is currently driven by SciPy minimization over the parameter vector exposed by the parameterization.

- `StructuredControlConfig(use_gradients=True)` uses the parameter-space pullback supplied by the structured family and hardware model.
- `StructuredControlConfig(use_gradients=False)` keeps the same public surface but lets SciPy use derivative-free updates for methods such as `Powell` or `Nelder-Mead`.

The same `ControlProblem` can combine:

- `StateTransferObjective`,
- `DensityMatrixTransferObjective`,
- `UnitaryObjective`,
- `CustomControlObjective`,
- waveform penalties in parameter, command, or physical domains.

## Extension Workflows

The module now includes two higher-level workflows built on top of the base solver surfaces.

Gate-time search:

- `GateTimeOptimizationConfig`
- `GateTimeCandidate`
- `GateTimeOptimizationResult`
- `optimize_gate_time_with_grape(...)`
- `optimize_gate_time_with_structured_control(...)`

These helpers scale the propagation grid to a list of candidate durations, solve each candidate problem, and return the full duration-versus-objective table. CPU-thread parallelism is controlled through `GateTimeOptimizationConfig(max_workers=...)`. For GPU-backed inner solves, pair the outer loop with `GrapeConfig(engine="jax", jax_device="gpu")`.

Hybrid structured-to-GRAPE refinement:

- `build_grape_refinement_problem(...)`
- `StructuredToGrapeResult`
- `solve_structured_then_grape(...)`

This workflow runs a coarse structured optimization first, lifts the resulting command waveform into a piecewise-constant GRAPE parameterization, and then performs local GRAPE cleanup from that warm start.

## Example Workflows

Interactive and script entry points:

- `examples/callable_optimal_control_demo.py`
- `examples/optimal_control_gate_time_sweep_demo.py`
- `examples/structured_to_grape_refinement_demo.py`
- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- `examples/hardware_constrained_grape_demo.py`
- `examples/grape_storage_subspace_gate_demo.py`
- `examples/structured_optimal_control_demo.py`
- `examples/readout_emptying_demo.py`
- `examples/studies/readout_emptying/`

The structured demo writes real study artifacts under:

- `outputs/structured_optimal_control_demo/gaussian_drag/`
- `outputs/structured_optimal_control_demo/fourier_basis/`

Each artifact bundle includes:

- `result.json`
- `parameters.csv`
- `waveforms.csv`
- `history.csv`
- `waveforms.png`
- `spectrum.png`
- `optimization_history.png`

## Important Conventions

- Hamiltonian, frame, and Hilbert-space conventions match `cqed_sim.sim` exactly.
- Amplitudes remain in `rad/s`; time remains in seconds unless the whole study consistently chooses another unit system.
- The structured solver optimizes pulse-family parameters, not raw time samples.
- Structured results export back into standard repository `Pulse` objects through `ControlResult.to_pulses(...)`.

## Limitations / Non-Goals

- The structured backend still assumes closed-system dense propagation.
- The GRAPE backend supports open-system and density-matrix objectives only on the NumPy engine; the JAX engine remains closed-system only.
- Free-final-time optimization is currently exposed as an explicit duration sweep, not as a differentiable inner solver.
- Built-in pulse families are intentionally minimal; `CallablePulseFamily` and `CallableParameterization` are the supported public extension points for study-specific waveform families.
- `CustomControlObjective` currently runs only on the closed-system state-vector path; it is not yet wired into the density-matrix / Lindblad propagation route.
- `evaluate_control_with_simulator(...)` reports the built-in state-transfer and unitary objectives; custom objective replay reporting remains study-specific.

## References

- `documentations/api/optimal_control.md`
- `documentations/tutorials/optimal_control.md`
- `examples/callable_optimal_control_demo.py`
- `examples/optimal_control_gate_time_sweep_demo.py`
- `examples/structured_to_grape_refinement_demo.py`
- `examples/structured_optimal_control_demo.py`
- `examples/hardware_constrained_grape_demo.py`
