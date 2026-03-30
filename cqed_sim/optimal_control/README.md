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
- `solve_structured_control(problem, config)` and `StructuredControlSolver` for parameter-space optimization over smooth pulse families.
- `StructuredPulseFamily` plus concrete families including:
  - `GaussianDragPulseFamily`,
  - `FourierSeriesPulseFamily`.
- `StructuredControlChannel` and `StructuredPulseParameterization` to map a flat parameter vector onto one or more repository control terms.
- `build_structured_control_problem_from_model(...)` to stay on the standard model-builder path instead of creating a parallel study-only abstraction.
- `HardwareModel` and `HardwareMap` stages for explicit command-to-physical waveform transforms.
- `CustomControlObjective` for user-defined objectives that return both a scalar cost and a physical-waveform gradient.
- `save_structured_control_artifacts(...)` to persist optimized parameters, waveform tables, spectra, and optimization-history plots for studies.

## Structured Pulse Families

The central structured-control abstraction is a smooth complex envelope

$$
u(t; \theta)
$$

defined by a small set of named parameters.

The current built-in families are:

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
- `UnitaryObjective`,
- `CustomControlObjective`,
- waveform penalties in parameter, command, or physical domains.

## Example Workflows

Interactive and script entry points:

- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- `examples/hardware_constrained_grape_demo.py`
- `examples/grape_storage_subspace_gate_demo.py`
- `examples/structured_optimal_control_demo.py`

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

- The structured backend still assumes closed-system dense propagation, just like the current GRAPE path.
- Only two pulse families are built in today; the framework is designed to add more families cleanly.
- `CustomControlObjective` currently runs on the NumPy engine only; the JAX engine remains targeted at the built-in fidelity objectives.
- `evaluate_control_with_simulator(...)` reports the built-in state-transfer and unitary objectives; custom objective replay reporting remains study-specific.# `cqed_sim.optimal_control`

The `optimal_control` module provides a model-backed GRAPE (Gradient Ascent Pulse Engineering) implementation for optimizing quantum gate pulses directly against the `cqed_sim` physics stack. It is the direct optimal-control layer of the library, complementary to the gate-sequence synthesis in `cqed_sim.map_synthesis`.

## Relevance in `cqed_sim`

GRAPE-based optimal control is relevant when:

- a gate target cannot be reached by the standard parametric pulse builders in `cqed_sim.pulses`,
- fine-grained leakage suppression, robustness, or multi-objective trade-offs are needed,
- or when map synthesis via `cqed_sim.map_synthesis` provides a pulse sequence that needs further refinement at the waveform level.

The module is simulator-backed: it uses the same Hamiltonian and physical conventions as `cqed_sim.sim` to compute gradients, so the optimized pulses are directly compatible with the rest of the runtime stack.

## Main Capabilities

- **`solve_grape(problem, config)`**: The main entry point. Takes a `ControlProblem` and a `GrapeConfig`, runs GRAPE, and returns a `GrapeResult`.
- **`solve_grape_multistart(problem, config, multistart_config)`**: Runs GRAPE from multiple random starting points and returns all results sorted best-first. Supports optional parallel execution via `GrapeMultistartConfig(max_workers=N)`.
- **`GrapeSolver`**: Object-oriented variant that exposes `.solve(problem)` and iteration-level hooks.
- **Control problem construction**: `build_control_problem_from_model(...)` and `build_control_system_from_model(...)` create a `ControlProblem` directly from a `cqed_sim` model, compiled time grid, and drive channel spec.
- **Hardware-aware control pipeline**: `resolve_control_schedule(...)` exposes the full parameter -> command -> physical waveform pipeline so users can inspect the exact waveform seen by propagation.
- **Objectives**: `UnitaryObjective` (maximize unitary fidelity), `StateTransferObjective` / `multi_state_transfer_objective` (maximize state transfer fidelity), and `state_preparation_objective`. An objective can also be derived from a `unitary_synthesis` target via `objective_from_unitary_synthesis_target(...)`.
- **Penalties**: `AmplitudePenalty`, `SlewRatePenalty`, `LeakagePenalty` — additive regularization terms to enforce hardware and physics constraints.
- **Hardware-aware penalties**: `BoundPenalty`, `BoundaryConditionPenalty`, and `IQRadiusPenalty` extend the additive framework to parameter, command, or physical waveform domains.
- **Parameterization**: `PiecewiseConstantParameterization` and `HeldSampleParameterization` expose parameter-space values separately from the propagation-grid command waveform.
- **Hardware maps**: `HardwareModel`, `FirstOrderLowPassHardwareMap`, `BoundaryWindowHardwareMap`, and `SmoothIQRadiusLimitHardwareMap` let the optimizer propagate through differentiable hardware transforms.
- **Initial guesses**: `zero_control_schedule`, `random_control_schedule`, `warm_start_schedule`.
- **Evaluation**: `evaluate_control_with_simulator(...)` replays either the command waveform or the physical post-hardware waveform through the `cqed_sim.sim` runtime.
- **Result types**: `GrapeResult` (full optimization history and final schedule), `ControlResult` (final schedule and fidelity), `GrapeIterationRecord` (per-iteration diagnostics). Results now expose command and physical waveforms plus hardware diagnostics.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `solve_grape(problem, config)` | Main GRAPE entry point |
| `solve_grape_multistart(problem, config, multistart_config)` | Multi-start GRAPE — returns all restart results sorted best-first |
| `GrapeSolver` | Object-oriented GRAPE solver |
| `GrapeConfig` | Solver hyperparameters (iterations, learning rate, tolerance) |
| `GrapeMultistartConfig` | Multi-start settings (n_restarts, max_workers, return_all) |
| `ControlProblem` | Full problem specification (system + objective + penalties) |
| `build_control_problem_from_model(...)` | Build a `ControlProblem` from a `cqed_sim` model |
| `ControlSystem` | Hamiltonian and drive-term specification |
| `resolve_control_schedule(...)` | Inspect parameter, command, and physical waveforms for a schedule |
| `UnitaryObjective` | Gate fidelity objective |
| `StateTransferObjective` | State-transfer fidelity objective |
| `AmplitudePenalty` | Amplitude regularization |
| `SlewRatePenalty` | Bandwidth regularization |
| `BoundPenalty` | Soft bound-violation penalty on parameter, command, or physical waveforms |
| `BoundaryConditionPenalty` | Soft zero-start / zero-end penalty |
| `IQRadiusPenalty` | Soft radial I/Q envelope penalty |
| `LeakagePenalty` | Leakage-suppression penalty |
| `PiecewiseConstantParameterization` | Control schedule parameterization |
| `HeldSampleParameterization` | Coarse-sample sample-and-hold parameterization |
| `HardwareModel` | Sequential hardware transform model from command to physical waveform |
| `FirstOrderLowPassHardwareMap` | Differentiable first-order bandwidth limit |
| `BoundaryWindowHardwareMap` | Hard zero-start / zero-end boundary window |
| `SmoothIQRadiusLimitHardwareMap` | Differentiable radial I/Q amplitude limiter |
| `GrapeResult` | Full optimization result |
| `evaluate_control_with_simulator(...)` | Validate result with full simulator |

## Usage Guidance

### Single-start GRAPE

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.optimal_control import (
    GrapeConfig, UnitaryObjective, AmplitudePenalty, LeakagePenalty,
    build_control_problem_from_model, solve_grape,
)
from cqed_sim.core import snap_op

# Define model and target
model = DispersiveTransmonCavityModel(...)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
target_U = snap_op(model, angles=[0, np.pi, 0, 0, 0, 0, 0, 0])

# Build control problem
problem = build_control_problem_from_model(
    model=model,
    frame=frame,
    t_total=500.0e-9,
    n_steps=250,
    channel_specs=[...],  # ModelControlChannelSpec per drive channel
    objective=UnitaryObjective(target=target_U),
    penalties=[AmplitudePenalty(weight=1e-3), LeakagePenalty(weight=0.1)],
)

# Run GRAPE (single start)
result = solve_grape(problem, GrapeConfig(maxiter=500))
print(result.metrics["nominal_fidelity"])
```

### Multi-start GRAPE

```python
from cqed_sim.optimal_control import GrapeConfig, GrapeMultistartConfig, solve_grape_multistart

# Run 6 independent random restarts, serial execution
results = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=200, seed=0, random_scale=0.2),
    multistart_config=GrapeMultistartConfig(n_restarts=6, max_workers=1),
)
best = results[0]  # sorted best-first
print(f"Best fidelity: {best.metrics['nominal_fidelity']:.6f}")

# Optional: parallel execution (beneficial only for long-running optimizations)
results_parallel = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=500, seed=0),
    multistart_config=GrapeMultistartConfig(n_restarts=4, max_workers=4),
)
```

**Note on parallelism:** Thread-based parallelism (`mp_context="thread"`) is the default and recommended strategy.  It has zero startup overhead.  Process-based contexts (`"spawn"`, `"loky"`) are available for full isolation if needed.

For an interactive walkthrough with pulse export, see:
`tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`

For a standalone comparison between unconstrained and hardware-aware GRAPE, see:
`examples/hardware_constrained_grape_demo.py`

For a benchmarking harness covering larger optimization cases, see:
`benchmarks/run_optimal_control_benchmarks.py`

## Important Assumptions / Conventions

- GRAPE propagators are computed on a piecewise-constant propagation grid: the Hamiltonian is held constant within each time step of `PiecewiseConstantTimeGrid`.
- Gradients are computed analytically using the co-state / adjoint method; no finite differences are used.
- When a hardware model is active, gradients flow through the parameterization map and the hardware map before reaching the schedule parameters.
- The Hamiltonian convention and frame convention match `cqed_sim.sim` exactly. Optimized pulses export directly into the `Pulse`/`SequenceCompiler`/`simulate_sequence` pipeline.
- Amplitude units are `rad/s`; time units are seconds.
- The internal control pipeline is explicit: parameter values -> command waveform -> physical waveform -> Hamiltonian coefficients.
- The unitary fidelity metric is the normalized Hilbert–Schmidt inner product over the full Hilbert space. For leakage-sensitive targets, use `LeakagePenalty` or define a subspace-projected objective.

## Relationships to Other Modules

- **`cqed_sim.sim`**: GRAPE uses the same Hamiltonian assembly and shares propagator logic; `evaluate_control_with_simulator(...)` validates results via the full QuTiP solver.
- **`cqed_sim.map_synthesis`**: synthesis can produce a pulse sequence that is then refined at the waveform level by GRAPE; `objective_from_unitary_synthesis_target(...)` bridges the two.
- **`cqed_sim.core`**: provides the model and frame that the control problem is built from.

## Limitations / Non-Goals

- GRAPE optimization is local; it converges to a local optimum. Use `solve_grape_multistart` to run multiple restarts and return the best result.
- The default engine (`engine="numpy"`) uses a NumPy-based propagator path with manual `expm_frechet` gradient computation.  The optional JAX engine (`engine="jax"`) provides JIT-compiled propagation and automatic differentiation with optional GPU acceleration.  See [JAX engine](#jax-engine) below.
- Ensemble robustness optimization (averaging over parameter uncertainty) is supported through `ModelEnsembleMember` but is not the default mode.
- The first hardware-aware extension currently targets held-sample parameterization, first-order low-pass filtering, I/Q radial limits, and boundary windows. Quantization-aware gradients, Fourier/spline bases, and projection-based hard constraints are deferred.
- Parallel multi-start defaults to thread-based execution (`mp_context="thread"`), which has zero startup overhead.  Process-based contexts (`"spawn"`, `"loky"`) are also available.

## JAX Engine

The GRAPE solver supports an optional JAX-accelerated engine that replaces the NumPy propagation + manual adjoint gradient with:

1. **JIT-compiled forward propagation** via `jax.lax.scan` and `jax.scipy.linalg.expm`.
2. **Automatic reverse-mode differentiation** via `jax.value_and_grad`, eliminating manual `expm_frechet` calls.
3. **GPU support**: when JAX is configured with a GPU device (e.g. `pip install jax[cuda12]`), all propagation and gradient computation runs on GPU with no code changes.

### Usage

```python
from cqed_sim.optimal_control import GrapeConfig, solve_grape

# CPU (default)
result = solve_grape(problem, config=GrapeConfig(engine="jax"))

# GPU (when JAX GPU is installed)
result = solve_grape(problem, config=GrapeConfig(engine="jax", jax_device="gpu"))
```

### Requirements

- CPU: `pip install jax` (already an optional dependency).
- GPU: `pip install jax[cuda12]` (or the appropriate CUDA variant).

### Performance Notes

- **First evaluation** incurs JIT compilation overhead (~1-5 s depending on system size).  Subsequent evaluations are fast.
- For **small systems** (dim ≤ 10) with few time steps, the NumPy engine may be faster due to JIT overhead.
- For **larger systems** (dim > 20) or many time steps, the JAX engine is significantly faster, especially on GPU.
- **Thread-based multi-start** + JAX engine provides near-ideal parallel scaling on CPU, since XLA computation is fully GIL-free.

## Parallelism

Multi-start GRAPE supports three parallelism strategies via `GrapeMultistartConfig(mp_context=...)`:

| Context | Startup Overhead | GIL Behavior | Best For |
|---------|-----------------|--------------|----------|
| `"thread"` (default) | Zero | NumPy/SciPy release GIL; JAX/XLA fully GIL-free | Most workloads |
| `"loky"` | Near-zero (reusable pool) | Full process isolation | Large problems, multi-GPU |
| `"spawn"` | ~4-5 s per worker (Windows) | Full process isolation | Fallback |

```python
from cqed_sim.optimal_control import GrapeMultistartConfig, solve_grape_multistart

# Thread-based (default, recommended)
results = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=200, engine="jax"),
    multistart_config=GrapeMultistartConfig(n_restarts=8, max_workers=4),
)
```

## References

- GRAPE method: Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," J. Magn. Reson. 172 (2005).
- Tutorial notebook: `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- Benchmark harness: `benchmarks/run_optimal_control_benchmarks.py`
