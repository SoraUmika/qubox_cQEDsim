# `cqed_sim.optimal_control`

The `optimal_control` module provides a model-backed GRAPE (Gradient Ascent Pulse Engineering) implementation for optimizing quantum gate pulses directly against the `cqed_sim` physics stack. It is the direct optimal-control layer of the library, complementary to the gate-sequence synthesis in `cqed_sim.unitary_synthesis`.

## Relevance in `cqed_sim`

GRAPE-based optimal control is relevant when:

- a gate target cannot be reached by the standard parametric pulse builders in `cqed_sim.pulses`,
- fine-grained leakage suppression, robustness, or multi-objective trade-offs are needed,
- or when unitary synthesis via `cqed_sim.unitary_synthesis` provides a pulse sequence that needs further refinement at the waveform level.

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

**Note on parallelism:** On Windows, `spawn` process startup overhead (~4–5 s per worker) dominates for short optimizations. Use `max_workers > 1` only when each individual GRAPE run takes several seconds or more.

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
- **`cqed_sim.unitary_synthesis`**: synthesis can produce a pulse sequence that is then refined at the waveform level by GRAPE; `objective_from_unitary_synthesis_target(...)` bridges the two.
- **`cqed_sim.core`**: provides the model and frame that the control problem is built from.

## Limitations / Non-Goals

- GRAPE optimization is local; it converges to a local optimum. Use `solve_grape_multistart` to run multiple restarts and return the best result.
- The current implementation uses a NumPy-based propagator path. It does not exploit JAX JIT or GPU acceleration on the gradient computation. GPU support is deferred until a dense-matrix propagator path that bypasses QuTiP is adopted.
- Ensemble robustness optimization (averaging over parameter uncertainty) is supported through `ModelEnsembleMember` but is not the default mode.
- The first hardware-aware extension currently targets held-sample parameterization, first-order low-pass filtering, I/Q radial limits, and boundary windows. Quantization-aware gradients, Fourier/spline bases, and projection-based hard constraints are deferred.
- Parallel multi-start via `max_workers > 1` carries significant process-startup overhead on Windows (spawn context). For short optimizations (< ~1 s per restart), serial execution is faster.

## References

- GRAPE method: Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," J. Magn. Reson. 172 (2005).
- Tutorial notebook: `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- Benchmark harness: `benchmarks/run_optimal_control_benchmarks.py`
