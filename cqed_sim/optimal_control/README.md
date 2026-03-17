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
- **`GrapeSolver`**: Object-oriented variant that exposes `.solve(problem)` and iteration-level hooks.
- **Control problem construction**: `build_control_problem_from_model(...)` and `build_control_system_from_model(...)` create a `ControlProblem` directly from a `cqed_sim` model, compiled time grid, and drive channel spec.
- **Objectives**: `UnitaryObjective` (maximize unitary fidelity), `StateTransferObjective` / `multi_state_transfer_objective` (maximize state transfer fidelity), and `state_preparation_objective`. An objective can also be derived from a `unitary_synthesis` target via `objective_from_unitary_synthesis_target(...)`.
- **Penalties**: `AmplitudePenalty`, `SlewRatePenalty`, `LeakagePenalty` — additive regularization terms to enforce hardware and physics constraints.
- **Parameterization**: `PiecewiseConstantParameterization` wraps a `PiecewiseConstantTimeGrid` and the control schedule as optimizable parameters.
- **Initial guesses**: `zero_control_schedule`, `random_control_schedule`, `warm_start_schedule`.
- **Evaluation**: `evaluate_control_with_simulator(...)` replays a converged control schedule through the `cqed_sim.sim` runtime to validate the GRAPE result against the full solver.
- **Result types**: `GrapeResult` (full optimization history and final schedule), `ControlResult` (final schedule and fidelity), `GrapeIterationRecord` (per-iteration diagnostics).

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `solve_grape(problem, config)` | Main GRAPE entry point |
| `GrapeSolver` | Object-oriented GRAPE solver |
| `GrapeConfig` | Solver hyperparameters (iterations, learning rate, tolerance) |
| `ControlProblem` | Full problem specification (system + objective + penalties) |
| `build_control_problem_from_model(...)` | Build a `ControlProblem` from a `cqed_sim` model |
| `ControlSystem` | Hamiltonian and drive-term specification |
| `UnitaryObjective` | Gate fidelity objective |
| `StateTransferObjective` | State-transfer fidelity objective |
| `AmplitudePenalty` | Amplitude regularization |
| `SlewRatePenalty` | Bandwidth regularization |
| `LeakagePenalty` | Leakage-suppression penalty |
| `PiecewiseConstantParameterization` | Control schedule parameterization |
| `GrapeResult` | Full optimization result |
| `evaluate_control_with_simulator(...)` | Validate result with full simulator |

## Usage Guidance

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

# Run GRAPE
result = solve_grape(problem, GrapeConfig(n_iter=500, learning_rate=0.01))
print(result.final_fidelity)
```

For an interactive walkthrough with pulse export, see:
`tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`

For a benchmarking harness covering larger optimization cases, see:
`benchmarks/run_optimal_control_benchmarks.py`

## Important Assumptions / Conventions

- GRAPE propagators are computed using the piecewise-constant (PWC) approximation: the Hamiltonian is held constant within each time step.
- Gradients are computed analytically using the co-state / adjoint method; no finite differences are used.
- The Hamiltonian convention and frame convention match `cqed_sim.sim` exactly. Optimized pulses export directly into the `Pulse`/`SequenceCompiler`/`simulate_sequence` pipeline.
- Amplitude units are `rad/s`; time units are seconds.
- The unitary fidelity metric is the normalized Hilbert–Schmidt inner product over the full Hilbert space. For leakage-sensitive targets, use `LeakagePenalty` or define a subspace-projected objective.

## Relationships to Other Modules

- **`cqed_sim.sim`**: GRAPE uses the same Hamiltonian assembly and shares propagator logic; `evaluate_control_with_simulator(...)` validates results via the full QuTiP solver.
- **`cqed_sim.unitary_synthesis`**: synthesis can produce a pulse sequence that is then refined at the waveform level by GRAPE; `objective_from_unitary_synthesis_target(...)` bridges the two.
- **`cqed_sim.core`**: provides the model and frame that the control problem is built from.

## Limitations / Non-Goals

- GRAPE optimization is local; it converges to a local optimum. Robustness to local minima depends on the initial guess and the number of restarts.
- The current implementation uses a NumPy-based propagator path. It does not exploit JAX JIT or GPU acceleration on the gradient computation.
- Ensemble robustness optimization (averaging over parameter uncertainty) is supported through `ModelEnsembleMember` but is not the default mode.
- GRAPE does not enforce hardware constraints such as AWG sample rate or DAC range beyond the `SlewRatePenalty` and `AmplitudePenalty` soft penalties.

## References

- GRAPE method: Khaneja et al., "Optimal control of coupled spin dynamics: design of NMR pulse sequences by gradient ascent algorithms," J. Magn. Reson. 172 (2005).
- Tutorial notebook: `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- Benchmark harness: `benchmarks/run_optimal_control_benchmarks.py`
