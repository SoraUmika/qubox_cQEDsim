# API Reference: Optimal Control (`cqed_sim.optimal_control`)

The `cqed_sim.optimal_control` package provides the direct-control layer of the library. It supports two public optimization styles on top of the same `ControlProblem` abstraction:

- GRAPE on a piecewise-constant propagation grid,
- structured, hardware-aware parameter-space optimization over smooth pulse families.

It also exposes first-class extension points for user-defined waveform maps and pulse families, plus high-level workflow helpers for outer-loop gate-time search and structured-to-GRAPE refinement.

!!! note "Current scope"
    The structured backend remains closed-system. GRAPE also supports a NumPy density-matrix / Lindblad path when `ControlSystem` members include `collapse_operators` or when the problem includes `DensityMatrixTransferObjective(...)`. The JAX engine remains closed-system only.

---

## Core Problem Objects

```python
from cqed_sim.optimal_control import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
    ModelControlChannelSpec,
    ModelEnsembleMember,
)
```

### `ControlTerm`

Represents one control Hamiltonian term.

Fields:

- `name`
- `operator`
- `amplitude_bounds`
- `export_channel`
- `drive_target`
- `quadrature`

`quadrature="I"` and `quadrature="Q"` follow the runtime drive convention used by `cqed_sim.sim.runner`:

$$
c(t) = I(t) + i Q(t).
$$

For model-backed control problems, `build_control_problem_from_model(...)` derives the Hermitian `Q`
operator as `+i(raising - lowering)`, so this exported envelope replays through the runtime pulse
stack with the same effective control Hamiltonian used during optimization.
Absolute positive drive frequencies remain a separate boundary translation handled through the
`cqed_sim.core` frequency helpers before any raw `Pulse.carrier` is assigned.

### `ControlSystem`

Represents one member of the control problem:

- `drift_hamiltonian`
- `control_operators`
- `collapse_operators`
- `weight`
- `label`

Attach `collapse_operators` to run GRAPE on the Lindblad superoperator path.

Multiple `ControlSystem` objects can be attached to the same `ControlProblem` for ensemble or worst-case optimization.

### `ControlProblem`

The solver-facing problem container:

- `parameterization`
- `systems`
- `objectives`
- `penalties`
- `ensemble_aggregate`
- `hardware_model`

Supported aggregation modes:

- `"mean"`
- `"worst"`

---

## General Parameterizations

```python
from cqed_sim.optimal_control import (
    CallableParameterization,
    ControlParameterSpec,
    FourierParameterization,
    HeldSampleParameterization,
    LinearInterpolatedParameterization,
    PiecewiseConstantParameterization,
)
```

### `ControlParameterSpec`

Describes one named optimizer variable for a callable waveform parameterization:

- `name`
- `lower_bound`
- `upper_bound`
- `default`
- `description`
- `units`

### `CallableParameterization`

Exposes an arbitrary parameter vector-to-waveform map without requiring a new parameterization subclass.

Required callable:

- `evaluator(values, time_grid, control_terms) -> waveform`

Optional callables:

- `pullback_evaluator(gradient_command, values, time_grid, control_terms, waveform) -> reduced_gradient`
- `metrics_evaluator(values, time_grid, control_terms, waveform) -> dict`

This is the direct extension point for ansatzes such as amplitude/phase maps, compressed Fourier or spline schedules, or other analytic waveform families that are easier to describe as a pure function than as a custom class.

`CallableParameterization` still resolves onto the standard command-waveform shape `(n_controls, n_time_slices)`, so the same objectives, penalties, hardware maps, pulse export, and replay path continue to work.

### Built-in rectangular and basis parameterizations

- `PiecewiseConstantParameterization`
- `HeldSampleParameterization`
- `FourierParameterization`
- `LinearInterpolatedParameterization`

These remain the standard choices when the optimizer variables are still fundamentally waveform samples or fixed linear basis coefficients.

Gate-duration changes are intentionally handled at the workflow level through `optimize_gate_time_with_grape(...)` or `optimize_gate_time_with_structured_control(...)`, rather than by assuming a differentiable free-final-time inner solver.

---

## Structured Pulse Families

```python
from cqed_sim.optimal_control import (
    CallablePulseFamily,
    PulseParameterSpec,
    StructuredPulseFamily,
    GaussianDragPulseFamily,
    FourierSeriesPulseFamily,
)
```

### `PulseParameterSpec`

Describes one named pulse parameter:

- `name`
- `lower_bound`
- `upper_bound`
- `default`
- `description`
- `units`

### `StructuredPulseFamily`

Abstract base class for smooth pulse families used by the structured backend.

Required behavior:

- expose named `parameter_specs`
- evaluate a complex envelope on a time grid
- return a Jacobian with respect to the pulse parameters

The built-in families are:

### `CallablePulseFamily`

Lets users define a smooth complex envelope directly from callables instead of subclassing `StructuredPulseFamily`.

Required callable:

- `evaluator(time_rel_s, duration_s, values) -> complex_envelope`

Optional callable:

- `jacobian_evaluator(time_rel_s, duration_s, values) -> jacobian`

When `jacobian_evaluator` is omitted, the family falls back to a finite-difference Jacobian through the existing `StructuredPulseFamily.waveform_and_jacobian(...)` path.

### `GaussianDragPulseFamily`

Uses a Gaussian envelope with a derivative quadrature correction:

$$
u(t; \theta) = A \left[g(t; \sigma, c) + i\,\alpha\,\frac{dg}{d\tau}\right] e^{i\phi}
$$

Optimized parameters:

- `amplitude`
- `sigma_fraction`
- `center_fraction`
- `phase_rad`
- `drag_alpha`

This is useful for hardware-realistic single-lobe qubit drives and DRAG-style correction studies.

### `FourierSeriesPulseFamily`

Represents a smooth complex envelope in a truncated real Fourier basis:

$$
u(t; \theta) = I(t; \theta_I) + i Q(t; \theta_Q)
$$

with

$$
I(t) = \sum_k a_k \cos\left(\frac{2\pi k t}{T}\right) + \sum_{k>0} b_k \sin\left(\frac{2\pi k t}{T}\right)
$$

and the same structure for `Q(t)`.

This is useful for band-limited basis optimization and later black-box search over a smaller parameter space.

---

## Structured Parameterization

```python
from cqed_sim.optimal_control import (
    StructuredControlChannel,
    StructuredPulseParameterization,
    build_structured_control_problem_from_model,
)
```

### `StructuredControlChannel`

Binds one `StructuredPulseFamily` to repository control terms through either:

- `export_channel`, or
- explicit `control_names`.

The same structured channel can drive:

- one `SCALAR` control term,
- one `I`-only or `Q`-only control term,
- one I/Q pair sharing an export channel.

### `StructuredPulseParameterization`

Maps a flat structured parameter vector onto the propagation-grid command waveform used by the solver.

Key properties and methods:

- `parameter_specs`
- `parameter_names()`
- `parameter_records(values)`
- `command_values(values)`
- `pullback(gradient_command, values)`
- `to_pulses(values, waveform_values=...)`

Unlike `PiecewiseConstantParameterization`, the structured parameterization does not assume one optimizer variable per control per time slice.

### `build_structured_control_problem_from_model(...)`

Convenience builder that:

1. reuses `ModelControlChannelSpec` and the existing model-level drive operators,
2. builds repository `ControlTerm` objects,
3. attaches a `StructuredPulseParameterization`,
4. returns a standard `ControlProblem`.

This keeps the structured workflow aligned with the same simulator-facing abstractions as GRAPE.

---

## Hardware and Transfer Pipeline

```python
from cqed_sim.optimal_control import (
    HardwareModel,
    FirstOrderLowPassHardwareMap,
    GainHardwareMap,
    DelayHardwareMap,
    BoundaryWindowHardwareMap,
    SmoothIQRadiusLimitHardwareMap,
    resolve_control_schedule,
)
```

The control pipeline is explicit:

$$
\theta \rightarrow u_{\mathrm{cmd}}(t; \theta) \rightarrow u_{\mathrm{phys}}(t) = \mathcal{H}[u_{\mathrm{cmd}}] \rightarrow H(t; u_{\mathrm{phys}}).
$$

`resolve_control_schedule(...)` returns:

- parameter values,
- command waveform,
- physical waveform,
- parameterization diagnostics,
- hardware reports and metrics.

Useful maps for structured studies include:

- `FirstOrderLowPassHardwareMap(...)`
- `GainHardwareMap(...)`
- `DelayHardwareMap(...)`
- `BoundaryWindowHardwareMap(...)`
- `SmoothIQRadiusLimitHardwareMap(...)`
- `FIRHardwareMap(...)`
- `FrequencyResponseHardwareMap(...)`

The identity transfer case is represented by `HardwareModel(maps=())` or `hardware_model=None` on the `ControlProblem`.

---

## Objectives

```python
from cqed_sim.optimal_control import (
    CustomControlObjective,
    CustomObjectiveContext,
    CustomObjectiveEvaluation,
    DensityMatrixTransferObjective,
    DensityMatrixTransferPair,
    StateTransferObjective,
    StateTransferPair,
    UnitaryObjective,
    multi_state_transfer_objective,
    objective_from_unitary_synthesis_target,
    state_preparation_objective,
)
```

### Built-in objectives

- `state_preparation_objective(initial, target)`
- `multi_state_transfer_objective(...)`
- `DensityMatrixTransferObjective(...)`
- `UnitaryObjective(target_operator=..., subspace=...)`

`DensityMatrixTransferObjective` accepts pure states or density matrices and evaluates a purity-normalized density overlap. For pure-state targets this reduces to standard fidelity.

### `CustomControlObjective`

Allows user-defined objectives inside the shared optimal-control evaluator.

The evaluator receives a `CustomObjectiveContext` with:

- the `ControlProblem`,
- the active `ControlSystem`,
- the current `ControlSchedule`,
- resolved command and physical waveforms,
- propagation data,
- the final unitary for that system.

It must return `CustomObjectiveEvaluation` containing:

- `cost`
- `gradient_physical`
- optional `metrics`

This lets advanced studies optimize custom control metrics while staying on the standard solver/result surface.

`CustomControlObjective` is currently limited to the closed-system state-vector path.

---

## Penalties

```python
from cqed_sim.optimal_control import (
    AmplitudePenalty,
    BoundPenalty,
    BoundaryConditionPenalty,
    IQRadiusPenalty,
    LeakagePenalty,
    SlewRatePenalty,
)
```

Supported penalties:

- `AmplitudePenalty(weight=..., reference=...)`
- `SlewRatePenalty(weight=...)`
- `BoundPenalty(weight=..., lower_bound=..., upper_bound=...)`
- `BoundaryConditionPenalty(weight=..., ramp_slices=...)`
- `IQRadiusPenalty(amplitude_max=..., weight=...)`
- `LeakagePenalty(subspace=..., weight=..., metric="average" | "worst")`

Waveform penalties can target one of three domains through `apply_to="parameter" | "command" | "physical"`.

---

## Solvers

```python
from cqed_sim.optimal_control import (
    GrapeConfig,
    GrapeSolver,
    StructuredControlConfig,
    StructuredControlSolver,
    solve_grape,
    solve_structured_control,
)
```

### `GrapeSolver`

The slice-level backend. Best suited for waveform-level refinement when you truly want one optimization variable per slice or one of the older command parameterizations such as `HeldSampleParameterization`.

Key `GrapeConfig` fields:

- `optimizer_method` — SciPy method name (`"L-BFGS-B"`, etc.) or Optax method (`"adam"`, `"adagrad"`, `"sgd"`, `"adamw"`).
- `engine` — `"numpy"` (default) or `"jax"`.
- `optax_learning_rate` — step size for Optax optimizers (default `1e-3`). Only used when `engine="jax"` and `optimizer_method` is an Optax name.
- `optax_grad_clip` — optional global L2 gradient clip for Optax optimizers.

Engine behavior summary:

- `engine="numpy"` supports both closed-system and density-matrix / Lindblad GRAPE.
- `engine="jax"` currently supports only closed-system objectives.

Optax methods are activated when `engine="jax"` **and** `optimizer_method` is one of `"adam"`, `"adagrad"`, `"sgd"`, or `"adamw"`. Requires the `optax` optional dependency group.

### `StructuredControlSolver`

The smooth parameter-space backend. It optimizes the parameter vector exposed by `StructuredPulseParameterization` instead of raw sample amplitudes.

Key configuration fields:

- `optimizer_method`
- `maxiter`
- `initial_guess="defaults" | "random" | explicit array`
- `use_gradients`
- `apply_hardware_in_forward_model`
- `report_command_reference`
- `engine`

`use_gradients=False` enables derivative-free SciPy methods on the same public problem surface.

---

## Workflow Extensions

```python
from cqed_sim.optimal_control import (
    GateTimeCandidate,
    GateTimeOptimizationConfig,
    GateTimeOptimizationResult,
    StructuredToGrapeResult,
    build_grape_refinement_problem,
    optimize_gate_time_with_grape,
    optimize_gate_time_with_structured_control,
    solve_structured_then_grape,
)
```

### Gate-time optimization

`optimize_gate_time_with_grape(...)` and `optimize_gate_time_with_structured_control(...)` perform an explicit outer-loop duration sweep:

1. scale the problem time grid to each candidate duration,
2. clone the parameterization on that scaled grid,
3. solve each candidate problem,
4. return a `GateTimeOptimizationResult` with all candidates and the best result.

`GateTimeOptimizationConfig(max_workers=N)` runs independent duration candidates in parallel on CPU threads. For JAX-backed inner solves, combine this with `GrapeConfig(engine="jax", jax_device="gpu")` to keep the propagation and gradient evaluation on GPU while the outer duration search is orchestrated from Python.

### Structured-to-GRAPE refinement

`build_grape_refinement_problem(...)` converts an existing control problem onto a piecewise-constant GRAPE parameterization while preserving the same systems, objectives, penalties, hardware model, and metadata.

`solve_structured_then_grape(...)` then:

1. solves the structured problem,
2. lifts the structured command waveform into the piecewise GRAPE parameterization,
3. runs GRAPE from that warm start,
4. returns a `StructuredToGrapeResult` containing both stages plus before/after metrics.

---

## Results and Artifact Export

```python
from cqed_sim.optimal_control import (
    ControlResult,
    GrapeResult,
    save_structured_control_artifacts,
)
```

`ControlResult` remains the shared result type. Structured solves return a `ControlResult` with:

- `backend="structured-control"`
- optimized parameter values in `schedule.values`
- resolved command and physical waveforms
- optimizer summary
- hardware reports
- parameterization diagnostics

### `save_structured_control_artifacts(...)`

Persists a study-ready artifact bundle containing:

- `result.json`
- `parameters.csv`
- `waveforms.csv`
- `history.csv`
- `waveforms.png`
- `spectrum.png`
- `optimization_history.png`

This is intended for end-to-end structured-control studies and for future closed-loop comparison against hardware runs.

---

## Example

```python
import numpy as np
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FirstOrderLowPassHardwareMap,
    FrameSpec,
    GaussianDragPulseFamily,
    GainHardwareMap,
    HardwareModel,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    StructuredControlChannel,
    StructuredControlConfig,
    build_structured_control_problem_from_model,
    solve_structured_control,
    state_preparation_objective,
)

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9,
    omega_q=2*np.pi*6e9,
    alpha=0.0,
    chi=0.0,
    kerr=0.0,
    n_cav=1,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

problem = build_structured_control_problem_from_model(
    model,
    frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=32, dt_s=4e-9),
    channel_specs=(
        ModelControlChannelSpec(
            name="qubit",
            target="qubit",
            quadratures=("I", "Q"),
            amplitude_bounds=(-8e7, 8e7),
            export_channel="qubit",
        ),
    ),
    structured_channels=(
        StructuredControlChannel(
            name="gaussian_drag",
            pulse_family=GaussianDragPulseFamily(default_phase=-0.5*np.pi),
            export_channel="qubit",
        ),
    ),
    objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
    hardware_model=HardwareModel(maps=(
        GainHardwareMap(gain=0.93, export_channels=("qubit",)),
        FirstOrderLowPassHardwareMap(cutoff_hz=28e6, export_channels=("qubit",)),
    )),
)

result = solve_structured_control(
    problem,
    config=StructuredControlConfig(maxiter=60, seed=7, initial_guess="random"),
)
```

See `examples/structured_optimal_control_demo.py` for a full study that generates artifact bundles for both Gaussian and Fourier pulse families.