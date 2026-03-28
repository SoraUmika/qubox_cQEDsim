# API Reference: Optimal Control (`cqed_sim.optimal_control`)

The `cqed_sim.optimal_control` package provides the direct-control layer of the library. It supports two public optimization styles on top of the same `ControlProblem` abstraction:

- GRAPE on a piecewise-constant propagation grid,
- structured, hardware-aware parameter-space optimization over smooth pulse families.

!!! note "Current scope"
    Both backends currently use closed-system dense propagation. The structured backend changes the optimization variable from raw slice amplitudes to a named parameter vector, but it preserves the same Hamiltonian, frame, and pulse-export conventions as the rest of `cqed_sim`.

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
c(t) = I(t) - i Q(t).
$$

### `ControlSystem`

Represents one closed-system member of the control problem:

- `drift_hamiltonian`
- `control_operators`
- `weight`
- `label`

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

## Structured Pulse Families

```python
from cqed_sim.optimal_control import (
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
u(t; \theta) = I(t; \theta_I) - i Q(t; \theta_Q)
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
- `UnitaryObjective(target_operator=..., subspace=...)`

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