# API Reference: Optimal Control (`cqed_sim.optimal_control`)

The `cqed_sim.optimal_control` package adds a first-class direct-control layer on top of the existing model, pulse, and simulator stack.

It is designed around a generic control problem abstraction, with GRAPE implemented as the first solver backend.

!!! note "Current scope"
    The current GRAPE backend is a closed-system dense optimizer on a piecewise-constant propagation grid. Command schedules can be either plain piecewise-constant controls or held-sample controls, and optional hardware maps can transform command waveforms into the physical waveforms used during propagation.

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

`quadrature="I"` and `quadrature="Q"` follow the repository drive convention used by `cqed_sim.sim.runner`: the exported complex baseband coefficient is

$$c(t) = I(t) - i Q(t).$$

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

Supported aggregation modes:

- `"mean"`
- `"worst"`

---

## Parameterization and Hardware Pipeline

```python
from cqed_sim.optimal_control import (
    ControlParameterization,
    ControlSchedule,
    HeldSampleParameterization,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    HardwareModel,
    FirstOrderLowPassHardwareMap,
    BoundaryWindowHardwareMap,
    SmoothIQRadiusLimitHardwareMap,
    resolve_control_schedule,
)
```

### `PiecewiseConstantTimeGrid`

```python
PiecewiseConstantTimeGrid.uniform(steps=16, dt_s=4.0e-9)
```

Key properties:

- `step_durations_s`
- `steps`
- `duration_s`
- `boundaries_s()`
- `midpoints_s()`

### `PiecewiseConstantParameterization`

Holds:

- ordered control terms
- time grid
- flatten/unflatten helpers
- hard bounds for optimizers
- pulse export back into repository `Pulse` objects

Useful methods:

- `zero_schedule()`
- `bounds()`
- `flatten(...)`
- `unflatten(...)`
- `clip(...)`
- `to_pulses(...)`

### `HeldSampleParameterization`

Stores coarse command samples with period `sample_period_s` and applies sample-and-hold onto the propagation grid.

This is the first structured parameterization for AWG-like update constraints.

### `ControlSchedule`

Concrete control values with shape `(n_controls, n_slices)`.

Useful methods:

- `flattened()`
- `clipped()`
- `command_values()`
- `to_pulses()`
- `max_abs_amplitude()`
- `rms_amplitude()`

### Hardware-aware resolution

`resolve_control_schedule(...)` exposes the full control pipeline:

- parameter-space values,
- command waveform values on the propagation grid,
- physical waveform values after the attached `HardwareModel`,
- parameterization and hardware diagnostics.

The first hardware maps are:

- `FirstOrderLowPassHardwareMap(...)`
- `BoundaryWindowHardwareMap(...)`
- `SmoothIQRadiusLimitHardwareMap(...)`

---

## Objectives

```python
from cqed_sim.optimal_control import (
    StateTransferObjective,
    StateTransferPair,
    UnitaryObjective,
    multi_state_transfer_objective,
    objective_from_unitary_synthesis_target,
    state_preparation_objective,
)
```

### `StateTransferObjective`

Weighted set of pure-state transfer pairs.

Use cases:

- state preparation
- multi-state transfer
- encoded logical-state transfer

Convenience constructors:

- `state_preparation_objective(...)`
- `multi_state_transfer_objective(...)`

### `UnitaryObjective`

Operator target matched through weighted probe-state transfer pairs.

Key fields:

- `target_operator`
- `subspace`
- `ignore_global_phase`
- `allow_diagonal_phase`
- `phase_blocks`
- `probe_states`
- `probe_strategy`

This makes the same objective usable for:

- full unitary synthesis on a truncated space
- subspace-selective logical gates
- block-phase-tolerant logical operators

### Adapting Existing Synthesis Targets

`objective_from_unitary_synthesis_target(...)` converts existing `cqed_sim.unitary_synthesis` targets into optimal-control objectives when the target is already expressible as a direct state or unitary task.

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

### Supported penalties

- `AmplitudePenalty(weight=..., reference=...)`
- `SlewRatePenalty(weight=...)`
- `BoundPenalty(weight=..., lower_bound=..., upper_bound=...)`
- `BoundaryConditionPenalty(weight=..., ramp_slices=...)`
- `IQRadiusPenalty(amplitude_max=..., weight=...)`
- `LeakagePenalty(subspace=..., weight=..., metric="average" | "worst")`

Leakage is evaluated against the propagated objective probe states and penalizes final population outside the retained subspace.

All waveform penalties can target one of three domains through `apply_to="parameter" | "command" | "physical"`.

---

## Model Builders

```python
from cqed_sim.optimal_control import (
    ModelControlChannelSpec,
    build_control_problem_from_model,
    build_control_system_from_model,
    build_control_terms_from_model,
)
```

These helpers turn a `cqed_sim` model into a direct-control problem without manually constructing dense operators.

Example:

```python
problem = build_control_problem_from_model(
    model,
    frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=8, dt_s=5.0e-9),
    channel_specs=(
        ModelControlChannelSpec(
            name="qubit",
            target="qubit",
            quadratures=("I", "Q"),
            amplitude_bounds=(-1.0e8, 1.0e8),
        ),
    ),
    objectives=(state_preparation_objective(psi_in, psi_target),),
)
```

Structured targets are supported through:

- string channel aliases such as `"qubit"`, `"storage"`, `"readout"`, `"sideband"`
- `TransmonTransitionDriveSpec(...)`
- `SidebandDriveSpec(...)`

`build_control_problem_from_model(...)` also accepts:

- `parameterization_cls=...` plus `parameterization_kwargs={...}` for structured command parameterizations such as `HeldSampleParameterization`
- `hardware_model=HardwareModel(...)` to attach a command-to-physical waveform transform directly to the problem

---

## GRAPE Solver

```python
from cqed_sim.optimal_control import (
    GrapeConfig, GrapeMultistartConfig, GrapeSolver,
    solve_grape, solve_grape_multistart,
)
```

### `GrapeConfig`

Key fields:

- `optimizer_method`
- `maxiter`
- `ftol`
- `gtol`
- `initial_guess`
- `random_scale`
- `seed`
- `history_every`
- `apply_hardware_in_forward_model`
- `report_command_reference`

The solver internally rescales physical control amplitudes into a dimensionless optimization vector using the configured control bounds. This avoids premature convergence caused by gradients that are numerically small only because the controls are expressed in `rad/s` and multiplied by very short time slices.

### `GrapeSolver`

```python
solver = GrapeSolver(GrapeConfig(maxiter=120, seed=7))
result = solver.solve(problem, initial_schedule=initial_schedule)
```

The solver returns a `GrapeResult` containing:

- optimized `ControlSchedule`
- resolved command and physical waveforms
- convergence history
- overall metrics
- per-system metrics
- nominal final unitary
- optimizer summary

### `solve_grape(...)`

Convenience wrapper around `GrapeSolver(...).solve(...)`.

### `GrapeMultistartConfig`

Configuration for a multi-start GRAPE run.

Key fields:

- `n_restarts` — number of independent random restarts (default: 4)
- `max_workers` — parallel worker processes; 1 = serial (default: 1)
- `mp_context` — multiprocessing start method; `"spawn"` required on Windows
- `return_all` — if `True`, return all restart results sorted best-first (default: `True`)

### `solve_grape_multistart(...)`

Runs GRAPE from multiple random starting points and returns results sorted by objective value (best first).

```python
from cqed_sim.optimal_control import GrapeConfig, GrapeMultistartConfig, solve_grape_multistart

results = solve_grape_multistart(
    problem,
    config=GrapeConfig(maxiter=200, seed=0),
    multistart_config=GrapeMultistartConfig(n_restarts=6, max_workers=1),
)
best = results[0]  # sorted best-first
```

**Windows note:** `spawn` process startup overhead (~4–5 s per worker) dominates for short optimizations. Only use `max_workers > 1` when each individual GRAPE run takes several seconds.

---

## Results, Replay, and Export

```python
from cqed_sim.optimal_control import (
    ControlEvaluationCase,
    ControlEvaluationResult,
    ControlResult,
    evaluate_control_with_simulator,
)

result.schedule.values
pulses, drive_ops, meta = result.to_pulses()
pulses_physical, drive_ops_physical, meta_physical = result.to_pulses(waveform="physical")
result.save("outputs/grape_result.json")
```

`ControlResult` is the common result surface for direct-control optimization runs. The current solver returns `GrapeResult`, which is the GRAPE-specific concrete result type.

`ControlResult.to_pulses()` exports the optimized command waveform as standard repository `Pulse` objects plus the corresponding `drive_ops` map. Passing `waveform="physical"` exports the post-hardware waveform instead.

### Simulator-backed replay

```python
nominal_replay = result.evaluate_with_simulator(
    problem,
    model=model,
    frame=frame,
)

noisy_replay = result.evaluate_with_simulator(
    problem,
    cases=(
        ControlEvaluationCase(
            model=model,
            frame=frame,
            noise=NoiseSpec(t1=2.0e-6, tphi=1.0e-6),
            label="noisy",
        ),
    ),
    waveform_mode="physical",
)
```

This replay path is evaluation-only. It keeps the optimizer closed-system, exports either the command or the physical waveform into runtime `Pulse` objects, replays those pulses through `simulate_sequence(...)`, and reports replay fidelities under nominal or noisy Lindblad dynamics.

For retained-subspace unitary objectives, replay also reports subspace leakage metrics.

Replay mode is selected through `waveform_mode="command" | "physical" | "problem_default"`.

The function form is:

```python
evaluate_control_with_simulator(problem, result.schedule, model=model, frame=frame)
```

### Benchmark harness

The repository benchmark script for larger GRAPE cases is:

- `benchmarks/run_optimal_control_benchmarks.py`

---

## Minimal End-to-End Example

```python
import numpy as np

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    UnitaryObjective,
    build_control_problem_from_model,
)
from cqed_sim.unitary_synthesis import Subspace


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.0e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=0.0,
    chi=0.0,
    kerr=0.0,
    n_cav=2,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
subspace = Subspace.custom(full_dim=4, indices=(0, 1), labels=("|g,0>", "|g,1>"))

problem = build_control_problem_from_model(
    model,
    frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9),
    channel_specs=(
        ModelControlChannelSpec(
            name="storage",
            target="storage",
            quadratures=("Q",),
            amplitude_bounds=(-1.0e8, 1.0e8),
        ),
    ),
    objectives=(
        UnitaryObjective(
            target_operator=rotation_y(np.pi / 2.0),
            subspace=subspace,
            ignore_global_phase=True,
        ),
    ),
)

result = GrapeSolver(GrapeConfig(maxiter=80, seed=7)).solve(
    problem,
    initial_schedule=np.array([[8.0e6]], dtype=float),
)
```

See also:

- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- `examples/grape_storage_subspace_gate_demo.py`
- `examples/hardware_constrained_grape_demo.py`
- `benchmarks/run_optimal_control_benchmarks.py`