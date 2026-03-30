# `cqed_sim.sim`

The `sim` module is the simulation runtime for `cqed_sim`. It assembles the time-dependent Hamiltonian from a model and a compiled pulse schedule, drives the QuTiP solver, and provides a collection of state extractors for computing physical observables from the resulting states.

## Relevance in `cqed_sim`

`sim` is the central execution engine. The standard library workflow is:

1. Define a model in `cqed_sim.core`.
2. Build pulses in `cqed_sim.pulses`.
3. Compile a timeline in `cqed_sim.sequence`.
4. Call `simulate_sequence(...)` or `prepare_simulation(...)` in `cqed_sim.sim`.
5. Extract observables from the result using the extractor functions in this module.

## Main Capabilities

### Simulation entry points

- **`simulate_sequence(model, compiled, psi0, drive_ops, config)`**: One-shot simulation. Runs the QuTiP solver for the given model, compiled channel schedule, initial state, and drive operator mapping. Returns a `SimulationResult`.
- **`prepare_simulation(model, compiled, drive_ops, config, e_ops)`**: Builds a reusable `SimulationSession` that can be called repeatedly with different initial states without re-assembling the Hamiltonian. Use this for parameter sweeps.
- **`SimulationSession.run(psi0)`**: Runs a single initial state against the prepared session.
- **`simulate_batch(session, states, max_workers)`**: Runs a batch of initial states through a *single* prepared session (same Hamiltonian, many initial states). Supports optional parallel execution.
- **`run_sweep(sessions, initial_states, max_workers)`**: Runs a list of (session, initial_state) pairs where *each pair has a different session* (e.g., different model parameters). Use this for parameter sweeps over detuning, chi, or any other quantity that changes the Hamiltonian. Supports optional parallel execution.

### Configuration

- **`SimulationConfig`**: Controls frame (`FrameSpec`), solver tolerances, time step, backend choice, noise model, and whether to store full state trajectories.
- **`NoiseSpec`**: Specifies collapse operators for open-system evolution. Supports `T1`, `T2`, and multilevel transmon decay via `transmon_t1=(T1_ge, T1_fe, ...)`.
- **`split_collapse_operators(...)`**: Splits noise channels into unmonitored Lindblad terms and a monitored bosonic-emission path for stochastic readout replay.
- **`hamiltonian_time_slices(...)`**: Returns the time-sliced Hamiltonian matrices for inspection without running the solver.

### Optional coupling terms

- **`cross_kerr(...)`, `self_kerr(...)`, `exchange(...)`**: Construct additional Hamiltonian terms via `CrossKerrSpec`, `SelfKerrSpec`, `ExchangeSpec`.
- **`TunableCoupler`**: Models a tunable coupling element between modes.

### State extractors

Two-mode:
- `reduced_qubit_state`, `reduced_cavity_state`, `reduced_transmon_state`
- `conditioned_bloch_xyz`, `conditioned_qubit_state`, `conditioned_population`
- `bloch_xyz_from_joint`, `cavity_moments`

Three-mode and readout-facing:
- `reduced_storage_state`, `reduced_readout_state`
- `readout_moments`, `readout_response_by_qubit_state`, `readout_photon_number`
- `storage_moments`, `storage_photon_number`

Generalized:
- `reduced_subsystem_state(state, subsystem_index)` — subsystem-index-based partial trace
- `subsystem_level_population`, `transmon_level_populations`
- `compute_shelving_leakage` — leakage diagnostic for shelving-style sideband benchmarks
- `joint_expectation`, `mode_moments`, `qubit_conditioned_mode_moments`, `qubit_conditioned_subsystem_state`
- `cavity_wigner`

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `simulate_sequence(...)` | One-shot simulation |
| `prepare_simulation(...)` | Build a reusable simulation session |
| `SimulationSession` | Reusable session; call `.run(psi0)` or `.run_many(states)` |
| `simulate_batch(...)` | Batch over many initial states with the same Hamiltonian |
| `run_sweep(sessions, states, max_workers)` | Sweep over (session, state) pairs with different Hamiltonians |
| `SimulationConfig` | Solver and frame configuration |
| `NoiseSpec` | Collapse operators for open-system evolution |
| `SimulationResult` | Result object: `.final_state`, `.states`, `.expect`, `.times` |
| `reduced_qubit_state(state)` | Partial trace to qubit |
| `conditioned_bloch_xyz(state, n)` | Bloch vector conditioned on Fock sector `n` |
| `readout_response_by_qubit_state(state)` | Readout conditioned on qubit state |

## Usage Guidance

### One-shot simulation

```python
from cqed_sim.sim import SimulationConfig, simulate_sequence

result = simulate_sequence(
    model,
    compiled,
    initial_state,
    drive_ops,
    config=SimulationConfig(frame=frame, max_step=2.0e-9),
)
rho_q = reduced_qubit_state(result.final_state)
```

### Batch simulation with session reuse

```python
from cqed_sim.sim import SimulationConfig, prepare_simulation, simulate_batch

session = prepare_simulation(
    model, compiled, drive_ops,
    config=SimulationConfig(frame=frame),
    e_ops={},  # omit expectation values for speed
)
results = simulate_batch(session, [psi_a, psi_b, psi_c], max_workers=1)
```

### Parameter sweep over Hamiltonians

```python
from cqed_sim.sim import prepare_simulation, run_sweep, SimulationConfig
from cqed_sim.sequence.scheduler import SequenceCompiler

chi_values = [chi_0 + k * delta_chi for k in range(20)]
sessions = [
    prepare_simulation(
        DispersiveTransmonCavityModel(..., chi=chi),
        compiled,
        drive_ops,
        config=SimulationConfig(frame=frame),
        e_ops={},
    )
    for chi in chi_values
]
results = run_sweep(sessions, [psi0] * len(chi_values))
# results[k] is the SimulationResult for chi_values[k]
```

### Open-system simulation

```python
from cqed_sim.sim import NoiseSpec, SimulationConfig

noise = NoiseSpec(
    t1_qubit=50.0e-6,
    t2_qubit=30.0e-6,
    t1_cavity=1.0e-3,
    transmon_t1=(50.0e-6, 20.0e-6),  # T1_ge, T1_fe for multilevel decay
)
result = simulate_sequence(model, compiled, psi0, drive_ops,
                           config=SimulationConfig(frame=frame, noise=noise))
```

## Important Assumptions / Conventions

- The solver backend is QuTiP by default. Dense NumPy and JAX backends are available via `SimulationConfig(backend=NumPyBackend())` for small systems.
- The time-dependent Hamiltonian is assembled in piecewise-constant form from the compiled channel schedule.
- Frame and carrier conventions follow `cqed_sim.core`: public wrappers should translate positive physical drive frequencies through the core helpers before assigning the raw low-level `Pulse.carrier = -omega_transition(frame)` expected by the runtime.
- Extractors that take `state` accept either a `qt.Qobj` density matrix or a ket; they trace out subsystems in the canonical tensor order (qubit first, then bosonic modes).
- `store_states=False` (default) omits the full trajectory from `SimulationResult` to reduce memory; set `True` only when you need time-resolved states.

## Relationships to Other Modules

- **`cqed_sim.core`**: provides the model and `FrameSpec`.
- **`cqed_sim.sequence`**: provides the `CompiledSequence` consumed here.
- **`cqed_sim.backends`**: optional alternative dense-solver backends selectable via `SimulationConfig`.
- **`cqed_sim.measurement`**: reuses the Hamiltonian assembly and `split_collapse_operators(...)` for continuous-readout replay.
- **`cqed_sim.observables`**: higher-level observable wrappers built on top of the extractors here.
- **`cqed_sim.tomo`**, **`cqed_sim.calibration`**: call `simulate_sequence(...)` internally.

## Limitations / Non-Goals

- Does not perform gate synthesis or pulse optimization — see `cqed_sim.map_synthesis` and `cqed_sim.optimal_control`.
- The QuTiP solver path does not currently exploit GPU acceleration or batch parallelism at the ODE level; speedup comes from session reuse and coarse CPU parallelism.
- Extractors assume the standard tensor ordering from `cqed_sim.core`; they will produce incorrect results if a non-standard ordering is used.
