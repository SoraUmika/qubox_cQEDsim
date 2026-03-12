# Running Simulations

The simulation engine solves the time-dependent Schrödinger or Lindblad master equation for a compiled pulse sequence applied to a cQED model.

---

## Basic Simulation

```python
from cqed_sim.sim import SimulationConfig, simulate_sequence

result = simulate_sequence(
    model,                          # Any cqed_sim model
    compiled,                       # CompiledSequence from SequenceCompiler
    model.basis_state(0, 0),        # Initial state |g, 0⟩
    {"q": "qubit"},                 # Drive operator mapping
    config=SimulationConfig(frame=frame),
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `model` | model object | Must have `operators()`, `subsystem_dims`, `static_hamiltonian()`, `drive_coupling_operators()` |
| `compiled` | `CompiledSequence` | Timeline from `SequenceCompiler.compile()` |
| `initial_state` | `qt.Qobj` | Initial ket or density matrix |
| `drive_ops` | `dict` | Maps pulse channel names to physical targets |
| `config` | `SimulationConfig` | Solver configuration |
| `c_ops` | `Sequence[qt.Qobj]` | Additional collapse operators |
| `noise` | `NoiseSpec` | Lindblad noise specification |
| `e_ops` | `dict[str, qt.Qobj]` | Custom observables to track |

### Drive Operator Mapping

The `drive_ops` dict maps pulse channel names to physical operator targets:

- **String targets:** `"qubit"`, `"cavity"`, `"storage"`, `"readout"`, `"sideband"`
- **Structured targets:** `TransmonTransitionDriveSpec(...)`, `SidebandDriveSpec(...)`

---

## SimulationConfig

```python
from cqed_sim.sim import SimulationConfig

config = SimulationConfig(
    frame=frame,              # FrameSpec for rotating frame
    atol=1e-8,                # Absolute tolerance
    rtol=1e-7,                # Relative tolerance
    max_step=2e-9,            # Maximum solver step size (s)
    store_states=False,       # Store intermediate states
    backend=None,             # None = QuTiP, or NumPyBackend/JaxBackend
)
```

---

## SimulationResult

```python
result.final_state         # qt.Qobj — state at end of simulation
result.states              # list[qt.Qobj] | None — trajectory if store_states=True
result.expectations        # dict[str, np.ndarray] — time series per observable
result.solver_result       # Raw QuTiP solver result
```

### Default Observables

When `e_ops` is not specified, the simulator tracks:

- $P_e$ — excited-state projector
- Mode quadratures and photon numbers for each bosonic mode

---

## Session Reuse for Sweeps

For parameter sweeps or multiple initial states, prepare the session once:

```python
from cqed_sim.sim import prepare_simulation, simulate_batch

session = prepare_simulation(
    model, compiled, {"q": "qubit"},
    config=SimulationConfig(frame=frame),
)

# Run with many initial states
results = simulate_batch(
    session,
    [model.basis_state(0, n) for n in range(4)],
    max_workers=1,
)
```

`SimulationSession` pre-computes the Hamiltonian, collapse operators, and solver options.

| Method | Description |
|---|---|
| `session.run(initial_state)` | Single-trajectory simulation |
| `session.run_many(initial_states)` | Parallel execution |

---

## Solver Selection

- **No backend, ket input, no noise:** Uses QuTiP's `sesolve` (Schrödinger equation)
- **No backend, density matrix or noise:** Uses QuTiP's `mesolve` (Lindblad master equation)
- **With backend (`NumPyBackend` / `JaxBackend`):** Dense piecewise-constant solver

```python
from cqed_sim.backends import NumPyBackend

config = SimulationConfig(frame=frame, backend=NumPyBackend())
```

!!! note
    Dense backends are intended for small systems and parity checks, not production-scale simulations.

---

## Hamiltonian Inspection

Inspect the time-dependent Hamiltonian before running:

```python
from cqed_sim.sim import hamiltonian_time_slices

H_list = hamiltonian_time_slices(model, compiled, drive_ops, frame)
# H_list = [H_0, [O_1_raising, coeff_1], [O_1_lowering, conj_1], ...]
```

This returns the Hamiltonian in QuTiP's time-dependent format.
