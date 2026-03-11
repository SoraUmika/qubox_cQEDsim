# `cqed_sim`

`cqed_sim` is the reusable cQED simulator library in this repository. It is intended to cover both clean Hamiltonian-level simulation and experiment-style protocol simulation for:

- qubit/transmon + storage-cavity systems,
- storage + transmon + readout-resonator systems,
- pulse-level schedules compiled onto explicit drive channels,
- deterministic open-system evolution,
- lightweight state-preparation and measurement wrappers.

Study code, audits, paper reproductions, and workflow-specific helpers are intentionally outside the core package under `examples/`.

## Package layout

Core library:

- `cqed_sim/core`
  - Hilbert-space conventions, frames, two-mode and three-mode dispersive models, ideal gates, manifold-frequency helpers.
- `cqed_sim/pulses`
  - `Pulse`, standard envelopes, standard pulse builders, calibration formulas, hardware distortion models.
- `cqed_sim/sequence`
  - `SequenceCompiler` and compiled-channel timeline assembly.
- `cqed_sim/sim`
  - Hamiltonian assembly, solver entry points, noise model, extractors, readout-conditioned response helpers.
- `cqed_sim/experiment`
  - Reusable state preparation, qubit measurement, and lightweight protocol wrappers built on the core simulator path.
- `cqed_sim/calibration`, `cqed_sim/observables`, `cqed_sim/operators`, `cqed_sim/tomo`, `cqed_sim/io`, `cqed_sim/plotting`, `cqed_sim/unitary_synthesis`
  - Reusable calibration, diagnostics, tomography, gate I/O, plotting, and synthesis helpers that remain part of the library surface.

Example-side code:

- `examples/workflows`
  - Sequential workflow helpers and SQR transfer artifacts.
- `examples/audits`
  - Convention-audit utilities.
- `examples/studies`
  - SNAP and related study code.
- `examples/paper_reproductions`
  - Paper-specific reproduction code.
- `examples/smoke_tests`
  - Smoke and integration checks for the moved example paths.

## Core conventions

- Two-mode tensor ordering is qubit first, storage second: `|q,n> = |q> tensor |n>`.
- Three-mode tensor ordering is qubit first, storage second, readout third: `|q,n_s,n_r>`.
- Computational basis is `|g> = |0>`, `|e> = |1>`.
- Internal Hamiltonian and frame frequencies are in `rad/s`; times are in `s`.
- Complex drive envelopes use `exp(+i * (omega * t + phase))`.
- Runtime dispersive terms use the excitation projector `n_q = b^\dagger b`; for a two-level qubit, `n_q = |e><e| = (I - sigma_z) / 2`.
- Runtime `chi` means the per-photon downward pull of the `|g,n> <-> |e,n>` transition frequency.
- Measurement helpers return exact probabilities by default and sampled outcomes only when `shots` is requested.

The canonical source of truth for physics conventions, caveats, and verified test coverage is:

- `physics_and_conventions/physics_conventions_report.tex`

## Common API

### Build a two-mode model and frame

```python
import numpy as np

from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec

model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.0e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=2.0 * np.pi * (-200.0e6),
    chi=2.0 * np.pi * (-2.84e6),
    kerr=2.0 * np.pi * (-2.0e3),
    n_cav=8,
    n_tr=2,
)

frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)
```

Useful entry points:

- `DispersiveTransmonCavityModel.static_hamiltonian(frame=...)`
- `DispersiveTransmonCavityModel.basis_state(q_level, cavity_level)`
- `manifold_transition_frequency(model, n, frame=...)`

### Build a three-mode storage-transmon-readout model

```python
import numpy as np

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec

model = DispersiveReadoutTransmonStorageModel(
    omega_s=2.0 * np.pi * 5.0e9,
    omega_r=2.0 * np.pi * 7.5e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=2.0 * np.pi * (-220.0e6),
    chi_s=2.0 * np.pi * 2.8e6,
    chi_r=2.0 * np.pi * 1.2e6,
    chi_sr=2.0 * np.pi * 15.0e3,
    kerr_s=2.0 * np.pi * (-2.0e3),
    kerr_r=2.0 * np.pi * (-30.0e3),
    n_storage=10,
    n_readout=12,
    n_tr=2,
)

frame = FrameSpec(
    omega_c_frame=model.omega_s,
    omega_q_frame=model.omega_q,
    omega_r_frame=model.omega_r,
)
```

Useful entry points:

- `DispersiveReadoutTransmonStorageModel.static_hamiltonian(frame=...)`
- `DispersiveReadoutTransmonStorageModel.basis_state(q_level, storage_level, readout_level)`
- `DispersiveReadoutTransmonStorageModel.qubit_transition_frequency(...)`
- `DispersiveReadoutTransmonStorageModel.storage_transition_frequency(...)`
- `DispersiveReadoutTransmonStorageModel.readout_transition_frequency(...)`

### Build pulses

```python
import numpy as np

from cqed_sim.io import RotationGate
from cqed_sim.pulses import build_rotation_pulse

gate = RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0)
pulses, drive_ops, meta = build_rotation_pulse(
    gate,
    {
        "duration_rotation_s": 100.0e-9,
        "rotation_sigma_fraction": 0.18,
    },
)
```

Common builders:

- `build_rotation_pulse(...)`
- `build_displacement_pulse(...)`
- `build_sqr_multitone_pulse(...)`

If you need a custom drive family, construct `Pulse(...)` directly and map its channel to a target such as `"qubit"`, `"storage"`, `"cavity"`, `"readout"`, or `"sideband"`.

### Compile and run low-level simulations

```python
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

compiled = SequenceCompiler(dt=2.0e-9).compile(
    pulses,
    t_end=max(pulse.t1 for pulse in pulses) + 2.0e-9,
)

result = simulate_sequence(
    model,
    compiled,
    model.basis_state(0, 0),
    drive_ops,
    config=SimulationConfig(frame=frame, max_step=2.0e-9),
)
```

Useful runtime entry points:

- `simulate_sequence(...)`
- `hamiltonian_time_slices(...)`
- `SimulationConfig(...)`
- `NoiseSpec(...)`

### State preparation and measurement

```python
from cqed_sim.experiment import (
    QubitMeasurementSpec,
    StatePreparationSpec,
    fock_state,
    measure_qubit,
    prepare_state,
    qubit_state,
)

initial = prepare_state(
    model,
    StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
)

measurement = measure_qubit(
    initial,
    QubitMeasurementSpec(shots=1024, seed=123),
)
```

Reusable preparation helpers:

- `prepare_state(...)`
- `prepare_ground_state(...)`
- `qubit_state(...)`, `qubit_level(...)`
- `fock_state(...)`, `coherent_state(...)`
- `amplitude_state(...)`, `density_matrix_state(...)`

Measurement helpers:

- `measure_qubit(...)`
- `QubitMeasurementSpec(...)`

`measure_qubit(...)` is intentionally lightweight. It computes exact qubit probabilities from the runtime state, optionally applies a confusion matrix via `p_observed = M @ p_latent` with `(g, e)` ordering, and optionally samples repeated-shot outcomes and synthetic Gaussian I/Q points.

## Experimental-style workflows

The recommended experiment-style path is:

1. Prepare an initial state with `StatePreparationSpec` and `prepare_state(...)`.
2. Build pulses or gate-derived pulse segments.
3. Compile onto a global timeline with `SequenceCompiler`.
4. Simulate with `simulate_sequence(...)` or use `SimulationExperiment` as a wrapper.
5. Extract reduced states, photon numbers, conditioned responses, or tomography observables.
6. Optionally sample qubit measurement outcomes with `measure_qubit(...)`.

The lightest wrapper for this flow is `SimulationExperiment`:

```python
import numpy as np

from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.experiment import (
    QubitMeasurementSpec,
    SimulationExperiment,
    StatePreparationSpec,
    fock_state,
    qubit_state,
)
from cqed_sim.pulses import Pulse


def square(t_rel):
    return np.ones_like(t_rel, dtype=np.complex128)


model = DispersiveTransmonCavityModel(
    omega_c=0.0,
    omega_q=0.0,
    alpha=0.0,
    chi=0.0,
    kerr=0.0,
    n_cav=3,
    n_tr=2,
)

experiment = SimulationExperiment(
    model=model,
    pulses=[Pulse("q", 0.0, 1.0, square, amp=np.pi / 4.0)],
    drive_ops={"q": "qubit"},
    dt=0.01,
    t_end=1.1,
    frame=FrameSpec(),
    state_prep=StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
    measurement=QubitMeasurementSpec(shots=2048, seed=7),
)

experiment_result = experiment.run()
```

`SimulationExperiment` is a reusable wrapper, not a separate solver. It still compiles with `SequenceCompiler` and simulates through `simulate_sequence(...)`.

## Performance-oriented usage

For high-throughput workloads, the recommended fast path is:

1. Compile once with `SequenceCompiler`.
2. Prepare the runtime once with `prepare_simulation(...)`.
3. Reuse the returned `SimulationSession` across many initial states or parameter loops.
4. Pass `e_ops={}` when you only need the final state and want to avoid expectation-value bookkeeping.

Example:

```python
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, prepare_simulation, simulate_batch

compiled = SequenceCompiler(dt=0.01).compile(pulses, t_end=1.1)
session = prepare_simulation(
    model,
    compiled,
    drive_ops,
    config=SimulationConfig(frame=frame),
    e_ops={},  # Fast path when only final states matter.
)

single = session.run(initial_state)
many = simulate_batch(session, [initial_state_a, initial_state_b], max_workers=1)
```

Practical guidance:

- Use `simulate_sequence(...)` for one-off runs.
- Use `prepare_simulation(...)` for repeated runs with the same model, compiled schedule, and drive mapping.
- Use `simulate_batch(...)` or `SimulationSession.run_many(...)` for sweep-style execution over many initial states.
- Keep `store_states=False` unless you actually need the full trajectory.
- Prefer serial prepared sessions for inner loops; on this Windows environment, multiprocessing with `spawn` only pays off for much coarser jobs than the small and medium benchmarks in `benchmarks/performance_audit.md`.

Performance artifacts:

- `benchmarks/run_performance_benchmarks.py`
- `benchmarks/performance_audit.md`

GPU note:

- No GPU backend is currently provided for the QuTiP solver path.
- The current architecture benefits more from cache/session reuse and coarse CPU parallelism than from GPU-oriented array acceleration.

## Extract common observables

Two-mode extractors:

```python
from cqed_sim.sim import conditioned_bloch_xyz, reduced_cavity_state, reduced_qubit_state

rho_q = reduced_qubit_state(result.final_state)
rho_c = reduced_cavity_state(result.final_state)
bloch_n0 = conditioned_bloch_xyz(result.final_state, n=0)
```

Three-mode and readout-facing helpers:

```python
from cqed_sim.sim import (
    readout_moments,
    readout_response_by_qubit_state,
    reduced_readout_state,
    reduced_storage_state,
)

rho_storage = reduced_storage_state(result.final_state)
rho_readout = reduced_readout_state(result.final_state)
moments = readout_moments(result.final_state)
conditioned = readout_response_by_qubit_state(result.final_state)
```

Also see:

- `mode_moments(...)`
- `storage_moments(...)`
- `storage_photon_number(...)`
- `readout_photon_number(...)`
- `joint_expectation(...)`
- `cqed_sim.observables.*` for phase, trajectory, and weakness diagnostics

## Ideal gates, calibration, tomography, and synthesis

Standard ideal gates:

- `qubit_rotation_xy(...)`
- `qubit_rotation_axis(...)`
- `displacement_op(...)`
- `snap_op(...)`
- `sqr_op(...)`

Reusable calibration entry points:

- `calibrate_sqr_gate(...)`
- `load_or_calibrate_sqr_gate(...)`
- `calibrate_guarded_sqr_target(...)`
- `benchmark_random_sqr_targets_vs_duration(...)`

Reusable tomography entry points:

- `run_all_xy(...)`
- `autocalibrate_all_xy(...)`
- `selective_pi_pulse(...)`
- `run_fock_resolved_tomo(...)`
- `calibrate_leakage_matrix(...)`

Unitary-synthesis entry points:

- `Subspace.qubit_cavity_block(...)`
- `make_target(...)`
- `UnitarySynthesizer(...).fit(...)`

Important caveat: `cqed_sim/unitary_synthesis` still uses a different drift-phase abstraction from the runtime Hamiltonian. In particular, its `sigma_z` sign and `chi` normalization are not identical to the runtime path. Use the conventions report before translating parameters between the runtime and synthesis layers.

## What stays in `examples/`

The core library intentionally does not contain:

- paper reproductions,
- one-off optimization studies,
- experiment-specific reconciliation scripts,
- legacy workflow glue,
- smoke/integration scripts for moved example paths.

Use `examples/` for those:

- `examples/workflows/sequential`
- `examples/workflows/sqr_transfer.py`
- `examples/audits/experiment_convention_audit.py`
- `examples/studies/*`
- `examples/paper_reproductions/*`

These are example or study APIs, not part of the canonical `cqed_sim` library surface.

## Tests

Core reusable-library suite:

```bash
pytest tests -q
```

Example-side verification:

```bash
pytest examples/audits/tests \
  examples/workflows/tests \
  examples/studies/tests \
  examples/paper_reproductions/tests \
  examples/smoke_tests/tests -q
```

The core suite is the canonical validation path for the reusable library. The example-side suites validate the moved workflow, study, audit, and reproduction code separately.
