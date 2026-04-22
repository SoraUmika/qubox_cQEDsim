# Getting Started

This page introduces the core mental model for working with `cqed_sim`.

!!! tip "First time here?"
    If you just want to run a simulation as fast as possible, jump to the [Quickstart](quickstart.md) page.
    Come back here for a deeper understanding of how the library is organized.

!!! tip "Not sure where to start?"
    Use [Choose by Goal](choose_by_goal.md) to pick a default model, tutorial, example, or contributor path by task.

---

## What Is `cqed_sim`?

`cqed_sim` is a pulse-level circuit-QED simulator built on [QuTiP](https://qutip.org/). It provides composable building blocks for:

- **System modeling** — transmon–cavity models with dispersive coupling, Kerr, anharmonicity
- **Rotating frames** — explicit frame management for slow-oscillation simulation
- **Pulse construction** — Gaussian, DRAG, square, and custom envelope pulses
- **Sequence compilation** — discretize pulse schedules onto a time grid
- **Time-domain simulation** — QuTiP ODE solver or dense piecewise-constant backends
- **State preparation and measurement** — tensor-product initial states, synthetic readout
- **Calibration and tomography** — SQR calibration, Rabi/Ramsey/T₁/T₂ targets, Fock tomography
- **Optimal control** — GRAPE with hardware-aware signal chain
- **RL control** — Gym-compatible environment for reinforcement learning experiments

---

## Core Simulation Pipeline

Every simulation in `cqed_sim` follows the same six-step pattern:

```
Model → Frame → State Prep → Pulses → Compile → Simulate → Extract / Measure
```

### 1. Model

Define the physical system. `cqed_sim` provides three model classes of increasing generality:

| Model | Modes | Use case |
|---|---|---|
| `DispersiveTransmonCavityModel` | qubit + storage | Default for most simulations |
| `DispersiveReadoutTransmonStorageModel` | qubit + storage + readout | Explicit readout resonator |
| `UniversalCQEDModel` | N modes | Arbitrary multi-mode Hamiltonians |

See [Defining Models](user_guides/defining_models.md) for details.

### 2. Frame

`FrameSpec` defines the rotating frame. Setting frame frequencies equal to bare mode frequencies removes fast oscillations:

```python
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
```

See [Rotating Frames](user_guides/frames.md).

### 3. State Preparation

Build a model-consistent tensor-product initial state:

```python
from cqed_sim.core import StatePreparationSpec, prepare_state, qubit_state, fock_state

initial = prepare_state(model, StatePreparationSpec(
    qubit=qubit_state("g"), storage=fock_state(0),
))
```

See [State Prep & Measurement](user_guides/state_prep_measurement.md).

### 4. Pulses and Compilation

Create `Pulse` objects and compile them onto a uniform time grid:

```python
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler

pulse = Pulse("q", 0.0, 80e-9, gaussian_envelope, amp=np.pi/2)
compiled = SequenceCompiler(dt=2e-9).compile([pulse], t_end=100e-9)
```

See [Pulse Construction](user_guides/pulse_construction.md) and [Sequence Compilation](user_guides/sequence_compilation.md).

### 5. Simulate

Run the time-domain solver:

```python
from cqed_sim.sim import SimulationConfig, simulate_sequence

result = simulate_sequence(
    model, compiled, initial,
    drive_ops={"q": "qubit"},
    config=SimulationConfig(frame=frame),
)
```

See [Running Simulations](user_guides/running_simulations.md).

### 6. Extract and Measure

Read out quantum state information or perform synthetic measurement:

```python
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

meas = measure_qubit(result.final_state, QubitMeasurementSpec(shots=2048, seed=42))
print(meas.counts)        # {"g": 24, "e": 2024}
print(meas.probabilities) # {"g": 0.012, "e": 0.988}
```

See [Extracting Observables](user_guides/extracting_observables.md) and [State Prep & Measurement](user_guides/state_prep_measurement.md).

---

## Learning Paths

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    Run your first simulation in 5 minutes.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-school:{ .lg .middle } **Tutorial Notebooks**

    ---

    Structured Jupyter curriculum from basics to GRAPE and RL.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-book-open-variant:{ .lg .middle } **User Guides**

    ---

    Step-by-step documentation for every part of the pipeline.

    [:octicons-arrow-right-24: Defining Models](user_guides/defining_models.md)

-   :material-atom:{ .lg .middle } **Physics & Conventions**

    ---

    Hamiltonian definitions, carrier sign, dispersive shift, units.

    [:octicons-arrow-right-24: Conventions](physics_conventions.md)

</div>
