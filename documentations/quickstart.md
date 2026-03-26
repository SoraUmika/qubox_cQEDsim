# Quickstart

This page gets you from a fresh install to a complete simulation in under 5 minutes.

---

## 1. Install

Clone the repository and install in editable mode:

```bash
pip install -e .
```

Verify:

```python
import cqed_sim
print(cqed_sim.__name__)  # "cqed_sim"
```

Dependencies (`numpy`, `scipy`, `qutip ≥ 5.0`, `matplotlib`, `pandas`) are installed automatically. See [Installation](installation.md) for optional JAX backend setup.

---

## 2. Define a system

All simulations start with a model and a rotating frame.

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec

# Two-mode dispersive model: qubit + microwave storage cavity
model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5.0e9,       # Cavity frequency (rad/s)
    omega_q=2 * np.pi * 6.0e9,       # Qubit frequency (rad/s)
    alpha=2 * np.pi * (-200e6),       # Transmon anharmonicity (rad/s)
    chi=2 * np.pi * (-2.84e6),        # Dispersive shift χ (rad/s)
    kerr=2 * np.pi * (-2e3),          # Cavity self-Kerr K (rad/s)
    n_cav=8,                          # Fock-space truncation
    n_tr=2,                           # Transmon levels (2 = qubit)
)

# Rotating frame at bare mode frequencies — removes fast oscillations
frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)
```

!!! tip "Units"
    Hamiltonian coefficients are in **rad/s**. Write `2 * np.pi * 5e9` for 5 GHz,
    not just `5e9`. Times (durations, t_end) are in **seconds**.

---

## 3. Prepare the initial state

```python
from cqed_sim.core import StatePreparationSpec, prepare_state, qubit_state, fock_state

initial = prepare_state(
    model,
    StatePreparationSpec(
        qubit=qubit_state("g"),    # qubit in ground state |g⟩
        storage=fock_state(0),     # cavity in vacuum |0⟩
    ),
)
```

Or use the shorthand:

```python
initial = model.basis_state(0, 0)  # |g, 0⟩ directly
```

---

## 4. Build a drive pulse

Drive the qubit with a Gaussian π-pulse:

```python
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope

# π-pulse on the qubit channel ("q")
# carrier = 0 because the qubit frame frequency removes the bare rotation
# amp = π/2 (half because envelope integrates to ~1 for a normalized Gaussian)
pulse = Pulse(
    channel="q",
    t0=0.0,
    duration=80e-9,
    envelope=gaussian_envelope,
    amp=np.pi / 2,        # total rotation = 2 * amp * integral(envelope)
    carrier=0.0,          # resonant in rotating frame
    phase=0.0,
)
```

!!! note "Carrier sign convention"
    `carrier` stores the angular frequency of the drive tone.
    For a transition at rotating-frame frequency `ω_transition`, set
    `carrier = -ω_transition`. In the resonant rotating frame, `carrier = 0.0`.
    See [Physics & Conventions](physics_conventions.md#drive-convention).

---

## 5. Compile the sequence

```python
from cqed_sim.sequence import SequenceCompiler

compiled = SequenceCompiler(dt=2e-9).compile([pulse], t_end=100e-9)
```

`SequenceCompiler(dt=...)` samples pulses onto a uniform grid at timestep `dt`.
`t_end` sets the total simulation window length.

---

## 6. Run the simulation

```python
from cqed_sim.sim import SimulationConfig, simulate_sequence

result = simulate_sequence(
    model,
    compiled,
    initial,
    drive_ops={"q": "qubit"},          # maps channel "q" to the qubit drive operator
    config=SimulationConfig(frame=frame),
)
```

`drive_ops` maps each pulse channel name to a drive target. The string `"qubit"` is a shorthand for the standard qubit drive; cavity drives use `"cavity"`.

---

## 7. Read out the result

```python
# Final qubit-cavity joint state
print(result.final_state)

# Expectation values (qubit population, photon number, etc.)
print(result.expectations)

# Measure the qubit (1000 shots, synthetic readout)
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

meas = measure_qubit(
    result.final_state,
    QubitMeasurementSpec(shots=1000, seed=42),
)
print(meas.counts)        # e.g., {"g": 12, "e": 988}
print(meas.probabilities) # e.g., {"g": 0.012, "e": 0.988}
```

---

## Complete script

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.core import StatePreparationSpec, prepare_state, qubit_state, fock_state
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

# System
model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5.0e9,
    omega_q=2 * np.pi * 6.0e9,
    alpha=2 * np.pi * (-200e6),
    chi=2 * np.pi * (-2.84e6),
    kerr=2 * np.pi * (-2e3),
    n_cav=8,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
initial = model.basis_state(0, 0)

# π-pulse
pulse = Pulse("q", 0.0, 80e-9, gaussian_envelope, amp=np.pi / 2)
compiled = SequenceCompiler(dt=2e-9).compile([pulse], t_end=100e-9)

# Simulate
result = simulate_sequence(
    model, compiled, initial,
    drive_ops={"q": "qubit"},
    config=SimulationConfig(frame=frame),
)

# Measure
meas = measure_qubit(result.final_state, QubitMeasurementSpec(shots=1000, seed=42))
print(meas.counts)
```

---

## What's next

<div class="grid cards" markdown>

-   :material-book-open-variant:{ .lg .middle } **User Guides**

    ---

    Step-by-step docs for every part of the workflow: models, frames, pulses, sequences, solvers, observables, noise.

    [:octicons-arrow-right-24: Start with Defining Models](user_guides/defining_models.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Structured Jupyter notebook curriculum — from a minimal dispersive model to GRAPE optimal control.

    [:octicons-arrow-right-24: Tutorial overview](tutorials/index.md)

-   :material-atom:{ .lg .middle } **Physics & Conventions**

    ---

    Hamiltonian definitions, carrier sign convention, dispersive-shift sign, tensor ordering, and units.

    [:octicons-arrow-right-24: Conventions reference](physics_conventions.md)

-   :material-book-alphabet:{ .lg .middle } **API Reference**

    ---

    Complete public API for every submodule with signatures, parameters, and examples.

    [:octicons-arrow-right-24: API overview](api/overview.md)

</div>
