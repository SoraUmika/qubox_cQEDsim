# Tutorial: Sideband Swap

This tutorial demonstrates how to simulate a sideband transition — a common operation for transferring excitations between a transmon and a bosonic mode.

For the current workflow tutorial suite, start with:

- `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb`
- `tutorials/20_bosonic_and_sideband/02_detuned_sideband_synchronization.ipynb`
- `tutorials/20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb`
- `tutorials/20_bosonic_and_sideband/04_shelving_isolation.ipynb`

The earlier numbered notebook `tutorials/24_sideband_like_interactions.ipynb` still remains useful as a broader curriculum summary.

---

## Physics Background

A sideband drive applies a coupling between the transmon and a bosonic mode at the appropriate frequency. In the rotating frame, the effective sideband Hamiltonian is:

$$H_{\text{sb}} = \varepsilon(t) \, |u\rangle\langle \ell| \, a_m + \varepsilon^*(t) \, a_m^\dagger \, |\ell\rangle\langle u|$$

where $|\ell\rangle$ and $|u\rangle$ are the lower and upper transmon levels, and $a_m$ is the bosonic mode annihilation operator.

- **Red sideband:** excites the transmon while removing a photon from the mode (or vice versa)
- **Blue sideband:** excites both the transmon and the mode simultaneously (or de-excites both)

---

## Setup

```python
import numpy as np
from cqed_sim.core import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    SidebandDriveSpec,
    carrier_for_transition_frequency,
)

model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5.0e9,
    omega_q=2 * np.pi * 6.0e9,
    alpha=2 * np.pi * (-200e6),
    chi=2 * np.pi * (-2.84e6),
    n_cav=8,
    n_tr=3,   # Need at least 3 levels for sideband
)

frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)
```

---

## Building a Sideband Pulse

```python
from cqed_sim.pulses import build_sideband_pulse

target = SidebandDriveSpec(
    mode="storage",
    lower_level=0,
    upper_level=1,
    sideband="red",
)

# Get the sideband transition frequency
omega_sb = model.sideband_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=1,
    sideband="red",
    frame=frame,
)

pulses, drive_ops, meta = build_sideband_pulse(
    target,
    duration_s=500e-9,
    amplitude_rad_s=2 * np.pi * 1e6,
    channel="sideband",
    carrier=carrier_for_transition_frequency(omega_sb),
)
```

---

## Running the Simulation

```python
from cqed_sim.core import (
    StatePreparationSpec,
    qubit_state,
    fock_state,
    prepare_state,
)
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

# Start with qubit excited, cavity empty: |e, 0⟩
initial = prepare_state(
    model,
    StatePreparationSpec(qubit=qubit_state("e"), storage=fock_state(0)),
)

compiled = SequenceCompiler(dt=2e-9).compile(pulses, t_end=550e-9)

result = simulate_sequence(
    model, compiled, initial, drive_ops,
    config=SimulationConfig(frame=frame, store_states=True),
)
```

---

## Observing the Swap

```python
from cqed_sim.sim import (
    reduced_qubit_state,
    storage_photon_number,
    transmon_level_populations,
)

# At the end
pops = transmon_level_populations(result.final_state)
n_s = storage_photon_number(result.final_state)

print(f"Transmon populations: {pops}")
print(f"Storage ⟨n⟩: {n_s:.3f}")
```

A successful red sideband swap transfers $|e, 0\rangle \to |g, 1\rangle$, with the transmon relaxing and the cavity gaining a photon.

### Population Dynamics vs. Pulse Duration

Sweeping the sideband pulse duration reveals the Rabi oscillation between transmon and cavity:

![Sideband Swap Dynamics](../assets/images/tutorials/sideband_swap_dynamics.png)

The qubit excited-state probability $P(e)$ drops as the cavity photon number $\langle n \rangle$ rises, completing a full swap cycle.

---

## Existing Examples

- `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb` — basic sideband swap
- `tutorials/20_bosonic_and_sideband/02_detuned_sideband_synchronization.ipynb` — detuned sideband synchronization
- `tutorials/20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb` — sequential storage reset
- `tutorials/20_bosonic_and_sideband/04_shelving_isolation.ipynb` — shelving with a multilevel sideband
- `tutorials/30_advanced_protocols/02_open_system_sideband_degradation.ipynb` — sideband with open-system noise
