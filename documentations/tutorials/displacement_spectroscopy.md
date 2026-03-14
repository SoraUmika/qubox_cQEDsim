# Tutorial: Displacement & Qubit Spectroscopy

The primary guided notebook path for this topic now lives in the top-level `tutorials/` curriculum:

- `tutorials/03_cavity_displacement_basics.ipynb`
- `tutorials/06_qubit_spectroscopy.ipynb`
- `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`

This page remains as a compact topical summary of the same physics.

---

## Physics Background

In a dispersive qubit–cavity system, the qubit transition frequency shifts with cavity photon number:

$$\omega_{ge}(n) = \omega_{ge}(0) + \chi \cdot n$$

By displacing the cavity into a coherent state and then sweeping a qubit drive frequency, one can resolve individual photon-number peaks in the spectroscopy signal.

---

## Setup

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec

model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5.0e9,
    omega_q=2 * np.pi * 6.0e9,
    alpha=2 * np.pi * (-200e6),
    chi=2 * np.pi * (-2.84e6),
    kerr=2 * np.pi * (-2e3),
    n_cav=15,
    n_tr=2,
)

frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)
```

---

## Step 1: Displace the Cavity

Create a coherent state by applying a displacement pulse:

```python
from cqed_sim.io import DisplacementGate
from cqed_sim.pulses import build_displacement_pulse

gate = DisplacementGate(index=0, name="displace", re=2.0, im=0.0)
disp_pulses, disp_ops, _ = build_displacement_pulse(
    gate,
    {"duration_displacement_s": 200e-9},
)
```

---

## Step 2: Sweep Qubit Drive Frequency

After displacing, apply a weak qubit probe at different frequencies and measure the qubit excitation probability:

```python
from cqed_sim.core import carrier_for_transition_frequency
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence
from cqed_sim.sim import reduced_qubit_state
from functools import partial

# Sweep detunings around the bare qubit frequency
detunings_hz = np.linspace(-15e6, 5e6, 201)
pe_values = []

for det_hz in detunings_hz:
    omega_probe = 2 * np.pi * det_hz  # Detuning in rotating frame
    carrier = carrier_for_transition_frequency(omega_probe)

    probe_pulse = Pulse(
        "q", 300e-9, 500e-9,
        partial(gaussian_envelope, sigma=0.25),
        carrier=carrier,
        amp=0.01 * np.pi,
    )

    all_pulses = disp_pulses + [probe_pulse]
    drive_ops = {**disp_ops, "q": "qubit"}

    compiled = SequenceCompiler(dt=2e-9).compile(all_pulses, t_end=850e-9)

    result = simulate_sequence(
        model, compiled, model.basis_state(0, 0), drive_ops,
        config=SimulationConfig(frame=frame),
    )

    rho_q = reduced_qubit_state(result.final_state)
    pe = rho_q[1, 1].real
    pe_values.append(pe)
```

---

## Step 3: Analyze the Spectrum

The spectrum shows peaks at each photon-number manifold, separated by χ:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(detunings_hz / 1e6, pe_values)
plt.xlabel("Qubit drive detuning (MHz)")
plt.ylabel("P(e)")
plt.title("Displacement + Qubit Spectroscopy")
plt.grid(True, alpha=0.3)
plt.show()
```

For a coherent state $|\alpha=2\rangle$, you should see peaks at:

- $\delta = 0$ (n=0 manifold)
- $\delta = \chi / (2\pi)$ (n=1 manifold)
- $\delta = 2\chi / (2\pi)$ (n=2 manifold)
- etc.

with amplitudes following the Poisson distribution $P(n) = e^{-|\alpha|^2} |\alpha|^{2n} / n!$.

---

## Related Repo Assets

- Guided notebooks: `tutorials/03_cavity_displacement_basics.ipynb`, `tutorials/06_qubit_spectroscopy.ipynb`, `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`
- Standalone script: `examples/displacement_qubit_spectroscopy.py`

Use the numbered notebooks when you want the full teaching flow and the standalone script when you want one compact executable example.
