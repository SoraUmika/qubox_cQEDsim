# Tutorial: Displacement & Qubit Spectroscopy

The primary guided notebook path for this topic:

- `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
- `tutorials/10_core_workflows/04_selective_gaussian_number_splitting.ipynb`

Foundational background:

- `tutorials/03_cavity_displacement_basics.ipynb`
- `tutorials/06_qubit_spectroscopy.ipynb`
- `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`

---

## Physics Background

### The Dispersive Hamiltonian

When a transmon qubit (frequency $\omega_q$, anharmonicity $\alpha$) is coupled to a cavity mode (frequency $\omega_c$) with vacuum Rabi coupling $g$, and the detuning $\Delta = \omega_q - \omega_c$ satisfies $g \ll |\Delta|$, the Jaynes-Cummings model reduces to the dispersive Hamiltonian via a Schrieffer-Wolff transformation:

$$H_{\text{disp}} = \omega_c \, a^\dagger a + \frac{\omega_q + \chi a^\dagger a}{2} \sigma_z + \frac{K}{2}(a^\dagger)^2 a^2$$

The key parameter is the **dispersive shift**:

$$\chi \approx \frac{-g^2 \alpha}{\Delta(\Delta + \alpha)}$$

where $\alpha < 0$ for a transmon. With typical parameters $g/2\pi \sim 100$ MHz, $\Delta/2\pi \sim 1$ GHz, $\alpha/2\pi \sim -200$ MHz, one obtains $\chi/2\pi \sim -1$ to $-10$ MHz.

The $K$ term is the **cavity self-Kerr**, arising from the same dispersive coupling; for a storage cavity far detuned from the transmon it is usually small ($K/2\pi \sim$ kHz range).

### Number-Split Qubit Spectrum

Rewriting $H_{\text{disp}}$ in the qubit subspace for a definite cavity state $|n\rangle$:

$$H|_n = \omega_c n + \left(\omega_q + \chi n\right)|e\rangle\langle e|$$

The qubit transition frequency in the $n$-photon manifold is:

$$\omega_{ge}(n) = \omega_{ge}(0) + \chi \cdot n$$

where $\omega_{ge}(0) = \omega_q + \chi \cdot 0 = \omega_q$ is the bare qubit frequency. Each photon number shifts the qubit resonance by $\chi$. When $|\chi|$ exceeds the qubit linewidth, these transitions are spectrally resolved — this is **photon-number splitting**.

For $\chi/2\pi = -2.84$ MHz, the peaks appear at:

| $n$ | Peak detuning (from $\omega_{ge}(0)$) |
|---|---|
| 0 | 0 MHz |
| 1 | −2.84 MHz |
| 2 | −5.68 MHz |
| 3 | −8.52 MHz |
| 4 | −11.36 MHz |

### Cavity Displacement and the Coherent State

A short resonant pulse applied to the cavity displaces it to a **coherent state** $|\alpha\rangle$:

$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_{n=0}^\infty \frac{\alpha^n}{\sqrt{n!}} |n\rangle$$

The photon-number distribution of a coherent state is Poissonian:

$$P(n) = e^{-|\alpha|^2} \frac{|\alpha|^{2n}}{n!}, \qquad \bar{n} = |\alpha|^2$$

For $\alpha = 2$: $\bar{n} = 4$, and the distribution peaks near $n = 3$–$5$.

### Peak Amplitudes as Photon-Number Weights

In the **weak, selective drive** limit, a long Gaussian qubit probe at frequency $\omega_{ge}(0) + \delta$ predominantly excites the $n$-th manifold when $\delta \approx n \chi$. The height of the $n$-th peak is proportional to $P(n)$ — the probability of finding $n$ photons in the cavity. This makes qubit spectroscopy a direct probe of the cavity Fock distribution.

The validity condition is $\Omega_R \ll |\chi| / 2\pi$ (probe Rabi frequency much smaller than the manifold spacing).

---

## Setup

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec

model = DispersiveTransmonCavityModel(
    omega_c = 2 * np.pi * 5.0e9,
    omega_q = 2 * np.pi * 6.0e9,
    alpha   = 2 * np.pi * (-200e6),
    chi     = 2 * np.pi * (-2.84e6),
    kerr    = 2 * np.pi * (-2e3),
    n_cav   = 15,
    n_tr    = 2,
)

frame = FrameSpec(
    omega_c_frame = model.omega_c,
    omega_q_frame = model.omega_q,
)
```

---

## Step 1: Displace the Cavity

Create a coherent state $|\alpha = 2\rangle$ by applying a displacement pulse:

```python
from cqed_sim.io import DisplacementGate
from cqed_sim.pulses import build_displacement_pulse

gate = DisplacementGate(index=0, name="displace", re=2.0, im=0.0)
disp_pulses, disp_ops, _ = build_displacement_pulse(
    gate, {"duration_displacement_s": 120e-9},
)
```

---

## Step 2: Sweep Qubit Drive Frequency

Apply a long, weak, Gaussian qubit probe at each detuning and record the qubit excitation probability:

```python
from cqed_sim.core import carrier_for_transition_frequency
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence, reduced_qubit_state
from functools import partial

detunings_hz = np.linspace(-16e6, 2e6, 181)
pe_values = []

for det_hz in detunings_hz:
    omega_probe = 2 * np.pi * det_hz
    carrier = carrier_for_transition_frequency(omega_probe)

    probe_pulse = Pulse(
        "q", 160e-9, 2.5e-6,
        partial(gaussian_envelope, sigma=0.18),
        carrier=carrier,
        amp=2 * np.pi * 0.04e6,   # Weak drive
    )

    all_pulses = disp_pulses + [probe_pulse]
    compiled = SequenceCompiler(dt=2e-9).compile(all_pulses, t_end=2.7e-6)
    result = simulate_sequence(
        model, compiled, model.basis_state(0, 0), {**disp_ops, "q": "qubit"},
        config=SimulationConfig(frame=frame),
    )
    rho_q = reduced_qubit_state(result.final_state)
    pe_values.append(float(np.real(rho_q[1, 1])))
```

---

## Step 3: Analyze the Spectrum

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(np.array(detunings_hz) / 1e6, pe_values)
plt.xlabel("Qubit drive detuning (MHz)")
plt.ylabel("$P(e)$")
plt.title(r"Number Splitting: $\omega_{ge}(n) = \omega_{ge}(0) + n\chi$")
plt.axvline(0, ls="--", color="gray", alpha=0.5, label="$n=0$")
for n in range(1, 6):
    plt.axvline(n * (-2.84), ls=":", color="gray", alpha=0.3)
plt.grid(alpha=0.3)
```

Running this sweep produces the following spectrum:

![Displacement + Qubit Spectroscopy](../assets/images/tutorials/displacement_spectroscopy.png)

For a coherent state $|\alpha = 2\rangle$, peaks appear at $n \cdot \chi / 2\pi$ for $n = 0, 1, 2, \ldots$ with amplitudes following the Poisson distribution $P(n) = e^{-4} 4^n / n!$.

---

## Connecting Peak Heights to Photon-Number Weights

The workflow notebook overlays the spectrum with a theory prediction: each peak modeled as a Gaussian of width $\sigma_\omega = 1/(2\sigma_t)$ (where $\sigma_t$ is the probe time-domain sigma) and area proportional to $P(n)$. The comparison validates:

1. That the displacement pulse prepared a coherent state with the correct amplitude
2. That the selective-probe condition is satisfied
3. That the peak heights accurately encode the Fock distribution

See [Number Splitting](number_splitting.md) for the detailed peak-height extraction and validation workflow.

---

## Related Repo Assets

- Guided notebooks: `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`, `tutorials/10_core_workflows/04_selective_gaussian_number_splitting.ipynb`, `tutorials/03_cavity_displacement_basics.ipynb`, `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`
- Standalone script: `examples/displacement_qubit_spectroscopy.py`

## See Also

- [Number Splitting](number_splitting.md) — photon-number discrimination via selective spectroscopy
- [Phase Space Conventions](phase_space_conventions.md) — Wigner function coordinates
- [Physics & Conventions](../physics_conventions.md) — dispersive Hamiltonian and sign conventions
