# Cross-Kerr Interaction and Conditional Phase

This tutorial covers the cross-Kerr interaction between two bosonic modes — a mechanism that produces a controlled phase shift conditioned on the photon occupation of one mode. The relevant notebook is:

- `tutorials/30_advanced_protocols/01_multimode_crosskerr.ipynb`

Background material:

- `tutorials/15_cross_kerr_and_conditional_phase_accumulation.ipynb`

---

## Physics Background

### Self-Kerr vs. Cross-Kerr

The Kerr family of interactions describes photon-number-dependent phase shifts arising from the fourth-order term of the Josephson potential:

**Self-Kerr** (single mode) — the frequency of a mode shifts with its own photon number:

$$H_K = \frac{K}{2} (a^\dagger)^2 a^2 = \frac{K}{2} \hat{n}(\hat{n}-1)$$

**Cross-Kerr** (two modes) — the frequency of one mode shifts with the photon number of another:

$$H_{\chi} = \chi_{sr} \, a_s^\dagger a_s \, a_r^\dagger a_r = \chi_{sr} \, \hat{n}_s \, \hat{n}_r$$

where $s$ denotes the storage mode and $r$ the readout mode. This interaction commutes with both number operators: it does not transfer photons between modes, it only accumulates phase.

### Conditional Phase Accumulation

Under $H_\chi$ alone, the time evolution is:

$$U(t) = e^{-i \chi_{sr} \hat{n}_s \hat{n}_r t}$$

For a state of the form $|n_s, n_r\rangle$, the accumulated phase is:

$$\phi(t) = \chi_{sr} \cdot n_s \cdot n_r \cdot t$$

The phase is **conditional**: it accumulates only when **both** modes are occupied. This is the basis for number-selective cross-Kerr gates.

### Physical Origin

In a three-mode dispersive system (storage + readout + transmon), the cross-Kerr between storage and readout arises from virtual transmon-mediated processes. If the transmon mediates coupling $g_s$ to the storage and $g_r$ to the readout, the effective cross-Kerr is approximately:

$$\chi_{sr} \approx -\frac{2 g_s^2 g_r^2}{\alpha} \left(\frac{1}{\Delta_s^2} + \frac{1}{\Delta_r^2}\right)$$

where $\alpha$ is the transmon anharmonicity and $\Delta_{s,r} = \omega_{s,r} - \omega_q$ are the storage/readout-qubit detunings. Physically, both modes "see" the same nonlinear oscillator (the transmon), so a photon in one mode shifts the effective impedance seen by the other.

### Observable: Relative Phase Between Storage Branches

Consider an initial state where the storage is in a superposition of $|0\rangle_s$ and $|1\rangle_s$, with the readout in a Fock state $|1\rangle_r$:

$$|\psi(0)\rangle = \frac{1}{\sqrt{2}}\left(|0_s, 1_r\rangle + |1_s, 1_r\rangle\right)$$

After time $t$:

- The $|0_s, 1_r\rangle$ component accumulates phase $\phi = \chi_{sr} \cdot 0 \cdot 1 \cdot t = 0$
- The $|1_s, 1_r\rangle$ component accumulates phase $\phi = \chi_{sr} \cdot 1 \cdot 1 \cdot t = \chi_{sr} \, t$

The relative phase between the two branches grows linearly: $\Delta\phi(t) = \chi_{sr} \, t$.

---

## Setup: Three-Mode Model

```python
import numpy as np
from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec

model = DispersiveReadoutTransmonStorageModel(
    omega_s   = 2 * np.pi * 5.0e9,    # Storage cavity frequency
    omega_r   = 2 * np.pi * 7.5e9,    # Readout resonator frequency
    omega_q   = 2 * np.pi * 6.0e9,    # Transmon qubit frequency
    chi_sr    = 2 * np.pi * 1.5e6,    # Storage-readout cross-Kerr
    chi_s     = 0.0,                   # Storage self-Kerr (set to 0 here)
    chi_r     = 0.0,                   # Readout self-Kerr (set to 0 here)
    n_storage = 4,
    n_readout = 4,
    n_tr      = 2,
)

frame = FrameSpec(
    omega_s_frame = model.omega_s,
    omega_r_frame = model.omega_r,
    omega_q_frame = model.omega_q,
)
```

---

## Simulating Conditional Phase Accumulation

```python
from cqed_sim.core import prepare_state, StatePreparationSpec
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence
import numpy as np

# Initial state: (|0_s, 1_r⟩ + |1_s, 1_r⟩) / √2
s0r1 = model.basis_state(0, 1, 0)   # storage=0, readout=1, qubit=g
s1r1 = model.basis_state(1, 1, 0)   # storage=1, readout=1, qubit=g
initial_state = (s0r1 + s1r1).unit()

# Free-evolution times to probe phase accumulation
times_ns = np.linspace(0, 700, 50)
phases = []

for t_ns in times_ns:
    t_s = t_ns * 1e-9
    # Free evolution: no pulses, just advance t_end
    compiled = SequenceCompiler(dt=2e-9).compile([], t_end=t_s)
    result = simulate_sequence(
        model, compiled, initial_state, {},
        config=SimulationConfig(frame=frame),
    )
    state = result.final_state

    # Extract relative phase between |0_s,1_r⟩ and |1_s,1_r⟩ branches
    amp_ref     = s0r1.overlap(state)
    amp_shifted = s1r1.overlap(state)
    phases.append(float(np.angle(amp_shifted / amp_ref)))
```

---

## Expected Results

The relative phase should grow linearly with time at a rate equal to $\chi_{sr}$:

$$\Delta\phi(t) = \chi_{sr} \cdot t$$

For $\chi_{sr}/2\pi = 1.5$ MHz, the phase accumulates $2\pi$ radians in a time $T = 1/\chi_{sr} \approx 667$ ns.

```python
import matplotlib.pyplot as plt

# Theory prediction
theory_times_ns = np.linspace(0, 700, 200)
theory_phases = 2 * np.pi * 1.5e6 * (theory_times_ns * 1e-9)

plt.figure(figsize=(8, 4))
plt.plot(times_ns, phases, "o", ms=5, label="Simulation")
plt.plot(theory_times_ns, theory_phases % (2 * np.pi) - np.pi,
         "--", linewidth=1.5, label=r"Theory: $\chi_{sr} \cdot t$")
plt.xlabel("Free evolution time (ns)")
plt.ylabel("Relative phase (rad)")
plt.title(r"Cross-Kerr Conditional Phase: $\Delta\phi = \chi_{sr} \cdot t$")
plt.legend()
plt.tight_layout()
```

The simulation points overlay with the theory line. The population of each branch remains constant — cross-Kerr is a purely phase-accumulating interaction with no number transfer.

---

## Conditional Phase Gate

A controlled-phase gate $CZ(\phi) = \text{diag}(1, 1, 1, e^{i\phi})$ on the storage-readout subspace $\{|0_s, 0_r\rangle, |0_s, 1_r\rangle, |1_s, 0_r\rangle, |1_s, 1_r\rangle\}$ is implemented by free evolution for a time $t = \phi / \chi_{sr}$. For $\phi = \pi$, the required time is:

$$t_{\pi} = \frac{\pi}{\chi_{sr}} = \frac{1}{2 f_{\chi_{sr}}}$$

For $\chi_{sr}/2\pi = 1.5$ MHz: $t_\pi \approx 333$ ns.

---

## Multi-Mode Cross-Kerr

In a three-mode system (storage, readout, transmon), the full cross-Kerr Hamiltonian includes all pairwise cross-Kerr terms. The notebook `01_multimode_crosskerr.ipynb` demonstrates how different pairs of modes accumulate different phases during free evolution, and how these interactions can be decoupled or exploited for multi-mode gate operations.

The total dispersive Hamiltonian in the multi-mode case is:

$$H = \sum_j \omega_j a_j^\dagger a_j + \sum_{j < k} \chi_{jk} \hat{n}_j \hat{n}_k + \sum_j \frac{K_j}{2} \hat{n}_j(\hat{n}_j - 1)$$

where $j, k$ index all modes (storage, readout, transmon), $\chi_{jk}$ are cross-Kerr couplings, and $K_j$ are self-Kerr rates.

---

## Physical Significance

Cross-Kerr interactions are central to several cQED applications:

- **Dispersive readout** — the cavity frequency shift $\chi$ on the qubit state is itself a storage-qubit cross-Kerr, enabling non-destructive qubit measurement
- **Controlled-phase gates** — conditional phase between two bosonic modes
- **QND photon detection** — a single photon in one mode shifts the spectroscopy of another
- **Entangled-state generation** — evolved from $|+\rangle_s |+\rangle_r$ under $H_\chi$, one obtains entangled cat states at $t = T/4$

---

## Related Notebooks

- `tutorials/30_advanced_protocols/01_multimode_crosskerr.ipynb` — three-mode simulation
- `tutorials/15_cross_kerr_and_conditional_phase_accumulation.ipynb` — foundational curriculum
- `tutorials/16_storage_cavity_coherent_state_dynamics.ipynb` — coherent-state evolution in storage

## See Also

- [Kerr Free Evolution](kerr_free_evolution.md) — self-Kerr dynamics and phase-space collapse
- [Sideband Swap](sideband_swap.md) — photon transfer protocols
- [Physics & Conventions](../physics_conventions.md) — dispersive Hamiltonian conventions
