# Tutorial: Kerr Free Evolution

The primary guided workflow notebook: `tutorials/10_core_workflows/02_kerr_free_evolution.ipynb`.

Coordinate-convention companion: `tutorials/10_core_workflows/03_phase_space_coordinates_and_wigner_conventions.ipynb`.

Sign-check companion: `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`.

Foundational notebook: `tutorials/14_kerr_free_evolution.ipynb`.

---

## Physics Background

### Origin of Self-Kerr in Superconducting Circuits

The Josephson junction potential is:

$$U(\hat{\varphi}) = -E_J \cos\hat{\varphi} \approx E_J\!\left(-1 + \frac{\hat{\varphi}^2}{2} - \frac{\hat{\varphi}^4}{24} + \cdots\right)$$

After quantization, $\hat{\varphi} = \varphi_{\text{zpf}}(a + a^\dagger)$ where $\varphi_{\text{zpf}}$ is the zero-point phase fluctuation amplitude. The $\hat{\varphi}^2$ term gives the harmonic frequency; the $\hat{\varphi}^4$ term gives the leading nonlinearity. For a cavity with small zero-point fluctuations, this fourth-order term becomes the **self-Kerr** after normal-ordering:

$$H_K = \frac{K}{2} (a^\dagger)^2 a^2 = \frac{K}{2} \hat{n}(\hat{n}-1)$$

where $K < 0$ for a transmon-coupled cavity (the dispersive shift mediates an effective Kerr on the storage mode). The self-Kerr makes the cavity anharmonic: the transition $|n\rangle \to |n+1\rangle$ is shifted by $K \cdot n$ relative to the fundamental frequency.

For typical storage cavities coupled dispersively to a transmon, $K/2\pi \sim -1$ to $-10$ kHz — small enough that the cavity looks nearly harmonic for short times, but significant over microsecond timescales.

### Phase-Space Dynamics: Collapse and Revival

A coherent state $|\alpha\rangle$ can be written as a superposition of Fock states:

$$|\alpha\rangle = e^{-|\alpha|^2/2} \sum_n \frac{\alpha^n}{\sqrt{n!}} |n\rangle$$

Under $H_K$, each Fock component picks up a phase $e^{-i K n(n-1) t/2}$:

$$|\psi(t)\rangle = e^{-|\alpha|^2/2} \sum_n \frac{\alpha^n}{\sqrt{n!}} e^{-i K n(n-1) t/2} |n\rangle$$

Because $n(n-1)$ is quadratic in $n$, the phases are not commensurate for general $t$ — the coherent-state components **dephase**, causing the Wigner function to distort.

At special fractions of the **Kerr revival period** $T_K = 2\pi/|K|$:

| Time | Phase-space structure |
|---|---|
| $t = 0$ | Gaussian blob (coherent state) |
| $t = T_K/4$ | Crescent or banana shape (phases partially dephased) |
| $t = T_K/2$ | Schrödinger cat state — superposition of two opposite-phase coherents |
| $t = 3T_K/4$ | Crescent on the opposite side |
| $t = T_K$ | Full revival — coherent state restored |

### Schrödinger Cat at $T_K / 2$

At exactly $t = T_K/2$, the $n(n-1)$ phases conspire so that:

$$|\psi(T_K/2)\rangle = \frac{1}{\sqrt{2}}\left(e^{i\phi_+}|\alpha'\rangle + e^{i\phi_-}|-\alpha'\rangle\right)$$

for some real phases $\phi_\pm$ and $|\alpha'| \approx |\alpha|$. This is a **two-component Schrödinger cat state** — a quantum superposition of two macroscopically distinct coherent states. Its Wigner function displays two Gaussian lobes separated in phase space, connected by an interference fringe with oscillatory, possibly negative regions.

For $|\alpha| = 2$ and $K/2\pi = -2$ kHz, the revival time is $T_K = 1/2\,\text{kHz} = 500\, \mu\text{s}$. The $T_K/2$ cat appears at $250\, \mu\text{s}$.

### Why Negative Wigner Values?

The Wigner function $W(\beta)$ can be negative for non-classical states. Negativity is a signature of quantum coherence between the two lobes: it represents the interference fringes between $|\alpha'\rangle$ and $|-\alpha'\rangle$. A statistical mixture of two coherent states would have a non-negative Wigner function; the superposition has negative values in the interference region.

---

## Workflow Location

Use the guided notebook and the related repo-side scripts:

- `tutorials/10_core_workflows/02_kerr_free_evolution.ipynb`
- `tutorials/10_core_workflows/03_phase_space_coordinates_and_wigner_conventions.ipynb`
- `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`
- `examples/workflows/kerr_free_evolution.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`

---

## Library Building Blocks

The Kerr workflow is built from reusable library primitives:

- `DispersiveTransmonCavityModel` and `FrameSpec` from `cqed_sim.core`
- `StatePreparationSpec`, `coherent_state`, and `prepare_state(...)` from `cqed_sim.core`
- `reduced_cavity_state(...)` and `cavity_wigner(...)` from `cqed_sim.sim`

---

## Minimal Manual Pattern

```python
import numpy as np
from cqed_sim.core import (
    DispersiveTransmonCavityModel, FrameSpec,
    StatePreparationSpec, coherent_state, qubit_state, prepare_state,
)
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence, cavity_wigner, reduced_cavity_state

# System with non-trivial self-Kerr
model = DispersiveTransmonCavityModel(
    omega_c = 2 * np.pi * 5.0e9,
    omega_q = 2 * np.pi * 6.0e9,
    alpha   = 2 * np.pi * (-200e6),
    chi     = 2 * np.pi * (-2.84e6),
    kerr    = 2 * np.pi * (-2e3),    # K/2π = -2 kHz → T_K = 500 μs
    n_cav   = 25,
    n_tr    = 2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

# Prepare |g⟩ ⊗ |α=2⟩
initial_state = prepare_state(
    model,
    StatePreparationSpec(qubit=qubit_state("g"), storage=coherent_state(2.0)),
)

# Evolve to T_K/4 and compute Wigner function
T_K = 1.0 / abs(-2e3)              # Revival period in seconds
t_snap = T_K / 4

compiled = SequenceCompiler(dt=2e-9).compile([], t_end=t_snap)
result = simulate_sequence(model, compiled, initial_state, {},
                           config=SimulationConfig(frame=frame))

rho_c = reduced_cavity_state(result.final_state)
xvec, yvec, wigner = cavity_wigner(rho_c, coordinate="alpha")
```

---

## Wigner Function Snapshots

Evolving a coherent state $|\alpha = 2\rangle$ under $H_K = \frac{K}{2}(a^\dagger)^2 a^2$ at $K/2\pi = -2$ kHz produces the following phase-space snapshots:

![Kerr Free Evolution — Wigner Snapshots](../assets/images/tutorials/kerr_free_evolution_wigner.png)

The four panels show:

1. **$t = 0$**: Gaussian coherent blob centered at $(\text{Re}(\alpha), \text{Im}(\alpha)) = (2, 0)$ in alpha coordinates — this is the initial coherent state $|\alpha=2\rangle$
2. **$t = T_K/4$**: The blob has sheared and stretched into a crescent. The outer (higher-photon-number) part of the distribution precesses faster than the inner part, causing the characteristic banana shape
3. **$t = T_K/2$**: Two distinct Gaussian lobes separated by roughly $2|\alpha|$ in the angular direction, with oscillatory negative-valued interference fringes between them — a Schrödinger cat state
4. **$t = 3T_K/4$**: Crescent on the opposite side, mirror-symmetric to the $T_K/4$ shape

### Coordinate Note

The plot is drawn in coherent-state `alpha` coordinates, so the initial coherent blob is centered at $(2, 0)$ matching the displacement amplitude $\alpha = 2$. In quadrature coordinates, the same blob would appear at $(2\sqrt{2}, 0) \approx (2.83, 0)$.

See [Phase Space Conventions](phase_space_conventions.md) for a detailed comparison.

---

For the full guided walkthrough, use `tutorials/10_core_workflows/02_kerr_free_evolution.ipynb`. For the coordinate-convention companion, use `tutorials/10_core_workflows/03_phase_space_coordinates_and_wigner_conventions.ipynb`. For the sign-check companion, use `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`. For a compact executable script, use the example workflow module or standalone script.

---

## See Also

- [Phase Space Conventions](phase_space_conventions.md) — alpha vs. quadrature coordinates, √2 rescaling
- [Cross-Kerr Interaction](cross_kerr.md) — conditional phase between two modes
- [Physics & Conventions](../physics_conventions.md) — Kerr sign convention reference
