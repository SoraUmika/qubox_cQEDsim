# Floquet Driven Systems

The Floquet tutorial set shows how `cqed_sim` treats a periodically driven Hamiltonian as a first-class eigenproblem rather than a static model perturbed by a classical drive.

Workflow notebooks:

- `tutorials/50_floquet_driven_systems/01_sideband_quasienergy_scan.ipynb`
- `tutorials/50_floquet_driven_systems/02_branch_tracking_and_multiphoton_resonances.ipynb`

---

## Physics Background

### Floquet Theorem

For a time-periodic Hamiltonian $H(t) = H(t + T)$ with period $T = 2\pi/\Omega$, the Floquet theorem guarantees that the solutions of the Schrödinger equation take the form:

$$|\psi_n(t)\rangle = e^{-i\varepsilon_n t} |\phi_n(t)\rangle$$

where $|\phi_n(t + T)\rangle = |\phi_n(t)\rangle$ is a time-periodic **Floquet state** and $\varepsilon_n$ is the corresponding **quasienergy**. This is the temporal analog of Bloch's theorem for spatial periodicity in crystals.

Key properties:
- Quasienergies are defined **modulo $\Omega$** (the drive frequency): if $\varepsilon_n$ is a quasienergy, so is $\varepsilon_n + m\Omega$ for any integer $m$. This defines a Brillouin zone $\varepsilon \in [-\Omega/2, \Omega/2)$.
- The Floquet states $|\phi_n(t)\rangle$ can be expanded in Fourier modes: $|\phi_n(t)\rangle = \sum_m e^{im\Omega t} |\phi_n^{(m)}\rangle$. The index $m$ is the **photon number** of the drive field.
- Near resonance, Floquet states are hybridized superpositions of bare Hamiltonian eigenstates — the drive reorganizes the static energy structure.

### The Floquet Hamiltonian

To solve for quasienergies numerically, one constructs the extended-Hilbert-space **Floquet Hamiltonian**:

$$[H_F]_{nm} = H_{n-m} + m\Omega \, \delta_{nm}$$

where $H_k = \frac{1}{T}\int_0^T H(t) e^{-ik\Omega t} dt$ are the Fourier components of $H(t)$. Diagonalizing $H_F$ in a truncated photon-number sector gives the quasienergies and Floquet states.

### Avoided Crossings and Resonance Conditions

When two bare states $|a\rangle$ and $|b\rangle$ satisfy an **$n$-photon resonance** condition:

$$E_a - E_b = n \, \Omega$$

their Floquet quasienergies approach each other, leading to an **avoided crossing**. The minimum gap at the crossing is proportional to the drive matrix element $\langle a | V | b \rangle$ (the relevant Fourier component of the drive).

The physical consequence: near resonance, the periodic drive efficiently mixes bare states, causing Rabi-like oscillations at the rate of the gap. The minimum gap is the **sideband Rabi frequency** observable in time-domain experiments.

### Why Floquet Analysis vs. Perturbation Theory?

Static dressed-state intuition often fails when:

1. **The drive is strong** — multiple photon transitions become relevant and the perturbative expansion diverges
2. **Multi-photon resonances** are accidentally degenerate with the primary resonance
3. **Drive-induced level repulsion** shifts sideband frequencies away from their bare values
4. **Multiple drives** create a quasiperiodic problem that requires careful treatment

The Floquet approach treats the drive exactly as part of the eigenvalue problem, naturally handling all of these cases.

---

## Included Notebooks

### `50_floquet_driven_systems/01_sideband_quasienergy_scan.ipynb`

This notebook builds a transmon-storage sideband drive, sweeps the drive frequency across the red-sideband condition, and tracks the resulting quasienergy branches.

**What it teaches:**

- How to build a `FloquetProblem` from a physical model and periodic drive term
- How `run_floquet_sweep(...)` preserves branch identity across a parameter scan
- How the minimum adjacent quasienergy gap identifies the avoided-crossing resonance

**Setup:**

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.floquet import FloquetProblem, FloquetConfig, run_floquet_sweep

model = DispersiveTransmonCavityModel(
    omega_c = 2 * np.pi * 5.05e9,
    omega_q = 2 * np.pi * 6.25e9,
    alpha   = 2 * np.pi * (-250e6),
    chi     = 2 * np.pi * (-15e6),  # Stronger chi for visible splitting
    n_cav   = 6,
    n_tr    = 3,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
```

**Running the frequency sweep:**

```python
from cqed_sim.floquet import build_target_drive_term, solve_floquet

# Sweep drive detuning around the red sideband
scan_detunings_mhz = np.linspace(-0.25, 0.25, 51)
problems = []
for det_mhz in scan_detunings_mhz:
    freq_hz = model.sideband_transition_frequency(
        cavity_level=0, lower_level=0, upper_level=2,
        sideband="red", frame=frame
    ) / (2 * np.pi) + det_mhz * 1e6

    drive = build_target_drive_term(
        model, sideband_spec,
        amplitude=2 * np.pi * 0.03e6,
        frequency=2 * np.pi * freq_hz,
        waveform="cos",
    )
    problems.append(FloquetProblem(
        model=model, frame=frame,
        periodic_terms=(drive,),
        period=1.0 / abs(freq_hz),
    ))

sweep = run_floquet_sweep(
    problems, parameter_values=scan_detunings_mhz,
    config=FloquetConfig(n_time_samples=96),
)
```

**Expected output:**

The sweep produces two panels:

1. **Quasienergy branches vs. detuning** — multiple branches tracked with consistent labeling across the scan. Near the resonance condition (detuning = 0), two branches form an avoided crossing: they repel each other rather than crossing.

2. **Minimum quasienergy gap vs. detuning** — a sharp dip at the resonance. The depth of the dip gives the effective sideband coupling rate. The position of the minimum locates the true resonance frequency (which can differ from the bare sideband frequency due to drive-induced level repulsion).

### Generated Plot

The figure below shows the quasienergy branches and avoided-crossing gap produced by `tools/generate_tutorial_plots.py`:

![Floquet Quasienergy Scan](../assets/images/tutorials/floquet_quasienergy_scan.png)

---

### `50_floquet_driven_systems/02_branch_tracking_and_multiphoton_resonances.ipynb`

This notebook solves a single driven problem near a half-frequency qubit condition and identifies which bare energy gaps satisfy integer-harmonic resonance conditions.

**What it teaches:**

- How to solve one driven Hamiltonian with `solve_floquet(...)`
- How to inspect folded quasienergies and compare with bare transition energies
- How `identify_multiphoton_resonances(...)` reports the relevant harmonic resonance

**Quasienergy folding:**

```python
from cqed_sim.floquet import solve_floquet, identify_multiphoton_resonances

result = solve_floquet(
    problem,
    config=FloquetConfig(n_fourier_modes=3, n_time_samples=96),
)

# Quasienergies folded into [-Ω/2, Ω/2)
eps = result.quasienergies_mhz

# Identify which bare-state gap aligns with n*Ω
resonances = identify_multiphoton_resonances(
    result,
    model=model,
    max_order=4,
    gap_tolerance_mhz=0.5,
)
for r in resonances:
    print(f"  {r.lower_label} → {r.upper_label}: {r.order}-photon resonance at Ω = {r.drive_frequency_mhz:.2f} MHz")
```

**Physical interpretation:** When the drive frequency is $\Omega = (E_a - E_b)/n$ for some pair of bare states $(a, b)$ and integer $n$, the $n$-photon process resonantly couples those states. In the Floquet picture, this appears as a quasienergy near-degeneracy between the $|a, 0\rangle$ Floquet state and the $|b, n\rangle$ Floquet state (shifted by $n$ drive photons).

---

## Why This Set Exists

Many cQED control problems are genuinely periodic-drive problems. Static dressed-state intuition is often not enough once a drive is strong enough to reorganize the spectrum. These notebooks provide a compact workflow for studying that regime without dropping down to ad hoc calculations.

They also connect directly to the repository's sideband tooling, so they can be used as a bridge between the standard time-domain workflow tutorials and the more specialized Floquet API.

---

## Related References

- [Floquet Analysis API](../api/floquet.md)
- [User Guide: Floquet Analysis](../user_guides/floquet_analysis.md)
- [Dai 2025 Intrinsic Resonances](dai_2025_intrinsic_multiphoton_resonances.md)
- [Dai 2025 Readout-Assisted Resonances](dai_2025_readout_assisted_floquet_resonances.md)
- [Sideband Swap](sideband_swap.md)
