# Physics & Conventions

This page summarizes the canonical physics conventions used throughout `cqed_sim`. The authoritative source is `physics_and_conventions/physics_conventions_report.tex`.

!!! warning
    Do not assume sign conventions from other cQED packages. This simulator uses specific conventions for the dispersive shift, Kerr terms, and drive waveforms that must be respected for correct results.

---

## Units

| Quantity | Unit |
|---|---|
| Hamiltonian coefficients (ω_c, ω_q, χ, K, α) | rad/s |
| Time (t₀, duration, dt, delays) | seconds |
| Noise parameters T₁, T_φ | seconds |
| Decay rates κ | 1/s |
| Pulse carrier frequency | rad/s |
| Pulse phase | radians |

---

## Hilbert Space and Tensor Ordering

### Two-Mode (Qubit + Storage)

$$|q, n\rangle = |q\rangle_{\text{qubit}} \otimes |n\rangle_{\text{storage}}$$

Tensor ordering: **qubit first, storage second**.

Flat index: $\text{index}(q, n) = q \cdot n_{\text{cav}} + n$

### Three-Mode (Qubit + Storage + Readout)

$$|q, n_s, n_r\rangle = |q\rangle_{\text{qubit}} \otimes |n_s\rangle_{\text{storage}} \otimes |n_r\rangle_{\text{readout}}$$

Tensor ordering: **qubit first, storage second, readout third**.

### Universal Model

`UniversalCQEDModel` places the transmon first, followed by bosonic modes in declaration order.

---

## Computational Basis

- $|g\rangle = |0\rangle$ (ground state)
- $|e\rangle = |1\rangle$ (excited state)
- $\sigma_z|g\rangle = +|g\rangle$, $\sigma_z|e\rangle = -|e\rangle$ (QuTiP standard, unmodified)

The core Hamiltonian uses the **excitation number operator** $n_q = b^\dagger b$, not $\sigma_z/2$. For a two-level system:

$$n_q = |e\rangle\langle e| = \frac{I - \sigma_z}{2}$$

---

## Two-Mode Hamiltonian

The static Hamiltonian in the rotating frame is:

$$\frac{H_0}{\hbar} = \delta_c \, n_c + \delta_q \, n_q + \frac{\alpha}{2} \, b^{\dagger 2} b^2 + \frac{K}{2} \, n_c(n_c - 1) + \chi \, n_c \, n_q + \chi_2 \, n_c(n_c-1) \, n_q + \cdots$$

where:

- $\delta_c = \omega_c - \omega_c^{\text{frame}}$ — cavity detuning from rotating frame
- $\delta_q = \omega_q - \omega_q^{\text{frame}}$ — qubit detuning from rotating frame
- $\alpha$ — transmon anharmonicity (typically negative)
- $K$ — cavity self-Kerr
- $\chi$ — first-order dispersive shift

---

## Three-Mode Hamiltonian

$$\frac{H}{\hbar} = \delta_s \, n_s + \delta_r \, n_r + \delta_q \, n_q + \frac{\alpha}{2} b^{\dagger 2} b^2 + \chi_s \, n_s \, n_q + \chi_r \, n_r \, n_q + \chi_{sr} \, n_s \, n_r + \frac{K_s}{2} n_s(n_s-1) + \frac{K_r}{2} n_r(n_r-1)$$

---

## Dispersive Shift (χ)

**Convention:** The term $+\chi \, n_c \, n_q$ appears in the Hamiltonian with a **positive sign**.

The photon-number-dependent qubit transition frequency is:

$$\omega_{ge}(n) = \omega_{ge}(0) + \Xi(n)$$

where:

$$\Xi(n) = \chi \cdot n + \chi_2 \cdot n(n-1) + \chi_3 \cdot n(n-1)(n-2) + \cdots$$

**Physical meaning:**

- **Negative χ** → qubit transition frequency **decreases** with photon number
- **Positive χ** → qubit transition frequency **increases** with photon number

### Higher-Order Dispersive Terms

Higher-order coefficients use **falling-factorial** form, not ordinary polynomial powers:

- $\chi_2$ multiplies $n(n-1)$, not $n^2$
- $\chi_3$ multiplies $n(n-1)(n-2)$, not $n^3$

These are stored in `chi_higher = (chi_2, chi_3, ...)` on the model.

---

## Self-Kerr (K)

The cavity self-Kerr enters as:

$$+\frac{K}{2} \, n_c(n_c - 1)$$

- **Positive K** raises adjacent bosonic transition spacings
- **Negative K** (typical for superconducting cavities) lowers them

Higher-order Kerr terms follow the same falling-factorial convention stored in `kerr_higher`.

---

## Rotating Frame

`FrameSpec` defines the rotating frame:

```python
FrameSpec(
    omega_c_frame=...,  # Cavity/storage frame frequency (rad/s)
    omega_q_frame=...,  # Qubit frame frequency (rad/s)
    omega_r_frame=...,  # Readout frame frequency (rad/s)
)
```

- `FrameSpec(0, 0, 0)` is the **lab frame** (no rotation)
- Setting frame frequencies equal to model frequencies removes bare rotations

!!! note
    The storage-mode frame frequency is stored in the legacy field `omega_c_frame`. An alias `omega_s_frame` exists as a property for three-mode clarity.

---

## Energy Spectrum Reference

`compute_energy_spectrum(...)` and `model.energy_spectrum(...)` diagonalize the
model's static Hamiltonian in the selected frame and then subtract the bare
vacuum-state energy before reporting `EnergySpectrum.energies`.

The vacuum basis state means all subsystems are in level `0`:

- two-mode: `|g,0⟩`
- three-mode: `|g,0,0⟩`
- cavity-only: `|0⟩`

This keeps the reported spectrum anchored at a physically clear zero of energy
without changing which Hamiltonian was diagonalized. In rotating frames, some
vacuum-referenced excited levels can therefore appear at negative energy.

For intuitive ladder plots and absolute dressed level spacings, prefer the lab
frame `FrameSpec()`. For detuning-style analysis, a rotating frame can still be
useful, but the resulting energies must be interpreted in that frame.

---

## Drive Convention

### Complex Envelope

$$\varepsilon(t) = \text{amp} \cdot \text{envelope}(t_{\text{rel}}) \cdot e^{i(\text{carrier} \cdot t + \text{phase})}$$

where $t_{\text{rel}} = (t - t_0) / \text{duration}$.

**Sign convention:** The waveform uses $e^{+i\omega t}$ (positive exponent) throughout.

### Drive Hamiltonian

$$H_{\text{drive}} = \varepsilon(t) \, b^\dagger + \varepsilon^*(t) \, b$$

### Carrier–Transition Relationship

Because of the $e^{+i\omega t}$ sign convention, a transition at rotating-frame frequency $\omega_{\text{transition}}$ is resonantly addressed by:

$$\text{Pulse.carrier} = -\omega_{\text{transition}}$$

Use `carrier_for_transition_frequency(...)` and `transition_frequency_from_carrier(...)` to convert between these.

---

## Anharmonicity (α)

The transmon anharmonicity is defined as:

$$\frac{\alpha}{2} \, b^{\dagger 2} b^2$$

where $\alpha$ is typically **negative** for transmon qubits (e.g., $\alpha \approx 2\pi \times (-200\,\text{MHz})$).

---

## Dissipation

When open-system dynamics are enabled via `NoiseSpec`:

| Source | Collapse Operator |
|---|---|
| Qubit relaxation (aggregate) | $\sqrt{\gamma_1} \, b$ |
| Qubit relaxation (per-transition) | $\sqrt{1/T_{1,j}} \, \|j{-}1\rangle\langle j\|$ for each level |
| Qubit dephasing (two-level) | $\sqrt{\gamma_\phi} \, \sigma_z$ |
| Qubit dephasing (multilevel) | $\sqrt{\gamma_\phi} \, n_q$ |
| Storage pure dephasing | $\sqrt{1/T_{\phi,s}} \, n_s$ |
| Readout pure dephasing | $\sqrt{1/T_{\phi,r}} \, n_r$ |
| Cavity decay | $\sqrt{\kappa(n_{\text{th}}+1)} \, a$ |
| Cavity thermal excitation | $\sqrt{\kappa \cdot n_{\text{th}}} \, a^\dagger$ |

where $\gamma_1 = 1/T_1$ and $\gamma_\phi = 1/(2T_\phi)$.

For storage-mode Ramsey data, the helper `cqed_sim.sim.pure_dephasing_time_from_t1_t2(...)` uses

$$
\frac{1}{T_2} = \frac{1}{2T_1} + \frac{1}{T_\phi}
$$

so that

$$
\frac{1}{T_\phi} = \max\left(0, \frac{1}{T_2} - \frac{1}{2T_1}\right).
$$

Per-transition ancilla decay is activated by setting `NoiseSpec(transmon_t1=(T1_ge, T1_fe, ...))`.

---

## Sequential Sideband Reset Approximation

The staged storage-reset workflow uses the explicit three-mode storage-transmon-readout model together with two driven effective red sidebands:

- a storage sideband that transfers $|g,n_s,0_r\rangle \leftrightarrow |f,n_s-1,0_r\rangle$
- a readout sideband that transfers $|f,n_s,0_r\rangle \leftrightarrow |g,n_s,1_r\rangle$

The lossy readout mode is then emptied by the Lindblad readout decay rate $\kappa_r$. This is a controlled effective-model approximation of the experimental reset channel rather than an autonomous microscopic exchange Hamiltonian between the transmon and readout.

---

## Confusion Matrix Convention

For qubit measurement, the confusion matrix uses **(g, e) column ordering**:

$$p_{\text{obs}} = M \cdot p_{\text{latent}}$$

where $p_{\text{latent}} = (p_g, p_e)^T$.

---

## Cross-Reference

The canonical, detailed physics reference is:

```
physics_and_conventions/physics_conventions_report.tex
```

All APIs in `cqed_sim` follow the conventions documented there. If any discrepancy is found between the code and that document, it should be treated as a bug.
