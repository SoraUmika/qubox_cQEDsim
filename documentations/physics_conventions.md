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

## Holographic Quantum-Algorithm Conventions

The generic holographic package in `cqed_sim.quantum_algorithms.holographic_sim`
uses a different tensor structure from the main cQED runtime:

$$|\sigma\rangle_{\text{physical}} \otimes |b\rangle_{\text{bond}}$$

The **physical register is first** and the persistent **bond register is second**.

For a dense joint unitary $U$ on `physical ⊗ bond`, the implemented channel
prepares a fixed physical reference state $\lvert 0 \rangle$ by default and uses

$$
\rho_{\text{bond}} \mapsto \sum_\sigma K_\sigma \rho_{\text{bond}} K_\sigma^\dagger,
\qquad
K_\sigma = \langle \sigma | U | 0 \rangle
$$

When the channel is constructed from a right-canonical MPS tensor, the stored MPS
matrices $V_\sigma$ are converted to standard channel-orientation Kraus operators
through

$$K_\sigma = V_\sigma^\dagger$$

`burn-in` means repeated application of the same holographic channel before the
observable insertions begin. Observable schedules always refer to measurements
performed on the physical register.

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

## RL Wrapper Conventions

The RL-facing stack in `cqed_sim.rl_control` does not define a second simulator convention set. It reuses the same runtime model, pulse, frame, measurement, and solver semantics as the rest of the package.

- `HybridSystemConfig` keeps all Hamiltonian coefficients in `rad/s` and all durations in `s`.
- When `use_model_rotating_frame=True` and the configured `FrameSpec` is left at zero, the environment convenience layer uses the model's bare storage and qubit frequencies as the rotating frame. This is a numerical convenience, not a new physical convention.
- The reduced regime uses `DispersiveTransmonCavityModel`; the fuller pulse regime uses `UniversalCQEDModel` with the same tensor ordering and sign conventions already documented on this page.
- Measurement-like observations and rewards are derived from `QubitMeasurementSpec` and therefore inherit the same confusion-matrix convention: latent `(g, e)` probabilities are mapped into reported `(g, e)` probabilities by left multiplication with the `2 x 2` confusion matrix.
- `render_diagnostics()` exposes simulator-only quantities such as the full state, reduced states, Wigner grids, and compiled pulses. Those diagnostics are intentionally richer than the measurement-like observation returned to a policy.

---

## Optimal-Control Conventions

The direct optimal-control layer in `cqed_sim.optimal_control` does not introduce a second Hamiltonian or waveform convention set. It reuses the same model operators, frame semantics, and pulse-runtime sign conventions already documented on this page.

- All drift and control Hamiltonian coefficients remain in `rad/s`, and all slice durations remain in `s`.
- The current backend is a dense closed-system GRAPE solver for piecewise-constant controls.
- Model-backed control problems are built from the existing static Hamiltonian and drive-operator helpers rather than from a separate tensor-ordering or Hamiltonian-assembly path.
- Exported rotating-frame controls use the same complex baseband convention as the pulse runtime:

$$c(t) = I(t) - i Q(t)$$

- That export rule is what makes the optimized real-valued Hermitian `I/Q` control channels replay correctly through standard `Pulse`, `SequenceCompiler`, and `simulate_sequence(...)` workflows.
- Leakage penalties are defined relative to retained logical subspaces in the same truncated Hilbert space used by the rest of the simulator.
- Simulator-backed replay through `evaluate_control_with_simulator(...)` or `ControlResult.evaluate_with_simulator(...)` is an evaluation path only. It replays the optimized schedule through the standard runtime with optional `NoiseSpec` Lindblad terms and reports the resulting objective probe-state fidelities.
- For retained-subspace unitary objectives, that replay path also reports subspace leakage. This is a runtime diagnostic, not a claim that the current GRAPE optimizer itself is doing open-system optimization.

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

When that inferred rate vanishes, the helper returns `None`, which the runtime interprets
as "do not add an extra pure-dephasing term".

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

---

## Gate Library (`cqed_sim.gates`)

The `cqed_sim.gates` subpackage provides ideal unitary gates for all
subsystem types. The conventions below are enforced in the implementation
and validated by `tests/test_42_gates.py`.

### Single-Qubit Gates

All single-qubit gates act on the basis $\{|g\rangle, |e\rangle\}$ with
$|g\rangle = \text{basis}(2,0)$, $|e\rangle = \text{basis}(2,1)$.

Rotation convention:
$$R_{\hat{n}}(\theta) = \exp\!\left(-i\frac{\theta}{2}\,\hat{n}\cdot\vec{\sigma}\right)$$

Equatorial rotation (`rphi`):
$$R_\phi(\theta) = \exp\!\left[-i\frac{\theta}{2}\left(\cos\phi\,X + \sin\phi\,Y\right)\right]$$

At $\phi=0$ this equals $R_x(\theta)$; at $\phi=\pi/2$ it equals $R_y(\theta)$.

Named gates are *exact ideal matrices*, not Hamiltonian-generated approximations:

| Gate | Matrix |
|---|---|
| $X$ | $\begin{pmatrix}0&1\\1&0\end{pmatrix}$ |
| $Y$ | $\begin{pmatrix}0&-i\\i&0\end{pmatrix}$ |
| $Z$ | $\begin{pmatrix}1&0\\0&-1\end{pmatrix}$ |
| $H$ | $\frac{1}{\sqrt{2}}\begin{pmatrix}1&1\\1&-1\end{pmatrix}$ |
| $S$ | $\text{diag}(1, i)$ |
| $T$ | $\text{diag}(1, e^{i\pi/4})$ |

### Multilevel Transmon Transition-Selective Gates

General adjacent-level rotation between levels $j$ and $k$ in a `dim`-level
transmon:
$$R^{j,k}_\phi(\theta) = \exp\!\left[-i\frac{\theta}{2}\left(e^{-i\phi}|j\rangle\langle k| + e^{i\phi}|k\rangle\langle j|\right)\right]$$

This is an *ideal unitary* (exact matrix exponential). All levels outside
$\{|j\rangle, |k\rangle\}$ are unaffected (identity on the complement).

Aliases:

- `r_ge(θ, φ, dim=3)` — $R^{0,1}_\phi(\theta)$. At `dim=2` matches the qubit `rphi`.
- `r_ef(θ, φ, dim=3)` — $R^{1,2}_\phi(\theta)$.

### Bosonic Cavity Gates

All bosonic gates are *exact ideal unitaries* within the Fock-space truncation.

| Gate | Defining equation | Notes |
| --- | --- | --- |
| Displacement | $D(\alpha) = \exp(\alpha a^\dagger - \alpha^* a)$ | Ergonomic wrapper of `qt.displace` |
| Oscillator rotation | $R(\theta) = \exp(-i\theta a^\dagger a)$ | $\lvert n\rangle \to e^{-in\theta}\lvert n\rangle$ |
| Parity | $\Pi = \exp(i\pi a^\dagger a)$ | $\lvert n\rangle \to (-1)^n\lvert n\rangle$. Equal to $R(-\pi)$. |
| Squeezing | $S(\zeta) = \exp\!\left(\tfrac{1}{2}\zeta^* a^2 - \tfrac{1}{2}\zeta(a^\dagger)^2\right)$ | Uses `qt.squeeze` |
| Self-Kerr evolution | $U_K(t) = \exp\!\left[-i\tfrac{K}{2}t\,\hat{n}(\hat{n}-1)\right]$ | Diagonal; see `kerr_evolution` |
| SNAP | $S(\{\phi_n\}) = \sum_n e^{i\phi_n}\lvert n\rangle\langle n\rvert$ | Accepts `{n: phase}` dict or dense array |

**Sign convention for `kerr_evolution`:** the phase accumulated by Fock state
$|n\rangle$ is $\exp[-i(K/2)t\,n(n-1)]$. Here $K$ is the same sign as in the
Hamiltonian $+K/2\,a^{\dagger 2}a^2$ used throughout the simulator (typically
$K < 0$ for a transmon-like self-Kerr).

### Qubit-Cavity Conditional and Interaction Gates

**Tensor ordering: qubit first, cavity second.**
Basis order: $|g,0\rangle, |g,1\rangle, \ldots, |e,0\rangle, \ldots$

#### Dispersive-phase gate — two conventions

The `convention` parameter of `dispersive_phase` selects one of:

**`convention="n_e"` (default, matches `UniversalCQEDModel`):**

$$H = \chi\,\hat{n}_\text{cav}\,\lvert e\rangle\langle e\rvert$$

$$U = \lvert g\rangle\langle g\rvert\otimes I + \lvert e\rangle\langle e\rvert\otimes e^{-i\chi t\hat{n}}$$

**`convention="z"` (Pauli-Z style):**

$$H = \tfrac{\chi}{2}\,\hat{n}_\text{cav}\,Z$$

$$U = e^{-i\chi t\hat{n}/2}\otimes\lvert g\rangle\langle g\rvert + e^{+i\chi t\hat{n}/2}\otimes\lvert e\rangle\langle e\rvert$$

The `"n_e"` convention matches the dispersive term used in `UniversalCQEDModel`
($+\chi n_c n_q$ with $n_q = |e\rangle\langle e|$). The two conventions differ
by a photon-number-dependent global phase on each qubit branch; the *relative*
dispersive phase between the two branches is the same.

#### Conditional displacement — two call forms

$$CD(\alpha)
= |g\rangle\langle g|\otimes D(+\alpha) + |e\rangle\langle e|\otimes D(-\alpha)
\quad\text{(symmetric form)}$$

$$CD(\alpha_g, \alpha_e)
= |g\rangle\langle g|\otimes D(\alpha_g) + |e\rangle\langle e|\otimes D(\alpha_e)
\quad\text{(general form)}$$

#### SQR tensor ordering

`sqr` and `multi_sqr` use **cavity first, qubit second**
(natural for $|n\rangle\langle n|\otimes R_\phi(\theta)$):

$$U_\text{SQR}(\theta,\phi;n) = |n\rangle\langle n|\otimes R_\phi(\theta) + \sum_{m\neq n}|m\rangle\langle m|\otimes I$$

The legacy `sqr_op(thetas, phis)` in `cqed_sim.core.ideal_gates` uses **qubit
first, cavity second** with a dense array interface only.

#### Jaynes–Cummings and blue-sideband gates

Qubit first, cavity second.
$$H_\text{JC} = g\!\left(\sigma_+\otimes a + \sigma_-\otimes a^\dagger\right), \quad
U_\text{JC}(t) = e^{-itH_\text{JC}}$$
$$H_\text{blue} = g\!\left(\sigma_+\otimes a^\dagger + \sigma_-\otimes a\right), \quad
U_\text{blue}(t) = e^{-itH_\text{blue}}$$

These are *Hamiltonian-generated unitaries* (computed via matrix exponential of
the full operator in the truncated Hilbert space).

#### Beam splitter

Mode a first, mode b second.
$$H_\text{BS} = g\!\left(a^\dagger b + ab^\dagger\right), \quad
U_\text{BS}(t) = e^{-itH_\text{BS}}$$

`beam_splitter(g, t, dim_a, dim_b)` is equivalent to
`beamsplitter_unitary(dim_a, dim_b, g*t)`.

### Two-Qubit Gates

Control first, target second. Basis order: $|gg\rangle, |ge\rangle, |eg\rangle, |ee\rangle$.

All two-qubit gates are *exact ideal matrices* (not Hamiltonian-generated).

| Gate | Type |
|---|---|
| CNOT, CZ, CP(φ) | Ideal unitary matrices |
| SWAP, iSWAP, √iSWAP | Ideal unitary matrices |

Note: iSWAP and √iSWAP can be approximately generated by an exchange
Hamiltonian $H = J(\sigma_+\sigma_- + \sigma_-\sigma_+)$, but `iswap_gate()`
and `sqrt_iswap_gate()` return the exact target matrices regardless of
physical generation pathway.

---

## Cross-Reference

The canonical, detailed physics reference is:

```text
physics_and_conventions/physics_conventions_report.tex
```

All APIs in `cqed_sim` follow the conventions documented there. If any discrepancy is found between the code and that document, it should be treated as a bug.
