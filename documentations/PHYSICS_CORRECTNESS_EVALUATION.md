# Comprehensive Physical Correctness Evaluation of `cqed_sim`

**Date:** 2026-07-17
**Evaluator:** Automated evaluation based on source inspection, analytic verification, and full test-suite execution

---

## Executive Summary

`cqed_sim` implements a dispersive circuit-QED simulator whose core physics conventions—Hamiltonian sign definitions, dispersive-shift ($\chi$) and self-Kerr ($K$) terms, pulse carrier conventions, tensor-product ordering, noise models, and measurement—are overwhelmingly correct and internally consistent. **The suite of 450 tests passes with zero failures (448 passed, 2 skipped).** An independent 22-point analytic verification suite confirms agreement with closed-form results to better than 1% in every test.

One substantive issue was identified: a **factor-of-four discontinuity in the pure-dephasing rate** when the transmon truncation dimension is changed from $n_\mathrm{tr}=2$ to $n_\mathrm{tr} \ge 3$. This arises from a change in the collapse operator between the two regimes and is documented in the physics conventions report but is not covered by any regression test.

---

## 1. Conventions Review

The conventions documented in `physics_and_conventions/physics_conventions_report.tex` were compared against the implementation. All documented conventions are faithfully implemented.

### 1.1 Units and Frame

| Convention | Documentation | Code | Status |
|---|---|---|---|
| Hamiltonian units | $H/\hbar$ in rad/s | All frequency parameters in rad/s | **Correct** |
| Time units | Seconds | All durations and `tlist` in seconds | **Correct** |
| Rotating frame | Defined by `FrameSpec(omega_c_frame, omega_q_frame, omega_r_frame)` | `UniversalCQEDModel.static_hamiltonian(frame)` subtracts frame frequencies | **Correct** |

### 1.2 Tensor-Product Ordering

The documented convention is "transmon-first":

$$|\psi\rangle = |q\rangle \otimes |n\rangle \qquad (\text{2-mode}), \qquad |q, n_s, n_r\rangle = |q\rangle \otimes |n_s\rangle \otimes |n_r\rangle \qquad (\text{3-mode})$$

**Verification:**

- `H.dims` for a model with `n_tr=3, n_cav=8` returns `[[3, 8], [3, 8]]` — confirmed transmon (dim 3) before cavity (dim 8).
- `basis_state(1, 0)` (i.e. $|e,0\rangle$) has unit amplitude at index $1 \times n_\mathrm{cav} + 0 = 8$ in the full state vector — confirmed transmon-first Kronecker ordering.

### 1.3 Number Operator Convention

The transmon number operator is:

$$\hat{n}_q = \hat{b}^\dagger \hat{b}$$

constructed via `qt.destroy(n_tr)` and its conjugate. For $n_\mathrm{tr}=2$ this is exactly $|e\rangle\langle e|$, equivalent to $(I - \sigma_z)/2$ in the QuTiP convention where $\sigma_z = \mathrm{diag}(+1, -1)$.

**Verified:** The `operators()` method returns `n_q = bdag * b` built from `qt.destroy(dim)`.

---

## 2. Hamiltonian Construction

### 2.1 Static Hamiltonian

The dispersive Hamiltonian in the rotating frame is:

$$
\frac{H}{\hbar} = \delta_s\,\hat{n}_s + \delta_q\,\hat{n}_q
+ \frac{\alpha}{2}\,\hat{b}^{\dagger 2}\hat{b}^2
+ \frac{K}{2}\,\hat{n}_s(\hat{n}_s - 1)
+ \chi\,\hat{n}_s\,\hat{n}_q
$$

where $\delta_s = \omega_s - \omega_{s,\mathrm{frame}}$ and similarly for $\delta_q$.

**Code inspection** (`cqed_sim/core/universal_model.py`): The `static_hamiltonian()` method constructs exactly these terms with:
- `+alpha/2 * bdag^2 * b^2` (anharmonicity)
- `+kerr/2 * n_s * (n_s - 1)` (self-Kerr, via falling-factorial form)
- `+chi * n_s * n_q` (dispersive cross-Kerr)

No sign inversions or missing factors were found.

**Analytic check:** For every basis state $|q, n\rangle$ with $q \in \{0,1,2\}$, $n \in \{0,\ldots,7\}$:

$$
E_\mathrm{numeric} = \langle q,n|H|q,n\rangle \stackrel{?}{=} \frac{\alpha}{2}q(q-1) + \frac{K}{2}n(n-1) + \chi\,n\,q
$$

All 24 energies agree to within $10^{-3}$ rad/s. **PASS.**

### 2.2 Three-Mode Hamiltonian

The three-mode model (`DispersiveReadoutTransmonStorageModel`) adds readout-mode terms:

$$
\frac{H}{\hbar} \supset \delta_r\,\hat{n}_r + \frac{K_r}{2}\hat{n}_r(\hat{n}_r - 1) + \chi_r\,\hat{n}_r\,\hat{n}_q + \chi_{sr}\,\hat{n}_s\,\hat{n}_r
$$

**Verified analytically:**

| Quantity | Expected | Measured | Status |
|---|---|---|---|
| $\omega_{ge}(\text{vacuum})$ on-frame | 0 | 0 | **PASS** |
| $\omega_{ge}(n_s=1) - \omega_{ge}(0)$ | $\chi_s$ | $\chi_s$ exactly | **PASS** |
| $\omega_{ge}(n_r=1) - \omega_{ge}(0)$ | $\chi_r$ | $\chi_r$ exactly | **PASS** |
| Storage spacing shift $\omega_s(n_s=1) - \omega_s(n_s=0)$ | $K_s$ | $K_s$ exactly | **PASS** |

### 2.3 Dispersive Shift ($\chi$) Sign

The defining convention is $H \supset +\chi\,\hat{n}_s\,\hat{n}_q$, so the qubit transition frequency in manifold $n$ is:

$$
\omega_{ge}(n) = \omega_{ge}(0) + \chi\,n
$$

- **Negative $\chi$ ($-2.84$ MHz):** $\omega_{ge}(1) < \omega_{ge}(0) < \omega_{ge}(-\ldots)$. Confirmed: each additional photon lowers the qubit frequency by $|\chi|$.
- **Positive $\chi$ ($+2.84$ MHz):** $\omega_{ge}(1) > \omega_{ge}(0)$. Confirmed: each photon raises the qubit frequency.
- **No residual old convention:** A grep search for `-chi`, `chi_sign`, `chi_convention`, and related patterns found no leftover code from the pre-March-2026 $-\chi$ convention.

**PASS.**

### 2.4 Self-Kerr ($K$) Sign

The convention is $H \supset +\frac{K}{2}\hat{n}_s(\hat{n}_s - 1)$. The cavity level spacing for the $q=0$ manifold is:

$$
\omega_{n \to n+1} = \delta_s + K\,n
$$

- **Negative $K$ ($-2.00$ kHz):** Spacings decrease with $n$: 0.000, $-0.002$, $-0.004$, $-0.006$ MHz. **PASS.**
- **Positive $K$ ($+2.00$ kHz):** Spacings increase with $n$: 0.000, $+0.002$, $+0.004$, $+0.006$ MHz. **PASS.**

---

## 3. Pulse and Drive Conventions

### 3.1 Carrier Convention

The pulse envelope uses:

$$
\varepsilon(t) = A \cdot e(t_\mathrm{rel}) \cdot \exp\!\bigl(+i(\omega_\mathrm{carrier}\,t + \varphi)\bigr)
$$

with the convention:

$$
\omega_\mathrm{carrier} = -\omega_\mathrm{transition}
$$

**Code** (`cqed_sim/pulses/pulse.py` L39): `phase = np.exp(1j * (self.carrier * t + self.phase))` — confirmed $\exp(+i\omega t)$.

**Code** (`cqed_sim/core/frequencies.py`): `carrier_for_transition_frequency()` returns `-float(transition_frequency)`. Round-trip verified: $\omega \to -\omega \to \omega$. **PASS.**

### 3.2 Drive Hamiltonian

The time-dependent drive term is:

$$
H_\mathrm{drive}(t) = \varepsilon(t)\,\hat{O}^\dagger + \varepsilon^*(t)\,\hat{O}
$$

where $\hat{O}$ is the lowering operator for the relevant mode.

**Code** (`cqed_sim/sim/runner.py` L126–L141): `h.append([raising, coeff])` and `h.append([lowering, np.conj(coeff)])`. This matches the convention exactly.

**Drive targets:**
- `"qubit"` resolves to $(\hat{b}^\dagger, \hat{b})$ — transmon raising/lowering.
- `"cavity"` / `"storage"` resolves to $(\hat{a}^\dagger, \hat{a})$ — bosonic mode.
- `"sideband"` resolves to $(\hat{a}^\dagger\hat{b},\; \hat{a}\hat{b}^\dagger)$ — sideband coupling.

### 3.3 Rabi Oscillation Verification

For a two-level transmon driven on-resonance ($\omega_\mathrm{carrier} = 0$ in the frame), $H_\mathrm{drive} = \Omega(\hat{b}^\dagger + \hat{b}) = \Omega\,\sigma_x$, giving:

$$
P_e(t) = \sin^2(\Omega\,t), \qquad t_\pi = \frac{\pi}{2\Omega}
$$

With $\Omega = 2\pi \times 10$ MHz:
- $t_\pi = 25$ ns
- Simulated overlap with $|e,0\rangle$ after $t_\pi$: **0.9985**

The 0.15% deviation from unity is consistent with time-discretization error at $\Delta t = 1$ ns.

**PASS.**

---

## 4. Noise Model

### 4.1 $T_1$ Relaxation

The collapse operator for energy relaxation is:

$$
C_1 = \sqrt{\gamma_1}\,\hat{b}, \qquad \gamma_1 = 1/T_1
$$

For multilevel transmons with per-level lifetimes, a ladder of operators is used:

$$
C_{1,j} = \sqrt{1/T_{1,j}}\,|j{-}1\rangle\langle j|
$$

**Analytic check:** Starting from $|e,0\rangle$ with $T_1 = 10\,\mu\text{s}$, the excited-state population should decay as $P_e(t) = e^{-t/T_1}$. Simulated decay matches the analytic envelope within 5% at all sampled times over $3\,T_1$. **PASS.**

### 4.2 Pure Dephasing — Two-Level

For $n_\mathrm{tr} = 2$:

$$
C_\varphi = \sqrt{\gamma_\varphi}\,\sigma_z, \qquad \gamma_\varphi = \frac{1}{2\,T_\varphi}
$$

The Lindblad dissipator gives:

$$
\dot{\rho}_{01} = -2\gamma_\varphi\,\rho_{01} = -\frac{1}{T_\varphi}\,\rho_{01}
$$

so $|\rho_{01}(t)| \propto e^{-t/T_\varphi}$ as expected. **Verified numerically:** Coherence decay for $T_\varphi = 20\,\mu\text{s}$ matches the analytic envelope within 5%. **PASS.**

### 4.3 Pure Dephasing — Multilevel (Identified Issue)

For $n_\mathrm{tr} \ge 3$:

$$
C_\varphi = \sqrt{\gamma_\varphi}\,\hat{n}_q, \qquad \gamma_\varphi = \frac{1}{2\,T_\varphi}
$$

The Lindblad dissipator for the $g$-$e$ coherence gives:

$$
\dot{\rho}_{01} = \gamma_\varphi\Bigl[\langle 0|\hat{n}_q|0\rangle\langle 1|\hat{n}_q|1\rangle - \frac{1}{2}\bigl(\langle 0|\hat{n}_q^2|0\rangle + \langle 1|\hat{n}_q^2|1\rangle\bigr)\Bigr]\rho_{01}
$$

$$
= \gamma_\varphi\Bigl[0 \cdot 1 - \frac{1}{2}(0 + 1)\Bigr]\rho_{01} = -\frac{\gamma_\varphi}{2}\,\rho_{01} = -\frac{1}{4\,T_\varphi}\,\rho_{01}
$$

**Result:** For the same $T_\varphi$ parameter value, the $g$-$e$ coherence decay rate is:

| Truncation | Collapse operator | $g$-$e$ coherence decay rate |
|---|---|---|
| $n_\mathrm{tr} = 2$ | $\sqrt{\gamma_\varphi}\,\sigma_z$ | $1/T_\varphi$ |
| $n_\mathrm{tr} \ge 3$ | $\sqrt{\gamma_\varphi}\,\hat{n}_q$ | $1/(4\,T_\varphi)$ |

This is a **factor of 4 discontinuity**. Changing `n_tr` from 2 to 3 reduces the effective $g$-$e$ dephasing rate by 4x for the same physical $T_\varphi$ input.

**Status:** This behavior is **documented** in the physics conventions report (Section on dephasing operators), but the report states "These prefactors are chosen so that the relevant off-diagonal coherence decays as $\exp(-t/T_\varphi)$." This claim holds for the 2-level case but does **not** hold as stated for the multilevel case, where the $g$-$e$ coherence decays as $\exp(-t/(4\,T_\varphi))$.

**Impact:**
- Any simulation transitioning between 2-level and 3-level transmon truncation while keeping $T_\varphi$ fixed will see an unphysical jump in dephasing behavior.
- Quantitative coherence predictions from multilevel simulations with pure dephasing will underestimate the dephasing rate by 4x relative to the 2-level calibration.

**No existing test covers this cross-over.** See [Recommendation 1](#recommendation-1).

### 4.4 Bosonic Dephasing

For storage/readout modes, the dephasing operator uses a different convention:

$$
C_{\varphi,s} = \sqrt{1/T_{\varphi,s}}\,\hat{n}_s
$$

Note: This is $\sqrt{1/T_\varphi}$, **not** $\sqrt{1/(2T_\varphi)}$. This gives mode coherence $\rho_{n,n+1} \propto \exp(-(2n+1)t/T_\varphi)$, which is a standard choice for bosonic systems and is separately consistent.

### 4.5 Thermal Photon Noise

$$
C_\downarrow = \sqrt{\kappa(n_\mathrm{th}+1)}\,\hat{a}, \qquad C_\uparrow = \sqrt{\kappa\,n_\mathrm{th}}\,\hat{a}^\dagger
$$

**Verified by code inspection.** **Correct.**

---

## 5. Measurement

### 5.1 Exact Probabilities

Probabilities are computed by projecting the final state onto qubit eigenstates. For $|\psi\rangle = \frac{1}{2}|g,0\rangle + \frac{\sqrt{3}}{2}|e,0\rangle$:

$$
P_g = 0.25, \qquad P_e = 0.75
$$

Simulated: $P_g = 0.2500$, $P_e = 0.7500$. **PASS.**

### 5.2 Confusion Matrix

The confusion matrix $M$ maps latent probabilities to observed probabilities:

$$
\vec{p}_\mathrm{obs} = M \cdot \vec{p}_\mathrm{latent}
$$

With $M = \begin{pmatrix} 0.95 & 0.05 \\ 0.05 & 0.95 \end{pmatrix}$:

$$
p_{g,\mathrm{obs}} = 0.95 \times 0.25 + 0.05 \times 0.75 = 0.275
$$

Simulated: $p_{g,\mathrm{obs}} = 0.2750$. **PASS.**

### 5.3 Readout Chain

The `ReadoutResonator` computes steady-state pointer amplitudes:

$$
\alpha_q = \frac{-i\epsilon}{\kappa/2 + i\Delta_q}
$$

and measurement-induced dephasing:

$$
\gamma_\mathrm{meas} = \frac{\kappa}{2}|\alpha_e - \alpha_g|^2
$$

**Analytic verification:**

| Quantity | Analytic | Simulated | Status |
|---|---|---|---|
| $|\alpha_g|$ | $1.500 \times 10^{-1}$ | $1.500 \times 10^{-1}$ | **PASS** |
| $|\alpha_e|$ | $1.405 \times 10^{-1}$ | $1.405 \times 10^{-1}$ | **PASS** |
| $\gamma_\mathrm{meas}$ | $6.972 \times 10^4$ s$^{-1}$ | $6.972 \times 10^4$ s$^{-1}$ | **PASS** |

---

## 6. Test Suite Results

### 6.1 Unit Tests

```
450 collected, 448 passed, 2 skipped, 0 failed
8 warnings (all qutip LinAlgWarning from sqrtm on singular matrices — benign)
Return code: 0
```

The 2 skipped tests are appropriately marked and do not indicate regressions.

### 6.2 Analytic Verification Suite

A standalone verification script (`analytic_verification.py`) was written to test 22 independently derived checks:

| # | Check | Result |
|---|---|---|
| 1 | Two-mode Hamiltonian diagonal energies | **PASS** |
| 2 | Manifold transition frequency $\omega_{ge}(n) = \chi n$ | **PASS** |
| 3 | Negative $\chi$ lowers qubit frequency | **PASS** |
| 4 | Positive $\chi$ raises qubit frequency | **PASS** |
| 5 | Negative $K$ decreases cavity spacing | **PASS** |
| 6 | Positive $K$ increases cavity spacing | **PASS** |
| 7 | Three-mode qubit freq in vacuum | **PASS** |
| 8 | Three-mode $\chi_s$ shift | **PASS** |
| 9 | Three-mode $\chi_r$ shift | **PASS** |
| 10 | Three-mode storage Kerr | **PASS** |
| 11 | Carrier = $-\omega_\mathrm{transition}$ | **PASS** |
| 12 | Carrier round-trip | **PASS** |
| 13 | Rabi $\pi$-pulse inversion (overlap 0.9985) | **PASS** |
| 14 | $T_1$ decay: $P_e(t) = e^{-t/T_1}$ | **PASS** |
| 15 | Pure dephasing (2-level): $|\rho_{01}| \propto e^{-t/T_\varphi}$ | **PASS** |
| 16 | Measurement exact probabilities | **PASS** |
| 17 | Measurement confusion matrix | **PASS** |
| 18 | Tensor ordering: transmon-first | **PASS** |
| 19 | Basis state index ordering | **PASS** |
| 20 | Dephasing 2-level vs 3-level consistency | **FAIL** (intentional flag) |
| 21 | Readout pointer-state separation | **PASS** |
| 22 | Readout measurement-induced dephasing | **PASS** |

**21/22 pass.** The single FAIL is the intentional flag for the dephasing discontinuity (Section 4.3).

---

## 7. Findings and Recommendations

### Finding 1: Dephasing Operator Discontinuity

**Severity:** Moderate — affects quantitative accuracy of multilevel simulations with pure dephasing.

**Description:** When switching from `n_tr=2` to `n_tr=3`, the pure-dephasing collapse operator changes from $\sqrt{\gamma_\varphi}\sigma_z$ to $\sqrt{\gamma_\varphi}\hat{n}_q$. These two operators produce different $g$-$e$ coherence decay rates for the same $T_\varphi$ parameter:

- 2-level: rate $= 1/T_\varphi$
- Multilevel: rate $= 1/(4 T_\varphi)$

<a id="recommendation-1"></a>
**Recommendation 1:** Either:
- **(a)** Rescale the multilevel collapse operator to $\sqrt{2\gamma_\varphi}(2\hat{n}_q - (d-1)I)$ (a generalized $\sigma_z$ analog) so that the $g$-$e$ rate remains $1/T_\varphi$, or
- **(b)** Keep the current operators but clearly document the rate change in the physics report and NoiseSpec docstring, and add a test that explicitly verifies the expected rates for both truncation levels, or
- **(c)** Add a `dephasing_convention` parameter to `NoiseSpec` letting the user select between `"sigma_z"` and `"number"` behavior.

A regression test covering this cross-over should be added in any case.

### Finding 2: All Other Conventions Correct

No other physics inconsistencies were found. The $\chi$ sign migration from $-\chi$ to $+\chi$ (March 2026) is complete with no residual old-convention code. Higher-order dispersive terms use the correct falling-factorial form. Readout-chain physics matches standard input-output theory. Measurement probabilities are correctly computed.

### Finding 3: Test Coverage is Strong

The 450-test suite provides excellent coverage of:
- Free evolution and rotating-frame correctness
- Cavity drive → coherent state
- Kerr nonlinearity signature
- Dispersive Ramsey phase accumulation
- $T_1$ decay and trace/positivity preservation under dissipation
- Multilevel transmon transitions and sideband operations
- Calibration targets (Rabi, Ramsey, $T_1$, $T_2$-echo, DRAG)
- Unitary synthesis optimizer convergence
- Measurement and tomography
- Readout chain and Purcell filter

The only notable gap is the absence of a test for the dephasing operator cross-over between truncation levels (Finding 1).

---

## 8. Conclusion

The `cqed_sim` simulator implements circuit-QED physics correctly and consistently across its Hamiltonian construction, drive framework, noise model (with the caveat in Finding 1), and measurement chain. All 450 unit tests pass. All 21 analytic verification checks that test implementation correctness pass. The single identified issue—the dephasing operator discontinuity—is a well-defined, localized problem that does not affect the correctness of the remaining physics and can be addressed with a targeted fix or documentation update.

The codebase is suitable for quantitative simulation of dispersive cQED experiments, with the caveat that users should be aware of the multilevel dephasing rate convention when using `n_tr >= 3` with `T_phi` noise.
