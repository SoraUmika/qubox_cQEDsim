# SQR Block-Phase Internal Note

## 1. Problem Statement

The original question was whether a multitone SQR pulse could realize the full truncated-space target

\[
U_{\mathrm{target}} =
\sum_{n=0}^{N}
e^{i \lambda_n}
\lvert n \rangle \langle n \rvert
\otimes
R_n(\theta_n, \phi_n)
\]

on the joint qubit+cavity Hilbert space.

The important distinction was:

- Matching the conditional qubit action inside each Fock-resolved `2 x 2` block is only an `SU(2)` problem.
- Matching the full joint unitary also requires the correct inter-Fock block phases `lambda_n`.

The first study showed that these are not the same problem. The follow-up showed that, in the qubit-only fixed-duration setting, they are structurally decoupled.

## 2. Model and Conventions

- Tensor ordering: `qubit tensor cavity`.
- Internal simulator frequency units: rad/s.
- Reported physical parameters: `chi / 2pi = -2.84 MHz`, `K / 2pi = -30 kHz`.
- Base Hamiltonian:

\[
H_0 =
\omega_c a^\dagger a
+
\omega_q n_q
-
\chi a^\dagger a\, n_q
+
\frac{K}{2} a^{\dagger 2} a^2
\]

  where the code path actually uses `n_q = b^\dagger b`, not `sigma_z`, inside the truncated two-level qubit model.
- Main truncations studied: `N = 2, 3, 4`, with the most detailed studies at `N = 3`.
- Pulse families tested across the two completed studies:
  - naive multitone SQR,
  - extended per-tone multitone SQR,
  - chirped/loop-capable multitone SQR,
  - piecewise IQ,
  - fixed-total segmented wait-then-correct IQ,
  - composite phase-switched multitone.
- Main diagnostics:
  - full truncated-space fidelity,
  - block-gauge / block-SU(2) fidelity,
  - extracted block phases,
  - phase RMS relative to target,
  - explicit post-SNAP benchmark,
  - off-block norm,
  - local sensitivity of block phase vs block-SU(2) metrics.

## 3. Summary of the First Study

The first proof-of-concept established three points.

- Qubit-only SQR could often match the conditional block `SU(2)` action reasonably well on small truncations.
- Some durations landed naturally close to useful Kerr-like phase patterns.
- Across naive, extended, chirped, and phase-aware SQR families, the extracted relative block phases stayed essentially unchanged at fixed duration.

Representative examples from `outputs/sqr_block_phase_study`:

- `uniform_pi_zero_N3_T700ns`: naive `F_full = 0.4591`, best phase-aware `F_full = 0.8727`, but the implemented relative block phase stayed pinned near `[0, 0.0199, 0.1720, 0.4565]`.
- `uniform_pi_kerr_cancel_N3_T700ns`: best phase-aware run reached `F_full = 0.9681`, but again by landing near the native drift phase, not by independently steering it.

That suggested the useful block phase was inherited from the drift rather than synthesized as a new degree of freedom by the qubit-only pulse family.

## 4. Summary of the Follow-Up Study

The follow-up in `outputs/sqr_block_phase_followup` was designed to separate "poor ansatz" from "structural limitation".

The key results were:

- Highly flexible fixed-duration families still failed to move the implemented block phase away from the drift baseline.
- Phase-steering distances were numerically negligible:
  - phase-aware multitone: `5.77e-08 rad`,
  - piecewise IQ: `3.53e-06 rad`,
  - segmented wait IQ: `2.62e-06 rad`,
  - composite multitone: `8.31e-06 rad`.
- Retargeting the objective to a deliberately far phase profile changed the optimizer result for the `SU(2)` blocks but not the implemented block phase:
  - multitone retargeting moved the phase by only `3.32e-08 rad`,
  - piecewise IQ retargeting moved it by effectively `0`.
- The duration/truncation diagnostic matched the static-trace prediction on `N = 2, 3, 4` and `T = 450, 700, 1000 ns` down to `1e-8` to `1e-5 rad`.
- The local sensitivity diagnostic showed that block-phase directions were tiny compared to block-`SU(2)` directions:
  - for the multitone family, `||J_phase||_F = 5.16e-08` versus `||J_SU2||_F = 1.96e-01`.

This is no longer well explained as an optimizer failure.

## 5. Structural Interpretation

The likely reason is simple.

- In each Fock sector `n`, the qubit-only drive contributes only traceless `b + b_dag` terms inside the corresponding `2 x 2` block.
- Those controls can reshape the block `SU(2)` content.
- But they do not independently change the block determinant at fixed total time.
- The determinant phase therefore remains fixed by the static block trace and total duration.

That is exactly what the numerical follow-up found.

The right interpretation is:

- qubit-only SQR controls the conditional `SU(2)` part,
- drift plus total duration sets the block phase,
- and fixed-duration qubit-only SQR does not provide arbitrary built-in SNAP control.

## 6. Practical Implication

The default design route should therefore be:

\[
\text{timing / drift co-design}
\;+\;
\text{high-quality qubit-only SQR for the block SU(2)}
\;+\;
\text{small explicit SNAP cleanup}.
\]

This is Route B.

The important practical shift is that the duration choice should be treated as a compiler-level phase-shaping variable, not as something the qubit drive can later "repair".

## 7. Route B Prototype

The Route B prototype is implemented in `examples/sqr_route_b_enlarged_control.py`, with outputs in `outputs/sqr_route_b_enlarged_control`.

Three required representative targets plus one extra stress case were tested:

- uniform `pi` rotations with zero target phase,
- structured `(theta_n, phi_n)` rotations with zero target phase,
- uniform `pi` rotations with quadratic target phase,
- uniform `pi` rotations with a designed non-drift phase pattern.

Key observations:

- `structured_zero`: moving from a naive fast `450 ns` reference to a co-designed `700 ns` choice improved post-SNAP fidelity from `0.8512` to `0.9689`, while reducing residual SNAP RMS from `1.5429` to `0.2441 rad`.
- `uniform_pi_quadratic`: `450 ns -> 700 ns` improved post-SNAP fidelity from `0.5429` to `0.7695`, while reducing residual SNAP RMS from `3.5822` to `0.1050 rad`.
- `uniform_pi_zero`: `450 ns -> 700 ns` reduced SNAP burden strongly (`1.4550 -> 0.2441 rad`) but slightly lowered post-SNAP fidelity (`0.8079 -> 0.7697`).
- `uniform_pi_random`: a pure phase-distance choice (`500 ns`) improved post-SNAP fidelity (`0.7863 -> 0.8997`) but increased the residual correction RMS (`1.8426 -> 3.9431 rad`).

The practical lesson is that Route B should be implemented as a small Pareto search, not as "minimize phase distance only".

Recommended Route B workflow:

1. Scan a small duration/timing grid using the static drift phase model.
2. Optimize the qubit-only SQR on a shortlist of candidate timings.
3. Evaluate the Pareto tradeoff between:
   - post-SNAP final fidelity,
   - residual SNAP correction RMS / max amplitude,
   - and raw SQR-only fidelity.
4. Choose the operating point that balances SNAP burden and total fidelity for the experiment at hand.

## 8. Remaining Open Question

The main open question is no longer whether qubit-only SQR can synthesize arbitrary built-in SNAP phase.

The real open question is:

> What is the minimal enlargement of the control set that begins to recover independent block-phase synthesis?

## 9. Enlarged-Control Study

A final targeted enlarged-control study was run in the same script.

Two enlargements were tested:

1. A minimal cavity-side displacement conjugation

\[
D(-\alpha)\, U_{\mathrm{SQR}}\, D(\alpha)
\]

as the smallest cavity-assisted primitive that was easy to approximate cleanly in the present environment.

2. An explicit cavity-diagonal SNAP-like phase assist applied to the same qubit-only SQR baseline.

### 9.1 Displacement Conjugation

This did not materially recover arbitrary block-phase control.

- For `zero`, `kerr_cancel`, `random_small_a`, and `random_medium_a`, the optimized displacement stayed essentially at zero and gave the same fidelity as the qubit-only baseline.
- For `natural_drift`, it produced only a tiny numerical increase (`0.7697 -> 0.7700`) while adding nonzero off-block norm (`~2.1e-2`).
- For `far_from_drift`, it improved `F_full` only from `0.0456` to `0.0490`, while generating large off-block structure (`off_block_norm ~ 2.48`) and still not recovering the target phase pattern.

So a generic minimal cavity displacement is not enough.

### 9.2 Explicit SNAP Assist

This immediately restored the missing phase degree of freedom.

- `zero`: `0.7450 -> 0.7697`
- `random_small_a`: `0.7345 -> 0.7697`
- `random_medium_a`: `0.7374 -> 0.7697`
- `far_from_drift`: `0.0456 -> 0.8789`

The post-assist fidelity is then limited mainly by the underlying qubit-only block-`SU(2)` quality, not by the block phase.

## 10. Final Recommendation

The current evidence supports the following practical recommendation.

- Route B should remain the default workflow.
- The qubit-only fixed-duration limitation should now be treated as structural, not as an optimizer limitation.
- Timing co-design is useful, but it is a low-dimensional compiler knob, not arbitrary block-phase control.
- A minimal generic cavity displacement does not materially unlock the missing phase degree of freedom.
- If arbitrary one-shot per-Fock phase synthesis is required, cavity-assisted control is likely necessary.
- Within the current simulator stack, the smallest effective enlargement is an explicit cavity-diagonal phase primitive, i.e. a SNAP-like assist.

In short:

> Use Route B by default. Add explicit SNAP cleanup when arbitrary block phase matters. Reserve more ambitious cavity-assisted control development for tasks that genuinely need one-shot arbitrary per-Fock phase synthesis.
