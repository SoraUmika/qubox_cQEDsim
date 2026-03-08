# Block-Phase Controllability Follow-Up

## Model and Conventions

- Tensor ordering: `qubit otimes cavity` throughout.
- Internal Hamiltonian units: angular frequency in rad/s. Reported user-facing `chi / 2pi` and `K / 2pi` are in Hz.
- Parameters: `chi / 2pi = -2.840 MHz`, `K / 2pi = -30.0 kHz`.
- Base Hamiltonian: dispersive `chi` plus cavity self-Kerr, with qubit-only drive control.

## Structural Result

- For each fixed Fock sector `n`, the qubit-only drive contributes only traceless `b + b_dag` terms inside the `2 x 2` block.
- Therefore `Tr H_n(t)` is set entirely by the static drift block, so `det U_n(T)` and the block phase `lambda_n` are fixed by the static block trace and the total duration `T`.
- Consequence: at fixed total duration, no qubit-only pulse family in this simulator can independently steer the relative block phases. Pulse-shape freedom only changes the SU(2) part of each block.

## New Pulse Families Tested

- Phase-aware multitone SQR with per-tone amplitude, phase, and detuning freedom.
- Piecewise-constant IQ control with ten segments over the full gate.
- Segmented piecewise IQ with a fixed inserted wait window and active correction segments.
- A simple composite two-segment phase-switched multitone sequence.

## Fixed-Duration Evidence

- `piecewise_iq`: `F_full=0.9671`, `F_block=1.0000`, `phase_RMS=0.2441 rad`, `steering_from_drift=3.527e-06 rad`, `max|lambda_impl-lambda_pred|=6.997e-06 rad`.
- `D_extended_phase`: `F_full=0.7450`, `F_block=0.7697`, `phase_RMS=0.2441 rad`, `steering_from_drift=5.770e-08 rad`, `max|lambda_impl-lambda_pred|=1.152e-07 rad`.
- `A_naive`: `F_full=0.4591`, `F_block=0.4709`, `phase_RMS=0.2441 rad`, `steering_from_drift=6.652e-08 rad`, `max|lambda_impl-lambda_pred|=1.329e-07 rad`.
- `piecewise_wait`: `F_full=0.2818`, `F_block=0.9874`, `phase_RMS=0.2441 rad`, `steering_from_drift=2.625e-06 rad`, `max|lambda_impl-lambda_pred|=5.203e-06 rad`.
- `composite_multitone`: `F_full=0.1573`, `F_block=0.3056`, `phase_RMS=0.2441 rad`, `steering_from_drift=8.311e-06 rad`, `max|lambda_impl-lambda_pred|=1.644e-05 rad`.

These runs cover much more flexible fixed-duration families than the first pass, but the extracted relative block phases still stay numerically pinned to the static-trace prediction.

## Retargeting Spot Checks

- `D_extended_phase` retargeted away from zero-phase: `F_full=0.3194`, `phase_RMS=0.9663 rad`, `phase_shift_vs_zero_target=3.323e-08 rad`.
- `piecewise_iq` retargeted away from zero-phase: `F_full=0.4202`, `phase_RMS=0.9663 rad`, `phase_shift_vs_zero_target=0.000e+00 rad`.

Changing the phase objective changes the optimizer outcome for the SU(2) blocks, but not the implemented fixed-T block phase itself.

## Reachability Scan

- `D_extended_phase`: maximum observed steering distance over the target scan was `5.770e-08 rad`. Near-drift targets reached `F_full=0.7697`; the farthest tested target dropped to `F_full=0.3190`.
- `piecewise_iq`: maximum observed steering distance over the target scan was `3.527e-06 rad`. Near-drift targets reached `F_full=1.0000`; the farthest tested target dropped to `F_full=0.4202`.
- `piecewise_wait`: maximum observed steering distance over the target scan was `2.625e-06 rad`. Near-drift targets reached `F_full=0.2563`; the farthest tested target dropped to `F_full=0.2420`.
- `composite_multitone`: maximum observed steering distance over the target scan was `8.311e-06 rad`. Near-drift targets reached `F_full=0.1666`; the farthest tested target dropped to `F_full=0.1216`.

That is the central reachability result: the fixed-T reachable set in block-phase space is effectively a single point for each `(N, T)`.

## Sensitivity Diagnostic

- `D_extended_phase`: `||J_phase||_F=5.160e-08`, `||J_SU2||_F=1.962e-01`, ratio `2.629e-07`.
- `piecewise_iq`: `||J_phase||_F=1.075e-11`, `||J_SU2||_F=8.706e-11`, ratio `1.235e-01`.
- `piecewise_wait`: `||J_phase||_F=4.375e-12`, `||J_SU2||_F=2.849e-08`, ratio `1.536e-04`.

Locally, nearby parameter directions still move the SU(2) quality, but they do not open meaningful block-phase directions.

## Timing / Truncation Diagnostic

- Varying the total duration changes the block phases exactly as predicted by the static block traces. That provides a one-dimensional drift-assisted phase-shaping axis.
- The same pinning relation was verified on `N = 2, 3, 4` truncations.

## Bottom Line

- The lack of independent built-in SNAP control is not just a limitation of the original Gaussian/chirped ansatz families.
- Under the current truncated dispersive-plus-Kerr Hamiltonian with qubit-only drive and fixed total duration, the relative block phases are fundamentally pinned by drift.
- Segmented control, inserted waits at fixed total duration, and flexible piecewise IQ improve the SU(2) blocks but do not materially enlarge the reachable block-phase set.
- The practical interpretation is `Route B`: treat the native block phase as drift-assisted timing/compiler structure, then apply a smaller explicit SNAP-style correction when arbitrary per-Fock phase synthesis is required.
- If truly arbitrary block-phase synthesis is needed in one shot, the control set must be enlarged beyond qubit-only SQR, e.g. by a primitive that changes the block traces or an explicit cavity/SNAP-like operation.

## Files

- `outputs/sqr_block_phase_followup/fixed_duration_runs.csv`
- `outputs/sqr_block_phase_followup/reachability_table.csv`
- `outputs/sqr_block_phase_followup/retargeting_spotchecks.csv`
- `outputs/sqr_block_phase_followup/duration_truncation_table.csv`
- `outputs/sqr_block_phase_followup/sensitivity_table.csv`
- `outputs/sqr_block_phase_followup/summary.json`
