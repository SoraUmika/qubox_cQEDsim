# SQR Built-In SNAP-like Phase Proof-of-Concept

## Conventions and Model

- Tensor ordering: `qubit otimes cavity`, with block indices selected via `qubit_cavity_block_indices(...)`.
- Internal simulator frequency units are rad/s. User-facing chi and Kerr values are reported in Hz.
- Hamiltonian terms enabled in this study: dispersive chi and cavity Kerr; `chi2 = chi3 = alpha = 0` for the initial proof-of-concept model.
- Drive family: multitone Gaussian qubit drive using the existing SQR tone builder and the existing QuTiP propagator path.

## Main Findings

- The naive baseline can keep the block-gauge fidelity above the true full-unitary fidelity, which confirms that endpoint-style rotation agreement is not sufficient for the stronger joint-unitary target.
  Representative naive zero-phase row: `F_full=0.5038`, `F_block-gauge=0.5065`, `phase_rms=0.1000 rad`.
- For the zero-phase target, the phase-aware optimizer improves the block SU(2) action, but it does not materially change the extracted per-Fock phase profile. The gap to the ideal post-SNAP benchmark therefore remains phase-limited.
  N=3 zero-phase, naive vs phase-aware: `F_full=0.4591 -> 0.8727`, `phase_rms=0.2441 -> 0.2441 rad`, `ideal-post-SNAP benchmark=0.9022`.
- For the Kerr-cancel target at 700 ns, the natural drift-induced phase profile already lies close to the Kerr-like target. The phase-aware optimizer then improves the full fidelity mainly by fixing the SU(2) blocks, not by independently steering the block phases.
  Chirped rotation-only vs phase-aware: `F_full=0.0013 -> 0.9681`, `phase_rms=0.0377 -> 0.0377 rad`.
- Comparing phase-aware extended detuning control to phase-aware chirped control indicates whether extra loop-capable freedom matters.
  `extended F_full=0.8623, phase_rms=0.0497 rad`; `chirped F_full=0.9681, phase_rms=0.0377 rad`.
- As the target rotations become more structured in `(theta_n, phi_n)`, the reachable fidelity drops relative to the uniform-pi cases. That is consistent with the larger simultaneous burden of matching both per-block SU(2) content and block phases.
  Best structured-target row: `D_extended_phase` with `F_full=0.9686`.

## Files

- Main summary table: `outputs/sqr_block_phase_study/summary_table.csv`
- Per-block table: `outputs/sqr_block_phase_study/block_table.csv`
- Full JSON payload: `outputs/sqr_block_phase_study/summary.json`

## Conclusion

On small truncated spaces, the evidence supports only a qualified and limited proof of concept.
The multitone SQR family can realize the desired conditional rotations reasonably well, and for certain durations it can exploit the natural chi/Kerr drift to land close to a useful Kerr-like block-phase pattern.
But across fixed `(N, T)` studies the extracted relative block phases are essentially unchanged across the naive, extended, chirped, and phase-aware constructions. That means the built-in phase is mostly inherited from the drift, not independently controllable.
The ideal explicit-SNAP benchmark still closes the remaining gap (`Delta F ~= 0.0005` on the best Kerr-target row), so a separate SNAP-like correction remains the cleaner route for arbitrary phase profiles.
