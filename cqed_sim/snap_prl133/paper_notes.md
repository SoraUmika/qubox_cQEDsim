# Landgraf PRL 133 (2024) Mapping Notes

Primary references:
- J. Landgraf et al., *Fast Quantum Control of Cavities Using an Improved Protocol without Coherent Errors*, PRL 133, 260802 (2024), DOI: 10.1103/PhysRevLett.133.260802.
- arXiv preprint: 2310.10498.
- APS supplemental material (consulted for protocol details and geometric error interpretation).

## Convention and Symbol Mapping

This implementation follows the project dispersive sign convention:

- `omega_ge(n) = omega_ge(0) - n * chi` when higher-order terms are off.

Paper-to-code variable mapping:

- paper `T` (slow-stage duration) -> code `SnapRunConfig.duration`
- paper envelope `Omega_0(t)` -> code `landgraf_envelope(t_rel)` then scaled by `base_amp`
- paper target manifold tones `omega_n` -> code `manifold_transition_frequency(model, n, frame)`
- paper per-tone amplitude correction `A_n` -> code `SnapToneParameters.amplitudes[n]`
- paper per-tone detuning `delta_n` -> code `SnapToneParameters.detunings[n]`
- paper per-tone phase `phi_n` -> code `SnapToneParameters.phases[n]`
- paper coherent phase error component -> code `ManifoldErrors.dtheta`
- paper longitudinal error component -> code `ManifoldErrors.dlambda`
- paper transversal error component -> code `ManifoldErrors.dalpha`
- paper iterative update with learning rate -> code `optimize_snap_prl133(..., learning_rate=...)`

## Model Assumptions

- Dispersive qubit-cavity model in rotating frame near `omega_q`.
- Closed-system unitary benchmarks (primary reproduction path), `sesolve`-equivalent flow.
- Optional open-system extension available in core simulator but not required for baseline coherent-error elimination checks.

## Notes on Reproduction Fidelity

- The algorithm is implemented as an interpretable low-parameter per-tone iterative optimizer.
- It preserves the original tone family by construction: no extra optimization frequencies are introduced.
- Reproduction targets in this repository are CI-oriented and deterministic; they are reduced-size numerical analogs of paper claims.

