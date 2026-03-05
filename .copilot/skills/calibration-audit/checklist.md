# Calibration Audit Checklist

## Artifact Loading
- [ ] Calibration JSON parses without error.
- [ ] All expected fields present: `sqr_name`, `max_n`, `levels[]`.
- [ ] Per-level arrays: `n`, `skipped`, `initial_loss`, `optimized_loss`, `optimized_params`.

## Parameter Bounds
- [ ] `d_lambda` within `(-0.5, 0.5)` for all levels.
- [ ] `d_alpha` within `(-π, π)` for all levels.
- [ ] `d_omega_hz` within `(-2e6, 2e6)` for all levels.
- [ ] No parameter within 5% of bound edge.

## Convergence
- [ ] Improvement factor ≥ 10× for all active levels.
- [ ] `optimized_loss ≤ 1e-3` for all active levels.
- [ ] No level has `optimized_loss > initial_loss` (no divergence).
- [ ] Skipped levels have `|theta_target| < theta_cutoff`.

## Physics Consistency
- [ ] `st_chi_hz` matches `experiment_mapping.md`.
- [ ] Detuning convention matches: `Δ(n) = 2π(χn + χ₂n² + χ₃n³)`.
- [ ] `duration_sqr_s` in physically reasonable range (0.1–5 μs).
- [ ] `max_n_cal` ≤ `cavity_fock_cutoff`.
- [ ] Qubit T1/T2 values match device specs.

## Guard-Band (if applicable)
- [ ] `F_logical ≥ 0.99` for all benchmark targets.
- [ ] `epsilon_guard ≤ 0.01` for all benchmark targets.
- [ ] Success rate ≥ 80% per target class.
- [ ] Fidelity improves monotonically with duration (within noise).

## Final Verdict
- [ ] All checks pass → **PASS**
- [ ] Non-critical warnings → **WARN** (list specific items)
- [ ] Any blocking failure → **FAIL** (list specific items + remediation)
