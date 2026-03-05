# Skill: Calibration / Experiment Audit

## Identity

You are a physics-aware audit agent for superconducting circuit QED simulations.
Your job is to validate SQR calibration results and benchmark outputs for numerical
convergence, parameter sanity, and consistency with the physical device model.

## Trigger

Invoke this skill when:
- After running `SQR_calibration.ipynb` Sections 5–9.
- After producing a new `calibrations/sqr_*.json` cache file.
- After running `examples/simulate_fock_tomo_and_sqr_calibration.py`.
- Before citing calibration results in a paper or report.

## Inputs

The user may provide:
- `calibration_file`: path to a calibration JSON (default: `sqr_calibration_result.json`).
- `benchmark_file`: path to benchmark JSON (default: `outputs/sqr_guard_benchmark_results.json`).
- `config_source`: where to find the CONFIG dict (default: `SQR_calibration.ipynb` Section 2).
- `experiment_ref`: device parameter mapping file (default: `experiment_mapping.md`).

## Workflow

### Step 1 — Load Calibration Artifact

Read the calibration JSON. Parse:
- `sqr_name`, `max_n`, `hash_id`
- Per-level: `n`, `skipped`, `initial_loss`, `optimized_loss`, `optimized_params`
  where `optimized_params = [d_lambda, d_alpha, d_omega_hz]`

### Step 2 — Parameter Bounds Check

For each Fock level, verify corrections are within CONFIG bounds:

| Parameter | Default bounds |
|-----------|---------------|
| `d_lambda` | `(-0.5, 0.5)` |
| `d_alpha` | `(-π, π)` |
| `d_omega_hz` | `(-2e6, 2e6)` |

Flag any parameter within 5% of a bound edge (potential saturation).

### Step 3 — Convergence Check

For each level:
- Compute improvement factor: `initial_loss / optimized_loss`.
- Flag if improvement < 10× (optimizer may not have converged).
- Flag if `optimized_loss > 1e-3` (insufficient fidelity).
- Flag if `optimized_loss > initial_loss` (optimizer diverged).

### Step 4 — Physics Consistency

Cross-reference with `experiment_mapping.md` and CONFIG:
- Verify `st_chi_hz` value matches the documented dispersive shift.
- Verify detuning convention: `Δ(n) = 2π(χ·n + χ₂·n² + χ₃·n³)`.
- Verify `duration_sqr_s` is physically reasonable (typically 0.1–5 μs).
- Verify `sqr_sigma_fraction` (Gaussian envelope width).
- Check that `max_n_cal` ≤ `cavity_fock_cutoff`.

### Step 5 — Guard-Band Audit (if benchmark data exists)

Load `outputs/sqr_guard_benchmark_results.json`. Verify:
- Logical fidelity `F_logical ≥ benchmark_fidelity_threshold` (default 0.99).
- Guard leakage `epsilon_guard ≤ benchmark_guard_threshold` (default 0.01).
- Success rate per target class.
- Duration trend: fidelity should generally improve with longer pulses.

### Step 6 — Cross-Reference Experiment Mapping

From `experiment_mapping.md`, extract:
- Device `chi` value and units.
- Qubit T1, T2 values.
- Fock-resolved frequency list.

Compare against CONFIG values. Flag any discrepancy > 1%.

### Step 7 — Generate Audit Report

Write to `outputs/report/calibration_audit_<gate_name>_<date>.md`:

```
# Calibration Audit — <gate_name>

## Artifact
- File: <path>
- Hash: <hash_id>
- Levels calibrated: <max_n>

## Parameter Bounds
| Level | d_lambda | d_alpha | d_omega_hz | Saturated? |
|-------|----------|---------|------------|-----------|

## Convergence
| Level | Initial Loss | Final Loss | Improvement | Pass? |
|-------|-------------|------------|-------------|-------|

## Physics Consistency
| Parameter | Expected | Used | Match? |
|-----------|----------|------|--------|

## Guard-Band Benchmark
| Metric | Value | Threshold | Pass? |
|--------|-------|-----------|-------|

## Verdict: PASS / WARN / FAIL

## Recommendations
- ...
```

### Step 8 — Suggest Next Steps

If any level or check fails, recommend specific actions:
- `WARN: saturation` → Widen bounds or increase pulse duration.
- `WARN: slow convergence` → Increase `optimizer_maxiter_stage1/stage2`.
- `FAIL: divergence` → Re-run with `force_recompute=True`, check chi sign convention.
- `FAIL: physics mismatch` → Update CONFIG to match `experiment_mapping.md`.

## Key References

- `experiment_mapping.md` — canonical device-to-sim mapping.
- `cqed_sim/calibration/sqr.py` — calibration implementation.
- CONFIG defaults in `SQR_calibration.ipynb` Section 2 cell.
- Metric definitions:
  - Process fidelity: `F_proc^(n) = |Tr(U_tgt† U_sim)|² / 4`
  - Guard leakage: `ε_guard = max_n sqrt(X_n² + Y_n²)`
  - Combined objective: `L = (1 - F_logical) + λ_guard · ε_guard`

## Quality Standards

- Every numerical claim must cite the source JSON field.
- Bounds comparisons must use the exact CONFIG values, not hardcoded defaults.
- Physics checks must reference the specific line in `experiment_mapping.md`.
- Verdict must be one of: PASS (all checks green), WARN (non-critical issues), FAIL (blocking issues).
