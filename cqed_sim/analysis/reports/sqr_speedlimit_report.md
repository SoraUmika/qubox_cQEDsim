# SQR Speed-Limit Report

## Study Definition
- Seed: `1234`
- Matched subspace: `n = 0..2` with `logical_n = 3`
- Guard levels: `1`
- Dispersive parameter: `chi = -2.840000 MHz`
- Phase-1 ordering convention: `U_test = U_pi,ideal @ U_SQR`, meaning the selective SQR acts first on the state and the ideal fast pi is applied after it.
- Leakage note: in the minimal dispersive qubit-drive-only model used for the main sweep, photon number is conserved, so subspace leakage for `n<=n_match` probe states is identically zero. The reported guard metric is therefore a selectivity proxy on out-of-support manifolds, not literal dynamical leakage.

## Phase 1
### phase1_selective_flip_n0
- Description: Selective pattern: only n=0 is flipped by the SQR. The tested gate is U_test = U_pi_ideal @ U_SQR.
  - `F >= 0.9900`: `750.0 ns`
  - `F >= 0.9990`: not reached
  - `F >= 0.9999`: not reached
- Best point: `T = 750.0 ns`, `sigma = 0.30`, `F_subspace = 0.991496`, `guard = 4.450e-02`

## Phase 2
### phase2_alternating_flips
- Description: Alternating selective flips across Fock levels.
  - `F >= 0.9900`: `500.0 ns`
  - `F >= 0.9990`: `500.0 ns`
  - `F >= 0.9999`: not reached

### phase2_mixed_angles
- Description: Mixed pi/2 and pi rotations with nontrivial XY phases.
  - `F >= 0.9900`: `500.0 ns`
  - `F >= 0.9990`: `1000.0 ns`
  - `F >= 0.9999`: not reached

### phase2_seeded_random
- Description: Deterministic seeded random target over the matched subspace.
  - `F >= 0.9900`: `500.0 ns`
  - `F >= 0.9990`: `1000.0 ns`
  - `F >= 0.9999`: not reached

### phase2_cluster_like
- Description: Cluster-relevant heuristic block pattern on n <= n_match.
  - `F >= 0.9900`: `500.0 ns`
  - `F >= 0.9990`: `1000.0 ns`
  - `F >= 0.9999`: not reached

## 16 ns Unconditional Pi Validation
- `n <= 1`: `F_subspace = 0.999679`, `worst block = 0.998728`, `detuning = -0.065 MHz`, `sigma = 0.120`
- `n <= 2`: `F_subspace = 0.998934`, `worst block = 0.994917`, `detuning = -0.129 MHz`, `sigma = 0.120`

## Configuration Snapshot
```json
{
  "seed": 1234,
  "n_match": 2,
  "guard_levels": 1,
  "chi_hz": -2840000.0,
  "chi2_hz": 0.0,
  "chi3_hz": 0.0,
  "kerr_hz": 0.0,
  "omega_q_hz": 0.0,
  "omega_c_hz": 0.0,
  "qubit_alpha_hz": 0.0,
  "durations_ns": [
    50,
    75,
    100,
    150,
    200,
    300,
    500,
    750,
    1000
  ],
  "sigma_fractions": [
    0.15,
    0.2,
    0.25,
    0.3
  ],
  "multistart": 1,
  "dt_s": 2e-09,
  "max_step_s": 2e-09,
  "optimizer_maxiter_stage1": 3,
  "optimizer_maxiter_stage2": 4,
  "d_lambda_bounds": [
    -0.5,
    0.5
  ],
  "d_alpha_bounds": [
    -3.141592653589793,
    3.141592653589793
  ],
  "d_omega_hz_bounds": [
    -2000000.0,
    2000000.0
  ],
  "lambda_guard": 0.1,
  "weight_mode": "uniform",
  "fidelity_thresholds": [
    0.99,
    0.999,
    0.9999
  ],
  "fast_pi_duration_s": 1.6e-08,
  "fast_pi_dt_s": 2.5e-10,
  "fast_pi_sigma_bounds": [
    0.12,
    0.3
  ],
  "fast_pi_detuning_hz_bounds": [
    -6000000.0,
    6000000.0
  ],
  "fast_pi_multistart": 4,
  "fast_pi_validate_levels": [
    1,
    2
  ],
  "representative_duration_ns": 200,
  "output_root": "outputs\\analysis\\sqr_speedlimit_multitone_gaussian",
  "report_path": "cqed_sim\\analysis\\reports\\sqr_speedlimit_report.md",
  "progress_every": 1,
  "qutip_nsteps_sqr_calibration": 100000,
  "fast_pi_qutip_nsteps": 250000
}
```