# API Reference — SQR Calibration (`cqed_sim.calibration`)

The calibration module optimizes per-manifold correction parameters for Selective Qubit Rotation (SQR) gates.

---

## SQRCalibrationResult

**Module path:** `cqed_sim.calibration.sqr`

```python
@dataclass
class SQRCalibrationResult:
    sqr_name: str
    max_n: int
    d_lambda: list[float]        # Amplitude corrections per Fock level
    d_alpha: list[float]         # Phase corrections per Fock level
    d_omega_rad_s: list[float]   # Frequency corrections per Fock level (rad/s)
    theta_target: list[float]
    phi_target: list[float]
    initial_loss: list[float]
    optimized_loss: list[float]
    levels: list[SQRLevelCalibration] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
```

| Property / Method | Description |
|---|---|
| `d_omega_hz` | Frequency corrections in Hz |
| `correction_for_n(n)` | `(d_lambda_n, d_alpha_n, d_omega_n)` for Fock level n |
| `improvement_summary()` | Dict with mean/max improvement metrics |
| `to_dict()` / `from_dict(payload)` | Serialization |

---

## Core Calibration Functions

| Function | Signature | Description |
|---|---|---|
| `calibrate_sqr_gate(gate, config)` | `(SQRGate, Mapping) -> SQRCalibrationResult` | Two-stage optimization (Powell → L-BFGS-B) per manifold |
| `load_or_calibrate_sqr_gate(gate, config, cache_dir)` | `-> SQRCalibrationResult` | Cache-aware calibration with config hash matching |
| `calibrate_all_sqr_gates(gates, config, cache_dir)` | `-> dict[str, SQRCalibrationResult]` | Calibrate all SQR gates in a sequence |
| `export_calibration_result(result, path, config)` | `-> Path` | Write JSON |
| `load_calibration_result(path)` | `-> SQRCalibrationResult` | Read JSON |

---

## Evaluation and Benchmarking

| Function | Signature | Description |
|---|---|---|
| `evaluate_sqr_gate_levels(gate, config, corrections)` | `-> list[dict]` | Per-Fock-level fidelity evaluation |
| `extract_effective_qubit_unitary(n, theta, phi, config, ...)` | `-> (ndarray, dict)` | 2×2 qubit unitary from time-dependent Hamiltonian |
| `target_qubit_unitary(theta, phi)` | `-> ndarray` | Ideal SQR target unitary |
| `conditional_process_fidelity(target, simulated)` | `-> float` | Process fidelity, clipped to [0, 1] |
| `conditional_loss(params, n, theta, phi, config)` | `-> float` | Optimization objective: 1 − fidelity + regularization |

---

## Random Target Benchmarking

### RandomSQRTarget

```python
@dataclass(frozen=True)
class RandomSQRTarget:
    target_id: str
    target_class: str              # "iid", "smooth", "hard", "sparse"
    logical_n: int
    guard_levels: int
    theta: tuple[float, ...]
    phi: tuple[float, ...]
```

### GuardedBenchmarkResult

```python
@dataclass
class GuardedBenchmarkResult:
    # target_id, target_class, duration_s, logical_n, guard_levels,
    # lambda_guard, weight_mode, poisson_alpha, logical_fidelity,
    # epsilon_guard, loss_total, success, converged, iterations,
    # objective_evaluations, calibration, per_manifold, convergence_trace, metadata
```

### Functions

| Function | Description |
|---|---|
| `generate_random_sqr_targets(logical_n, guard_levels, n_targets_per_class, seed, ...)` | Generate random SQR targets across classes |
| `calibrate_guarded_sqr_target(target, config, ...)` | Optimize with guard-level leakage penalty |
| `benchmark_random_sqr_targets_vs_duration(config, durations, targets, ...)` | Sweep durations × targets |
| `benchmark_results_table(results)` | Convert to record dicts |
| `summarize_duration_benchmark(results)` | Group by duration with statistics |
