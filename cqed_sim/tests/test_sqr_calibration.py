from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.calibration.sqr import (
    GuardedBenchmarkResult,
    RandomSQRTarget,
    benchmark_random_sqr_targets_vs_duration,
    benchmark_results_table,
    calibrate_guarded_sqr_target,
    calibrate_sqr_gate,
    conditional_process_fidelity,
    evaluate_guarded_sqr_target,
    extract_effective_qubit_unitary,
    generate_random_sqr_targets,
    summarize_duration_benchmark,
    target_qubit_unitary,
)
from cqed_sim.io.gates import SQRGate
from cqed_sim.observables.bloch import bloch_xyz_from_joint


def _normalized_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["n_cav_dim"] = int(config.get("n_cav_dim", int(config["cavity_fock_cutoff"]) + 1))
    config.setdefault("omega_q_hz", 0.0)
    config.setdefault("omega_c_hz", 0.0)
    config.setdefault("qubit_alpha_hz", 0.0)
    config.setdefault("max_n_cal", min(2, int(config["cavity_fock_cutoff"])))
    config.setdefault("st_K_hz", 0.0)
    config.setdefault("st_K2_hz", 0.0)
    config.setdefault("use_rotating_frame", True)
    config.setdefault("qb_T1_relax_ns", None)
    config.setdefault("qb_T2_ramsey_ns", None)
    config.setdefault("qb_T2_echo_ns", None)
    config.setdefault("t2_source", "ramsey")
    config.setdefault("cavity_kappa_1_per_s", 0.0)
    config.setdefault("optimizer_method_stage1", "Powell")
    config.setdefault("optimizer_method_stage2", "L-BFGS-B")
    config.setdefault("d_lambda_bounds", (-0.5, 0.5))
    config.setdefault("d_alpha_bounds", (-np.pi, np.pi))
    config.setdefault("d_omega_hz_bounds", (-2.0e6, 2.0e6))
    config.setdefault("regularization_lambda", 1.0e-6)
    config.setdefault("regularization_alpha", 1.0e-6)
    config.setdefault("regularization_omega", 1.0e-18)
    config.setdefault("optimizer_maxiter_stage1", 12)
    config.setdefault("optimizer_maxiter_stage2", 18)
    config.setdefault("qutip_nsteps_sqr_calibration", 100000)
    return config


def _calibration_test_gate() -> SQRGate:
    return SQRGate(
        index=0,
        name="sqr_cal_test",
        theta=(np.pi / 8.0, np.pi / 3.0, np.pi / 5.0),
        phi=(0.0, np.pi / 4.0, -np.pi / 6.0),
    )


def _benchmark_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 6,
            "n_cav_dim": 7,
            "duration_sqr_s": 4.0e-7,
            "sqr_sigma_fraction": 1.0 / 6.0,
            "dt_s": 4.0e-9,
            "max_step_s": 4.0e-9,
            "st_chi_hz": -2.840421354241756e6,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "max_n_cal": 3,
            "sqr_theta_cutoff": 1.0e-10,
            "optimizer_method_stage1": "Powell",
            "optimizer_method_stage2": "L-BFGS-B",
            "optimizer_maxiter_stage1": 6,
            "optimizer_maxiter_stage2": 8,
            "qutip_nsteps_sqr_calibration": 50000,
        }
    )
    return config


def _identity_target(logical_n: int = 2, guard_levels: int = 1) -> RandomSQRTarget:
    total = logical_n + guard_levels
    return RandomSQRTarget(
        target_id="identity_target",
        target_class="identity",
        logical_n=logical_n,
        guard_levels=guard_levels,
        theta=tuple(0.0 for _ in range(total)),
        phi=tuple(0.0 for _ in range(total)),
    )


def _boundary_target(logical_n: int = 2, guard_levels: int = 1, theta: float = np.pi / 2.0) -> RandomSQRTarget:
    total = logical_n + guard_levels
    thetas = [0.0] * total
    phis = [0.0] * total
    thetas[logical_n - 1] = float(theta)
    phis[logical_n - 1] = np.pi / 3.0
    return RandomSQRTarget(
        target_id="boundary_target",
        target_class="boundary",
        logical_n=logical_n,
        guard_levels=guard_levels,
        theta=tuple(thetas),
        phi=tuple(phis),
    )


def _assert_close(actual, expected, atol: float, label: str) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{label} mismatch. expected={expected}, actual={actual}")


def _test_rotation_convention_sanity(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update({"st_chi_hz": 0.0, "st_chi2_hz": 0.0, "st_chi3_hz": 0.0})
    unitary, _ = extract_effective_qubit_unitary(
        n=0,
        theta_target=np.pi / 2.0,
        phi_target=0.0,
        config=config,
    )
    final_state = qt.Qobj(unitary @ np.array([[1.0], [0.0]], dtype=np.complex128), dims=[[2], [1]])
    bloch = np.asarray(bloch_xyz_from_joint(qt.tensor(final_state, qt.basis(1, 0))))
    _assert_close(bloch, np.array([0.0, -1.0, 0.0]), atol=5.0e-2, label="Rotation convention")


def _test_process_fidelity_sanity(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update({"st_chi_hz": 0.0, "st_chi2_hz": 0.0, "st_chi3_hz": 0.0})
    simulated, _ = extract_effective_qubit_unitary(
        n=0,
        theta_target=np.pi / 3.0,
        phi_target=np.pi / 6.0,
        config=config,
    )
    target = target_qubit_unitary(np.pi / 3.0, np.pi / 6.0)
    fidelity = conditional_process_fidelity(target, simulated)
    if not fidelity > 0.995:
        raise AssertionError(f"Process fidelity sanity failed: {fidelity}")


def _test_optimizer_improvement(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    gate = _calibration_test_gate()
    result = calibrate_sqr_gate(gate, config)
    improved = [level.n for level in result.levels if (not level.skipped) and level.optimized_loss < level.initial_loss]
    if not improved:
        raise AssertionError("Expected optimizer to improve at least one calibrated manifold.")


def _test_random_targets_seeded_deterministic(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    targets_a = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=7)
    targets_b = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=7)
    if targets_a != targets_b:
        raise AssertionError("Random target generator is not deterministic for a fixed seed.")
    results_a = benchmark_random_sqr_targets_vs_duration(config, [4.0e-7], targets_a[:2], lambda_guard=0.1)
    results_b = benchmark_random_sqr_targets_vs_duration(config, [4.0e-7], targets_b[:2], lambda_guard=0.1)
    rows_a = benchmark_results_table(results_a)
    rows_b = benchmark_results_table(results_b)
    if rows_a != rows_b:
        raise AssertionError("Benchmark summary changed across repeated seeded runs.")


def _test_calibration_repeatability(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    target = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=11, target_classes=("iid",))[0]
    result_a = calibrate_guarded_sqr_target(target, config, lambda_guard=0.1)
    result_b = calibrate_guarded_sqr_target(target, config, lambda_guard=0.1)
    if not np.isclose(result_a.logical_fidelity, result_b.logical_fidelity, atol=1.0e-10):
        raise AssertionError("Logical fidelity changed across identical reruns.")
    if not np.isclose(result_a.epsilon_guard, result_b.epsilon_guard, atol=1.0e-10):
        raise AssertionError("Guard metric changed across identical reruns.")


def _test_fidelity_bounded(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    targets = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=5, target_classes=("iid", "sparse"))
    results = benchmark_random_sqr_targets_vs_duration(config, [4.0e-7], targets, lambda_guard=0.1)
    for result in results:
        if not (0.0 <= result.logical_fidelity <= 1.0):
            raise AssertionError("Logical fidelity left [0,1].")
        for row in result.per_manifold:
            if not (0.0 <= row["process_fidelity"] <= 1.0):
                raise AssertionError("Per-manifold fidelity left [0,1].")


def _test_identity_target_gives_high_fidelity(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    result = calibrate_guarded_sqr_target(_identity_target(), config, lambda_guard=0.1)
    if not result.logical_fidelity > 0.999999:
        raise AssertionError(f"Identity target logical fidelity too low: {result.logical_fidelity}")
    if not result.epsilon_guard < 1.0e-12:
        raise AssertionError(f"Identity target guard leakage too high: {result.epsilon_guard}")


def _test_guard_penalty_reduces_boundary_leakage(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    config.update({"duration_sqr_s": 2.0e-7, "dt_s": 2.0e-9, "max_step_s": 2.0e-9})
    target = _boundary_target(theta=np.pi)
    no_guard = calibrate_guarded_sqr_target(target, config, lambda_guard=0.0)
    with_guard = calibrate_guarded_sqr_target(target, config, lambda_guard=0.2)
    if not with_guard.epsilon_guard < no_guard.epsilon_guard:
        raise AssertionError("Guard penalty did not reduce boundary leakage.")
    if not with_guard.logical_fidelity >= no_guard.logical_fidelity - 0.2:
        raise AssertionError("Guard penalty degraded logical fidelity too strongly.")


def _test_boundary_leakage_visible_when_not_optimized(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    config.update({"duration_sqr_s": 1.5e-7, "dt_s": 2.0e-9, "max_step_s": 2.0e-9})
    metrics = evaluate_guarded_sqr_target(_boundary_target(theta=np.pi), config, corrections={}, lambda_guard=0.0)
    if not metrics["epsilon_guard"] > 1.0e-3:
        raise AssertionError("Boundary leakage was not visible in the short-pulse regime.")


def _test_selectivity_improves_with_duration(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    target = _boundary_target(theta=np.pi)
    short_config = dict(config)
    short_config.update({"duration_sqr_s": 1.5e-7, "dt_s": 2.0e-9, "max_step_s": 2.0e-9})
    long_config = dict(config)
    long_config.update({"duration_sqr_s": 9.0e-7, "dt_s": 6.0e-9, "max_step_s": 6.0e-9})
    short_metrics = evaluate_guarded_sqr_target(target, short_config, corrections={}, lambda_guard=0.0)
    long_metrics = evaluate_guarded_sqr_target(target, long_config, corrections={}, lambda_guard=0.0)
    if not long_metrics["epsilon_guard"] < short_metrics["epsilon_guard"]:
        raise AssertionError("Guard leakage did not improve with longer duration.")


def _test_success_rate_nondecreasing_with_duration(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    targets = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=13, target_classes=("iid", "sparse"))
    results = benchmark_random_sqr_targets_vs_duration(
        config,
        [2.0e-7, 8.0e-7],
        targets,
        lambda_guard=0.15,
        fidelity_threshold=0.8,
        guard_threshold=0.2,
    )
    summary = summarize_duration_benchmark(results)
    if not summary[-1]["success_rate"] + 1.0e-12 >= summary[0]["success_rate"]:
        raise AssertionError("Success rate decreased with longer duration.")


def _test_objective_nonincreasing_near_convergence(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    target = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=17, target_classes=("smooth",))[0]
    result = calibrate_guarded_sqr_target(target, config, lambda_guard=0.1)
    tail = np.asarray([row["best_loss_total"] for row in result.convergence_trace[-5:]], dtype=float)
    if tail.size >= 2 and np.any(np.diff(tail) > 1.0e-12):
        raise AssertionError("Best-so-far objective increased near convergence.")


def _test_iteration_cap_respected(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    config.update({"optimizer_maxiter_stage1": 4, "optimizer_maxiter_stage2": 5})
    target = generate_random_sqr_targets(logical_n=3, guard_levels=1, n_targets_per_class=1, seed=19, target_classes=("hard",))[0]
    result = calibrate_guarded_sqr_target(target, config, lambda_guard=0.1)
    if int(result.metadata["stage1_nit"]) > int(config["optimizer_maxiter_stage1"]):
        raise AssertionError("Stage 1 exceeded iteration cap.")
    if int(result.metadata["stage2_nit"]) > int(config["optimizer_maxiter_stage2"]):
        raise AssertionError("Stage 2 exceeded iteration cap.")


def _test_dissipative_guarded_path_uses_channel_solver(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    config.update(
        {
            "qb_T1_relax_ns": 400.0,
            "qb_T2_ramsey_ns": 260.0,
            "t2_source": "ramsey",
        }
    )
    target = _boundary_target(theta=np.pi / 2.0)
    coherent = evaluate_guarded_sqr_target(target, _benchmark_config(base_config), corrections={}, lambda_guard=0.1)
    dissipative = evaluate_guarded_sqr_target(target, config, corrections={}, lambda_guard=0.1)
    if dissipative["simulation_mode"] != "channel":
        raise AssertionError("Dissipative guarded evaluation did not switch to channel mode.")
    if "mesolve" not in str(dissipative["execution_path"]):
        raise AssertionError("Dissipative guarded evaluation did not use the shared mesolve path.")
    if not (0.0 <= dissipative["logical_fidelity"] <= 1.0):
        raise AssertionError("Dissipative logical fidelity left [0,1].")
    if not dissipative["logical_fidelity"] <= coherent["logical_fidelity"] + 1.0e-9:
        raise AssertionError("Dissipation unexpectedly improved logical fidelity.")


def _test_benchmark_cell_runtime_budget(base_config: dict[str, Any]) -> None:
    config = _benchmark_config(base_config)
    config.update({"optimizer_maxiter_stage1": 3, "optimizer_maxiter_stage2": 4})
    targets = generate_random_sqr_targets(logical_n=2, guard_levels=1, n_targets_per_class=1, seed=23, target_classes=("iid", "sparse"))
    start = time.perf_counter()
    _ = benchmark_random_sqr_targets_vs_duration(config, [2.5e-7, 7.5e-7], targets, lambda_guard=0.1)
    elapsed = time.perf_counter() - start
    # Keep this as a smoke-level budget check, but allow variability across
    # laptops/VMs where BLAS threading and CPU frequency scaling differ.
    runtime_budget_s = 35.0
    if not elapsed < runtime_budget_s:
        raise AssertionError(f"Reduced benchmark runtime exceeded budget: {elapsed:.3f} s")


def run_sqr_calibration_sanity_suite(base_config: dict[str, Any]) -> list[dict[str, str]]:
    tests = [
        ("Calibration Test 1: rotation convention sanity", lambda: _test_rotation_convention_sanity(base_config)),
        ("Calibration Test 2: process fidelity sanity", lambda: _test_process_fidelity_sanity(base_config)),
        ("Calibration Test 3: optimizer improvement", lambda: _test_optimizer_improvement(base_config)),
        ("Benchmark Test 1: random targets are seeded deterministic", lambda: _test_random_targets_seeded_deterministic(base_config)),
        ("Benchmark Test 2: calibration repeatability", lambda: _test_calibration_repeatability(base_config)),
        ("Benchmark Test 3: fidelity boundedness", lambda: _test_fidelity_bounded(base_config)),
        ("Benchmark Test 4: identity target remains identity", lambda: _test_identity_target_gives_high_fidelity(base_config)),
        ("Benchmark Test 5: guard penalty reduces boundary leakage", lambda: _test_guard_penalty_reduces_boundary_leakage(base_config)),
        ("Benchmark Test 6: boundary leakage visible without guard optimization", lambda: _test_boundary_leakage_visible_when_not_optimized(base_config)),
        ("Benchmark Test 7: selectivity improves with duration", lambda: _test_selectivity_improves_with_duration(base_config)),
        ("Benchmark Test 8: success rate nondecreasing with duration", lambda: _test_success_rate_nondecreasing_with_duration(base_config)),
        ("Benchmark Test 9: objective nonincreasing near convergence", lambda: _test_objective_nonincreasing_near_convergence(base_config)),
        ("Benchmark Test 10: iteration cap respected", lambda: _test_iteration_cap_respected(base_config)),
        ("Benchmark Test 11: runtime budget", lambda: _test_benchmark_cell_runtime_budget(base_config)),
    ]
    results = []
    for label, fn in tests:
        fn()
        results.append({"label": label, "status": "PASS"})
    return results


def test_sqr_calibration_optimizer_improves_loss():
    config = _normalized_config(
        {
            "cavity_fock_cutoff": 8,
            "n_cav_dim": 9,
            "duration_sqr_s": 1.0e-6,
            "sqr_sigma_fraction": 1.0 / 6.0,
            "dt_s": 5.0e-9,
            "max_step_s": 5.0e-9,
            "st_chi_hz": -2.0e5,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "max_n_cal": 2,
            "sqr_theta_cutoff": 1.0e-10,
            "optimizer_method_stage1": "Powell",
            "optimizer_method_stage2": "L-BFGS-B",
        }
    )
    _test_optimizer_improvement(config)


def test_random_targets_seeded_deterministic():
    _test_random_targets_seeded_deterministic(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_calibration_repeatability():
    _test_calibration_repeatability(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_fidelity_bounded():
    _test_fidelity_bounded(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_identity_target_gives_high_fidelity():
    _test_identity_target_gives_high_fidelity(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_guard_penalty_reduces_boundary_leakage():
    _test_guard_penalty_reduces_boundary_leakage(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_boundary_leakage_visible_when_not_optimized():
    _test_boundary_leakage_visible_when_not_optimized(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_selectivity_improves_with_duration():
    _test_selectivity_improves_with_duration(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_success_rate_nondecreasing_with_duration():
    _test_success_rate_nondecreasing_with_duration(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_objective_nonincreasing_near_convergence():
    _test_objective_nonincreasing_near_convergence(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_iteration_cap_respected():
    _test_iteration_cap_respected(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_benchmark_cell_runtime_budget():
    _test_benchmark_cell_runtime_budget(_benchmark_config({"cavity_fock_cutoff": 6}))


def test_dissipative_guarded_path_uses_channel_solver():
    _test_dissipative_guarded_path_uses_channel_solver(_benchmark_config({"cavity_fock_cutoff": 6}))
