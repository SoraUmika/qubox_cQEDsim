from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import DisplacementGate, RotationGate, SQRGate
from cqed_sim.observables.bloch import bloch_xyz_from_joint, reduced_qubit_state
from cqed_sim.observables.fock import (
    conditional_phase_diagnostics,
    fock_resolved_bloch_diagnostics,
    relative_phase_family_diagnostics,
    wrapped_phase_error,
)
from cqed_sim.observables.weakness import comparison_metrics
from cqed_sim.operators.basic import purity
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.extractors import conditioned_bloch_xyz
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from examples.workflows.sequential.common import build_initial_state
from examples.workflows.sequential.ideal import ideal_gate_unitary, run_case_a
from examples.workflows.sequential.pulse_calibrated import run_case_d
from examples.workflows.sequential.pulse_open import run_case_c
from examples.workflows.sequential.pulse_unitary import run_case_b
from examples.workflows.sequential.trajectories import simulate_gate_bloch_trajectory
from cqed_sim.sequence.scheduler import SequenceCompiler


def _normalized_config(base_config: dict[str, Any]) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["n_cav_dim"] = int(config.get("n_cav_dim", int(config["cavity_fock_cutoff"]) + 1))
    return config


def _assert_close(actual, expected, atol: float, label: str) -> None:
    if not np.allclose(actual, expected, atol=atol, rtol=0.0):
        raise AssertionError(f"{label} mismatch. expected={expected}, actual={actual}")


def _baseline_case_a(gates, config: dict[str, Any]) -> dict[str, np.ndarray]:
    state = build_initial_state(config, n_cav_dim=int(config["n_cav_dim"]))
    xs = []
    ys = []
    zs = []
    ns = []

    def record(current_state):
        rho = current_state if current_state.isoper else current_state.proj()
        x, y, z = bloch_xyz_from_joint(rho)
        rho_c = qt.ptrace(rho, 1)
        a = qt.destroy(int(config["n_cav_dim"]))
        xs.append(float(x))
        ys.append(float(y))
        zs.append(float(z))
        ns.append(float(np.real((rho_c * a.dag() * a).tr())))

    record(state)
    for gate in gates:
        unitary = ideal_gate_unitary(gate, int(config["n_cav_dim"]))
        state = unitary * state if not state.isoper else unitary * state * unitary.dag()
        record(state)
    return {
        "x": np.asarray(xs, dtype=float),
        "y": np.asarray(ys, dtype=float),
        "z": np.asarray(zs, dtype=float),
        "n": np.asarray(ns, dtype=float),
    }


def _minimal_track_from_states(states: list[qt.Qobj], case: str, gate_type: str = "Test") -> dict[str, Any]:
    snapshots = []
    for idx, state in enumerate(states):
        rho = state if state.isoper else state.proj()
        snapshots.append(
            {
                "index": idx,
                "state": rho,
                "rho_c": qt.ptrace(rho, 1),
                "gate_type": "INIT" if idx == 0 else gate_type,
                "top_label": "0:INIT" if idx == 0 else f"{idx}:{gate_type}",
            }
        )
    return {
        "case": case,
        "indices": np.arange(len(states), dtype=int),
        "snapshots": snapshots,
    }


def baseline_vs_refactor_sanity(base_config: dict[str, Any]) -> dict[str, float]:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 6,
            "n_cav_dim": 7,
            "omega_c_hz": 0.0,
            "omega_q_hz": 0.0,
            "qubit_alpha_hz": 0.0,
            "st_chi_hz": 0.0,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "st_K_hz": 0.0,
            "st_K2_hz": 0.0,
        }
    )
    gates = [
        RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
        DisplacementGate(index=1, name="disp", re=0.2, im=-0.1),
        SQRGate(index=2, name="sqr", theta=(0.0, np.pi / 4.0), phi=(0.0, np.pi / 2.0)),
    ]
    baseline = _baseline_case_a(gates, config)
    refactor = run_case_a(gates, config)
    metrics = {
        "x_max_abs_err": float(np.max(np.abs(refactor["x"] - baseline["x"]))),
        "y_max_abs_err": float(np.max(np.abs(refactor["y"] - baseline["y"]))),
        "z_max_abs_err": float(np.max(np.abs(refactor["z"] - baseline["z"]))),
        "n_max_abs_err": float(np.max(np.abs(refactor["n"] - baseline["n"]))),
    }
    for key, value in metrics.items():
        if value > 1.0e-12:
            raise AssertionError(f"Baseline vs refactor mismatch for {key}: {value}")
    return metrics


def _test_1_ideal_rotation_sanity(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config["cavity_fock_cutoff"] = 0
    config["n_cav_dim"] = 1
    gate = RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0)
    out = ideal_gate_unitary(gate, 1) * build_initial_state(config, n_cav_dim=1)
    _assert_close(np.asarray(bloch_xyz_from_joint(out)), np.array([0.0, -1.0, 0.0]), atol=2.0e-3, label="Test 1")
    if not np.isclose(purity(out), 1.0, atol=1.0e-10):
        raise AssertionError("Test 1 purity was not preserved.")


def _test_2_case_b_displacement_sanity(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 18,
            "n_cav_dim": 19,
            "omega_c_hz": 0.0,
            "st_chi_hz": 0.0,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "st_K_hz": 0.0,
            "st_K2_hz": 0.0,
        }
    )
    gate = DisplacementGate(index=0, name="disp", re=0.25, im=-0.15)
    track = run_case_b([gate], config, case_label="Case B test")
    a = qt.destroy(int(config["n_cav_dim"]))
    actual = qt.expect(a, track["snapshots"][-1]["rho_c"])
    if not np.isclose(actual, gate.alpha, atol=5.0e-2):
        raise AssertionError("Test 2 pulse-level displacement missed target alpha.")


def _test_3_sqr_conditionality() -> None:
    gate = SQRGate(index=0, name="sqr", theta=(0.0, np.pi / 2.0), phi=(0.0, np.pi / 2.0))
    psi = qt.tensor(qt.basis(2, 0), (qt.basis(4, 0) + qt.basis(4, 1)).unit())
    out = ideal_gate_unitary(gate, 4) * psi
    if not purity(reduced_qubit_state(out)) < 0.999:
        raise AssertionError("Test 3 expected qubit mixedness after conditional rotation.")


def _test_4_decoherence_limits() -> None:
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=2,
        n_tr=2,
    )
    compiled = SequenceCompiler(dt=0.02).compile([], t_end=1.0)
    res_t1 = simulate_sequence(model, compiled, model.basis_state( 1,0), {}, SimulationConfig(), noise=NoiseSpec(t1=1.0))
    if not np.isclose(bloch_xyz_from_joint(res_t1.final_state)[2], 1.0 - 2.0 * math.e ** (-1.0), atol=0.06):
        raise AssertionError("Test 4 T1 decay mismatch.")
    plus = qt.tensor((qt.basis(2, 0) + qt.basis(2, 1)).unit(), qt.basis(2, 0))
    res_tphi = simulate_sequence(model, compiled, plus, {}, SimulationConfig(), noise=NoiseSpec(tphi=1.0))
    if not np.isclose(bloch_xyz_from_joint(res_tphi.final_state)[0], math.e ** (-1.0), atol=0.06):
        raise AssertionError("Test 4 Tphi decay mismatch.")


def _test_5_case_a_vs_case_b_ideal_limit(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 12,
            "n_cav_dim": 13,
            "omega_c_hz": 0.0,
            "omega_q_hz": 0.0,
            "qubit_alpha_hz": 0.0,
            "st_chi_hz": 0.0,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "st_K_hz": 0.0,
            "st_K2_hz": 0.0,
            "duration_displacement_s": 1.0,
            "duration_rotation_s": 1.0,
            "dt_s": 2.5e-4,
            "max_step_s": 2.5e-4,
        }
    )
    gates = [
        RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
        DisplacementGate(index=1, name="disp", re=0.2, im=-0.1),
    ]
    case_a = run_case_a(gates, config)
    case_b = run_case_b(gates, config, case_label="Case B test")
    metrics = comparison_metrics(case_a, case_b)
    for key in ("x_rmse", "y_rmse", "z_rmse", "n_rmse"):
        if not metrics[key] < 4.0e-2:
            raise AssertionError(f"Test 5 {key} too large: {metrics[key]}")


def _test_6_case_b_case_c_shapes(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 8,
            "n_cav_dim": 9,
            "omega_c_hz": 0.0,
            "omega_q_hz": 0.0,
            "qubit_alpha_hz": 0.0,
            "st_chi_hz": 0.0,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "st_K_hz": 0.0,
            "st_K2_hz": 0.0,
            "duration_displacement_s": 1.0,
            "duration_rotation_s": 1.0,
            "duration_sqr_s": 1.0,
            "dt_s": 2.5e-3,
            "max_step_s": 2.5e-3,
        }
    )
    gates = [
        RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
        DisplacementGate(index=1, name="disp", re=0.1, im=0.0),
        SQRGate(index=2, name="sqr", theta=(0.0, np.pi / 6.0), phi=(0.0, 0.0)),
    ]
    case_b = run_case_b(gates, config, case_label="Case B shape test")
    case_c = run_case_c(gates, config, case_label="Case C shape test")
    for track in (case_b, case_c):
        expected_len = len(gates) + 1
        if track["x"].shape != (expected_len,):
            raise AssertionError("Unexpected Bloch output shape.")
        if not np.all(np.isfinite(track["x"])) or not np.all(np.isfinite(track["y"])) or not np.all(np.isfinite(track["z"])):
            raise AssertionError("Non-finite Bloch values detected.")


def _test_7_case_d_shapes_and_calibration(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 5,
            "n_cav_dim": 6,
            "omega_c_hz": 0.0,
            "omega_q_hz": 0.0,
            "qubit_alpha_hz": 0.0,
            "st_chi_hz": -2.0e5,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "st_K_hz": 0.0,
            "st_K2_hz": 0.0,
            "duration_displacement_s": 48.0e-9,
            "duration_rotation_s": 16.0e-9,
            "duration_sqr_s": 1.0e-6,
            "dt_s": 2.0e-8,
            "max_step_s": 2.0e-8,
            "max_n_cal": 1,
            "optimizer_method_stage1": "Powell",
            "optimizer_method_stage2": "L-BFGS-B",
            "optimizer_maxiter_stage1": 6,
            "optimizer_maxiter_stage2": 8,
            "d_lambda_bounds": (-0.5, 0.5),
            "d_alpha_bounds": (-np.pi, np.pi),
            "d_omega_hz_bounds": (-5.0e5, 5.0e5),
            "regularization_lambda": 1.0e-6,
            "regularization_alpha": 1.0e-6,
            "regularization_omega": 1.0e-18,
            "calibration_cache_dir": str(Path("calibrations") / "_test_cache"),
            "calibration_force_recompute": True,
            "case_d_include_dissipation": False,
        }
    )
    gates = [
        RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
        SQRGate(index=1, name="sqr_case_d", theta=(0.0, np.pi / 4.0), phi=(0.0, np.pi / 5.0)),
    ]
    case_d = run_case_d(gates, config, case_label="Case D test")
    if case_d["x"].shape != (len(gates) + 1,):
        raise AssertionError("Unexpected Case D Bloch output shape.")
    if not np.all(np.isfinite(case_d["x"])) or not np.all(np.isfinite(case_d["y"])) or not np.all(np.isfinite(case_d["z"])):
        raise AssertionError("Non-finite Case D Bloch values detected.")
    summaries = case_d["metadata"].get("calibration_summaries", {})
    if not summaries:
        raise AssertionError("Case D did not attach calibration summaries.")
    first_summary = next(iter(summaries.values()))
    if not first_summary["improved_levels"]:
        raise AssertionError("Case D calibration did not improve any manifold.")


def _test_8_gate_indexed_diagnostics(base_config: dict[str, Any]) -> None:
    config = _normalized_config(base_config)
    config.update(
        {
            "cavity_fock_cutoff": 5,
            "n_cav_dim": 6,
            "omega_c_hz": 0.0,
            "omega_q_hz": 0.0,
            "qubit_alpha_hz": 0.0,
            "st_chi_hz": 0.0,
            "st_chi2_hz": 0.0,
            "st_chi3_hz": 0.0,
            "st_K_hz": 0.0,
            "st_K2_hz": 0.0,
            "duration_displacement_s": 1.0,
            "duration_rotation_s": 1.0,
            "duration_sqr_s": 1.0,
            "dt_s": 5.0e-3,
            "max_step_s": 5.0e-3,
            "phase_track_max_n": 2,
        }
    )
    gates = [
        DisplacementGate(index=0, name="disp", re=0.2, im=0.0),
        RotationGate(index=1, name="x45", theta=np.pi / 4.0, phi=0.0),
    ]
    track = run_case_b(gates, config, case_label="Case B diagnostics test")
    bloch = fock_resolved_bloch_diagnostics(track, max_n=int(config["phase_track_max_n"]))
    phase = relative_phase_family_diagnostics(track, max_n=int(config["phase_track_max_n"]))
    if not np.array_equal(bloch["n_values"], phase["n_values"]):
        raise AssertionError("Gate diagnostics used mismatched n ranges.")
    expected_shape = (int(config["phase_track_max_n"]) + 1, len(gates) + 1)
    if (
        bloch["x"].shape != expected_shape
        or phase["families"]["ground"]["phase"].shape != expected_shape
        or phase["families"]["excited"]["phase"].shape != expected_shape
    ):
        raise AssertionError("Unexpected gate-diagnostic array shape.")
    trajectory = simulate_gate_bloch_trajectory(
        track,
        gates,
        config,
        gate_index=2,
        conditioned_n_levels=[0, 1],
        probability_threshold=1.0e-8,
    )
    if trajectory["times_s"].shape[0] != trajectory["x"].shape[0]:
        raise AssertionError("Trajectory times and Bloch arrays have mismatched lengths.")
    if not np.all(np.isfinite(trajectory["x"])) or not np.all(np.isfinite(trajectory["y"])) or not np.all(np.isfinite(trajectory["z"])):
        raise AssertionError("Non-finite unconditional trajectory values detected.")


def _test_9_phase_unwrap_continuity_across_branch_cut() -> None:
    n_cav = 3

    def qubit_dm(phase: float) -> qt.Qobj:
        coherence = 0.5 * np.exp(-1j * phase)
        return qt.Qobj(
            np.array([[0.5, coherence], [np.conjugate(coherence), 0.5]], dtype=np.complex128),
            dims=[[2], [2]],
        )

    state_a = qt.tensor( qubit_dm(np.pi - 5.0e-2),qt.basis(n_cav, 0).proj())
    state_b = qt.tensor( qubit_dm(-np.pi + 4.0e-2),qt.basis(n_cav, 0).proj())
    track = _minimal_track_from_states([state_a, state_b], case="phase continuity test", gate_type="Rotation")
    diag = conditional_phase_diagnostics(track, max_n=0, probability_threshold=1.0e-8, unwrap=True, coherence_threshold=1.0e-8)
    delta = float(diag["phase"][0, 1] - diag["phase"][0, 0])
    if abs(delta) > 0.2:
        raise AssertionError(f"Unwrapped phase jumped across the branch cut: {delta}")

    ideal = conditional_phase_diagnostics(
        _minimal_track_from_states([state_a], case="ideal phase reference"),
        max_n=0,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    simulated = conditional_phase_diagnostics(
        _minimal_track_from_states([state_b], case="simulated phase reference"),
        max_n=0,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    error = float(wrapped_phase_error(simulated, ideal)[0, 0])
    if abs(error) > 0.2:
        raise AssertionError(f"Ratio-based wrapped phase error should stay small across the branch cut, got {error}")


def _test_10_relative_phase_definition() -> None:
    n_cav = 4
    n_target = 2
    theta = 0.73
    psi = (qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0)) + np.exp(1j * theta) * qt.tensor( qt.basis(2, 1),qt.basis(n_cav, n_target))).unit()
    diag = conditional_phase_diagnostics(
        _minimal_track_from_states([psi], case="relative phase definition test", gate_type="SQR"),
        max_n=n_target,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    phase = float(diag["phase"][n_target, 0])
    wrapped = (phase - theta + np.pi) % (2.0 * np.pi) - np.pi
    if abs(wrapped) > 1.0e-10:
        raise AssertionError(f"Relative phase definition mismatch: expected {theta}, got {phase}")
    if not np.isnan(diag["phase"][0, 0]) or not np.isnan(diag["phase"][1, 0]):
        raise AssertionError("Unpopulated |e,n> manifolds should be masked in the controlled-superposition test.")


def _test_11_global_phase_invariance() -> None:
    n_cav = 4
    n_target = 1
    theta = -1.11
    alpha = 0.42
    psi = (qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0)) + np.exp(1j * theta) * qt.tensor( qt.basis(2, 1),qt.basis(n_cav, n_target))).unit()
    psi_shifted = np.exp(1j * alpha) * psi
    diag_a = conditional_phase_diagnostics(
        _minimal_track_from_states([psi], case="global phase baseline"),
        max_n=n_target,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    diag_b = conditional_phase_diagnostics(
        _minimal_track_from_states([psi_shifted], case="global phase shifted"),
        max_n=n_target,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    phase_a = float(diag_a["phase"][n_target, 0])
    phase_b = float(diag_b["phase"][n_target, 0])
    wrapped = (phase_a - phase_b + np.pi) % (2.0 * np.pi) - np.pi
    if abs(wrapped) > 1.0e-12:
        raise AssertionError(f"Global phase invariance failed: {phase_a} vs {phase_b}")


def _test_11b_ground_relative_phase_definition() -> None:
    n_cav = 4
    n_target = 2
    theta = -0.37
    psi = (qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0)) + np.exp(1j * theta) * qt.tensor( qt.basis(2, 0),qt.basis(n_cav, n_target))).unit()
    diag = relative_phase_family_diagnostics(
        _minimal_track_from_states([psi], case="ground relative phase definition test", gate_type="Displacement"),
        max_n=n_target,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    phase_ground = float(diag["families"]["ground"]["phase"][n_target, 0])
    wrapped = (phase_ground - theta + np.pi) % (2.0 * np.pi) - np.pi
    if abs(wrapped) > 1.0e-10:
        raise AssertionError(f"Ground-family relative phase mismatch: expected {theta}, got {phase_ground}")
    if not np.isnan(diag["families"]["excited"]["phase"][n_target, 0]):
        raise AssertionError("Excited-family phase should be masked for a pure ground-manifold superposition.")


def _test_12_conditioned_bloch_matches_known_state() -> None:
    n_cav = 5
    rho_q = ((qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()).proj()
    state = qt.tensor( rho_q,qt.basis(n_cav, 2).proj())
    conditioned = np.asarray(conditioned_bloch_xyz(state, n=2, fallback="nan")[:3], dtype=float)
    unconditional = np.asarray(bloch_xyz_from_joint(state), dtype=float)
    _assert_close(conditioned, unconditional, atol=1.0e-12, label="Conditioned Bloch known state")


def _test_13_bloch_bounds_or_masking() -> None:
    n_cav = 3
    state = qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0))
    track = _minimal_track_from_states([state], case="bloch masking test")
    diagnostics = fock_resolved_bloch_diagnostics(track, max_n=1, probability_threshold=1.0e-6)
    if not diagnostics["valid"][0, 0]:
        raise AssertionError("Populated manifold should remain valid.")
    if diagnostics["valid"][1, 0]:
        raise AssertionError("Empty manifold should be masked.")
    if not (np.isnan(diagnostics["x"][1, 0]) and np.isnan(diagnostics["y"][1, 0]) and np.isnan(diagnostics["z"][1, 0])):
        raise AssertionError("Masked Bloch entries should be NaN for low-population manifolds.")
    values = np.asarray([diagnostics["x"][0, 0], diagnostics["y"][0, 0], diagnostics["z"][0, 0]], dtype=float)
    if np.any(np.abs(values) > 1.0 + 1.0e-9):
        raise AssertionError(f"Conditioned Bloch values exceeded physical bounds: {values}")


def _test_14_manifold_isolation() -> None:
    n_cav = 4
    theta = -0.61
    psi = (qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0)) + np.exp(1j * theta) * qt.tensor( qt.basis(2, 1),qt.basis(n_cav, 1))).unit()
    diag = conditional_phase_diagnostics(
        _minimal_track_from_states([psi], case="manifold isolation test", gate_type="SQR"),
        max_n=2,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    if np.isnan(diag["phase"][1, 0]):
        raise AssertionError("n=1 phase should be present in the manifold-isolation test.")
    if not np.isnan(diag["phase"][0, 0]) or not np.isnan(diag["phase"][2, 0]):
        raise AssertionError("Only the populated |e,1> manifold should carry a defined relative phase.")


def _test_15_phase_family_bundle_contains_both() -> None:
    n_cav = 4
    theta_ground = 0.21
    theta_excited = -0.64
    psi = (
        qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0))
        + np.exp(1j * theta_ground) * qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 1))
        + np.exp(1j * theta_excited) * qt.tensor( qt.basis(2, 1),qt.basis(n_cav, 2))
    ).unit()
    diag = relative_phase_family_diagnostics(
        _minimal_track_from_states([psi], case="phase family bundle test", gate_type="SQR"),
        max_n=2,
        probability_threshold=1.0e-8,
        unwrap=False,
        coherence_threshold=1.0e-8,
    )
    if set(diag["families"]) != {"ground", "excited"}:
        raise AssertionError("Combined phase diagnostics must expose both ground and excited families.")
    ground_phase = float(diag["families"]["ground"]["phase"][1, 0])
    excited_phase = float(diag["families"]["excited"]["phase"][2, 0])
    if abs(((ground_phase - theta_ground + np.pi) % (2.0 * np.pi)) - np.pi) > 1.0e-10:
        raise AssertionError("Ground-family phase was not extracted correctly from the combined bundle.")
    if abs(((excited_phase - theta_excited + np.pi) % (2.0 * np.pi)) - np.pi) > 1.0e-10:
        raise AssertionError("Excited-family phase was not extracted correctly from the combined bundle.")


def run_notebook_sanity_suite(base_config: dict[str, Any]) -> list[dict[str, str]]:
    tests = [
        ("Test 1: ideal rotation sanity", lambda: _test_1_ideal_rotation_sanity(base_config)),
        ("Test 2: Case B displacement sanity", lambda: _test_2_case_b_displacement_sanity(base_config)),
        ("Test 3: SQR conditionality", _test_3_sqr_conditionality),
        ("Test 4: decoherence limits", _test_4_decoherence_limits),
        ("Test 5: Case A vs Case B ideal limit", lambda: _test_5_case_a_vs_case_b_ideal_limit(base_config)),
        ("Test 6: Case B/Case C shape and finiteness", lambda: _test_6_case_b_case_c_shapes(base_config)),
        ("Test 7: Case D calibrated SQR path", lambda: _test_7_case_d_shapes_and_calibration(base_config)),
        ("Test 8: gate-indexed diagnostics", lambda: _test_8_gate_indexed_diagnostics(base_config)),
        ("Test 9: relative phase definition", _test_10_relative_phase_definition),
        ("Test 10: branch cut continuity", _test_9_phase_unwrap_continuity_across_branch_cut),
        ("Test 11: global phase invariance", _test_11_global_phase_invariance),
        ("Test 11b: ground relative phase definition", _test_11b_ground_relative_phase_definition),
        ("Test 12: conditioned Bloch matches known state", _test_12_conditioned_bloch_matches_known_state),
        ("Test 13: Bloch bounds or masking", _test_13_bloch_bounds_or_masking),
        ("Test 14: manifold isolation", _test_14_manifold_isolation),
        ("Test 15: phase family bundle contains both", _test_15_phase_family_bundle_contains_both),
    ]
    results = []
    for label, fn in tests:
        fn()
        results.append({"label": label, "status": "PASS"})
    return results


def test_baseline_vs_refactor_case_a():
    base_config = {
        "cavity_fock_cutoff": 8,
        "n_cav_dim": 9,
        "initial_qubit": "g",
        "initial_cavity_kind": "fock",
        "initial_cavity_fock": 0,
        "initial_cavity_alpha": {"re": 0.0, "im": 0.0},
        "initial_cavity_amplitudes": None,
        "wigner_every_gate": True,
        "wigner_points": 21,
        "wigner_extent": 3.0,
        "omega_c_hz": 0.0,
        "omega_q_hz": 0.0,
        "qubit_alpha_hz": 0.0,
        "st_chi_hz": 0.0,
        "st_chi2_hz": 0.0,
        "st_chi3_hz": 0.0,
        "st_K_hz": 0.0,
        "st_K2_hz": 0.0,
        "use_rotating_frame": True,
        "qb_T1_relax_ns": 9812.873848245112,
        "qb_T2_ramsey_ns": 6324.73112712837,
        "qb_T2_echo_ns": 8070.0,
        "t2_source": "ramsey",
        "cavity_kappa_1_per_s": 0.0,
        "duration_displacement_s": 32.0e-9,
        "duration_rotation_s": 64.0e-9,
        "duration_sqr_s": 1.0e-6,
        "rotation_sigma_fraction": 0.18,
        "sqr_sigma_fraction": 0.18,
        "sqr_theta_cutoff": 1.0e-10,
        "dt_s": 1.0e-9,
        "max_step_s": 1.0e-9,
    }
    baseline_vs_refactor_sanity(base_config)


def test_relative_phase_definition():
    _test_10_relative_phase_definition()


def test_branch_cut_continuity():
    _test_9_phase_unwrap_continuity_across_branch_cut()


def test_global_phase_invariance():
    _test_11_global_phase_invariance()


def test_ground_relative_phase_definition():
    _test_11b_ground_relative_phase_definition()


def test_conditioned_bloch_matches_known_state():
    _test_12_conditioned_bloch_matches_known_state()


def test_bloch_bounds_or_masking():
    _test_13_bloch_bounds_or_masking()


def test_manifold_isolation():
    _test_14_manifold_isolation()


def test_phase_family_bundle_contains_both():
    _test_15_phase_family_bundle_contains_both()
