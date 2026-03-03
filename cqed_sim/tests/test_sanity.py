from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import DisplacementGate, RotationGate, SQRGate
from cqed_sim.observables.bloch import bloch_xyz_from_joint, reduced_qubit_state
from cqed_sim.observables.weakness import comparison_metrics
from cqed_sim.operators.basic import purity
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from cqed_sim.simulators.common import build_initial_state
from cqed_sim.simulators.ideal import ideal_gate_unitary, run_case_a
from cqed_sim.simulators.pulse_open import run_case_c
from cqed_sim.simulators.pulse_unitary import run_case_b
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
        rho_c = qt.ptrace(rho, 0)
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
    psi = qt.tensor((qt.basis(4, 0) + qt.basis(4, 1)).unit(), qt.basis(2, 0))
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
    res_t1 = simulate_sequence(model, compiled, model.basis_state(0, 1), {}, SimulationConfig(), noise=NoiseSpec(t1=1.0))
    if not np.isclose(bloch_xyz_from_joint(res_t1.final_state)[2], 1.0 - 2.0 * math.e ** (-1.0), atol=0.06):
        raise AssertionError("Test 4 T1 decay mismatch.")
    plus = qt.tensor(qt.basis(2, 0), (qt.basis(2, 0) + qt.basis(2, 1)).unit())
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


def run_notebook_sanity_suite(base_config: dict[str, Any]) -> list[dict[str, str]]:
    tests = [
        ("Test 1: ideal rotation sanity", lambda: _test_1_ideal_rotation_sanity(base_config)),
        ("Test 2: Case B displacement sanity", lambda: _test_2_case_b_displacement_sanity(base_config)),
        ("Test 3: SQR conditionality", _test_3_sqr_conditionality),
        ("Test 4: decoherence limits", _test_4_decoherence_limits),
        ("Test 5: Case A vs Case B ideal limit", lambda: _test_5_case_a_vs_case_b_ideal_limit(base_config)),
        ("Test 6: Case B/Case C shape and finiteness", lambda: _test_6_case_b_case_c_shapes(base_config)),
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
