from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer
from cqed_sim.unitary_synthesis.backends import simulate_sequence
from cqed_sim.unitary_synthesis.sequence import (
    ConditionalPhaseSQR,
    Displacement,
    DriftPhaseModel,
    GateSequence,
    SQR,
    drift_phase_table,
    drift_phase_unitary,
)


def _make_multi_gate_sequence(n_cav: int = 4) -> GateSequence:
    gates = [
        SQR(name="sqr_a", theta_n=[0.1] * n_cav, phi_n=[0.0] * n_cav, duration=100e-9),
        Displacement(name="disp_a", alpha=0.02 + 0.01j, duration=90e-9),
        SQR(name="sqr_b", theta_n=[0.2] * n_cav, phi_n=[0.1] * n_cav, duration=120e-9),
        Displacement(name="disp_b", alpha=-0.01 + 0.03j, duration=80e-9),
    ]
    return GateSequence(gates=gates, n_cav=n_cav)


def test_t1_time_policy_enforcement_modes() -> None:
    policy = {
        "default": {"optimize": True, "bounds": (20e-9, 2e-6), "init": 200e-9},
        "SQR": {"optimize": True, "bounds": (50e-9, 1.2e-6), "init": 400e-9},
        "Displacement": {"optimize": True, "bounds": (30e-9, 0.9e-6), "init": 300e-9},
    }

    seq_type = _make_multi_gate_sequence()
    seq_type.configure_time_parameters(policy, mode="per-type")
    sqr_groups = [g.time_group for g in seq_type.gates if g.type == "SQR"]
    assert len(set(sqr_groups)) == 1

    sqr_param_idx = next(i for i, p in enumerate(seq_type.time_params) if p.group == sqr_groups[0])
    vals = seq_type.get_time_vector(active_only=False)
    vals[sqr_param_idx] = 900e-9
    seq_type.set_time_vector(vals, active_only=False)
    sqr_durations = [g.duration for g in seq_type.gates if g.type == "SQR"]
    assert np.allclose(sqr_durations, [900e-9, 900e-9])

    seq_instance = _make_multi_gate_sequence()
    seq_instance.configure_time_parameters(policy, mode="per-instance")
    sqr_groups_inst = [g.time_group for g in seq_instance.gates if g.type == "SQR"]
    assert len(set(sqr_groups_inst)) == 2

    idx0 = next(i for i, p in enumerate(seq_instance.time_params) if p.group == sqr_groups_inst[0])
    idx1 = next(i for i, p in enumerate(seq_instance.time_params) if p.group == sqr_groups_inst[1])
    vals = seq_instance.get_time_vector(active_only=False)
    old_1 = vals[idx1]
    vals[idx0] = 850e-9
    seq_instance.set_time_vector(vals, active_only=False)
    assert np.isclose(seq_instance.time_params[idx0].value, 850e-9)
    assert np.isclose(seq_instance.time_params[idx1].value, old_1)

    freeze_policy = {
        "default": {"optimize": True, "bounds": (20e-9, 2e-6), "init": 200e-9},
        "SQR": {"optimize": False, "bounds": (50e-9, 1.2e-6), "init": 400e-9},
        "Displacement": {"optimize": True, "bounds": (30e-9, 0.9e-6), "init": 300e-9},
    }
    seq_freeze = _make_multi_gate_sequence()
    seq_freeze.configure_time_parameters(freeze_policy, mode="per-type")
    active_types = {p.gate_type for p in seq_freeze.active_time_params()}
    assert "SQR" not in active_types
    assert "Displacement" in active_types

    seq_hybrid = _make_multi_gate_sequence()
    seq_hybrid.configure_time_parameters(
        policy,
        mode="hybrid",
        shared_groups={"sqr_a": "shared_sqr", "sqr_b": "shared_sqr"},
    )
    sqr_groups_hybrid = [g.time_group for g in seq_hybrid.gates if g.type == "SQR"]
    disp_groups_hybrid = [g.time_group for g in seq_hybrid.gates if g.type == "Displacement"]
    assert len(set(sqr_groups_hybrid)) == 1
    assert len(set(disp_groups_hybrid)) == 2


def test_t2_drift_phase_formula_matches_pulse_drift_only() -> None:
    n_cav = 4
    duration = 320e-9
    drift = DriftPhaseModel(chi=2.5e6, chi2=-1.2e5, kerr=0.8e5, kerr2=-1.1e4)

    gate = ConditionalPhaseSQR(
        name="cp",
        phases_n=[0.0] * n_cav,
        duration=duration,
        drift_model=drift,
        include_drift=True,
    )
    seq = GateSequence(gates=[gate], n_cav=n_cav)
    sub = Subspace.qubit_cavity_block(n_match=n_cav - 1, n_cav=n_cav)

    ideal = simulate_sequence(seq, sub, backend="ideal").full_operator
    pulse = simulate_sequence(seq, sub, backend="pulse").full_operator
    analytic = np.asarray(drift_phase_unitary(n_cav=n_cav, duration=duration, model=drift).full(), dtype=np.complex128)

    assert np.allclose(ideal, analytic, atol=1e-12)
    assert np.allclose(pulse, analytic, atol=1e-12)


def test_t3_conditional_relative_phase_matches_expected() -> None:
    n_cav = 5
    duration = 410e-9
    drift = DriftPhaseModel(chi=2.8e6, chi2=-2.2e5, kerr=0.0, kerr2=0.0)

    gate = ConditionalPhaseSQR(
        name="cp",
        phases_n=[0.0] * n_cav,
        duration=duration,
        drift_model=drift,
        include_drift=True,
    )
    seq = GateSequence(gates=[gate], n_cav=n_cav)
    sub = Subspace.qubit_cavity_block(n_match=3, n_cav=n_cav)

    direct_u = np.asarray(drift_phase_unitary(n_cav=n_cav, duration=duration, model=drift).full(), dtype=np.complex128)
    pulse_u = simulate_sequence(seq, sub, backend="pulse").full_operator

    for n in [0, 1, 2]:
        expected = np.exp(-1j * (drift.chi * n + drift.chi2 * (n * (n - 1))) * duration)

        ratio_direct = direct_u[n_cav + n, n_cav + n] / direct_u[n, n]
        ratio_pulse = pulse_u[n_cav + n, n_cav + n] / pulse_u[n, n]

        assert np.isclose(ratio_direct, expected, atol=1e-12)
        assert np.isclose(ratio_pulse, expected, atol=1e-12)

    table = drift_phase_table(n_cav=n_cav, duration=duration, model=drift)
    expected_delta = np.asarray([(drift.chi * n + drift.chi2 * (n * (n - 1))) * duration for n in range(n_cav)], dtype=float)
    assert np.allclose(table.phase_delta, expected_delta, atol=1e-12)


def test_t4_time_optimization_improves_conditional_phase_target() -> None:
    n_match = 3
    sub = Subspace.qubit_cavity_block(n_match=n_match)
    drift = {
        "chi": 3.1e6,
        "chi2": -2.0e5,
        "kerr": 1.7e5,
        "kerr2": -0.8e4,
        "frame": "rotating_omega_c_omega_q",
    }

    target_duration = 600e-9
    target_gate = ConditionalPhaseSQR(
        name="target_cp",
        phases_n=[0.0] * (n_match + 1),
        duration=target_duration,
        include_drift=True,
        drift_model=DriftPhaseModel(**drift),
    )
    target = GateSequence(gates=[target_gate], n_cav=n_match + 1).unitary(backend="ideal")

    policy_fixed = {
        "default": {"optimize": False, "bounds": (80e-9, 1.6e-6), "init": 120e-9},
        "CondPhaseSQR": {"optimize": False, "bounds": (80e-9, 1.6e-6), "init": 120e-9},
    }
    no_time = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["CondPhaseSQR"],
        optimize_times=False,
        time_policy=policy_fixed,
        leakage_weight=1.0,
        drift_config=drift,
        seed=77,
    ).fit(target=target, multistart=1, maxiter=160)

    policy_opt = {
        "default": {"optimize": True, "bounds": (80e-9, 1.6e-6), "init": 120e-9},
        "CondPhaseSQR": {"optimize": True, "bounds": (80e-9, 1.6e-6), "init": 120e-9},
    }
    with_time = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["CondPhaseSQR"],
        optimize_times=True,
        time_policy=policy_opt,
        leakage_weight=1.0,
        drift_config=drift,
        seed=77,
    ).fit(target=target, init_guess="random", multistart=4, maxiter=300)

    f_no = no_time.report["metrics"]["fidelity"]
    f_yes = with_time.report["metrics"]["fidelity"]
    assert f_yes > f_no + 1e-4
    assert f_yes > 0.98
    assert with_time.report["metrics"]["leakage_worst"] < 1e-8


def test_t5_report_includes_phase_decomposition() -> None:
    n_match = 3
    sub = Subspace.qubit_cavity_block(n_match=n_match)
    drift = {"chi": 2.2e6, "chi2": -1.6e5, "kerr": 1.1e5, "kerr2": -0.7e4}

    target = GateSequence(
        gates=[
            SQR(
                name="sqr_t",
                theta_n=[0.2] * (n_match + 1),
                phi_n=[0.0] * (n_match + 1),
                duration=300e-9,
                include_conditional_phase=True,
                drift_model=DriftPhaseModel(**drift),
            ),
            ConditionalPhaseSQR(
                name="cp_t",
                phases_n=[0.0] * (n_match + 1),
                duration=350e-9,
                include_drift=True,
                drift_model=DriftPhaseModel(**drift),
            ),
        ],
        n_cav=n_match + 1,
    ).unitary(backend="ideal")

    synth = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["SQR", "CondPhaseSQR"],
        optimize_times=True,
        include_conditional_phase_in_sqr=True,
        drift_config=drift,
        seed=99,
    )
    result = synth.fit(target=target, multistart=1, maxiter=120)

    phase_rows = result.report.get("phase_decomposition", [])
    assert len(phase_rows) >= 2
    for row in phase_rows:
        assert "duration" in row
        assert "phi_g" in row
        assert "phi_e" in row
        assert "delta_phi" in row
        assert "contributions" in row
        assert "chi_chi2" in row["contributions"]
        assert "kerr_kerr2" in row["contributions"]
        assert len(row["n"]) == n_match + 1
