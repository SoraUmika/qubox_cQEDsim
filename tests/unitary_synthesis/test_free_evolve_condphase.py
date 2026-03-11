from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer
from cqed_sim.unitary_synthesis.backends import simulate_sequence
from cqed_sim.unitary_synthesis.metrics import subspace_unitary_fidelity
from cqed_sim.unitary_synthesis.sequence import (
    ConditionalPhaseSQR,
    DriftPhaseModel,
    FreeEvolveCondPhase,
    GateSequence,
    drift_phase_table,
    drift_phase_unitary,
)


def _identity_up_to_global(u: np.ndarray, atol: float = 1e-10) -> bool:
    d = u.shape[0]
    overlap = np.trace(u) / max(d, 1)
    if abs(overlap) == 0.0:
        return False
    phase = np.exp(-1j * np.angle(overlap))
    return np.linalg.norm(phase * u - np.eye(d, dtype=np.complex128)) < atol


def test_fec1_identity_when_drift_terms_absent() -> None:
    rng = np.random.default_rng(123)
    n_cav = 5
    sub = Subspace.qubit_cavity_block(n_match=3, n_cav=n_cav)
    model = DriftPhaseModel(chi=0.0, chi2=0.0, kerr=0.0, kerr2=0.0, delta_c=0.0, delta_q=0.0)

    for t_wait in rng.uniform(20e-9, 900e-9, size=5):
        gate = FreeEvolveCondPhase(name="wait", duration=float(t_wait), drift_model=model)
        seq = GateSequence(gates=[gate], n_cav=n_cav)
        ideal = simulate_sequence(seq, sub, backend="ideal")
        pulse = simulate_sequence(seq, sub, backend="pulse")

        assert _identity_up_to_global(ideal.full_operator)
        assert _identity_up_to_global(pulse.full_operator)

        i_sub = np.eye(sub.dim, dtype=np.complex128)
        f_ideal = subspace_unitary_fidelity(ideal.subspace_operator, i_sub, gauge="global")
        f_pulse = subspace_unitary_fidelity(pulse.subspace_operator, i_sub, gauge="global")
        assert np.isclose(f_ideal, 1.0, atol=1e-12)
        assert np.isclose(f_pulse, 1.0, atol=1e-12)


def test_fec2_analytic_phases_match_pulse_drift_evolution() -> None:
    n_cav = 6
    sub = Subspace.qubit_cavity_block(n_match=4, n_cav=n_cav)
    model = DriftPhaseModel(chi=2.7e6, chi2=-2.2e5, kerr=1.1e5, kerr2=-8.0e3, delta_c=3.0e4, delta_q=-5.0e4)

    for t_wait in [80e-9, 220e-9, 600e-9]:
        gate = FreeEvolveCondPhase(name="wait", duration=t_wait, drift_model=model)
        seq = GateSequence(gates=[gate], n_cav=n_cav)

        ideal = simulate_sequence(seq, sub, backend="ideal")
        pulse = simulate_sequence(seq, sub, backend="pulse")
        analytic = np.asarray(drift_phase_unitary(n_cav=n_cav, duration=t_wait, model=model).full(), dtype=np.complex128)

        assert np.allclose(ideal.full_operator, analytic, atol=1e-12)
        assert np.allclose(pulse.full_operator, analytic, atol=1e-12)
        assert np.allclose(ideal.subspace_operator, pulse.subspace_operator, atol=1e-12)


def test_fec3_relative_phase_scaling() -> None:
    n_cav = 6
    sub = Subspace.qubit_cavity_block(n_match=4, n_cav=n_cav)
    model = DriftPhaseModel(chi=2.9e6, chi2=-1.9e5, kerr=7.0e4, kerr2=-5.0e3, delta_c=0.0, delta_q=0.0)

    t1 = 120e-9
    t2 = 450e-9

    seq1 = GateSequence(gates=[FreeEvolveCondPhase(name="w1", duration=t1, drift_model=model)], n_cav=n_cav)
    seq2 = GateSequence(gates=[FreeEvolveCondPhase(name="w2", duration=t2, drift_model=model)], n_cav=n_cav)
    u1 = simulate_sequence(seq1, sub, backend="pulse").full_operator
    u2 = simulate_sequence(seq2, sub, backend="pulse").full_operator

    table1 = drift_phase_table(n_cav=n_cav, duration=t1, model=model)
    table2 = drift_phase_table(n_cav=n_cav, duration=t2, model=model)

    dt = t2 - t1
    for n in [1, 2, 3]:
        expected = (model.chi * n + model.chi2 * (n * (n - 1))) * dt

        ideal_diff = table2.phase_delta[n] - table1.phase_delta[n]
        assert np.isclose(ideal_diff, expected, atol=1e-12)

        ratio1 = u1[n_cav + n, n_cav + n] / u1[n, n]
        ratio2 = u2[n_cav + n, n_cav + n] / u2[n, n]
        pulse_diff = ratio2 / ratio1
        assert np.isclose(pulse_diff, np.exp(-1j * expected), atol=1e-10)


def test_fec4_equivalence_to_zero_drive_condphase_sqr() -> None:
    n_cav = 5
    sub = Subspace.qubit_cavity_block(n_match=3, n_cav=n_cav)
    model = DriftPhaseModel(chi=2.5e6, chi2=-1.5e5, kerr=0.8e5, kerr2=-6.0e3)
    t_wait = 370e-9

    free_gate = FreeEvolveCondPhase(name="wait", duration=t_wait, drift_model=model)
    cond_gate = ConditionalPhaseSQR(
        name="cond",
        phases_n=[0.0] * n_cav,
        duration=t_wait,
        drift_model=model,
        include_drift=True,
    )

    free_seq = GateSequence(gates=[free_gate], n_cav=n_cav)
    cond_seq = GateSequence(gates=[cond_gate], n_cav=n_cav)

    free_ideal = simulate_sequence(free_seq, sub, backend="ideal")
    cond_ideal = simulate_sequence(cond_seq, sub, backend="ideal")
    free_pulse = simulate_sequence(free_seq, sub, backend="pulse")
    cond_pulse = simulate_sequence(cond_seq, sub, backend="pulse")

    assert np.allclose(free_ideal.full_operator, cond_ideal.full_operator, atol=1e-12)
    assert np.allclose(free_pulse.full_operator, cond_pulse.full_operator, atol=1e-12)
    assert np.allclose(free_ideal.subspace_operator, cond_ideal.subspace_operator, atol=1e-12)
    assert np.allclose(free_pulse.subspace_operator, cond_pulse.subspace_operator, atol=1e-12)


def test_fec5_synthesis_benefits_from_free_evolution_gate() -> None:
    n_match = 3
    sub = Subspace.qubit_cavity_block(n_match=n_match)
    drift = {
        "chi": 2.0e6,
        "chi2": -5.0e4,
        "kerr": 1.1e5,
        "kerr2": -9.0e3,
        "delta_c": 0.0,
        "delta_q": 0.0,
    }

    t_target = 550e-9
    target = GateSequence(
        gates=[FreeEvolveCondPhase(name="target_wait", duration=t_target, drift_model=DriftPhaseModel(**drift))],
        n_cav=n_match + 1,
    ).unitary(backend="ideal")

    no_free = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["SNAP"],
        optimize_times=False,
        leakage_weight=10.0,
        drift_config=drift,
        seed=8,
    ).fit(target=target, multistart=1, maxiter=180)

    with_free = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["FreeEvolveCondPhase"],
        optimize_times=True,
        time_policy={
            "default": {"optimize": True, "bounds": (80e-9, 1.2e-6), "init": 120e-9},
            "FreeEvolveCondPhase": {"optimize": True, "bounds": (80e-9, 1.2e-6), "init": 120e-9},
        },
        leakage_weight=10.0,
        drift_config=drift,
        seed=8,
    ).fit(target=target, init_guess="heuristic", multistart=1, maxiter=240)

    f_no = no_free.report["metrics"]["fidelity"]
    f_with = with_free.report["metrics"]["fidelity"]

    assert f_with > f_no + 0.15
    assert f_with > 0.995
    assert with_free.report["metrics"]["leakage_worst"] < 1e-8
