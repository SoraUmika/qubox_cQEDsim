from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.unitary_synthesis.backends import simulate_sequence
from cqed_sim.unitary_synthesis.sequence import (
    CavityBlockPhase,
    ConditionalPhaseSQR,
    Displacement,
    DriftPhaseModel,
    FreeEvolveCondPhase,
    GateSequence,
    QubitRotation,
    SNAP,
    SQR,
)
from cqed_sim.unitary_synthesis.subspace import Subspace


def _is_unitary(u: np.ndarray, tol: float = 1e-9) -> bool:
    i = np.eye(u.shape[0], dtype=np.complex128)
    return np.linalg.norm(u.conj().T @ u - i) < tol


def test_c1_qubit_rotation_correctness() -> None:
    n_cav = 4
    g1 = QubitRotation(name="r0", theta=0.0, phi=0.2, duration=50e-9)
    g2 = QubitRotation(name="rx", theta=np.pi, phi=0.0, duration=50e-9)
    g3 = QubitRotation(name="ry", theta=np.pi, phi=np.pi / 2, duration=50e-9)
    for gate in [g1, g2, g3]:
        u = np.asarray(gate.ideal_unitary(n_cav).full())
        assert _is_unitary(u)
    u0 = np.asarray(g1.ideal_unitary(n_cav).full())
    assert np.allclose(u0, np.eye(2 * n_cav), atol=1e-12)


def test_c2_displacement_sanity() -> None:
    n_cav = 22
    g0 = Displacement(name="d0", alpha=0.0, duration=100e-9)
    u0 = np.asarray(g0.ideal_unitary(n_cav).full())
    assert np.allclose(u0, np.eye(2 * n_cav), atol=1e-12)

    alpha = 0.3 + 0.2j
    beta = -0.1 + 0.25j
    d1 = qt.displace(n_cav, alpha)
    d2 = qt.displace(n_cav, beta)
    lhs = d1 * d2
    rhs = np.exp(1j * np.imag(alpha * np.conjugate(beta))) * qt.displace(n_cav, alpha + beta)
    lhs_arr = lhs.full()
    rhs_arr = rhs.full()
    keep = 12
    assert np.allclose(lhs_arr[:keep, :keep], rhs_arr[:keep, :keep], atol=2e-2)

    vacuum = qt.basis(n_cav, 0)
    state = qt.displace(n_cav, alpha) * vacuum
    n_exp = qt.expect(qt.num(n_cav), state)
    assert np.isclose(n_exp, abs(alpha) ** 2, atol=2e-2)


def test_c3_snap_diagonal_and_composition() -> None:
    n_cav = 6
    p1 = np.linspace(0, 0.4, n_cav)
    p2 = np.linspace(-0.1, 0.2, n_cav)
    g1 = SNAP(name="s1", phases=list(p1), duration=100e-9)
    g2 = SNAP(name="s2", phases=list(p2), duration=100e-9)
    u1 = np.asarray(g1.ideal_unitary(n_cav).full())
    assert np.allclose(u1, np.diag(np.diag(u1)))
    u12 = np.asarray((g2.ideal_unitary(n_cav) * g1.ideal_unitary(n_cav)).full())
    g3 = SNAP(name="s3", phases=list((p1 + p2 + np.pi) % (2 * np.pi) - np.pi), duration=100e-9)
    u3 = np.asarray(g3.ideal_unitary(n_cav).full())
    assert np.allclose(np.abs(np.diag(u12)), np.abs(np.diag(u3)), atol=1e-12)


def test_c3b_cavity_block_phase_targets_selected_blocks() -> None:
    n_cav = 5
    gate = CavityBlockPhase(name="bp", phases=[0.2, -0.45], fock_levels=(1, 3), duration=80e-9)
    u = np.asarray(gate.ideal_unitary(n_cav).full())
    for level in range(n_cav):
        idx = [level, n_cav + level]
        block = u[np.ix_(idx, idx)]
        if level == 1:
            expected = np.exp(1j * 0.2) * np.eye(2)
        elif level == 3:
            expected = np.exp(1j * -0.45) * np.eye(2)
        else:
            expected = np.eye(2)
        assert np.allclose(block, expected, atol=1e-12)
    report = gate.phase_decomposition(n_cav)
    assert report is not None
    assert report["realization"] == "ideal-only"


def test_c3c_cavity_block_phase_ideal_backend_matches_gate() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2, n_cav=4)
    gate = CavityBlockPhase(name="bp", phases=[0.0, 0.35], fock_levels=(0, 2), duration=60e-9)
    seq = GateSequence(gates=[gate], n_cav=4)
    res = simulate_sequence(seq, sub, backend="ideal")
    target = np.asarray(gate.ideal_unitary(4).full())
    assert np.allclose(res.full_operator, target, atol=1e-12)


def test_c4_sqr_block_structure() -> None:
    n_cav = 5
    theta = [0.0, 0.1, 0.0, 0.2, 0.0]
    phi = [0.0, 0.0, 0.0, 0.3, 0.0]
    g = SQR(name="q", theta_n=theta, phi_n=phi, duration=100e-9)
    u = np.asarray(g.ideal_unitary(n_cav).full())
    for n in range(n_cav):
        idx = [n, n_cav + n]
        block = u[np.ix_(idx, idx)]
        offdiag_norm = np.linalg.norm(u[idx, :]) - np.linalg.norm(block)
        assert abs(offdiag_norm) < 1e-8
        if np.isclose(theta[n], 0.0):
            assert np.allclose(block, np.eye(2), atol=1e-12)


def test_d1_pulse_backend_returns_unitary_without_dissipation() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    seq = GateSequence(gates=[QubitRotation(name="r", theta=0.4, phi=0.1, duration=80e-9)], n_cav=3)
    res = simulate_sequence(seq, sub, backend="pulse")
    assert _is_unitary(res.full_operator)


def test_d2_pulse_vs_ideal_agree_reference_duration() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    seq = GateSequence(gates=[QubitRotation(name="r", theta=0.4, phi=0.2, duration=120e-9)], n_cav=3)
    ideal = simulate_sequence(seq, sub, backend="ideal")
    pulse = simulate_sequence(seq, sub, backend="pulse")
    assert np.allclose(ideal.subspace_operator, pulse.subspace_operator, atol=1e-10)


def test_d3_solver_convergence_proxy() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    seq = GateSequence(
        gates=[QubitRotation(name="r1", theta=0.4, phi=0.1, duration=100e-9), QubitRotation(name="r2", theta=0.2, phi=-0.4, duration=90e-9)],
        n_cav=3,
    )
    coarse = simulate_sequence(seq, sub, backend="pulse", dt=10e-9)
    fine = simulate_sequence(seq, sub, backend="pulse", dt=1e-9)
    assert np.linalg.norm(coarse.subspace_operator - fine.subspace_operator) < 1e-8


def test_d4_truncation_stability() -> None:
    sub_small = Subspace.qubit_cavity_block(n_match=2, n_cav=4)
    sub_big = Subspace.qubit_cavity_block(n_match=2, n_cav=6)
    seq_small = GateSequence(gates=[Displacement(name="d", alpha=0.05 + 0.02j, duration=120e-9)], n_cav=4)
    seq_big = GateSequence(gates=[Displacement(name="d", alpha=0.05 + 0.02j, duration=120e-9)], n_cav=6)
    r1 = simulate_sequence(seq_small, sub_small, backend="pulse")
    r2 = simulate_sequence(seq_big, sub_big, backend="pulse")
    assert np.linalg.norm(r1.subspace_operator - r2.subspace_operator) < 5e-2


def test_conditional_phase_gate_exists_and_is_unitary() -> None:
    gate = ConditionalPhaseSQR(name="cp", phases_n=[0.1, -0.2, 0.3], duration=100e-9)
    u = np.asarray(gate.ideal_unitary(3).full())
    assert _is_unitary(u)


def test_free_evolve_condphase_gate_exists_and_is_unitary() -> None:
    gate = FreeEvolveCondPhase(
        name="wait",
        duration=120e-9,
        drift_model=DriftPhaseModel(chi=2.1e6, chi2=-1.2e5, kerr=0.8e5, kerr2=-6.0e3),
    )
    u = np.asarray(gate.ideal_unitary(4).full())
    up = np.asarray(gate.pulse_unitary(4).full())
    assert _is_unitary(u)
    assert _is_unitary(up)
