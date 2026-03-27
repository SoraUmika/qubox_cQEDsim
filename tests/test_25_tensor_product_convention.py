from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_dims, qubit_cavity_index
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.ideal_gates import embed_cavity_op, embed_qubit_op, sqr_op
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.gates.coupled import (
    conditional_displacement,
    conditional_rotation,
    controlled_parity,
    controlled_snap,
    dispersive_phase,
    jaynes_cummings,
    multi_sqr,
    sqr,
)
from cqed_sim.operators.basic import joint_basis_state, tensor_qubit_cavity
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.extractors import conditioned_bloch_xyz, reduced_cavity_state, reduced_qubit_state
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence

ATOL = 1e-10


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def test_basis_state_flat_indices_follow_qubit_then_cavity():
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)

    expected = {
        (0, 0): 0,
        (1, 0): 4,
        (0, 1): 1,
    }
    for (q, n), flat_index in expected.items():
        psi = model.basis_state(q, n)
        manual = qt.tensor(qt.basis(2, q), qt.basis(4, n))
        assert (psi - manual).norm() < 1e-12
        assert int(np.argmax(np.abs(np.asarray(psi.full()).ravel()))) == flat_index


def test_operator_embeddings_act_on_qubit_and_cavity_slots():
    n_cav = 5
    state = qt.tensor(qt.basis(2, 1), qt.basis(n_cav, 2))

    z_q = embed_qubit_op(qt.sigmaz(), n_cav)
    n_c = embed_cavity_op(qt.num(n_cav), 2)

    assert np.isclose(qt.expect(z_q, state), -1.0, atol=1e-12)
    assert np.isclose(qt.expect(n_c, state), 2.0, atol=1e-12)


def test_static_hamiltonian_diagonal_uses_qubit_cavity_order():
    omega_c = 1.7
    omega_q = 5.2
    chi = 0.4
    kerr = -0.15
    n_cav = 4

    model = DispersiveTransmonCavityModel(
        omega_c=omega_c,
        omega_q=omega_q,
        alpha=0.0,
        chi=chi,
        kerr=kerr,
        n_cav=n_cav,
        n_tr=2,
    )
    diag = np.real(np.diag(np.asarray(model.static_hamiltonian().full(), dtype=np.complex128)))

    assert np.isclose(diag[0], 0.0, atol=1e-12)  # |g,0>
    assert np.isclose(diag[1], omega_c, atol=1e-12)  # |g,1>
    assert np.isclose(diag[2], 2.0 * omega_c + kerr, atol=1e-12)  # |g,2>
    assert np.isclose(diag[n_cav], omega_q, atol=1e-12)  # |e,0>
    assert np.isclose(diag[n_cav + 1], omega_q + omega_c + chi, atol=1e-12)  # |e,1>


def test_conditioned_bloch_uses_fock_projector_on_second_subsystem():
    n_cav = 4
    state = qt.tensor((qt.basis(2, 0) + qt.basis(2, 1)).unit(), qt.basis(n_cav, 2))

    x2, y2, z2, p2, valid2 = conditioned_bloch_xyz(state, n=2)
    x0, y0, z0, p0, valid0 = conditioned_bloch_xyz(state, n=0, fallback="nan")

    assert valid2
    assert np.isclose(p2, 1.0, atol=1e-12)
    assert np.allclose([x2, y2, z2], [1.0, 0.0, 0.0], atol=1e-12)
    assert not valid0
    assert np.isclose(p0, 0.0, atol=1e-12)
    assert np.isnan(x0) and np.isnan(y0) and np.isnan(z0)


def test_state_prep_evolution_and_measurement_keep_qubit_cavity_consistent():
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    pulse = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=1.1)

    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 1),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec()),
    )

    rho_q = reduced_qubit_state(result.final_state)
    rho_c = reduced_cavity_state(result.final_state)

    assert np.isclose(float(np.real((rho_c * qt.basis(3, 1).proj()).tr())), 1.0, atol=1e-6)
    assert np.isclose(float(np.real((rho_c * qt.num(3)).tr())), 1.0, atol=1e-6)
    assert np.allclose(
        [
            float(np.real((rho_q * qt.sigmax()).tr())),
            float(np.real((rho_q * qt.sigmay()).tr())),
            float(np.real((rho_q * qt.sigmaz()).tr())),
        ],
        [0.0, -1.0, 0.0],
        atol=5e-2,
    )


# ── Regression tests: subsystem ordering is qubit ⊗ cavity everywhere ────


def test_conventions_qubit_cavity_dims_shape():
    """qubit_cavity_dims returns [n_qubit, n_cav] for both row and column."""
    dims = qubit_cavity_dims(2, 5)
    assert dims == [[2, 5], [2, 5]]


def test_conventions_flat_index_qubit_major():
    """Flat index = q * n_cav + n confirms qubit is major index."""
    n_cav = 6
    assert qubit_cavity_index(n_cav, 0, 0) == 0   # |g,0>
    assert qubit_cavity_index(n_cav, 0, 3) == 3   # |g,3>
    assert qubit_cavity_index(n_cav, 1, 0) == 6   # |e,0>
    assert qubit_cavity_index(n_cav, 1, 5) == 11  # |e,5>


def test_joint_basis_state_matches_tensor_qubit_then_cavity():
    """joint_basis_state produces qt.tensor(qubit_ket, cavity_ket)."""
    n_cav = 5
    for q_label in ["g", "e"]:
        q_idx = 0 if q_label == "g" else 1
        for n in range(n_cav):
            state = joint_basis_state(n_cav, q_label, n)
            manual = qt.tensor(qt.basis(2, q_idx), qt.basis(n_cav, n))
            assert (state - manual).norm() < ATOL


def test_tensor_qubit_cavity_helper_order():
    """tensor_qubit_cavity(op_q, op_c) == qt.tensor(op_q, op_c)."""
    op_q = qt.sigmaz()
    op_c = qt.num(4)
    result = tensor_qubit_cavity(op_q, op_c)
    expected = qt.tensor(op_q, op_c)
    np.testing.assert_allclose(result.full(), expected.full(), atol=ATOL)


def test_partial_trace_qubit_is_subsystem_zero():
    """Tracing over cavity (subsystem 1) yields the qubit state."""
    n_cav = 4
    psi_q = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
    psi_c = qt.basis(n_cav, 2)
    psi = qt.tensor(psi_q, psi_c)
    rho_q = reduced_qubit_state(psi)
    np.testing.assert_allclose(rho_q.full(), (psi_q * psi_q.dag()).full(), atol=ATOL)


def test_partial_trace_cavity_is_subsystem_one():
    """Tracing over qubit (subsystem 0) yields the cavity state."""
    n_cav = 4
    psi_q = qt.basis(2, 1)
    psi_c = qt.coherent(n_cav, 0.5)
    psi = qt.tensor(psi_q, psi_c)
    rho_c = reduced_cavity_state(psi)
    np.testing.assert_allclose(rho_c.full(), (psi_c * psi_c.dag()).full(), atol=1e-8)


# ── SQR gate ordering regression ─────────────────────────────────────────


def test_sqr_uses_qubit_first_cavity_second():
    """sqr() from gates.coupled must use qubit ⊗ cavity ordering."""
    n_cav = 5
    U = sqr(np.pi, 0.0, n=2, cavity_dim=n_cav)
    # Verify dimensions match qubit_dim * cavity_dim
    assert U.shape == (2 * n_cav, 2 * n_cav)

    # |g,2> should map to -i|e,2> for Rx(pi)
    psi_in = qt.tensor(qt.basis(2, 0), qt.basis(n_cav, 2))
    psi_out = U * psi_in
    expected = qt.tensor(-1j * qt.basis(2, 1), qt.basis(n_cav, 2))
    np.testing.assert_allclose(psi_out.full(), expected.full(), atol=ATOL)

    # |g,0> should be unchanged (non-target level)
    psi_in0 = qt.tensor(qt.basis(2, 0), qt.basis(n_cav, 0))
    psi_out0 = U * psi_in0
    np.testing.assert_allclose(psi_out0.full(), psi_in0.full(), atol=ATOL)


def test_multi_sqr_uses_qubit_first_cavity_second():
    """multi_sqr() from gates.coupled must use qubit ⊗ cavity ordering."""
    n_cav = 4
    thetas = np.zeros(n_cav)
    phis = np.zeros(n_cav)
    U = multi_sqr(thetas, phis, n_cav)
    # Identity should match qt.tensor(qubit_eye, cavity_eye)
    expected_id = qt.tensor(qt.qeye(2), qt.qeye(n_cav))
    np.testing.assert_allclose(U.full(), expected_id.full(), atol=ATOL)


def test_sqr_composable_with_other_qubit_cavity_gates():
    """sqr() and dispersive_phase() live in the same qubit ⊗ cavity space."""
    n_cav = 4
    U_sqr = sqr(np.pi / 2, 0.0, n=1, cavity_dim=n_cav)
    U_disp = dispersive_phase(1.0, 1.0, n_cav)
    composite = U_disp * U_sqr
    # Must produce a valid unitary in the same space
    prod = (composite.dag() * composite).full()
    np.testing.assert_allclose(prod, np.eye(2 * n_cav), atol=ATOL)


def test_sqr_op_and_sqr_produce_same_unitary():
    """sqr_op from ideal_gates and sqr from gates.coupled must agree."""
    n_cav = 4
    thetas = np.array([0.0, np.pi / 3, 0.0, -np.pi / 4])
    phis = np.array([0.0, 0.5, 0.0, 1.2])
    U_dense = sqr_op(thetas, phis)
    U_sparse = multi_sqr(thetas, phis, n_cav)
    np.testing.assert_allclose(U_dense.full(), U_sparse.full(), atol=ATOL)


# ── Gate dimension consistency ────────────────────────────────────────────


def test_all_coupled_gates_have_qubit_times_cavity_dimension():
    """All qubit-cavity gates produce (qubit_dim * cavity_dim) square matrices."""
    n_cav = 5
    q_dim = 2
    expected_dim = q_dim * n_cav

    gates = [
        dispersive_phase(0.1, 1.0, n_cav),
        conditional_rotation(0.5, n_cav),
        sqr(np.pi, 0.0, n=0, cavity_dim=n_cav),
        multi_sqr(np.zeros(n_cav), np.zeros(n_cav), n_cav),
        controlled_parity(n_cav),
        controlled_snap({1: 0.5}, n_cav),
        conditional_displacement(0.3, cavity_dim=n_cav),
        jaynes_cummings(0.01, 1.0, n_cav),
    ]
    for gate in gates:
        assert gate.shape == (expected_dim, expected_dim), (
            f"Gate with shape {gate.shape} does not match expected ({expected_dim}, {expected_dim})"
        )


# ── Embedding consistency ─────────────────────────────────────────────────


def test_embed_qubit_op_acts_only_on_qubit():
    """embed_qubit_op(sigma_x) should flip qubit while leaving cavity unchanged."""
    n_cav = 4
    X = embed_qubit_op(qt.sigmax(), n_cav)
    psi = qt.tensor(qt.basis(2, 0), qt.basis(n_cav, 3))  # |g,3>
    result = X * psi
    expected = qt.tensor(qt.basis(2, 1), qt.basis(n_cav, 3))  # |e,3>
    np.testing.assert_allclose(result.full(), expected.full(), atol=ATOL)


def test_embed_cavity_op_acts_only_on_cavity():
    """embed_cavity_op(a) should lower cavity n while leaving qubit unchanged."""
    n_cav = 4
    a_emb = embed_cavity_op(qt.destroy(n_cav), 2)
    psi = qt.tensor(qt.basis(2, 1), qt.basis(n_cav, 2))  # |e,2>
    result = a_emb * psi
    expected = np.sqrt(2) * qt.tensor(qt.basis(2, 1), qt.basis(n_cav, 1))  # sqrt(2)|e,1>
    np.testing.assert_allclose(result.full(), expected.full(), atol=ATOL)
