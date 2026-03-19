"""Tests for the cqed_sim.operators module.

Covers: sigma_x/y/z, tensor_qubit_cavity, embed_qubit_op, embed_cavity_op,
build_qubit_state, joint_basis_state, as_dm, purity,
destroy_cavity, create_cavity, number_operator, fock_projector.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim.operators import (
    as_dm,
    build_qubit_state,
    create_cavity,
    destroy_cavity,
    embed_cavity_op,
    embed_qubit_op,
    fock_projector,
    joint_basis_state,
    number_operator,
    purity,
    sigma_x,
    sigma_y,
    sigma_z,
    tensor_qubit_cavity,
)


# ---------------------------------------------------------------------------
# Pauli operators
# ---------------------------------------------------------------------------

class TestPauliOperators:
    def test_sigma_x_shape_and_hermiticity(self):
        sx = sigma_x()
        assert sx.shape == (2, 2)
        assert sx.isherm
        np.testing.assert_allclose(sx.full(), qt.sigmax().full())

    def test_sigma_y_shape_and_hermiticity(self):
        sy = sigma_y()
        assert sy.shape == (2, 2)
        assert sy.isherm
        np.testing.assert_allclose(sy.full(), qt.sigmay().full())

    def test_sigma_z_shape_and_hermiticity(self):
        sz = sigma_z()
        assert sz.shape == (2, 2)
        assert sz.isherm
        np.testing.assert_allclose(sz.full(), qt.sigmaz().full())

    def test_pauli_anticommutation(self):
        sx, sy, sz = sigma_x(), sigma_y(), sigma_z()
        I2 = qt.qeye(2)
        np.testing.assert_allclose((sx * sy + sy * sx).full(), np.zeros((2, 2)), atol=1e-14)
        np.testing.assert_allclose((sx * sz + sz * sx).full(), np.zeros((2, 2)), atol=1e-14)
        np.testing.assert_allclose((sy * sz + sz * sy).full(), np.zeros((2, 2)), atol=1e-14)

    def test_pauli_squares_are_identity(self):
        I2 = qt.qeye(2).full()
        for op in (sigma_x(), sigma_y(), sigma_z()):
            np.testing.assert_allclose((op * op).full(), I2, atol=1e-14)


# ---------------------------------------------------------------------------
# Tensor and embedding
# ---------------------------------------------------------------------------

class TestTensorAndEmbed:
    def test_tensor_qubit_cavity_dims(self):
        op_q = qt.sigmax()
        op_c = qt.destroy(5)
        result = tensor_qubit_cavity(op_q, op_c)
        assert result.dims == [[2, 5], [2, 5]]

    def test_embed_qubit_op_dims(self):
        op = embed_qubit_op(qt.sigmaz(), 4)
        assert op.dims == [[2, 4], [2, 4]]
        # Check that qubit part acts correctly on |g> ⊗ |0>
        psi = qt.tensor(qt.basis(2, 0), qt.basis(4, 0))
        result = op * psi
        expected = qt.tensor(qt.sigmaz() * qt.basis(2, 0), qt.basis(4, 0))
        np.testing.assert_allclose(result.full(), expected.full(), atol=1e-14)

    def test_embed_cavity_op_dims(self):
        a = qt.destroy(5)
        op = embed_cavity_op(a, n_tr=2)
        assert op.dims == [[2, 5], [2, 5]]


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

class TestBuildQubitState:
    @pytest.mark.parametrize("label", ["g", "e", "+x", "-x", "+y", "-y"])
    def test_valid_labels_produce_unit_vectors(self, label):
        psi = build_qubit_state(label)
        assert psi.shape == (2, 1)
        np.testing.assert_allclose(psi.norm(), 1.0, atol=1e-14)

    def test_g_is_ground(self):
        np.testing.assert_allclose(build_qubit_state("g").full(), qt.basis(2, 0).full())

    def test_e_is_excited(self):
        np.testing.assert_allclose(build_qubit_state("e").full(), qt.basis(2, 1).full())

    def test_plus_x_superposition(self):
        psi = build_qubit_state("+x")
        # |+x> = (|g> + |e>) / sqrt(2)
        expected = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        np.testing.assert_allclose(psi.full(), expected.full(), atol=1e-14)

    def test_invalid_label_raises(self):
        with pytest.raises(ValueError, match="Unsupported qubit label"):
            build_qubit_state("invalid")


class TestJointBasisState:
    def test_ground_zero_photon(self):
        psi = joint_basis_state(5, "g", 0)
        expected = qt.tensor(qt.basis(2, 0), qt.basis(5, 0))
        np.testing.assert_allclose(psi.full(), expected.full(), atol=1e-14)

    def test_excited_two_photon(self):
        psi = joint_basis_state(5, "e", 2)
        expected = qt.tensor(qt.basis(2, 1), qt.basis(5, 2))
        np.testing.assert_allclose(psi.full(), expected.full(), atol=1e-14)

    def test_dims(self):
        psi = joint_basis_state(4, "g", 1)
        assert psi.dims[0] == [2, 4]


# ---------------------------------------------------------------------------
# Density matrix and purity
# ---------------------------------------------------------------------------

class TestAsDmAndPurity:
    def test_as_dm_from_ket(self):
        psi = qt.basis(2, 0)
        rho = as_dm(psi)
        assert rho.isoper
        np.testing.assert_allclose(rho.full(), (psi * psi.dag()).full(), atol=1e-14)

    def test_as_dm_from_dm_is_identity(self):
        rho = qt.ket2dm(qt.basis(2, 0))
        assert as_dm(rho) is rho

    def test_purity_of_pure_state(self):
        psi = qt.basis(2, 0)
        np.testing.assert_allclose(purity(psi), 1.0, atol=1e-12)

    def test_purity_of_maximally_mixed(self):
        rho = qt.qeye(2) / 2
        np.testing.assert_allclose(purity(rho), 0.5, atol=1e-12)


# ---------------------------------------------------------------------------
# Cavity operators
# ---------------------------------------------------------------------------

class TestCavityOperators:
    def test_destroy_cavity_shape(self):
        a = destroy_cavity(5)
        assert a.shape == (5, 5)

    def test_create_cavity_is_dagger(self):
        a = destroy_cavity(5)
        adag = create_cavity(5)
        np.testing.assert_allclose(adag.full(), a.dag().full(), atol=1e-14)

    def test_number_operator_eigenvalues(self):
        N = 6
        n_op = number_operator(N)
        eigenvalues = n_op.eigenenergies()
        np.testing.assert_allclose(sorted(eigenvalues), np.arange(N), atol=1e-12)

    def test_number_operator_expectation(self):
        n_op = number_operator(5)
        for n in range(5):
            psi = qt.basis(5, n)
            np.testing.assert_allclose(qt.expect(n_op, psi), float(n), atol=1e-12)

    def test_fock_projector_is_projector(self):
        P = fock_projector(5, 2)
        # P^2 = P
        np.testing.assert_allclose((P * P).full(), P.full(), atol=1e-14)
        # Tr(P) = 1
        np.testing.assert_allclose(P.tr(), 1.0, atol=1e-14)

    def test_fock_projector_picks_correct_state(self):
        P = fock_projector(5, 3)
        for n in range(5):
            psi = qt.basis(5, n)
            expected = 1.0 if n == 3 else 0.0
            np.testing.assert_allclose(qt.expect(P, psi), expected, atol=1e-14)

    def test_commutation_relation(self):
        N = 10
        a = destroy_cavity(N)
        adag = create_cavity(N)
        commutator = a * adag - adag * a
        # [a, a†] = 1 holds for all but the last diagonal entry
        # in a truncated Fock space (finite-N artifact).
        diag = np.diag(commutator.full().real)
        np.testing.assert_allclose(diag[:N - 1], 1.0, atol=1e-12)
