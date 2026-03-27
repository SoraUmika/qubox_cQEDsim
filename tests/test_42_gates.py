"""
Comprehensive physics-aware tests for the cqed_sim.gates library.

Covers:
  1. Unitarity for all gate families
  2. Basis-action (known eigenvalue / transformation) tests
  3. Dimensional-structure tests
  4. Convention / sign / ordering tests
  5. API ergonomics (dict vs. dense, keyword aliases)
  6. Cross-checks against analytic forms
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

# Top-level imports so the public API is what is exercised
import cqed_sim as cs

# ─── tolerance ───────────────────────────────────────────────────────────────
ATOL = 1e-10


def assert_unitary(U: qt.Qobj, atol: float = ATOL) -> None:
    """Assert U†U ≈ I."""
    dim = U.shape[0]
    prod = (U.dag() * U).full()
    np.testing.assert_allclose(prod, np.eye(dim), atol=atol,
                               err_msg="U†U ≠ I (gate is not unitary)")


def assert_close(A: qt.Qobj, B: qt.Qobj, atol: float = ATOL) -> None:
    np.testing.assert_allclose(A.full(), B.full(), atol=atol)


def ket(dim: int, n: int) -> qt.Qobj:
    return qt.basis(dim, n)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Single-qubit gates
# ─────────────────────────────────────────────────────────────────────────────

class TestSingleQubitGates:
    """Tests for cqed_sim.gates.qubit.*"""

    def test_identity_shape(self):
        I = cs.identity_gate()
        assert I.shape == (2, 2)
        assert_unitary(I)

    def test_pauli_unitarity(self):
        for G in [cs.x_gate(), cs.y_gate(), cs.z_gate()]:
            assert G.shape == (2, 2)
            assert_unitary(G)

    def test_h_gate_unitarity_and_shape(self):
        H = cs.h_gate()
        assert H.shape == (2, 2)
        assert_unitary(H)

    def test_s_gate_unitarity(self):
        assert_unitary(cs.s_gate())
        assert_unitary(cs.s_dag_gate())

    def test_s_squared_is_z(self):
        """S² = Z (up to global phase consistent with exact matrices)."""
        S2 = cs.s_gate() * cs.s_gate()
        Z = cs.z_gate()
        # They should be equal (both diag(1,-1) after squaring diag(1,i))
        assert_close(S2, Z)

    def test_s_dag_is_s_inverse(self):
        assert_close(cs.s_gate() * cs.s_dag_gate(), cs.identity_gate())

    def test_t_gate_unitarity(self):
        assert_unitary(cs.t_gate())
        assert_unitary(cs.t_dag_gate())

    def test_t_squared_is_s(self):
        T2 = cs.t_gate() * cs.t_gate()
        assert_close(T2, cs.s_gate())

    def test_t_dag_is_t_inverse(self):
        assert_close(cs.t_gate() * cs.t_dag_gate(), cs.identity_gate())

    def test_rx_unitarity(self):
        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]:
            assert_unitary(cs.rx(theta))

    def test_ry_unitarity(self):
        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]:
            assert_unitary(cs.ry(theta))

    def test_rz_unitarity(self):
        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi, 2 * np.pi]:
            assert_unitary(cs.rz(theta))

    def test_rphi_unitarity(self):
        for theta in [0.0, np.pi / 2, np.pi]:
            for phi in [0.0, np.pi / 4, np.pi / 2, np.pi]:
                assert_unitary(cs.rphi(theta, phi))

    def test_rx_pi_flips_g_to_minus_i_e(self):
        """Rx(π)|g⟩ = -i|e⟩  (standard rotation convention)."""
        g = ket(2, 0)
        e = ket(2, 1)
        result = cs.rx(np.pi) * g
        expected = -1j * e
        assert_close(result, expected)

    def test_ry_pi_flips_g_to_minus_e(self):
        """Ry(π)|g⟩ = -i|e⟩ with real axis convention... actually Ry(π)|g⟩ = |e⟩."""
        # R_y(π) = [[cos(π/2), -sin(π/2)], [sin(π/2), cos(π/2)]] ... no, let's compute
        # R_y(θ) = exp(-iθ/2 Y) = cos(θ/2) I - i sin(θ/2) Y
        # Y = [[0,-i],[i,0]]; -iY = [[0,-1],[1,0]]
        # At θ=π: 0*I + (-i)*(-i)*[[0,1],[-1,0]] ... let me just check numerically
        g = ket(2, 0)
        result = cs.ry(np.pi) * g
        # exp(-iπ/2 Y)|0> = (cos(π/2)I - i sin(π/2) Y)|0> = -i Y |0> = -i * i|1> = |1>
        expected = -1j * (1j * ket(2, 1))  # -i * (i|e>) = |e>
        # Actually: -i*Y|g> = -i*(i|e>) = |e>  (since Y|g> = i|e>)
        expected = ket(2, 1)
        assert_close(result, expected)

    def test_rx_at_zero_is_identity(self):
        assert_close(cs.rx(0.0), cs.identity_gate())

    def test_rphi_at_phi0_equals_rx(self):
        for theta in [0.3, np.pi / 3, np.pi]:
            assert_close(cs.rphi(theta, 0.0), cs.rx(theta))

    def test_rphi_at_phi_halfpi_equals_ry(self):
        for theta in [0.3, np.pi / 3, np.pi]:
            assert_close(cs.rphi(theta, np.pi / 2), cs.ry(theta))

    def test_h_maps_g_to_plus_x(self):
        """H|g⟩ = |+x⟩ = (|g⟩+|e⟩)/√2."""
        plus_x = (ket(2, 0) + ket(2, 1)).unit()
        result = cs.h_gate() * ket(2, 0)
        assert_close(result, plus_x)

    def test_h_maps_e_to_minus_x(self):
        """H|e⟩ = |-x⟩ = (|g⟩-|e⟩)/√2."""
        minus_x = (ket(2, 0) - ket(2, 1)).unit()
        result = cs.h_gate() * ket(2, 1)
        assert_close(result, minus_x)

    def test_x_gate_not(self):
        """X|g⟩ = |e⟩, X|e⟩ = |g⟩."""
        assert_close(cs.x_gate() * ket(2, 0), ket(2, 1))
        assert_close(cs.x_gate() * ket(2, 1), ket(2, 0))

    def test_z_gate_phases(self):
        """Z|g⟩ = |g⟩, Z|e⟩ = -|e⟩."""
        assert_close(cs.z_gate() * ket(2, 0), ket(2, 0))
        assert_close(cs.z_gate() * ket(2, 1), -ket(2, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Multilevel transmon gates
# ─────────────────────────────────────────────────────────────────────────────

class TestTransmonGates:
    """Tests for cqed_sim.gates.transmon.*"""

    def test_transition_rotation_shape(self):
        U = cs.transition_rotation(3, 0, 1, np.pi / 2)
        assert U.shape == (3, 3)

    def test_transition_rotation_unitarity(self):
        for dim in [2, 3, 4, 5]:
            for theta in [0.1, np.pi / 2, np.pi]:
                for phi in [0.0, np.pi / 4]:
                    U = cs.transition_rotation(dim, 0, 1, theta, phi)
                    assert_unitary(U)

    def test_transition_rotation_leaves_other_levels_alone(self):
        """Levels not involved in the transition are unaffected."""
        dim = 4
        U = cs.transition_rotation(dim, 0, 1, np.pi)
        # Level 2 should pass through unchanged
        lev2 = ket(dim, 2)
        assert_close(U * lev2, lev2)
        # Level 3 should pass through unchanged
        lev3 = ket(dim, 3)
        assert_close(U * lev3, lev3)

    def test_transition_rotation_acts_on_subspace(self):
        """Full π rotation between levels 0 and 1 maps |0⟩ → ±i|1⟩."""
        dim = 3
        U = cs.transition_rotation(dim, 0, 1, np.pi, phi=0.0)
        result = U * ket(dim, 0)
        # Should be -i|1⟩ (X-like rotation); use numpy inner product (QuTiP-version-safe)
        overlap = abs(np.dot(ket(dim, 1).full().flatten().conj(), result.full().flatten()))
        assert abs(overlap - 1.0) < ATOL

    def test_r_ge_dim2_matches_rphi(self):
        """At dim=2, r_ge should match rphi from the qubit module."""
        for theta in [0.0, np.pi / 4, np.pi / 2, np.pi]:
            for phi in [0.0, np.pi / 4, np.pi / 2]:
                assert_close(cs.r_ge(theta, phi, dim=2), cs.rphi(theta, phi))

    def test_r_ge_unitarity(self):
        for theta in [0.0, np.pi / 2, np.pi]:
            assert_unitary(cs.r_ge(theta, phi=0.0, dim=3))

    def test_r_ef_unitarity(self):
        for theta in [0.0, np.pi / 2, np.pi]:
            assert_unitary(cs.r_ef(theta, phi=0.0, dim=3))

    def test_r_ef_requires_dim_ge_3(self):
        with pytest.raises(ValueError):
            cs.r_ef(np.pi, dim=2)

    def test_r_ef_leaves_g_alone(self):
        """r_ef acts on e-f subspace; |g⟩ = |0⟩ should be unchanged."""
        U = cs.r_ef(np.pi, phi=0.0, dim=3)
        g = ket(3, 0)
        assert_close(U * g, g)

    def test_transition_rotation_invalid_levels(self):
        with pytest.raises(ValueError):
            cs.transition_rotation(3, 0, 5, np.pi)

    def test_transition_rotation_same_level_raises(self):
        with pytest.raises(ValueError):
            cs.transition_rotation(3, 1, 1, np.pi)

    def test_r_ge_ef_successive_pi_pulses(self):
        """Two π-pulses: first g→e, then e→f in dim=3."""
        dim = 3
        U_ge = cs.r_ge(np.pi, 0.0, dim=dim)
        U_ef = cs.r_ef(np.pi, 0.0, dim=dim)
        # Start in |g⟩, end up in |f⟩ (up to global phase)
        state = U_ef * U_ge * ket(dim, 0)
        overlap = abs(np.dot(ket(dim, 2).full().flatten().conj(), state.full().flatten()))
        assert abs(overlap - 1.0) < ATOL


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Bosonic cavity gates
# ─────────────────────────────────────────────────────────────────────────────

class TestBosonicGates:
    """Tests for cqed_sim.gates.bosonic.*"""

    DIM = 12

    def test_displacement_shape(self):
        U = cs.displacement(0.5 + 0.3j, self.DIM)
        assert U.shape == (self.DIM, self.DIM)

    def test_displacement_unitarity(self):
        for alpha in [0.0, 0.5, 1.0 + 0.5j, -0.3 - 0.7j]:
            assert_unitary(cs.displacement(alpha, self.DIM))

    def test_displacement_at_zero_is_identity(self):
        assert_close(cs.displacement(0.0, self.DIM), qt.qeye(self.DIM))

    def test_oscillator_rotation_shape(self):
        U = cs.oscillator_rotation(np.pi / 4, self.DIM)
        assert U.shape == (self.DIM, self.DIM)

    def test_oscillator_rotation_unitarity(self):
        for theta in [0.0, np.pi / 4, np.pi, 2 * np.pi]:
            assert_unitary(cs.oscillator_rotation(theta, self.DIM))

    def test_oscillator_rotation_fock_eigenvalues(self):
        """R(θ)|n⟩ = exp(-inθ)|n⟩."""
        theta = np.pi / 3
        U = cs.oscillator_rotation(theta, self.DIM)
        for n in range(self.DIM):
            result = U * ket(self.DIM, n)
            expected = np.exp(-1j * n * theta) * ket(self.DIM, n)
            assert_close(result, expected)

    def test_oscillator_rotation_2pi_is_identity(self):
        """R(2π) = I (up to numerical precision)."""
        assert_close(cs.oscillator_rotation(2 * np.pi, self.DIM), qt.qeye(self.DIM))

    def test_parity_shape(self):
        assert cs.parity(self.DIM).shape == (self.DIM, self.DIM)

    def test_parity_unitarity(self):
        assert_unitary(cs.parity(self.DIM))

    def test_parity_fock_action(self):
        """Π|n⟩ = (-1)^n |n⟩."""
        P = cs.parity(self.DIM)
        for n in range(self.DIM):
            result = P * ket(self.DIM, n)
            expected = ((-1) ** n) * ket(self.DIM, n)
            assert_close(result, expected)

    def test_parity_squared_is_identity(self):
        P = cs.parity(self.DIM)
        assert_close(P * P, qt.qeye(self.DIM))

    def test_parity_equals_oscillator_rotation_at_minus_pi(self):
        """Π = exp(iπn̂) = R(-π)."""
        P = cs.parity(self.DIM)
        R = cs.oscillator_rotation(-np.pi, self.DIM)
        assert_close(P, R)

    def test_squeeze_unitarity(self):
        for zeta in [0.0, 0.3, 0.5 + 0.2j, -0.4j]:
            assert_unitary(cs.squeeze(zeta, self.DIM))

    def test_squeeze_at_zero_is_identity(self):
        assert_close(cs.squeeze(0.0, self.DIM), qt.qeye(self.DIM))

    def test_kerr_evolution_unitarity(self):
        for kerr in [-1e6, 0.0, 2e6]:
            for time in [0.0, 1e-6, 10e-6]:
                assert_unitary(cs.kerr_evolution(kerr, time, self.DIM))

    def test_kerr_evolution_diagonal(self):
        """U_K is diagonal in the Fock basis."""
        kerr = -2e6
        time = 5e-6
        U = cs.kerr_evolution(kerr, time, self.DIM)
        mat = U.full()
        off_diag = mat - np.diag(np.diag(mat))
        np.testing.assert_allclose(np.abs(off_diag), 0.0, atol=ATOL)

    def test_kerr_evolution_fock_phases(self):
        """U_K|n⟩ = exp[-i K/2 t n(n-1)] |n⟩."""
        kerr = -2e6
        time = 5e-6
        U = cs.kerr_evolution(kerr, time, self.DIM)
        for n in range(self.DIM):
            result = U * ket(self.DIM, n)
            phase = np.exp(-1j * 0.5 * kerr * time * n * (n - 1))
            expected = phase * ket(self.DIM, n)
            assert_close(result, expected)

    def test_kerr_at_zero_time_is_identity(self):
        assert_close(cs.kerr_evolution(-2e6, 0.0, self.DIM), qt.qeye(self.DIM))

    def test_snap_unitarity_dense(self):
        phases = np.linspace(0.0, 2 * np.pi, self.DIM)
        assert_unitary(cs.snap(phases, self.DIM))

    def test_snap_unitarity_dict(self):
        phases = {0: 0.0, 1: np.pi / 2, 3: np.pi}
        assert_unitary(cs.snap(phases, self.DIM))

    def test_snap_fock_action_dense(self):
        """S|n⟩ = exp(i φ_n)|n⟩."""
        phases = [0.1, 0.5, 1.0, 1.5, 2.0]
        dim = len(phases)
        U = cs.snap(phases, dim)
        for n in range(dim):
            result = U * ket(dim, n)
            expected = np.exp(1j * phases[n]) * ket(dim, n)
            assert_close(result, expected)

    def test_snap_fock_action_dict(self):
        """Dict form: specified levels get phase, unspecified get phase 0."""
        phase_dict = {1: np.pi / 2, 3: np.pi}
        dim = 6
        U = cs.snap(phase_dict, dim)
        for n in range(dim):
            result = U * ket(dim, n)
            expected_phase = phase_dict.get(n, 0.0)
            expected = np.exp(1j * expected_phase) * ket(dim, n)
            assert_close(result, expected)

    def test_snap_dense_dict_equivalence(self):
        """Dense and dict forms give the same gate."""
        phases_dense = [0.0, np.pi / 2, np.pi, 0.0, 0.0]
        phases_dict = {1: np.pi / 2, 2: np.pi}
        dim = 5
        U_dense = cs.snap(phases_dense, dim)
        U_dict = cs.snap(phases_dict, dim)
        assert_close(U_dense, U_dict)

    def test_snap_out_of_range_dict_raises(self):
        with pytest.raises(ValueError):
            cs.snap({15: 1.0}, dim=5)

    def test_snap_oversized_array_raises(self):
        with pytest.raises(ValueError):
            cs.snap([0.1, 0.2, 0.3], dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Qubit-cavity conditional gates
# ─────────────────────────────────────────────────────────────────────────────

class TestCoupledGates:
    """Tests for cqed_sim.gates.coupled.*"""

    QDIM = 2
    CDIM = 8

    def _g(self, n: int) -> qt.Qobj:
        return qt.tensor(ket(self.QDIM, 0), ket(self.CDIM, n))

    def _e(self, n: int) -> qt.Qobj:
        return qt.tensor(ket(self.QDIM, 1), ket(self.CDIM, n))

    # --- dispersive_phase -------------------------------------------------

    def test_dispersive_phase_shape(self):
        U = cs.dispersive_phase(-2e6, 5e-6, self.CDIM)
        assert U.shape == (self.QDIM * self.CDIM, self.QDIM * self.CDIM)

    def test_dispersive_phase_unitarity(self):
        for chi in [-2e6, 0.0, 1e6]:
            for t in [0.0, 1e-6, 5e-6]:
                assert_unitary(cs.dispersive_phase(chi, t, self.CDIM))

    def test_dispersive_phase_g_branch_unchanged(self):
        """|g,n⟩ is unaffected (convention='n_e')."""
        chi = -2e6
        time = 5e-6
        U = cs.dispersive_phase(chi, time, self.CDIM, convention="n_e")
        for n in range(self.CDIM):
            result = U * self._g(n)
            assert_close(result, self._g(n))

    def test_dispersive_phase_e_branch_acquires_phase(self):
        """|e,n⟩ → exp(-iχtn)|e,n⟩."""
        chi = -2e6
        time = 5e-6
        U = cs.dispersive_phase(chi, time, self.CDIM, convention="n_e")
        for n in range(self.CDIM):
            result = U * self._e(n)
            expected = np.exp(-1j * chi * time * n) * self._e(n)
            assert_close(result, expected)

    def test_dispersive_phase_z_convention_unitarity(self):
        assert_unitary(
            cs.dispersive_phase(-2e6, 5e-6, self.CDIM, convention="z")
        )

    def test_dispersive_phase_z_convention_both_branches(self):
        """With 'z' convention both branches acquire photon-number phases."""
        chi = -2e6
        time = 5e-6
        U = cs.dispersive_phase(chi, time, self.CDIM, convention="z")
        for n in range(self.CDIM):
            result_g = U * self._g(n)
            expected_g = np.exp(-1j * chi * time / 2.0 * n) * self._g(n)
            assert_close(result_g, expected_g)
            result_e = U * self._e(n)
            expected_e = np.exp(+1j * chi * time / 2.0 * n) * self._e(n)
            assert_close(result_e, expected_e)

    def test_dispersive_phase_bad_convention_raises(self):
        with pytest.raises(ValueError):
            cs.dispersive_phase(-2e6, 1e-6, self.CDIM, convention="bad")

    # --- conditional_rotation ---------------------------------------------

    def test_conditional_rotation_shape(self):
        U = cs.conditional_rotation(np.pi / 3, self.CDIM)
        assert U.shape == (self.QDIM * self.CDIM, self.QDIM * self.CDIM)

    def test_conditional_rotation_unitarity(self):
        for theta in [0.0, np.pi / 4, np.pi, 2 * np.pi]:
            assert_unitary(cs.conditional_rotation(theta, self.CDIM))

    def test_conditional_rotation_g_branch_unchanged(self):
        """Control on |e⟩: |g,n⟩ is unaffected."""
        U = cs.conditional_rotation(np.pi / 2, self.CDIM, control_state="e")
        for n in range(self.CDIM):
            assert_close(U * self._g(n), self._g(n))

    def test_conditional_rotation_e_branch(self):
        """|e,n⟩ → exp(-inθ)|e,n⟩."""
        theta = np.pi / 3
        U = cs.conditional_rotation(theta, self.CDIM, control_state="e")
        for n in range(self.CDIM):
            result = U * self._e(n)
            expected = np.exp(-1j * n * theta) * self._e(n)
            assert_close(result, expected)

    def test_conditional_rotation_control_g_flipped(self):
        """With control_state='g': |g,n⟩ rotates, |e,n⟩ unchanged."""
        theta = np.pi / 5
        U = cs.conditional_rotation(theta, self.CDIM, control_state="g")
        for n in range(min(3, self.CDIM)):
            result = U * self._g(n)
            expected = np.exp(-1j * n * theta) * self._g(n)
            assert_close(result, expected)
            assert_close(U * self._e(n), self._e(n))

    # --- conditional_displacement -----------------------------------------

    def test_conditional_displacement_shape_symmetric(self):
        U = cs.conditional_displacement(alpha=0.5, cavity_dim=self.CDIM)
        assert U.shape == (self.QDIM * self.CDIM, self.QDIM * self.CDIM)

    def test_conditional_displacement_unitarity(self):
        for alpha in [0.0, 0.5, 0.5 + 0.3j]:
            assert_unitary(
                cs.conditional_displacement(alpha=alpha, cavity_dim=self.CDIM)
            )

    def test_conditional_displacement_symmetric_sign(self):
        """CD(α): |g,0⟩ branch gets D(+α), |e,0⟩ branch gets D(-α)."""
        alpha = 0.5
        U = cs.conditional_displacement(alpha=alpha, cavity_dim=self.CDIM)
        D_plus = qt.displace(self.CDIM, alpha)
        D_minus = qt.displace(self.CDIM, -alpha)
        g0 = self._g(0)
        e0 = self._e(0)
        # Apply and compare with individual displacements on the cavity part
        result_g = U * g0
        expected_g = qt.tensor(ket(self.QDIM, 0), D_plus * ket(self.CDIM, 0))
        assert_close(result_g, expected_g)
        result_e = U * e0
        expected_e = qt.tensor(ket(self.QDIM, 1), D_minus * ket(self.CDIM, 0))
        assert_close(result_e, expected_e)

    def test_conditional_displacement_general_form(self):
        ag, ae = 0.4 + 0.1j, -0.2 + 0.3j
        U = cs.conditional_displacement(
            alpha_g=ag, alpha_e=ae, cavity_dim=self.CDIM
        )
        assert_unitary(U)

    def test_conditional_displacement_conflicting_args_raises(self):
        with pytest.raises(ValueError):
            cs.conditional_displacement(alpha=0.5, alpha_g=0.3, cavity_dim=self.CDIM)

    def test_conditional_displacement_missing_args_raises(self):
        with pytest.raises(ValueError):
            cs.conditional_displacement(alpha_g=0.3, cavity_dim=self.CDIM)

    # --- controlled_parity ------------------------------------------------

    def test_controlled_parity_shape(self):
        U = cs.controlled_parity(self.CDIM)
        assert U.shape == (self.QDIM * self.CDIM, self.QDIM * self.CDIM)

    def test_controlled_parity_unitarity(self):
        assert_unitary(cs.controlled_parity(self.CDIM))

    def test_controlled_parity_g_branch_trivial(self):
        """|g,n⟩ passes through unchanged."""
        U = cs.controlled_parity(self.CDIM)
        for n in range(self.CDIM):
            assert_close(U * self._g(n), self._g(n))

    def test_controlled_parity_e_branch_applies_parity(self):
        """|e,n⟩ → (-1)^n |e,n⟩."""
        U = cs.controlled_parity(self.CDIM)
        for n in range(self.CDIM):
            result = U * self._e(n)
            expected = ((-1) ** n) * self._e(n)
            assert_close(result, expected)

    # --- controlled_snap --------------------------------------------------

    def test_controlled_snap_unitarity(self):
        phases = {1: np.pi / 2, 3: np.pi}
        assert_unitary(cs.controlled_snap(phases, self.CDIM))

    def test_controlled_snap_g_branch_trivial(self):
        """|g,n⟩ passes through unchanged."""
        U = cs.controlled_snap({2: np.pi}, self.CDIM)
        for n in range(min(4, self.CDIM)):
            assert_close(U * self._g(n), self._g(n))

    def test_controlled_snap_e_branch_applies_snap(self):
        """|e,n⟩ → exp(iφ_n)|e,n⟩."""
        phase_dict = {1: np.pi / 2, 3: np.pi}
        U = cs.controlled_snap(phase_dict, self.CDIM)
        for n in range(min(5, self.CDIM)):
            result = U * self._e(n)
            expected = np.exp(1j * phase_dict.get(n, 0.0)) * self._e(n)
            assert_close(result, expected)

    # --- sqr (single-n) ---------------------------------------------------

    def test_sqr_shape(self):
        U = cs.sqr(np.pi, 0.0, n=2, cavity_dim=self.CDIM)
        # qubit first, cavity second (repository convention)
        assert U.shape == (self.QDIM * self.CDIM, self.QDIM * self.CDIM)

    def test_sqr_unitarity(self):
        for n in [0, 1, 3]:
            assert_unitary(cs.sqr(np.pi / 2, 0.0, n=n, cavity_dim=self.CDIM))

    def test_sqr_only_target_fock_rotated(self):
        """SQR: only Fock level n gets a qubit rotation; others unchanged."""
        target_n = 2
        theta = np.pi  # X-like rotation
        U = cs.sqr(theta, phi=0.0, n=target_n, cavity_dim=self.CDIM)

        # Non-target levels: qubit in |g⟩ should stay in |g⟩
        for m in [0, 1, 3, 4]:
            state = qt.tensor(ket(self.QDIM, 0), ket(self.CDIM, m))
            result = U * state
            # Qubit should still be in |g⟩ (up to global phase)
            expected = qt.tensor(ket(self.QDIM, 0), ket(self.CDIM, m))
            assert_close(result, expected)

        # Target level n=2: qubit |g⟩ should become -i|e⟩ (Rx(π) action)
        state = qt.tensor(ket(self.QDIM, 0), ket(self.CDIM, target_n))
        result = U * state
        expected = qt.tensor(-1j * ket(self.QDIM, 1), ket(self.CDIM, target_n))
        assert_close(result, expected)

    # --- multi_sqr --------------------------------------------------------

    def test_multi_sqr_unitarity_dense(self):
        thetas = np.linspace(0, np.pi, self.CDIM)
        phis = np.zeros(self.CDIM)
        assert_unitary(cs.multi_sqr(thetas, phis, self.CDIM))

    def test_multi_sqr_unitarity_dict(self):
        thetas = {0: np.pi / 2, 3: np.pi}
        phis = {0: 0.0, 3: np.pi / 4}
        assert_unitary(cs.multi_sqr(thetas, phis, self.CDIM))

    def test_multi_sqr_all_zero_is_identity(self):
        """All-zero rotations = identity."""
        U = cs.multi_sqr(
            np.zeros(self.CDIM), np.zeros(self.CDIM), self.CDIM
        )
        assert_close(U, qt.tensor(qt.qeye(self.QDIM), qt.qeye(self.CDIM)))

    # --- jaynes_cummings --------------------------------------------------

    def test_jc_shape(self):
        U = cs.jaynes_cummings(0.02e9, np.pi / (2 * 0.02e9), self.CDIM)
        assert U.shape == (self.QDIM * self.CDIM, self.QDIM * self.CDIM)

    def test_jc_unitarity(self):
        for g in [0.01e9, 0.05e9]:
            for t in [1e-9, 5e-9]:
                assert_unitary(cs.jaynes_cummings(g, t, self.CDIM))

    def test_jc_at_zero_time_is_identity(self):
        U = cs.jaynes_cummings(0.02e9, 0.0, self.CDIM)
        assert_close(U, qt.qeye(self.QDIM * self.CDIM))

    def test_jc_preserves_total_excitation_in_one_photon_subspace(self):
        """JC conserves total excitation number |e,0⟩ ↔ |g,1⟩."""
        g = 0.02e9
        # Half-oscillation time
        t_half = np.pi / (2 * g)
        U = cs.jaynes_cummings(g, t_half, self.CDIM)
        e0 = qt.tensor(ket(self.QDIM, 1), ket(self.CDIM, 0))
        g1 = qt.tensor(ket(self.QDIM, 0), ket(self.CDIM, 1))
        result = U * e0
        # Should be ±i * g1 (swap with a phase)
        overlap = abs(np.dot(g1.full().flatten().conj(), result.full().flatten()))
        assert abs(overlap - 1.0) < 1e-5

    # --- blue_sideband ----------------------------------------------------

    def test_blue_sideband_unitarity(self):
        for g in [0.01e9, 0.05e9]:
            for t in [1e-9, 5e-9]:
                assert_unitary(cs.blue_sideband(g, t, self.CDIM))

    # --- beam_splitter ----------------------------------------------------

    def test_beam_splitter_unitarity(self):
        for g in [0.01e9, 0.05e9]:
            for t in [1e-9, 5e-9]:
                assert_unitary(cs.beam_splitter(g, t, dim_a=5, dim_b=5))

    def test_beam_splitter_shape(self):
        U = cs.beam_splitter(0.02e9, 1e-9, dim_a=4, dim_b=6)
        assert U.shape == (4 * 6, 4 * 6)

    def test_beam_splitter_matches_beamsplitter_unitary(self):
        """beam_splitter(g, t, da, db) == beamsplitter_unitary(da, db, g*t)."""
        g, t, da, db = 0.02e9, 2e-9, 5, 5
        U_new = cs.beam_splitter(g, t, da, db)
        U_old = cs.beamsplitter_unitary(da, db, g * t)
        assert_close(U_new, U_old)


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Two-qubit gates
# ─────────────────────────────────────────────────────────────────────────────

class TestTwoQubitGates:
    """Tests for cqed_sim.gates.two_qubit.*"""

    def test_shapes(self):
        for gate_fn in [
            cs.cnot_gate, cs.cz_gate, cs.swap_gate,
            cs.iswap_gate, cs.sqrt_iswap_gate,
        ]:
            U = gate_fn()
            assert U.shape == (4, 4), f"{gate_fn.__name__} shape wrong"

    def test_controlled_phase_shape(self):
        assert cs.controlled_phase(np.pi / 2).shape == (4, 4)

    def test_unitarity(self):
        gates = [
            cs.cnot_gate(), cs.cz_gate(), cs.swap_gate(),
            cs.iswap_gate(), cs.sqrt_iswap_gate(),
            cs.controlled_phase(0.3), cs.controlled_phase(np.pi),
        ]
        for U in gates:
            assert_unitary(U)

    def test_cnot_action(self):
        """CNOT: X on target iff control = |e⟩."""
        gg = qt.tensor(ket(2, 0), ket(2, 0))
        ge = qt.tensor(ket(2, 0), ket(2, 1))
        eg = qt.tensor(ket(2, 1), ket(2, 0))
        ee = qt.tensor(ket(2, 1), ket(2, 1))
        U = cs.cnot_gate()
        assert_close(U * gg, gg)
        assert_close(U * ge, ge)
        assert_close(U * eg, ee)
        assert_close(U * ee, eg)

    def test_cz_diagonal(self):
        """CZ = diag(1,1,1,-1)."""
        expected = np.diag([1.0, 1.0, 1.0, -1.0])
        np.testing.assert_allclose(cs.cz_gate().full(), expected, atol=ATOL)

    def test_controlled_phase_at_pi_equals_cz(self):
        """CP(π) = CZ."""
        assert_close(cs.controlled_phase(np.pi), cs.cz_gate())

    def test_controlled_phase_at_0_is_identity(self):
        assert_close(cs.controlled_phase(0.0), qt.tensor(qt.qeye(2), qt.qeye(2)))

    def test_swap_action(self):
        """SWAP |ge⟩ = |eg⟩ etc."""
        ge = qt.tensor(ket(2, 0), ket(2, 1))
        eg = qt.tensor(ket(2, 1), ket(2, 0))
        gg = qt.tensor(ket(2, 0), ket(2, 0))
        ee = qt.tensor(ket(2, 1), ket(2, 1))
        U = cs.swap_gate()
        assert_close(U * ge, eg)
        assert_close(U * eg, ge)
        assert_close(U * gg, gg)
        assert_close(U * ee, ee)

    def test_swap_squared_is_identity(self):
        S = cs.swap_gate()
        assert_close(S * S, qt.tensor(qt.qeye(2), qt.qeye(2)))

    def test_iswap_matrix_entries(self):
        """Verify known matrix entries of iSWAP."""
        U = cs.iswap_gate().full()
        assert_close(qt.Qobj(U[np.ix_([1, 2], [1, 2])]),
                     qt.Qobj(np.array([[0, 1j], [1j, 0]])))

    def test_iswap_preserves_outer_states(self):
        """|gg⟩ and |ee⟩ unchanged by iSWAP."""
        U = cs.iswap_gate()
        gg = qt.tensor(ket(2, 0), ket(2, 0))
        ee = qt.tensor(ket(2, 1), ket(2, 1))
        assert_close(U * gg, gg)
        assert_close(U * ee, ee)

    def test_sqrt_iswap_squared_is_iswap(self):
        """√iSWAP · √iSWAP = iSWAP."""
        S = cs.sqrt_iswap_gate()
        assert_close(S * S, cs.iswap_gate())

    def test_sqrt_iswap_matrix_entries(self):
        """Verify the 2×2 middle block of √iSWAP."""
        U = cs.sqrt_iswap_gate().full()
        s = 1.0 / np.sqrt(2.0)
        block = U[np.ix_([1, 2], [1, 2])]
        expected = np.array([[s, 1j * s], [1j * s, s]])
        np.testing.assert_allclose(block, expected, atol=ATOL)


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: Cross-checks with analytic small-dim forms
# ─────────────────────────────────────────────────────────────────────────────

class TestAnalyticCrossChecks:
    """Cross-validate gate matrices against known analytic expressions."""

    def test_jc_analytic_dim4(self):
        """Jaynes–Cummings in a 2-photon truncated cavity (dim=3).

        Analytically, the 1-excitation subspace is a 2-level system:
          |e,0⟩ ↔ |g,1⟩ with Rabi splitting g√1.
        At t = π/(2g): |e,0⟩ → -i|g,1⟩.
        """
        g = 1.0  # normalised
        t = np.pi / (2 * g)
        cav_dim = 3
        U = cs.jaynes_cummings(g, t, cav_dim)
        e0 = qt.tensor(ket(2, 1), ket(cav_dim, 0))
        g1 = qt.tensor(ket(2, 0), ket(cav_dim, 1))
        result = U * e0
        # Should equal -i|g,1⟩
        expected = -1j * g1
        assert_close(result, expected, atol=1e-6)

    def test_displacement_small_alpha_fock(self):
        """For small alpha, D(α)|0⟩ ≈ |0⟩ + α|1⟩ + ... (coherent state)."""
        alpha = 0.1
        dim = 6
        D = cs.displacement(alpha, dim)
        result = (D * ket(dim, 0)).full().flatten()
        # Leading term: coefficient of |1⟩ should be ≈ alpha
        assert abs(result[1] - alpha) < 0.01

    def test_oscillator_rotation_analytic(self):
        """Analytic check: R(π/2)|1⟩ = exp(-iπ/2)|1⟩ = -i|1⟩."""
        U = cs.oscillator_rotation(np.pi / 2, 4)
        result = U * ket(4, 1)
        expected = np.exp(-1j * np.pi / 2) * ket(4, 1)
        assert_close(result, expected)

    def test_dispersive_full_period(self):
        """At t = 2π/|χ| the dispersive gate is identity (full period)."""
        chi = -2e6
        period = 2 * np.pi / abs(chi)
        U = cs.dispersive_phase(chi, period, 6)
        assert_close(U, qt.qeye(2 * 6))

    def test_transition_rotation_analytic_2x2_block(self):
        """In the {|0⟩,|1⟩} subspace, R^{0,1}(θ) = Rx-like matrix."""
        theta = np.pi / 3
        phi = np.pi / 4
        dim = 3
        U = cs.transition_rotation(dim, 0, 1, theta, phi)
        mat = U.full()
        # The 2×2 upper-left block should equal qubit_rotation_xy(theta, phi)
        R2 = cs.rphi(theta, phi).full()
        np.testing.assert_allclose(mat[:2, :2], R2, atol=ATOL)
        # Level 2 must be unchanged
        np.testing.assert_allclose(mat[2, 2], 1.0, atol=ATOL)
        np.testing.assert_allclose(mat[0, 2], 0.0, atol=ATOL)
        np.testing.assert_allclose(mat[2, 0], 0.0, atol=ATOL)
