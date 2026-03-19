"""Tests for the cqed_sim.observables module.

Covers: reduced_qubit_state, reduced_cavity_state, bloch_xyz_from_joint,
cavity_moments, cavity_wigner, selected_wigner_snapshots, wigner_negativity,
bloch_trajectory_from_states, wrapped_phase_error.
"""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim.observables import (
    bloch_xyz_from_joint,
    bloch_trajectory_from_states,
    cavity_moments,
    cavity_wigner,
    reduced_cavity_state,
    reduced_qubit_state,
    selected_wigner_snapshots,
    wigner_negativity,
    wrapped_phase_error,
)


N_CAV = 5  # Small cavity for fast tests


def _joint_state(qubit_label: str, n_photon: int, n_cav: int = N_CAV) -> qt.Qobj:
    """Build |qubit, n> joint state."""
    q_idx = 0 if qubit_label == "g" else 1
    return qt.tensor(qt.basis(2, q_idx), qt.basis(n_cav, n_photon))


# ---------------------------------------------------------------------------
# Reduced states
# ---------------------------------------------------------------------------

class TestReducedStates:
    def test_reduced_qubit_from_product_state(self):
        psi = _joint_state("g", 0)
        rho_q = reduced_qubit_state(psi)
        assert rho_q.shape == (2, 2)
        expected = qt.ket2dm(qt.basis(2, 0))
        np.testing.assert_allclose(rho_q.full(), expected.full(), atol=1e-12)

    def test_reduced_cavity_from_product_state(self):
        psi = _joint_state("e", 3)
        rho_c = reduced_cavity_state(psi)
        assert rho_c.shape == (N_CAV, N_CAV)
        expected = qt.ket2dm(qt.basis(N_CAV, 3))
        np.testing.assert_allclose(rho_c.full(), expected.full(), atol=1e-12)

    def test_reduced_qubit_from_entangled_state(self):
        # Bell-like state  (|g,0> + |e,1>) / sqrt(2)
        psi = (_joint_state("g", 0) + _joint_state("e", 1)).unit()
        rho_q = reduced_qubit_state(psi)
        # Should be maximally mixed qubit
        np.testing.assert_allclose(rho_q.full(), 0.5 * qt.qeye(2).full(), atol=1e-12)


# ---------------------------------------------------------------------------
# Bloch coordinates
# ---------------------------------------------------------------------------

class TestBlochXYZ:
    def test_ground_state_bloch(self):
        psi = _joint_state("g", 0)
        x, y, z = bloch_xyz_from_joint(psi)
        np.testing.assert_allclose(x, 0.0, atol=1e-12)
        np.testing.assert_allclose(y, 0.0, atol=1e-12)
        np.testing.assert_allclose(z, 1.0, atol=1e-12)

    def test_excited_state_bloch(self):
        psi = _joint_state("e", 0)
        x, y, z = bloch_xyz_from_joint(psi)
        np.testing.assert_allclose(z, -1.0, atol=1e-12)

    def test_plus_x_state_bloch(self):
        q_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
        psi = qt.tensor(q_plus, qt.basis(N_CAV, 0))
        x, y, z = bloch_xyz_from_joint(psi)
        np.testing.assert_allclose(x, 1.0, atol=1e-12)
        np.testing.assert_allclose(y, 0.0, atol=1e-12)
        np.testing.assert_allclose(z, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Cavity moments
# ---------------------------------------------------------------------------

class TestCavityMoments:
    def test_vacuum_moments(self):
        psi = _joint_state("g", 0)
        m = cavity_moments(psi)
        np.testing.assert_allclose(m["n"], 0.0, atol=1e-12)

    def test_fock_state_moments(self):
        psi = _joint_state("g", 3)
        m = cavity_moments(psi)
        np.testing.assert_allclose(m["n"], 3.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Wigner function
# ---------------------------------------------------------------------------

class TestCavityWigner:
    def test_wigner_vacuum(self):
        rho_c = qt.ket2dm(qt.basis(N_CAV, 0))
        xvec, yvec, w = cavity_wigner(rho_c, n_points=21, extent=3.0)
        assert w.shape == (21, 21)
        # Vacuum Wigner is positive everywhere
        assert np.all(w >= -1e-10)
        # Peak at origin
        mid = 10
        assert w[mid, mid] == pytest.approx(np.max(w), rel=1e-3)

    def test_wigner_fock1_has_negativity(self):
        rho_c = qt.ket2dm(qt.basis(N_CAV, 1))
        xvec, yvec, w = cavity_wigner(rho_c, n_points=21, extent=3.0)
        # |1> Wigner function is negative at origin
        mid = 10
        assert w[mid, mid] < 0


class TestWignerNegavitiy:
    def test_vacuum_zero_negativity(self):
        snapshot = {
            "wigner": {
                "xvec": np.linspace(-3, 3, 21),
                "yvec": np.linspace(-3, 3, 21),
                "w": np.abs(np.random.default_rng(42).random((21, 21))),  # all positive
            }
        }
        # Positive Wigner → zero negativity (up to numerical issues)
        neg = wigner_negativity(snapshot)
        assert neg >= 0.0

    def test_none_wigner_returns_nan(self):
        snapshot = {"wigner": None}
        assert np.isnan(wigner_negativity(snapshot))


class TestSelectedWignerSnapshots:
    def test_stride_selection(self):
        snapshots = [{"index": i, "data": i} for i in range(10)]
        track = {"wigner_snapshots": snapshots}
        chosen = selected_wigner_snapshots(track, stride=3)
        indices = [s["index"] for s in chosen]
        # Should include index 0, multiples of 3, and the last
        assert 0 in indices
        assert 9 in indices
        assert 3 in indices
        assert 6 in indices

    def test_empty_snapshots(self):
        track = {"wigner_snapshots": []}
        assert selected_wigner_snapshots(track, stride=2) == []


# ---------------------------------------------------------------------------
# Bloch trajectory
# ---------------------------------------------------------------------------

class TestBlochTrajectory:
    def test_trajectory_from_ground_states(self):
        states = [_joint_state("g", 0)] * 5
        traj = bloch_trajectory_from_states(states)
        assert traj["x"].shape == (5,)
        np.testing.assert_allclose(traj["z"], 1.0, atol=1e-12)

    def test_trajectory_with_conditioned_levels(self):
        states = [_joint_state("g", 0)] * 3
        traj = bloch_trajectory_from_states(states, conditioned_n_levels=[0, 1])
        assert 0 in traj["conditioned"]
        assert 1 in traj["conditioned"]
        # For |g,0>, conditioned on n=0 should give valid Bloch vector
        assert traj["conditioned"][0]["valid"].any()


# ---------------------------------------------------------------------------
# Wrapped phase error
# ---------------------------------------------------------------------------

class TestWrappedPhaseError:
    def test_zero_error_for_identical_phases(self):
        phases = np.array([0.0, 1.0, 2.0])
        err = wrapped_phase_error(phases, phases)
        np.testing.assert_allclose(err, 0.0, atol=1e-14)

    def test_wrapping_near_pi(self):
        sim = np.array([np.pi - 0.01])
        ideal = np.array([-np.pi + 0.01])
        err = wrapped_phase_error(sim, ideal)
        # Wrapped distance should be ~0.02, not ~2*pi - 0.02
        assert np.all(np.abs(err) < 0.03)
