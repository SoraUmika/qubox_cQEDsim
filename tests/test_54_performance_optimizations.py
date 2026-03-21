"""Tests verifying that performance optimizations are in place and correct.

These tests validate that caching, vectorization, and sparse-construction
optimizations produce the same results as naive implementations, and that
caches are actually populated after use.
"""
from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

# ---------------------------------------------------------------------------
# 1.  Vectorised Hamiltonian sampling produces identical results
# ---------------------------------------------------------------------------


def test_vectorised_hamiltonian_samples_match_naive():
    """Vectorised _dense_hamiltonian_samples gives the same result as the
    original per-timestep Python loop."""
    from cqed_sim.sim.solver import _coeff_samples, _dense_hamiltonian_samples, _operator_to_dense

    n_cav = 5
    tlist = np.linspace(0, 1e-6, 51)
    h0 = qt.tensor(qt.sigmaz(), qt.qeye(n_cav))
    h1 = qt.tensor(qt.sigmax(), qt.qeye(n_cav))
    coeff1 = np.cos(2 * np.pi * 5e6 * tlist)
    hamiltonian = [h0, [h1, coeff1]]

    # Vectorised implementation (current)
    result = _dense_hamiltonian_samples(hamiltonian, tlist)
    assert result.shape == (tlist.size, 2 * n_cav, 2 * n_cav)

    # Naive reference
    h0_dense = _operator_to_dense(h0)
    h1_dense = _operator_to_dense(h1)
    for idx in range(tlist.size):
        expected = h0_dense + coeff1[idx] * h1_dense
        np.testing.assert_allclose(result[idx], expected, atol=1e-14)


# ---------------------------------------------------------------------------
# 2.  Operator factory caching
# ---------------------------------------------------------------------------


def test_sigma_operators_are_cached():
    from cqed_sim.operators.basic import sigma_x, sigma_y, sigma_z

    assert sigma_x() is sigma_x()
    assert sigma_y() is sigma_y()
    assert sigma_z() is sigma_z()


def test_cavity_operators_are_cached():
    from cqed_sim.operators.cavity import create_cavity, destroy_cavity, fock_projector, number_operator

    n = 10
    assert destroy_cavity(n) is destroy_cavity(n)
    assert create_cavity(n) is create_cavity(n)
    assert number_operator(n) is number_operator(n)
    assert fock_projector(n, 3) is fock_projector(n, 3)


# ---------------------------------------------------------------------------
# 3.  Sparse Liouvillian matches dense for large dim
# ---------------------------------------------------------------------------


def test_sparse_liouvillian_matches_dense():
    """For dim >= threshold, the sparse-kron Liouvillian path must match the
    dense path exactly."""
    from cqed_sim.backends.numpy_backend import NumPyBackend, _SPARSE_KRON_DIM_THRESHOLD

    backend = NumPyBackend()
    dim = max(_SPARSE_KRON_DIM_THRESHOLD, 20)  # force sparse path
    rng = np.random.default_rng(42)
    h = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    h = (h + h.conj().T) / 2  # Hermitian
    c1 = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    c_ops = [c1]

    # Sparse path (current impl for large dim)
    L_sparse = backend.lindbladian(h, c_ops)

    # Dense reference (force small-dim path)
    identity = np.eye(dim, dtype=np.complex128)
    L_dense = -1j * (np.kron(identity, h) - np.kron(h.T, identity))
    for c in c_ops:
        c = np.asarray(c, dtype=np.complex128)
        cd_c = c.conj().T @ c
        L_dense += np.kron(c.conj(), c)
        L_dense -= 0.5 * np.kron(identity, cd_c)
        L_dense -= 0.5 * np.kron(cd_c.T, identity)

    np.testing.assert_allclose(L_sparse, L_dense, atol=1e-12)


# ---------------------------------------------------------------------------
# 4.  Subsystem projector caching
# ---------------------------------------------------------------------------


def test_subsystem_projector_is_cached():
    from cqed_sim.sim.extractors import _subsystem_projector

    dims = (2, 5)
    p1 = _subsystem_projector(dims, 0, 1)
    p2 = _subsystem_projector(dims, 0, 1)
    assert p1 is p2  # same object from cache


def test_subsystem_level_population_correct():
    from cqed_sim.sim.extractors import subsystem_level_population

    state = qt.tensor(qt.basis(2, 1), qt.basis(5, 3))
    assert abs(subsystem_level_population(state, 0, 1) - 1.0) < 1e-12
    assert abs(subsystem_level_population(state, 0, 0)) < 1e-12
    assert abs(subsystem_level_population(state, 1, 3) - 1.0) < 1e-12


# ---------------------------------------------------------------------------
# 5.  Mode annihilation operator caching
# ---------------------------------------------------------------------------


def test_mode_annihilation_operator_cached():
    from cqed_sim.sim.extractors import _mode_annihilation_operator

    a1 = _mode_annihilation_operator(10)
    a2 = _mode_annihilation_operator(10)
    assert a1 is a2


# ---------------------------------------------------------------------------
# 6.  Solver pre-conversion: ensure the record function is correct
# ---------------------------------------------------------------------------


def test_solver_store_states_false_skips_qobj():
    """When store_states=False, the solver should still compute expectations
    correctly but return an empty states list."""
    from cqed_sim.sim.solver import solve_with_backend

    h0 = qt.tensor(qt.sigmaz(), qt.qeye(3))
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(3, 0))
    obs = [qt.tensor(qt.sigmaz(), qt.qeye(3))]
    tlist = np.linspace(0, 1e-7, 11)

    result = solve_with_backend([h0], tlist, psi0, observables=obs, store_states=False)
    assert result.states == []
    assert len(result.expect) == 1
    assert len(result.expect[0]) == tlist.size


def test_solver_store_states_true_populates():
    from cqed_sim.sim.solver import solve_with_backend

    h0 = qt.tensor(qt.sigmaz(), qt.qeye(3))
    psi0 = qt.tensor(qt.basis(2, 0), qt.basis(3, 0))
    tlist = np.linspace(0, 1e-7, 11)

    result = solve_with_backend([h0], tlist, psi0, store_states=True)
    assert len(result.states) == tlist.size


# ---------------------------------------------------------------------------
# 7.  Calibration time-grid rounding
# ---------------------------------------------------------------------------


def test_calibration_time_grid_rounding():
    """Slightly different float inputs that represent the same physical
    duration should hit the same cache entry."""
    from cqed_sim.calibration.sqr import _build_time_grid

    g1 = _build_time_grid(1e-6, 1e-9)
    g2 = _build_time_grid(1e-6 + 1e-19, 1e-9 + 1e-19)
    assert g1 is g2  # same cached array
