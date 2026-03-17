from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis.metrics import (
    leakage_metrics,
    logical_block_phase_diagnostics,
    subspace_unitary_fidelity,
)
from cqed_sim.unitary_synthesis.subspace import Subspace


def test_b1_perfect_match_on_subspace_with_junk_outside() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2, n_cav=5)
    dim = sub.full_dim
    u = np.eye(dim, dtype=np.complex128)
    t = np.eye(dim, dtype=np.complex128)
    rng = np.random.default_rng(1)
    outside = [i for i in range(dim) if i not in sub.indices]
    rand = rng.standard_normal((len(outside), len(outside))) + 1j * rng.standard_normal((len(outside), len(outside)))
    q, _ = np.linalg.qr(rand)
    t[np.ix_(outside, outside)] = q
    f = subspace_unitary_fidelity(sub.restrict_operator(u), sub.restrict_operator(t))
    assert np.isclose(f, 1.0)


def test_b2_global_phase_invariance() -> None:
    rng = np.random.default_rng(2)
    d = 6
    x = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    q, _ = np.linalg.qr(x)
    f = subspace_unitary_fidelity(np.exp(1j * 0.37) * q, q, gauge="global")
    assert np.isclose(f, 1.0, atol=1e-12)


def test_b3_block_phase_gauge_toggle() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    d = sub.dim
    target = np.eye(d, dtype=np.complex128)
    phases = [0.2, -0.4, 0.7]
    actual = np.zeros((d, d), dtype=np.complex128)
    for n, phase in enumerate(phases):
        actual[2 * n : 2 * n + 2, 2 * n : 2 * n + 2] = np.exp(1j * phase) * np.eye(2)
    f_block = subspace_unitary_fidelity(actual, target, gauge="block", block_slices=sub.per_fock_blocks())
    f_plain = subspace_unitary_fidelity(actual, target, gauge="global")
    assert np.isclose(f_block, 1.0, atol=1e-12)
    assert f_plain < 0.99


def test_b4_leakage_sanity() -> None:
    sub = Subspace.qubit_cavity_block(n_match=1, n_cav=3)
    dim = sub.full_dim
    u = np.eye(dim, dtype=np.complex128)
    leak = 0.3
    i_in = sub.indices[0]
    i_out = [i for i in range(dim) if i not in sub.indices][0]
    u[i_in, i_in] = np.sqrt(1 - leak)
    u[i_out, i_in] = np.sqrt(leak)
    m = leakage_metrics(u, sub)
    assert m.worst >= m.average
    assert m.worst >= 0.29


def test_b5_logical_block_phase_diagnostics_extract_residuals_and_best_fit() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target = np.eye(sub.dim, dtype=np.complex128)
    phases = np.asarray((0.1, -0.35, 0.55), dtype=float)
    actual = np.zeros_like(target)
    for n, phase in enumerate(phases):
        actual[2 * n : 2 * n + 2, 2 * n : 2 * n + 2] = np.exp(1j * phase) * np.eye(2)

    baseline = logical_block_phase_diagnostics(actual, target, block_slices=sub.per_fock_blocks())
    assert np.allclose(baseline.block_phases_rad, phases, atol=1e-12)
    assert np.allclose(
        baseline.residual_block_phases_rad,
        np.array([0.0, -0.45, 0.45]),
        atol=1e-12,
    )
    corrected = logical_block_phase_diagnostics(
        actual,
        target,
        block_slices=sub.per_fock_blocks(),
        applied_correction_phases=baseline.best_fit_correction_phases_rad,
    )
    assert corrected.rms_block_phase_error_rad < 1.0e-12
    assert np.allclose(corrected.residual_block_phases_rad, 0.0, atol=1.0e-12)
