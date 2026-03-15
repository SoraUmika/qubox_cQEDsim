from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.unitary_synthesis.targets import make_target


def _old_cluster_small(which: str) -> np.ndarray:
    cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    sw = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    u1 = sw @ cz @ np.kron(hadamard, np.eye(2, dtype=np.complex128))
    if str(which).lower() == "u2":
        ry = np.asarray((1j * np.pi / 2.0 * qt.sigmay() / 2.0).expm().full(), dtype=np.complex128)
        return np.kron(ry, np.eye(2, dtype=np.complex128)) @ u1
    return u1


def test_cluster_reference_u1_matches_old_analytic_formula() -> None:
    current = make_target("cluster", n_match=1, which="u1")
    old = _old_cluster_small("u1")
    assert np.allclose(current, old, atol=1.0e-12)


def test_cluster_reference_u2_matches_old_analytic_formula() -> None:
    current = make_target("cluster", n_match=1, which="u2")
    old = _old_cluster_small("u2")
    assert np.allclose(current, old, atol=1.0e-12)