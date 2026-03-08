from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis.subspace import Subspace


def test_a1_deterministic_basis_ordering() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    assert sub.indices == (0, 3, 1, 4, 2, 5)
    assert sub.labels == ("|g,0>", "|e,0>", "|g,1>", "|e,1>", "|g,2>", "|e,2>")


def test_a2_projector_properties() -> None:
    sub = Subspace.qubit_cavity_block(n_match=3)
    p = sub.projector()
    assert np.allclose(p.conj().T, p)
    assert np.allclose(p @ p, p)
    assert np.linalg.matrix_rank(p) == sub.dim


def test_a3_embed_extract_roundtrip() -> None:
    rng = np.random.default_rng(7)
    sub = Subspace.qubit_cavity_block(n_match=3)
    psi = rng.standard_normal(sub.dim) + 1j * rng.standard_normal(sub.dim)
    psi /= np.linalg.norm(psi)
    full = sub.embed(psi)
    back = sub.extract(full)
    assert np.allclose(back, psi)
