from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cqed_sim.core.conventions import qubit_cavity_index


@dataclass(frozen=True)
class Subspace:
    r"""Subspace selector for the full qubit-cavity Hilbert space.

    Basis convention follows the repository ordering: |q>_qubit \otimes |n>_cavity,
    with qubit index 0='g', 1='e'.
    """

    full_dim: int
    indices: tuple[int, ...]
    labels: tuple[str, ...]
    kind: str = "custom"
    metadata: dict[str, int | str] | None = None

    @property
    def dim(self) -> int:
        return len(self.indices)

    @staticmethod
    def _qc_index(q: int, n: int, n_cav: int) -> int:
        return qubit_cavity_index(n_cav, q, n)

    @classmethod
    def qubit_cavity_block(cls, n_match: int, n_cav: int | None = None) -> "Subspace":
        n_cav = int(n_match + 1 if n_cav is None else n_cav)
        if n_match < 0 or n_match >= n_cav:
            raise ValueError("n_match must satisfy 0 <= n_match < n_cav.")
        indices: list[int] = []
        labels: list[str] = []
        for n in range(n_match + 1):
            indices.extend([cls._qc_index(0, n, n_cav), cls._qc_index(1, n, n_cav)])
            labels.extend([f"|g,{n}>", f"|e,{n}>"])
        return cls(
            full_dim=2 * n_cav,
            indices=tuple(indices),
            labels=tuple(labels),
            kind="qubit_cavity_block",
            metadata={"n_match": n_match, "n_cav": n_cav},
        )

    @classmethod
    def cavity_only(
        cls,
        n_match: int,
        qubit: str = "g",
        n_cav: int | None = None,
    ) -> "Subspace":
        n_cav = int(n_match + 1 if n_cav is None else n_cav)
        if qubit not in {"g", "e"}:
            raise ValueError("qubit must be 'g' or 'e'.")
        q = 0 if qubit == "g" else 1
        indices = tuple(cls._qc_index(q, n, n_cav) for n in range(n_match + 1))
        labels = tuple(f"|{qubit},{n}>" for n in range(n_match + 1))
        return cls(
            full_dim=2 * n_cav,
            indices=indices,
            labels=labels,
            kind="cavity_only",
            metadata={"n_match": n_match, "n_cav": n_cav, "qubit": qubit},
        )

    @classmethod
    def custom(
        cls,
        full_dim: int,
        indices: Iterable[int],
        labels: Iterable[str] | None = None,
    ) -> "Subspace":
        idx = tuple(int(i) for i in indices)
        if len(idx) == 0:
            raise ValueError("Custom subspace must have at least one index.")
        if min(idx) < 0 or max(idx) >= full_dim:
            raise ValueError("Subspace indices must lie in [0, full_dim).")
        if len(set(idx)) != len(idx):
            raise ValueError("Subspace indices must be unique.")
        if labels is None:
            lbs = tuple(f"|{i}>" for i in idx)
        else:
            lbs = tuple(str(x) for x in labels)
            if len(lbs) != len(idx):
                raise ValueError("labels length must match indices length.")
        return cls(full_dim=int(full_dim), indices=idx, labels=lbs, kind="custom", metadata={})

    def projector(self) -> np.ndarray:
        proj = np.zeros((self.full_dim, self.full_dim), dtype=np.complex128)
        proj[self.indices, self.indices] = 1.0
        return proj

    def embed(self, vec_sub: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec_sub, dtype=np.complex128).reshape(-1)
        if vec.size != self.dim:
            raise ValueError("Subspace vector length mismatch.")
        out = np.zeros(self.full_dim, dtype=np.complex128)
        out[list(self.indices)] = vec
        return out

    def extract(self, vec_full: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec_full, dtype=np.complex128).reshape(-1)
        if vec.size != self.full_dim:
            raise ValueError("Full-space vector length mismatch.")
        return vec[list(self.indices)]

    def restrict_operator(self, op_full: np.ndarray) -> np.ndarray:
        op = np.asarray(op_full, dtype=np.complex128)
        if op.shape != (self.full_dim, self.full_dim):
            raise ValueError("Full-space operator shape mismatch.")
        idx = np.asarray(self.indices, dtype=int)
        return op[np.ix_(idx, idx)]

    def per_fock_blocks(self) -> list[slice]:
        if self.kind != "qubit_cavity_block":
            raise ValueError("Block-phase gauge only applies to qubit_cavity_block subspaces.")
        return [slice(2 * i, 2 * i + 2) for i in range(self.dim // 2)]
