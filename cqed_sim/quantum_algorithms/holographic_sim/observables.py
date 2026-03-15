from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import as_complex_array, ensure_square_matrix, is_hermitian, projectors_from_eigenvectors


@dataclass(frozen=True)
class PhysicalObservable:
    matrix: np.ndarray
    label: str | None = None

    def __post_init__(self) -> None:
        matrix = ensure_square_matrix(self.matrix, name="observable")
        if not is_hermitian(matrix):
            raise ValueError("PhysicalObservable requires a Hermitian operator.")
        object.__setattr__(self, "matrix", matrix)

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    def eigendecomposition(self, *, descending: bool = True) -> tuple[np.ndarray, np.ndarray]:
        eigenvalues, eigenvectors = np.linalg.eigh(self.matrix)
        order = np.argsort(eigenvalues)
        if descending:
            order = order[::-1]
        return eigenvalues[order].astype(np.complex128), eigenvectors[:, order].astype(np.complex128)

    def projectors(self) -> tuple[np.ndarray, ...]:
        _, eigenvectors = self.eigendecomposition()
        return projectors_from_eigenvectors(eigenvectors)

    def to_record(self) -> dict[str, Any]:
        eigenvalues, _ = self.eigendecomposition()
        return {
            "label": self.label,
            "dim": self.dim,
            "eigenvalues": eigenvalues.tolist(),
            "matrix": self.matrix.tolist(),
        }


def as_observable(value: Any, *, label: str | None = None) -> PhysicalObservable:
    if isinstance(value, PhysicalObservable):
        if label is None or value.label == label:
            return value
        return PhysicalObservable(matrix=value.matrix, label=label)
    return PhysicalObservable(matrix=as_complex_array(value), label=label)


def identity(dim: int) -> PhysicalObservable:
    return PhysicalObservable(np.eye(int(dim), dtype=np.complex128), label=f"I_{int(dim)}")


def pauli_x() -> PhysicalObservable:
    return PhysicalObservable(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128), label="X")


def pauli_y() -> PhysicalObservable:
    return PhysicalObservable(np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128), label="Y")


def pauli_z() -> PhysicalObservable:
    return PhysicalObservable(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128), label="Z")

