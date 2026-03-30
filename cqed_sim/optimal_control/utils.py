from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.unitary_synthesis.subspace import Subspace


def as_complex_array(value: qt.Qobj | np.ndarray | Any, *, name: str = "value") -> np.ndarray:
    if isinstance(value, qt.Qobj):
        return np.asarray(value.full(), dtype=np.complex128)
    return np.asarray(value, dtype=np.complex128)


def as_square_matrix(value: qt.Qobj | np.ndarray | Any, *, name: str = "matrix") -> np.ndarray:
    matrix = as_complex_array(value, name=name)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return matrix


def normalize_state_vector(vector: np.ndarray, *, name: str = "state") -> np.ndarray:
    state = np.asarray(vector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(state))
    if norm <= 0.0:
        raise ValueError(f"{name} must have non-zero norm.")
    return state / norm


def as_state_vector(
    value: qt.Qobj | np.ndarray | Any,
    *,
    full_dim: int,
    subspace: Subspace | None = None,
    name: str = "state",
) -> np.ndarray:
    if isinstance(value, qt.Qobj):
        data = np.asarray(value.full(), dtype=np.complex128)
    else:
        data = np.asarray(value, dtype=np.complex128)

    if data.ndim == 2:
        if data.shape[1] == 1:
            data = data.reshape(-1)
        elif data.shape[0] == 1:
            data = data.reshape(-1)
        else:
            raise ValueError(f"{name} must be a ket/state vector, not a square operator.")
    elif data.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional or a column vector.")

    vector = np.asarray(data, dtype=np.complex128).reshape(-1)
    if subspace is not None and vector.size == subspace.dim:
        vector = subspace.embed(vector)
    if vector.size != int(full_dim):
        raise ValueError(f"{name} has dimension {vector.size}, expected {int(full_dim)}.")
    return normalize_state_vector(vector, name=name)


def dense_projector(subspace: Subspace) -> np.ndarray:
    projector = np.zeros((subspace.full_dim, subspace.full_dim), dtype=np.complex128)
    indices = np.asarray(subspace.indices, dtype=int)
    projector[np.ix_(indices, indices)] = np.eye(indices.size, dtype=np.complex128)
    return projector


def embed_subspace_operator(target: np.ndarray, subspace: Subspace, *, outside_identity: bool = True) -> np.ndarray:
    operator = np.asarray(target, dtype=np.complex128)
    if operator.shape != (subspace.dim, subspace.dim):
        raise ValueError(
            f"Target operator has shape {operator.shape}, expected {(subspace.dim, subspace.dim)} for the supplied subspace."
        )
    full = np.eye(subspace.full_dim, dtype=np.complex128) if outside_identity else np.zeros(
        (subspace.full_dim, subspace.full_dim), dtype=np.complex128
    )
    indices = np.asarray(subspace.indices, dtype=int)
    full[np.ix_(indices, indices)] = operator
    return full


def quadrature_operators(raising: qt.Qobj, lowering: qt.Qobj) -> tuple[np.ndarray, np.ndarray]:
    i_term = np.asarray((raising + lowering).full(), dtype=np.complex128)
    q_term = np.asarray((1j * (raising - lowering)).full(), dtype=np.complex128)
    return i_term, q_term


def finite_bound_scale(lower: float, upper: float, *, fallback: float = 1.0) -> float:
    if np.isfinite(lower) and np.isfinite(upper):
        return max(abs(float(lower)), abs(float(upper)), fallback)
    if np.isfinite(lower):
        return max(abs(float(lower)), fallback)
    if np.isfinite(upper):
        return max(abs(float(upper)), fallback)
    return float(fallback)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


__all__ = [
    "as_complex_array",
    "as_square_matrix",
    "normalize_state_vector",
    "as_state_vector",
    "dense_projector",
    "embed_subspace_operator",
    "quadrature_operators",
    "finite_bound_scale",
    "json_ready",
]