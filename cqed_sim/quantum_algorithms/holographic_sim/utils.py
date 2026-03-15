from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Iterator, Sequence

import numpy as np


def as_complex_array(value: Any) -> np.ndarray:
    """Return a dense complex array, unwrapping QuTiP-like `.full()` objects."""
    if hasattr(value, "full") and callable(value.full):
        value = value.full()
    return np.asarray(value, dtype=np.complex128)


def ensure_square_matrix(matrix: Any, *, name: str = "matrix") -> np.ndarray:
    arr = as_complex_array(matrix)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return arr


def ensure_positive_int(value: int, *, name: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{name} must be positive.")
    return ivalue


def basis_vector(dim: int, index: int = 0) -> np.ndarray:
    dim = ensure_positive_int(dim, name="dim")
    idx = int(index)
    if idx < 0 or idx >= dim:
        raise ValueError(f"basis index {idx} is outside the range [0, {dim - 1}].")
    vec = np.zeros(dim, dtype=np.complex128)
    vec[idx] = 1.0
    return vec


def normalize_state_vector(state: Any, *, dim: int | None = None, atol: float = 1.0e-12) -> np.ndarray:
    vec = np.asarray(as_complex_array(state), dtype=np.complex128).reshape(-1)
    if dim is not None and vec.size != int(dim):
        raise ValueError(f"State vector has dimension {vec.size}, expected {int(dim)}.")
    norm = float(np.linalg.norm(vec))
    if norm <= atol:
        raise ValueError("State vector norm is zero.")
    return vec / norm


def coerce_density_matrix(state: Any, *, dim: int | None = None, atol: float = 1.0e-12) -> np.ndarray:
    arr = as_complex_array(state)
    if arr.ndim == 1:
        vec = normalize_state_vector(arr, dim=dim, atol=atol)
        return np.outer(vec, vec.conj())
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Density matrices must be square.")
    if dim is not None and arr.shape[0] != int(dim):
        raise ValueError(f"Density matrix has dimension {arr.shape[0]}, expected {int(dim)}.")
    herm = 0.5 * (arr + arr.conj().T)
    trace = complex(np.trace(herm))
    if abs(trace) <= atol:
        raise ValueError("Density matrix trace is zero.")
    return herm / trace


def is_hermitian(matrix: Any, *, atol: float = 1.0e-10) -> bool:
    arr = ensure_square_matrix(matrix)
    return bool(np.linalg.norm(arr - arr.conj().T, ord="fro") <= atol)


def unitarity_error(unitary: Any) -> float:
    arr = ensure_square_matrix(unitary, name="unitary")
    ident = np.eye(arr.shape[0], dtype=np.complex128)
    return float(np.linalg.norm(arr.conj().T @ arr - ident, ord="fro"))


def validate_unitary(unitary: Any, *, atol: float = 1.0e-10) -> np.ndarray:
    arr = ensure_square_matrix(unitary, name="unitary")
    if unitarity_error(arr) > atol:
        raise ValueError("Input matrix is not unitary within tolerance.")
    return arr


def projectors_from_eigenvectors(eigenvectors: np.ndarray) -> tuple[np.ndarray, ...]:
    vecs = as_complex_array(eigenvectors)
    if vecs.ndim != 2:
        raise ValueError("eigenvectors must be a rank-2 array with basis vectors as columns.")
    return tuple(np.outer(vecs[:, idx], vecs[:, idx].conj()) for idx in range(vecs.shape[1]))


def partial_trace_joint(
    rho_joint: Any,
    *,
    physical_dim: int,
    bond_dim: int,
    trace_out: str,
) -> np.ndarray:
    rho = ensure_square_matrix(rho_joint, name="rho_joint")
    expected = int(physical_dim) * int(bond_dim)
    if rho.shape != (expected, expected):
        raise ValueError(f"joint density matrix has shape {rho.shape}, expected {(expected, expected)}.")
    rho4 = rho.reshape(int(physical_dim), int(bond_dim), int(physical_dim), int(bond_dim))
    if trace_out == "physical":
        return np.einsum("aibj,ab->ij", rho4, np.eye(int(physical_dim), dtype=np.complex128))
    if trace_out == "bond":
        return np.einsum("aibj,ij->ab", rho4, np.eye(int(bond_dim), dtype=np.complex128))
    raise ValueError("trace_out must be either 'physical' or 'bond'.")


def conditional_bond_state(
    rho_joint: Any,
    projector: Any,
    *,
    physical_dim: int,
    bond_dim: int,
    atol: float = 1.0e-15,
) -> tuple[np.ndarray | None, float]:
    rho = ensure_square_matrix(rho_joint, name="rho_joint")
    proj = ensure_square_matrix(projector, name="projector")
    if proj.shape != (int(physical_dim), int(physical_dim)):
        raise ValueError("Projector dimension does not match the physical Hilbert space.")
    rho4 = rho.reshape(int(physical_dim), int(bond_dim), int(physical_dim), int(bond_dim))
    bond_unnormalized = np.einsum("ab,aibj->ij", proj.conj(), rho4).astype(np.complex128)
    bond_unnormalized = 0.5 * (bond_unnormalized + bond_unnormalized.conj().T)
    prob = float(np.real(np.trace(bond_unnormalized)))
    if prob <= atol:
        return None, 0.0
    return bond_unnormalized / prob, prob


def state_overlap_probability(rho: Any, state: Any, *, dim: int | None = None) -> float:
    rho_dm = coerce_density_matrix(rho, dim=dim)
    vec = normalize_state_vector(state, dim=rho_dm.shape[0] if dim is None else dim)
    return float(np.real(vec.conj().T @ rho_dm @ vec))


def trace_distance(rho_a: Any, rho_b: Any) -> float:
    a = coerce_density_matrix(rho_a)
    b = coerce_density_matrix(rho_b, dim=a.shape[0])
    singular_values = np.linalg.svd(a - b, compute_uv=False)
    return 0.5 * float(np.sum(np.abs(singular_values)))


def no_progress(iterable: Iterable[Any], **_: Any) -> Iterator[Any]:
    return iter(iterable)


def progress_wrapper(iterable: Iterable[Any], *, enabled: bool, **kwargs: Any) -> Iterator[Any]:
    if not enabled:
        return no_progress(iterable)
    try:
        from tqdm import tqdm

        return tqdm(iterable, **kwargs)
    except Exception:
        return no_progress(iterable)


def json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.complexfloating):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if hasattr(value, "to_record") and callable(value.to_record):
        try:
            return json_ready(value.to_record())
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value
