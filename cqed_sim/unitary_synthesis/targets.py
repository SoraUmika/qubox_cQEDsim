from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_block_indices
from cqed_sim.core.ideal_gates import qubit_rotation_xy
from .metrics import choi_from_superoperator, matrix_unit, superoperator_from_choi, superoperator_from_kraus, superoperator_from_unitary
from .subspace import Subspace


def _full_dim(n_match: int) -> int:
    return 2 * (n_match + 1)


def _diag_phase(phases: np.ndarray) -> np.ndarray:
    n_cav = phases.size
    op_c = np.diag(np.exp(1j * phases))
    return np.kron(np.eye(2, dtype=np.complex128), op_c)


def _conditional_z(phases: np.ndarray) -> np.ndarray:
    n_cav = phases.size
    out = np.zeros((2 * n_cav, 2 * n_cav), dtype=np.complex128)
    for n, p in enumerate(phases):
        g_idx, e_idx = qubit_cavity_block_indices(n_cav, n)
        out[g_idx, g_idx] = np.exp(-0.5j * p)
        out[e_idx, e_idx] = np.exp(0.5j * p)
    return out


def _expand_unitary_qc(uni: np.ndarray, dim_q: int, dim_b: int, dim_c: int) -> np.ndarray:
    new_dim = dim_q * dim_c
    new_indices = [dim_c * iq + jb for iq in range(dim_q) for jb in range(dim_b)]
    expanded = np.eye(new_dim, dtype=np.complex128)
    for old_row, new_row in enumerate(new_indices):
        for old_col, new_col in enumerate(new_indices):
            expanded[new_row, new_col] = uni[old_row, old_col]
    return expanded


def _ghz_reference_target(n_cav: int) -> np.ndarray:
    cnot_ctrl1_tgt0 = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.complex128,
    )
    return _expand_unitary_qc(cnot_ctrl1_tgt0.conj().T, dim_q=2, dim_b=2, dim_c=n_cav)


def _cluster_reference_target(n_cav: int, which: str = "u1") -> np.ndarray:
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
    u1_small = sw @ cz @ np.kron(hadamard, np.eye(2, dtype=np.complex128))

    lowered = str(which).lower()
    if lowered in {"u2", "2", "second"}:
        ry = np.asarray((1j * np.pi / 2.0 * qt.sigmay() / 2.0).expm().full(), dtype=np.complex128)
        u_small = np.kron(ry, np.eye(2, dtype=np.complex128)) @ u1_small
    else:
        u_small = u1_small

    return _expand_unitary_qc(u_small, dim_q=2, dim_b=2, dim_c=n_cav)


def _default_reference_root() -> Path | None:
    """Return a reference-matrix search root or None if unavailable.

    Resolution order:
    1. ``CQED_REFERENCE_ROOT`` environment variable, if set.
    2. Sibling-repo relative path ``<repo_root>/../noah_repo/miscellaneous/sequential_simulation``.

    Returns ``None`` when no candidate path exists, which causes reference-
    matrix targets to raise an informative error at use time.
    """
    import os
    env_override = os.environ.get("CQED_REFERENCE_ROOT")
    if env_override is not None:
        p = Path(env_override)
        if p.exists():
            return p
        # Env var is set but path does not exist — warn rather than silently ignoring
        import warnings
        warnings.warn(
            f"CQED_REFERENCE_ROOT is set to '{env_override}' but that path does not exist. "
            "Reference-matrix targets will not be available.",
            UserWarning,
            stacklevel=2,
        )
        return None
    repo_root = Path(__file__).resolve().parents[2]
    candidate = repo_root.parent / "noah_repo" / "miscellaneous" / "sequential_simulation"
    return candidate if candidate.exists() else None


def _validate_unitary_matrix(matrix: np.ndarray, *, atol: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Target matrices must be square.")
    ident = np.eye(arr.shape[0], dtype=np.complex128)
    if np.linalg.norm(arr.conj().T @ arr - ident, ord="fro") > atol:
        raise ValueError("Target matrix is not unitary within tolerance.")
    return arr


def _as_state_qobj(
    state: qt.Qobj | np.ndarray,
    *,
    full_dim: int,
    subspace: Subspace | None = None,
    dims: Sequence[int] | None = None,
) -> qt.Qobj:
    if isinstance(state, qt.Qobj):
        obj = state
        source_dims = obj.dims
    else:
        arr = np.asarray(state, dtype=np.complex128)
        if arr.ndim == 1:
            obj = qt.Qobj(arr.reshape(-1))
        elif arr.ndim == 2:
            obj = qt.Qobj(arr)
        else:
            raise ValueError("State targets must be vectors or density matrices.")
        source_dims = None

    resolved_dims = [int(dim) for dim in dims] if dims is not None else None

    if obj.isket or (obj.shape[1] == 1 and obj.shape[0] > 1):
        vec = np.asarray(obj.full(), dtype=np.complex128).reshape(-1)
        if subspace is not None and vec.size == subspace.dim:
            vec = subspace.embed(vec)
        if vec.size != full_dim:
            raise ValueError(f"State vector has dimension {vec.size}, expected {full_dim}.")
        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec = vec / norm
        dims_payload = source_dims if source_dims is not None and int(np.prod(source_dims[0])) == full_dim else (
            [resolved_dims, [1] * len(resolved_dims)] if resolved_dims is not None and int(prod(resolved_dims)) == full_dim else [[full_dim], [1]]
        )
        return qt.Qobj(vec.reshape(-1), dims=dims_payload)

    matrix = np.asarray(obj.full(), dtype=np.complex128)
    if subspace is not None and matrix.shape == (subspace.dim, subspace.dim):
        embedded = np.zeros((full_dim, full_dim), dtype=np.complex128)
        idx = np.asarray(subspace.indices, dtype=int)
        embedded[np.ix_(idx, idx)] = matrix
        matrix = embedded
    if matrix.shape != (full_dim, full_dim):
        raise ValueError(f"State operator has shape {matrix.shape}, expected {(full_dim, full_dim)}.")
    dims_payload = source_dims if source_dims is not None and int(np.prod(source_dims[0])) == full_dim else (
        [resolved_dims, resolved_dims] if resolved_dims is not None and int(prod(resolved_dims)) == full_dim else [[full_dim], [full_dim]]
    )
    return qt.Qobj(matrix, dims=dims_payload)


def _expand_target_matrix(matrix: np.ndarray, *, full_dim: int, subspace: Subspace | None) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.shape == (full_dim, full_dim):
        return arr
    if subspace is not None and arr.shape == (subspace.dim, subspace.dim):
        embedded = np.eye(full_dim, dtype=np.complex128)
        idx = np.asarray(subspace.indices, dtype=int)
        embedded[np.ix_(idx, idx)] = arr
        return embedded
    raise ValueError(f"Target matrix shape {arr.shape} does not match full_dim={full_dim} or subspace dim.")


def _as_operator_qobj(
    operator: qt.Qobj | np.ndarray,
    *,
    full_dim: int,
    subspace: Subspace | None = None,
    dims: Sequence[int] | None = None,
) -> qt.Qobj:
    if isinstance(operator, qt.Qobj):
        obj = operator
        source_dims = obj.dims
    else:
        arr = np.asarray(operator, dtype=np.complex128)
        if arr.ndim != 2:
            raise ValueError("Observable targets must be operator matrices.")
        obj = qt.Qobj(arr)
        source_dims = None

    matrix = np.asarray(obj.full(), dtype=np.complex128)
    if subspace is not None and matrix.shape == (subspace.dim, subspace.dim):
        embedded = np.zeros((full_dim, full_dim), dtype=np.complex128)
        idx = np.asarray(subspace.indices, dtype=int)
        embedded[np.ix_(idx, idx)] = matrix
        matrix = embedded
    if matrix.shape != (full_dim, full_dim):
        raise ValueError(f"Observable operator has shape {matrix.shape}, expected {(full_dim, full_dim)}.")
    resolved_dims = [int(dim) for dim in dims] if dims is not None else None
    dims_payload = source_dims if source_dims is not None and int(np.prod(source_dims[0])) == full_dim else (
        [resolved_dims, resolved_dims] if resolved_dims is not None and int(prod(resolved_dims)) == full_dim else [[full_dim], [full_dim]]
    )
    return qt.Qobj(matrix, dims=dims_payload)


def _normalize_weights(weights: Sequence[float] | None, count: int, *, name: str) -> np.ndarray:
    if count <= 0:
        raise ValueError(f"{name} requires a positive count.")
    if weights is None:
        return np.full(count, 1.0 / count, dtype=float)
    arr = np.asarray(weights, dtype=float)
    if arr.shape != (count,):
        raise ValueError(f"{name} weights must have shape {(count,)}, received {arr.shape}.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError(f"{name} weights must sum to a positive value.")
    return arr / total


def _infer_dimension_from_object(value: qt.Qobj | np.ndarray) -> int:
    if isinstance(value, qt.Qobj):
        return int(value.shape[0])
    arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim == 1:
        return int(arr.size)
    if arr.ndim == 2:
        return int(arr.shape[0])
    raise ValueError("Could not infer a Hilbert-space dimension from the supplied object.")


def _validate_isometry_matrix(matrix: np.ndarray, *, atol: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("Isometry targets must be two-dimensional matrices.")
    arr = _validate_orthonormal_columns(arr, atol=atol, name="Target matrix")
    return arr


def _validate_square_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    return arr


def _validate_orthonormal_columns(
    matrix: np.ndarray,
    *,
    atol: float = 1.0e-8,
    name: str,
) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    ident = np.eye(arr.shape[1], dtype=np.complex128)
    if np.linalg.norm(arr.conj().T @ arr - ident, ord="fro") > atol:
        raise ValueError(f"{name} must have orthonormal columns within tolerance.")
    return arr


@dataclass(frozen=True)
class SubspaceSpec:
    basis_matrix: np.ndarray
    labels: tuple[str, ...] = field(default_factory=tuple)
    kind: str = "custom"
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        basis = _validate_orthonormal_columns(
            np.asarray(self.basis_matrix, dtype=np.complex128),
            atol=1.0e-8,
            name="SubspaceSpec.basis_matrix",
        )
        object.__setattr__(self, "basis_matrix", basis)
        if self.labels:
            labels = tuple(str(label) for label in self.labels)
            if len(labels) != int(basis.shape[1]):
                raise ValueError("SubspaceSpec.labels length must match the number of basis columns.")
            object.__setattr__(self, "labels", labels)
        else:
            object.__setattr__(self, "labels", tuple(f"|phi_{idx}>" for idx in range(int(basis.shape[1]))))
        object.__setattr__(self, "metadata", None if self.metadata is None else dict(self.metadata))

    @property
    def full_dim(self) -> int:
        return int(self.basis_matrix.shape[0])

    @property
    def dim(self) -> int:
        return int(self.basis_matrix.shape[1])

    @classmethod
    def from_basis_matrix(
        cls,
        basis_matrix: np.ndarray,
        *,
        labels: Sequence[str] | None = None,
        kind: str = "basis_matrix",
        metadata: dict[str, Any] | None = None,
    ) -> "SubspaceSpec":
        return cls(
            basis_matrix=np.asarray(basis_matrix, dtype=np.complex128),
            labels=tuple(labels or ()),
            kind=str(kind),
            metadata=metadata,
        )

    @classmethod
    def from_indices(
        cls,
        full_dim: int,
        indices: Sequence[int],
        *,
        labels: Sequence[str] | None = None,
        kind: str = "index_subset",
        metadata: dict[str, Any] | None = None,
    ) -> "SubspaceSpec":
        full_dim = int(full_dim)
        idx = tuple(int(index) for index in indices)
        if not idx:
            raise ValueError("SubspaceSpec.from_indices requires at least one basis index.")
        if min(idx) < 0 or max(idx) >= full_dim:
            raise ValueError("SubspaceSpec indices must lie in [0, full_dim).")
        if len(set(idx)) != len(idx):
            raise ValueError("SubspaceSpec indices must be unique.")
        basis = np.zeros((full_dim, len(idx)), dtype=np.complex128)
        for col, row in enumerate(idx):
            basis[int(row), col] = 1.0
        return cls(
            basis_matrix=basis,
            labels=tuple(labels or tuple(f"|{row}>" for row in idx)),
            kind=str(kind),
            metadata={} if metadata is None else dict(metadata),
        )

    @classmethod
    def from_basis_states(
        cls,
        basis_states: Sequence[qt.Qobj | np.ndarray],
        *,
        full_dim: int | None = None,
        labels: Sequence[str] | None = None,
        kind: str = "basis_states",
        metadata: dict[str, Any] | None = None,
    ) -> "SubspaceSpec":
        if not basis_states:
            raise ValueError("SubspaceSpec.from_basis_states requires at least one basis state.")
        resolved_full_dim = int(full_dim) if full_dim is not None else _infer_dimension_from_object(basis_states[0])
        columns: list[np.ndarray] = []
        for state in basis_states:
            obj = _as_state_qobj(state, full_dim=resolved_full_dim, subspace=None)
            if obj.isoper:
                raise ValueError("Subspace basis states must be state vectors, not density matrices.")
            columns.append(np.asarray(obj.full(), dtype=np.complex128).reshape(-1))
        basis = np.column_stack(columns)
        return cls(
            basis_matrix=basis,
            labels=tuple(labels or ()),
            kind=str(kind),
            metadata=metadata,
        )

    @classmethod
    def from_projector(
        cls,
        projector: np.ndarray,
        *,
        atol: float = 1.0e-8,
        labels: Sequence[str] | None = None,
        kind: str = "projector",
        metadata: dict[str, Any] | None = None,
    ) -> "SubspaceSpec":
        proj = _validate_square_matrix(projector, name="Subspace projector")
        herm = 0.5 * (proj + proj.conj().T)
        if np.linalg.norm(proj - herm, ord="fro") > atol:
            raise ValueError("Subspace projector must be Hermitian within tolerance.")
        if np.linalg.norm(herm @ herm - herm, ord="fro") > max(atol, 1.0e-6):
            raise ValueError("Subspace projector must be idempotent within tolerance.")
        eigvals, eigvecs = np.linalg.eigh(herm)
        mask = eigvals > 0.5
        if not np.any(mask):
            raise ValueError("Subspace projector has rank zero.")
        basis = eigvecs[:, mask]
        return cls(
            basis_matrix=basis,
            labels=tuple(labels or ()),
            kind=str(kind),
            metadata=metadata,
        )

    @classmethod
    def from_subspace(cls, subspace: Subspace) -> "SubspaceSpec":
        return cls.from_indices(
            subspace.full_dim,
            subspace.indices,
            labels=subspace.labels,
            kind=subspace.kind,
            metadata={} if subspace.metadata is None else dict(subspace.metadata),
        )

    @classmethod
    def qubit_cavity_branch(
        cls,
        *,
        n_cav: int,
        qubit: str = "g",
        fock_levels: Sequence[int] | None = None,
    ) -> "SubspaceSpec":
        n_cav = int(n_cav)
        if qubit not in {"g", "e"}:
            raise ValueError("qubit must be 'g' or 'e'.")
        levels = tuple(range(n_cav)) if fock_levels is None else tuple(int(level) for level in fock_levels)
        if not levels:
            raise ValueError("fock_levels must contain at least one level.")
        q_idx = 0 if qubit == "g" else 1
        indices = [qubit_cavity_block_indices(n_cav, level)[q_idx] for level in levels]
        return cls.from_indices(
            2 * n_cav,
            indices,
            labels=[f"|{qubit},{level}>" for level in levels],
            kind="qubit_cavity_branch",
            metadata={"n_cav": n_cav, "qubit": qubit, "levels": list(levels)},
        )

    def projector(self) -> np.ndarray:
        return np.asarray(self.basis_matrix @ self.basis_matrix.conj().T, dtype=np.complex128)

    def embed(self, coefficients: np.ndarray) -> np.ndarray:
        coeff = np.asarray(coefficients, dtype=np.complex128).reshape(-1)
        if coeff.size != self.dim:
            raise ValueError(f"Subspace coefficients have length {coeff.size}, expected {self.dim}.")
        return np.asarray(self.basis_matrix @ coeff, dtype=np.complex128)

    def coordinates(self, vector: np.ndarray) -> np.ndarray:
        vec = np.asarray(vector, dtype=np.complex128).reshape(-1)
        if vec.size != self.full_dim:
            raise ValueError(f"Full-space vector has length {vec.size}, expected {self.full_dim}.")
        return np.asarray(self.basis_matrix.conj().T @ vec, dtype=np.complex128)

    def restrict_operator(self, operator: np.ndarray) -> np.ndarray:
        op = np.asarray(operator, dtype=np.complex128)
        if op.shape != (self.full_dim, self.full_dim):
            raise ValueError(f"Operator has shape {op.shape}, expected {(self.full_dim, self.full_dim)}.")
        return np.asarray(self.basis_matrix.conj().T @ op @ self.basis_matrix, dtype=np.complex128)

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": str(self.kind),
            "full_dim": int(self.full_dim),
            "dim": int(self.dim),
            "labels": [str(label) for label in self.labels],
        }


def coerce_subspace_spec(
    spec: Any,
    *,
    full_dim: int | None = None,
    expected_dim: int | None = None,
    name: str = "subspace",
) -> SubspaceSpec:
    resolved: SubspaceSpec
    if isinstance(spec, SubspaceSpec):
        resolved = spec
    elif isinstance(spec, Subspace):
        resolved = SubspaceSpec.from_subspace(spec)
    elif isinstance(spec, qt.Qobj):
        arr = np.asarray(spec.full(), dtype=np.complex128)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
            resolved = SubspaceSpec.from_projector(arr)
        else:
            resolved = SubspaceSpec.from_basis_states([spec], full_dim=full_dim)
    elif isinstance(spec, np.ndarray):
        arr = np.asarray(spec, dtype=np.complex128)
        if arr.ndim == 2 and arr.shape[0] == arr.shape[1] and (full_dim is None or arr.shape[0] == int(full_dim)):
            resolved = SubspaceSpec.from_projector(arr)
        elif arr.ndim == 2:
            resolved = SubspaceSpec.from_basis_matrix(arr)
        elif arr.ndim == 1 and np.issubdtype(spec.dtype, np.integer):
            if full_dim is None:
                raise ValueError(f"{name} specified by indices requires full_dim.")
            resolved = SubspaceSpec.from_indices(int(full_dim), [int(value) for value in spec.tolist()])
        else:
            resolved = SubspaceSpec.from_basis_states([arr], full_dim=full_dim)
    elif isinstance(spec, Sequence) and spec and all(isinstance(value, (int, np.integer)) for value in spec):
        if full_dim is None:
            raise ValueError(f"{name} specified by indices requires full_dim.")
        resolved = SubspaceSpec.from_indices(int(full_dim), [int(value) for value in spec])
    elif isinstance(spec, Sequence):
        resolved = SubspaceSpec.from_basis_states(list(spec), full_dim=full_dim)
    else:
        raise TypeError(f"Could not coerce {name} of type {type(spec).__name__} into a SubspaceSpec.")

    if full_dim is not None and int(resolved.full_dim) != int(full_dim):
        raise ValueError(f"{name} full_dim={resolved.full_dim} does not match expected full_dim={int(full_dim)}.")
    if expected_dim is not None and int(resolved.dim) != int(expected_dim):
        raise ValueError(f"{name} dim={resolved.dim} does not match expected dim={int(expected_dim)}.")
    return resolved


def _subsystem_dims(value: Sequence[int] | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    dims = tuple(int(dim) for dim in value)
    if not dims or any(dim <= 0 for dim in dims):
        raise ValueError("subsystem_dims must contain positive subsystem dimensions.")
    return dims


def _state_dim_matches(state: qt.Qobj | np.ndarray, expected_dim: int) -> bool:
    try:
        return _infer_dimension_from_object(state) == int(expected_dim)
    except Exception:
        return False


def _coerce_operator_matrix(value: qt.Qobj | np.ndarray, *, name: str) -> np.ndarray:
    if isinstance(value, qt.Qobj):
        arr = np.asarray(value.full(), dtype=np.complex128)
    else:
        arr = np.asarray(value, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a matrix.")
    return arr


def _channel_dims_from_representation(
    *,
    choi: np.ndarray | None,
    superoperator: np.ndarray | None,
    kraus_operators: Sequence[np.ndarray] | None,
    unitary: np.ndarray | None,
    isometry: np.ndarray | None,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> tuple[int, int]:
    if unitary is not None:
        return int(unitary.shape[1]), int(unitary.shape[0])
    if isometry is not None:
        return int(isometry.shape[1]), int(isometry.shape[0])
    if kraus_operators is not None:
        return int(kraus_operators[0].shape[1]), int(kraus_operators[0].shape[0])
    if superoperator is not None:
        if input_dim is None or output_dim is None:
            if superoperator.shape[0] == superoperator.shape[1]:
                dim_float = np.sqrt(float(superoperator.shape[0]))
                dim = int(round(dim_float))
                if dim * dim != superoperator.shape[0]:
                    raise ValueError("Target superoperator dimension must be a perfect square when input_dim/output_dim are omitted.")
                return dim, dim
            raise ValueError("Rectangular target superoperators require explicit input_dim and output_dim.")
        if int(output_dim) * int(output_dim) != int(superoperator.shape[0]):
            raise ValueError("Target superoperator row dimension does not match output_dim^2.")
        if int(input_dim) * int(input_dim) != int(superoperator.shape[1]):
            raise ValueError("Target superoperator column dimension does not match input_dim^2.")
        return int(input_dim), int(output_dim)
    assert choi is not None
    if input_dim is None or output_dim is None:
        dim_float = np.sqrt(float(choi.shape[0]))
        dim = int(round(dim_float))
        if dim * dim != choi.shape[0]:
            raise ValueError("Rectangular target Choi matrices require explicit input_dim and output_dim.")
        return dim, dim
    if int(input_dim) * int(output_dim) != int(choi.shape[0]):
        raise ValueError("Target Choi dimension does not match input_dim * output_dim.")
    return int(input_dim), int(output_dim)


def _embed_subsystem_operator_with_environment(
    operator: np.ndarray,
    *,
    retained_subsystems: Sequence[int],
    subsystem_dims: Sequence[int],
    environment_state: qt.Qobj,
) -> qt.Qobj:
    keep = tuple(int(index) for index in retained_subsystems)
    full_dims = [int(dim) for dim in subsystem_dims]
    env = tuple(index for index in range(len(full_dims)) if index not in keep)
    keep_dims = [full_dims[index] for index in keep]
    operator_qobj = qt.Qobj(np.asarray(operator, dtype=np.complex128), dims=[keep_dims, keep_dims])
    env_state_qobj = environment_state if environment_state.isoper else environment_state.proj()
    combined = qt.tensor(operator_qobj, env_state_qobj)
    current_order = list(keep) + list(env)
    perm = [current_order.index(index) for index in range(len(full_dims))]
    return combined.permute(perm)


def _default_probe_states(
    *,
    full_dim: int,
    subspace: Subspace | None,
    strategy: str,
) -> list[qt.Qobj]:
    if subspace is None:
        base_dim = full_dim
        basis_vectors = [qt.Qobj(np.eye(base_dim, dtype=np.complex128)[:, i], dims=[[full_dim], [1]]) for i in range(base_dim)]
    else:
        base_dim = subspace.dim
        basis_vectors = []
        for i in range(base_dim):
            vec = np.zeros(base_dim, dtype=np.complex128)
            vec[i] = 1.0
            basis_vectors.append(qt.Qobj(subspace.embed(vec), dims=[[full_dim], [1]]))

    lowered = str(strategy).lower()
    if lowered == "basis":
        return basis_vectors

    if lowered == "basis_plus_uniform":
        uniform = np.ones(base_dim, dtype=np.complex128) / np.sqrt(base_dim)
        if subspace is not None:
            uniform = subspace.embed(uniform)
        basis_vectors.append(qt.Qobj(uniform, dims=[[full_dim], [1]]))
        return basis_vectors

    if lowered == "basis_plus_blocks":
        if subspace is None:
            return basis_vectors
        blocks = subspace.per_fock_blocks() if subspace.kind == "qubit_cavity_block" else [slice(0, subspace.dim)]
        for block in blocks:
            vec_sub = np.zeros(subspace.dim, dtype=np.complex128)
            vec_sub[block] = 1.0 / np.sqrt(max(block.stop - block.start, 1))
            basis_vectors.append(qt.Qobj(subspace.embed(vec_sub), dims=[[full_dim], [1]]))
        return basis_vectors

    raise ValueError(f"Unsupported open-system probe strategy '{strategy}'.")


@dataclass(frozen=True)
class TargetUnitary:
    matrix: np.ndarray
    ignore_global_phase: bool = False
    allow_diagonal_phase: bool = False
    phase_blocks: tuple[tuple[int, ...], ...] | None = None
    probe_states: tuple[qt.Qobj | np.ndarray, ...] = field(default_factory=tuple)
    open_system_probe_strategy: str = "basis_plus_uniform"

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _validate_unitary_matrix(self.matrix))
        if self.phase_blocks is not None:
            normalized = tuple(tuple(int(idx) for idx in block) for block in self.phase_blocks)
            object.__setattr__(self, "phase_blocks", normalized)
        object.__setattr__(self, "probe_states", tuple(self.probe_states))

    @property
    def dim(self) -> int:
        return int(self.matrix.shape[0])

    def resolved_gauge(self, fallback: str = "global") -> str:
        if self.phase_blocks:
            return "block"
        if self.allow_diagonal_phase:
            return "diagonal"
        if self.ignore_global_phase:
            return "global"
        return str(fallback)

    def resolved_blocks(self, subspace: Subspace | None = None, fallback: str = "global") -> tuple[tuple[int, ...], ...] | None:
        if self.phase_blocks:
            return self.phase_blocks
        if subspace is not None and self.resolved_gauge(fallback=fallback) == "block":
            return tuple(tuple(range(block.start, block.stop)) for block in subspace.per_fock_blocks())
        return None

    def resolved_probe_pairs(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> tuple[list[qt.Qobj], list[qt.Qobj]]:
        if self.probe_states:
            initial = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace) for state in self.probe_states]
        else:
            strategy = self.open_system_probe_strategy
            if self.allow_diagonal_phase:
                strategy = "basis"
            elif self.phase_blocks:
                initial = _default_probe_states(full_dim=full_dim, subspace=subspace, strategy="basis")
                if subspace is None:
                    for block in self.phase_blocks:
                        vec = np.zeros(full_dim, dtype=np.complex128)
                        vec[list(block)] = 1.0 / np.sqrt(max(len(block), 1))
                        initial.append(qt.Qobj(vec, dims=[[full_dim], [1]]))
                else:
                    for block in self.phase_blocks:
                        vec_sub = np.zeros(subspace.dim, dtype=np.complex128)
                        vec_sub[list(block)] = 1.0 / np.sqrt(max(len(block), 1))
                        initial.append(qt.Qobj(subspace.embed(vec_sub), dims=[[full_dim], [1]]))
            else:
                initial = _default_probe_states(full_dim=full_dim, subspace=subspace, strategy=strategy)

        target_full = _expand_target_matrix(self.matrix, full_dim=full_dim, subspace=subspace)
        outputs: list[qt.Qobj] = []
        for state in initial:
            if state.isoper:
                rho = np.asarray(state.full(), dtype=np.complex128)
                evolved = target_full @ rho @ target_full.conj().T
                outputs.append(qt.Qobj(evolved, dims=state.dims))
            else:
                vec = np.asarray(state.full(), dtype=np.complex128).reshape(-1)
                evolved = target_full @ vec
                outputs.append(qt.Qobj(evolved.reshape(-1), dims=state.dims))
        return initial, outputs

    def as_channel_target(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> "TargetChannel":
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.shape == (full_dim, full_dim):
            return TargetChannel(unitary=matrix)
        if subspace is not None and matrix.shape == (subspace.dim, subspace.dim):
            subspace_spec = SubspaceSpec.from_subspace(subspace)
            return TargetChannel(unitary=matrix, input_subspace=subspace_spec, output_subspace=subspace_spec)
        raise ValueError(
            f"TargetUnitary matrix shape {matrix.shape} does not match full_dim={full_dim} or subspace dim."
        )


class TargetStateMapping:
    def __init__(
        self,
        *,
        initial_states: Sequence[qt.Qobj | np.ndarray] | None = None,
        target_states: Sequence[qt.Qobj | np.ndarray] | None = None,
        initial_state: qt.Qobj | np.ndarray | None = None,
        target_state: qt.Qobj | np.ndarray | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        if initial_state is not None or target_state is not None:
            if initial_states is not None or target_states is not None:
                raise ValueError("Use either singular or plural state-mapping arguments, not both.")
            initial_states = [] if initial_state is None else [initial_state]
            target_states = [] if target_state is None else [target_state]

        if initial_states is None or target_states is None:
            raise ValueError("TargetStateMapping requires initial_states and target_states.")
        if len(initial_states) == 0 or len(target_states) == 0:
            raise ValueError("TargetStateMapping requires at least one initial/target state pair.")
        if len(initial_states) != len(target_states):
            raise ValueError("initial_states and target_states must have the same length.")

        self.initial_states = tuple(initial_states)
        self.target_states = tuple(target_states)
        if weights is None:
            self.weights = tuple(1.0 for _ in self.initial_states)
        else:
            if len(weights) != len(self.initial_states):
                raise ValueError("weights length must match the number of state pairs.")
            self.weights = tuple(float(w) for w in weights)

    def resolved_pairs(self, *, full_dim: int, subspace: Subspace | None = None) -> tuple[list[qt.Qobj], list[qt.Qobj], np.ndarray]:
        initial = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace) for state in self.initial_states]
        targets = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace) for state in self.target_states]
        weights = np.asarray(self.weights, dtype=float)
        if np.sum(weights) <= 0.0:
            raise ValueError("State-mapping weights must sum to a positive value.")
        return initial, targets, weights / np.sum(weights)


class TargetReducedStateMapping:
    def __init__(
        self,
        *,
        initial_states: Sequence[qt.Qobj | np.ndarray] | None = None,
        target_states: Sequence[qt.Qobj | np.ndarray] | None = None,
        retained_subsystems: Sequence[int],
        subsystem_dims: Sequence[int] | None = None,
        initial_state: qt.Qobj | np.ndarray | None = None,
        target_state: qt.Qobj | np.ndarray | None = None,
        weights: Sequence[float] | None = None,
    ) -> None:
        if initial_state is not None or target_state is not None:
            if initial_states is not None or target_states is not None:
                raise ValueError("Use either singular or plural reduced-state arguments, not both.")
            initial_states = [] if initial_state is None else [initial_state]
            target_states = [] if target_state is None else [target_state]

        if initial_states is None or target_states is None:
            raise ValueError("TargetReducedStateMapping requires initial_states and target_states.")
        if len(initial_states) == 0 or len(target_states) == 0:
            raise ValueError("TargetReducedStateMapping requires at least one initial/target state pair.")
        if len(initial_states) != len(target_states):
            raise ValueError("initial_states and target_states must have the same length.")

        kept = tuple(int(index) for index in retained_subsystems)
        if not kept:
            raise ValueError("TargetReducedStateMapping requires at least one retained subsystem.")
        dims = _subsystem_dims(subsystem_dims)
        if dims is not None and max(kept) >= len(dims):
            raise ValueError("retained_subsystems must index the provided subsystem_dims.")

        self.initial_states = tuple(initial_states)
        self.target_states = tuple(target_states)
        self.retained_subsystems = kept
        self.subsystem_dims = dims
        if weights is None:
            self.weights = tuple(1.0 for _ in self.initial_states)
        else:
            if len(weights) != len(self.initial_states):
                raise ValueError("weights length must match the number of reduced-state pairs.")
            self.weights = tuple(float(weight) for weight in weights)

    def infer_dimension(self) -> int:
        if self.subsystem_dims is not None:
            return int(prod(self.subsystem_dims))
        return _infer_dimension_from_object(self.initial_states[0])

    def resolved_data(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
        system_subsystem_dims: Sequence[int] | None = None,
    ) -> tuple[list[qt.Qobj], list[qt.Qobj], np.ndarray, tuple[int, ...], tuple[int, ...] | None]:
        dims = self.subsystem_dims if self.subsystem_dims is not None else _subsystem_dims(system_subsystem_dims)
        if dims is not None and int(prod(dims)) != int(full_dim):
            raise ValueError(f"Resolved subsystem_dims multiply to {int(prod(dims))}, expected full_dim={int(full_dim)}.")
        initial = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace, dims=dims) for state in self.initial_states]
        keep = tuple(int(index) for index in self.retained_subsystems)
        reduced_dim = int(prod(dims[index] for index in keep)) if dims is not None else None

        targets: list[qt.Qobj] = []
        for state in self.target_states:
            if _state_dim_matches(state, full_dim) or (subspace is not None and _state_dim_matches(state, subspace.dim)):
                targets.append(_as_state_qobj(state, full_dim=full_dim, subspace=subspace, dims=dims))
                continue
            if reduced_dim is not None and _state_dim_matches(state, reduced_dim):
                keep_dims = [dims[index] for index in keep]
                targets.append(_as_state_qobj(state, full_dim=reduced_dim, subspace=None, dims=keep_dims))
                continue
            if reduced_dim is None and len(keep) == 1:
                targets.append(_as_state_qobj(state, full_dim=_infer_dimension_from_object(state), subspace=None))
                continue
            raise ValueError(
                "Reduced-state targets must match either the full space, the synthesis subspace, or the retained-subsystem dimension."
            )

        weights = np.asarray(self.weights, dtype=float)
        if np.sum(weights) <= 0.0:
            raise ValueError("Reduced-state weights must sum to a positive value.")
        return initial, targets, weights / np.sum(weights), keep, dims


@dataclass(frozen=True)
class TargetIsometry:
    matrix: np.ndarray
    input_subspace: Any | None = None
    input_states: tuple[qt.Qobj | np.ndarray, ...] = field(default_factory=tuple)
    weights: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "matrix", _validate_isometry_matrix(self.matrix))
        object.__setattr__(self, "input_states", tuple(self.input_states))
        object.__setattr__(self, "weights", tuple(float(weight) for weight in self.weights))
        if self.weights and len(self.weights) != int(self.matrix.shape[1]):
            raise ValueError("TargetIsometry weights length must match the input dimension.")
        if self.input_states and len(self.input_states) != int(self.matrix.shape[1]):
            raise ValueError("TargetIsometry input_states length must match the input dimension.")

    @classmethod
    def from_basis_map(
        cls,
        *,
        target_states: Sequence[qt.Qobj | np.ndarray],
        input_subspace: Any | None = None,
        weights: Sequence[float] | None = None,
    ) -> "TargetIsometry":
        if not target_states:
            raise ValueError("TargetIsometry.from_basis_map requires at least one target state.")
        full_dim = _infer_dimension_from_object(target_states[0])
        columns: list[np.ndarray] = []
        for state in target_states:
            obj = _as_state_qobj(state, full_dim=full_dim, subspace=None)
            if obj.isoper:
                raise ValueError("TargetIsometry.from_basis_map expects pure-state image vectors, not density matrices.")
            columns.append(np.asarray(obj.full(), dtype=np.complex128).reshape(-1))
        matrix = np.column_stack(columns)
        return cls(
            matrix=matrix,
            input_subspace=input_subspace,
            weights=tuple(() if weights is None else tuple(float(weight) for weight in weights)),
        )

    @property
    def output_dim(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.matrix.shape[1])

    def infer_dimension(self) -> int:
        return int(self.output_dim)

    def resolved_input_subspace(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> SubspaceSpec:
        if self.input_subspace is not None:
            return coerce_subspace_spec(
                self.input_subspace,
                full_dim=full_dim,
                expected_dim=self.input_dim,
                name="TargetIsometry.input_subspace",
            )
        if self.input_states:
            return SubspaceSpec.from_basis_states(
                self.input_states,
                full_dim=full_dim,
                kind="target_isometry_input_basis",
            )
        if subspace is not None:
            subspace_spec = SubspaceSpec.from_subspace(subspace)
            if self.input_dim == subspace_spec.dim:
                return subspace_spec
            if self.input_dim < subspace_spec.dim:
                return SubspaceSpec.from_basis_matrix(
                    subspace_spec.basis_matrix[:, : self.input_dim],
                    labels=subspace_spec.labels[: self.input_dim],
                    kind="target_isometry_input_prefix",
                    metadata=subspace_spec.metadata,
                )
        if self.input_dim <= full_dim:
            return SubspaceSpec.from_indices(full_dim, range(self.input_dim), kind="logical_prefix")
        raise ValueError("TargetIsometry could not infer a default logical input basis for the requested input dimension.")

    def resolved_output_matrix(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> np.ndarray:
        matrix = np.asarray(self.matrix, dtype=np.complex128)
        if matrix.shape[0] == int(full_dim):
            return matrix
        if subspace is not None and matrix.shape[0] == int(subspace.dim):
            return np.column_stack([subspace.embed(matrix[:, column]) for column in range(matrix.shape[1])])
        raise ValueError(
            f"TargetIsometry output dimension {matrix.shape[0]} does not match full_dim={full_dim} or the active subspace."
        )

    def resolved_pairs(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> tuple[list[qt.Qobj], list[qt.Qobj], np.ndarray]:
        input_spec = self.resolved_input_subspace(full_dim=full_dim, subspace=subspace)
        initial = [
            qt.Qobj(input_spec.basis_matrix[:, column].reshape(-1), dims=[[full_dim], [1]])
            for column in range(input_spec.dim)
        ]

        output_matrix = self.resolved_output_matrix(full_dim=full_dim, subspace=subspace)
        targets: list[qt.Qobj] = []
        for column in range(self.input_dim):
            vec = np.asarray(output_matrix[:, column], dtype=np.complex128).reshape(-1)
            targets.append(qt.Qobj(vec.reshape(-1), dims=[[full_dim], [1]]))

        weights = _normalize_weights(self.weights if self.weights else None, self.input_dim, name="TargetIsometry")
        return initial, targets, weights

    def as_channel_target(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> "TargetChannel":
        return TargetChannel(
            isometry=self.resolved_output_matrix(full_dim=full_dim, subspace=subspace),
            input_subspace=self.resolved_input_subspace(full_dim=full_dim, subspace=subspace),
        )


class TargetChannel:
    def __init__(
        self,
        *,
        choi: qt.Qobj | np.ndarray | None = None,
        superoperator: qt.Qobj | np.ndarray | None = None,
        kraus_operators: Sequence[qt.Qobj | np.ndarray] | None = None,
        unitary: qt.Qobj | np.ndarray | None = None,
        isometry: qt.Qobj | np.ndarray | None = None,
        input_dim: int | None = None,
        output_dim: int | None = None,
        input_subspace: Any | None = None,
        output_subspace: Any | None = None,
        retained_subsystems: Sequence[int] | None = None,
        subsystem_dims: Sequence[int] | None = None,
        environment_state: qt.Qobj | np.ndarray | None = None,
        enforce_cptp: bool = False,
    ) -> None:
        specified = sum(
            value is not None
            for value in (
                choi,
                superoperator,
                kraus_operators,
                unitary,
                isometry,
            )
        )
        if specified != 1:
            raise ValueError(
                "TargetChannel requires exactly one of choi, superoperator, kraus_operators, unitary, or isometry."
            )

        unitary_arr = None if unitary is None else _validate_unitary_matrix(_coerce_operator_matrix(unitary, name="Target unitary"))
        isometry_arr = None if isometry is None else _validate_isometry_matrix(_coerce_operator_matrix(isometry, name="Target isometry"))
        kraus_arr = None
        if kraus_operators is not None:
            if not kraus_operators:
                raise ValueError("TargetChannel.kraus_operators requires at least one Kraus operator.")
            kraus_arr = tuple(_coerce_operator_matrix(op, name="Kraus operator") for op in kraus_operators)
            out_dim = int(kraus_arr[0].shape[0])
            in_dim = int(kraus_arr[0].shape[1])
            if any(matrix.shape != (out_dim, in_dim) for matrix in kraus_arr):
                raise ValueError("All Kraus operators must have the same shape.")

        super_arr = None if superoperator is None else _coerce_operator_matrix(superoperator, name="Target superoperator")
        choi_arr = None if choi is None else _validate_square_matrix(
            np.asarray(qt.Qobj(choi).full() if isinstance(choi, qt.Qobj) else choi, dtype=np.complex128),
            name="Target Choi matrix",
        )

        resolved_input_dim, resolved_output_dim = _channel_dims_from_representation(
            choi=choi_arr,
            superoperator=super_arr,
            kraus_operators=kraus_arr,
            unitary=unitary_arr,
            isometry=isometry_arr,
            input_dim=input_dim,
            output_dim=output_dim,
        )

        self.unitary = unitary_arr
        self.isometry = isometry_arr
        self.kraus_operators = kraus_arr
        self.superoperator = super_arr if super_arr is not None else (
            superoperator_from_unitary(unitary_arr if unitary_arr is not None else isometry_arr)
            if unitary_arr is not None or isometry_arr is not None
            else superoperator_from_kraus(kraus_arr)
            if kraus_arr is not None
            else superoperator_from_choi(choi_arr, input_dim=resolved_input_dim, output_dim=resolved_output_dim)
        )
        self.choi = choi_arr if choi_arr is not None else choi_from_superoperator(
            self.superoperator,
            input_dim=resolved_input_dim,
            output_dim=resolved_output_dim,
        )
        self.input_dim = int(resolved_input_dim)
        self.output_dim = int(resolved_output_dim)
        self.channel_dim = int(self.input_dim)
        self.input_subspace = input_subspace
        self.output_subspace = output_subspace
        self.retained_subsystems = None if retained_subsystems is None else tuple(int(index) for index in retained_subsystems)
        self.subsystem_dims = _subsystem_dims(subsystem_dims)
        self.environment_state = environment_state
        self.enforce_cptp = bool(enforce_cptp)

        if self.retained_subsystems is not None and (self.input_subspace is not None or self.output_subspace is not None):
            raise ValueError("TargetChannel.retained_subsystems cannot be combined with input_subspace or output_subspace.")
        if self.output_subspace is not None and self.retained_subsystems is not None:
            raise ValueError("TargetChannel.output_subspace cannot be combined with retained_subsystems.")
        if self.retained_subsystems is not None and not self.retained_subsystems:
            raise ValueError("TargetChannel.retained_subsystems must contain at least one subsystem index when provided.")
        if self.subsystem_dims is not None and self.retained_subsystems is not None and max(self.retained_subsystems) >= len(self.subsystem_dims):
            raise ValueError("TargetChannel.retained_subsystems must index the provided subsystem_dims.")
        if self.retained_subsystems is not None and self.subsystem_dims is not None:
            retained_dim = int(prod(self.subsystem_dims[index] for index in self.retained_subsystems))
            if retained_dim != self.input_dim or retained_dim != self.output_dim:
                raise ValueError(
                    f"TargetChannel retained subsystem dimension {retained_dim} does not match channel dimensions ({self.input_dim}, {self.output_dim})."
                )
        if self.retained_subsystems is not None and environment_state is None:
            raise ValueError("TargetChannel requires environment_state when retained_subsystems are specified.")

    @classmethod
    def from_choi(
        cls,
        choi: qt.Qobj | np.ndarray,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        input_subspace: Any | None = None,
        output_subspace: Any | None = None,
        **kwargs: Any,
    ) -> "TargetChannel":
        return cls(
            choi=choi,
            input_dim=input_dim,
            output_dim=output_dim,
            input_subspace=input_subspace,
            output_subspace=output_subspace,
            **kwargs,
        )

    @classmethod
    def from_superoperator(
        cls,
        superoperator: qt.Qobj | np.ndarray,
        *,
        input_dim: int | None = None,
        output_dim: int | None = None,
        input_subspace: Any | None = None,
        output_subspace: Any | None = None,
        **kwargs: Any,
    ) -> "TargetChannel":
        return cls(
            superoperator=superoperator,
            input_dim=input_dim,
            output_dim=output_dim,
            input_subspace=input_subspace,
            output_subspace=output_subspace,
            **kwargs,
        )

    @classmethod
    def from_kraus(
        cls,
        kraus_operators: Sequence[qt.Qobj | np.ndarray],
        *,
        input_subspace: Any | None = None,
        output_subspace: Any | None = None,
        **kwargs: Any,
    ) -> "TargetChannel":
        return cls(
            kraus_operators=kraus_operators,
            input_subspace=input_subspace,
            output_subspace=output_subspace,
            **kwargs,
        )

    @classmethod
    def from_unitary(
        cls,
        unitary: qt.Qobj | np.ndarray,
        *,
        input_subspace: Any | None = None,
        output_subspace: Any | None = None,
        **kwargs: Any,
    ) -> "TargetChannel":
        return cls(
            unitary=unitary,
            input_subspace=input_subspace,
            output_subspace=output_subspace,
            **kwargs,
        )

    @classmethod
    def from_isometry(
        cls,
        isometry: qt.Qobj | np.ndarray,
        *,
        input_subspace: Any | None = None,
        output_subspace: Any | None = None,
        **kwargs: Any,
    ) -> "TargetChannel":
        return cls(
            isometry=isometry,
            input_subspace=input_subspace,
            output_subspace=output_subspace,
            **kwargs,
        )

    def infer_dimension(self) -> int:
        return int(max(self.input_dim, self.output_dim))

    def infer_full_dimension(self) -> int | None:
        if self.subsystem_dims is not None:
            return int(prod(self.subsystem_dims))
        for spec in (self.input_subspace, self.output_subspace):
            if isinstance(spec, SubspaceSpec):
                return int(spec.full_dim)
            if isinstance(spec, Subspace):
                return int(spec.full_dim)
        if self.retained_subsystems is None and self.input_dim == self.output_dim and self.input_subspace is None and self.output_subspace is None:
            return int(self.input_dim)
        return None

    def resolved_data(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
        system_subsystem_dims: Sequence[int] | None = None,
    ) -> dict[str, Any]:
        dims = self.subsystem_dims if self.subsystem_dims is not None else _subsystem_dims(system_subsystem_dims)
        if dims is not None and int(prod(dims)) != int(full_dim):
            raise ValueError(f"Resolved subsystem_dims multiply to {int(prod(dims))}, expected full_dim={int(full_dim)}.")

        full_operator_inputs: list[qt.Qobj] = []
        reduction: dict[str, Any]
        if self.retained_subsystems is not None:
            if dims is None:
                raise ValueError("TargetChannel retained_subsystems require subsystem_dims or a system that exposes composite subsystem dimensions.")
            env = tuple(index for index in range(len(dims)) if index not in self.retained_subsystems)
            env_dims = [dims[index] for index in env]
            env_dim = int(prod(env_dims)) if env_dims else 1
            environment_state = _as_state_qobj(self.environment_state, full_dim=env_dim, subspace=None, dims=env_dims if env_dims else None)
            for col in range(self.input_dim):
                for row in range(self.input_dim):
                    full_operator_inputs.append(
                        _embed_subsystem_operator_with_environment(
                            matrix_unit(self.input_dim, row, col),
                            retained_subsystems=self.retained_subsystems,
                            subsystem_dims=dims,
                            environment_state=environment_state,
                        )
                    )
            reduction = {
                "kind": "partial_trace",
                "retained_subsystems": tuple(int(index) for index in self.retained_subsystems),
                "subsystem_dims": tuple(int(dim) for dim in dims),
            }
        else:
            if self.input_subspace is not None:
                input_spec = coerce_subspace_spec(
                    self.input_subspace,
                    full_dim=full_dim,
                    expected_dim=self.input_dim,
                    name="TargetChannel.input_subspace",
                )
            elif subspace is not None and self.input_dim == subspace.dim:
                input_spec = SubspaceSpec.from_subspace(subspace)
            elif self.input_dim == int(full_dim):
                input_spec = SubspaceSpec.from_indices(full_dim, range(full_dim), kind="full_space")
            else:
                raise ValueError(
                    f"TargetChannel input dimension {self.input_dim} does not match full_dim={int(full_dim)} or the active synthesis subspace."
                )

            for col in range(self.input_dim):
                for row in range(self.input_dim):
                    embedded = input_spec.basis_matrix @ matrix_unit(self.input_dim, row, col) @ input_spec.basis_matrix.conj().T
                    full_operator_inputs.append(_as_state_qobj(embedded, full_dim=full_dim, dims=dims))

            if self.output_subspace is not None:
                output_spec = coerce_subspace_spec(
                    self.output_subspace,
                    full_dim=full_dim,
                    expected_dim=self.output_dim,
                    name="TargetChannel.output_subspace",
                )
                reduction = {
                    "kind": "basis",
                    "basis_matrix": np.asarray(output_spec.basis_matrix, dtype=np.complex128),
                    "labels": tuple(output_spec.labels),
                }
            elif subspace is not None and self.output_dim == subspace.dim:
                output_spec = SubspaceSpec.from_subspace(subspace)
                reduction = {
                    "kind": "basis",
                    "basis_matrix": np.asarray(output_spec.basis_matrix, dtype=np.complex128),
                    "labels": tuple(output_spec.labels),
                }
            elif self.output_dim == int(full_dim):
                reduction = {"kind": "none"}
            else:
                raise ValueError(
                    f"TargetChannel output dimension {self.output_dim} does not match full_dim={int(full_dim)} or the active synthesis subspace."
                )

        if self.enforce_cptp:
            hermitian_choi = 0.5 * (self.choi + self.choi.conj().T)
            min_eig = float(np.min(np.linalg.eigvalsh(hermitian_choi)).real)
            if min_eig < -1.0e-8:
                raise ValueError("TargetChannel target Choi matrix is not completely positive within tolerance.")

        return {
            "channel_dim": int(self.channel_dim),
            "input_dim": int(self.input_dim),
            "output_dim": int(self.output_dim),
            "target_choi": np.asarray(self.choi, dtype=np.complex128),
            "target_superoperator": np.asarray(self.superoperator, dtype=np.complex128),
            "probe_operators": full_operator_inputs,
            "reduction": reduction,
            "retained_subsystems": None if self.retained_subsystems is None else tuple(int(index) for index in self.retained_subsystems),
            "subsystem_dims": None if dims is None else tuple(int(dim) for dim in dims),
            "enforce_cptp": bool(self.enforce_cptp),
        }


class ObservableTarget:
    def __init__(
        self,
        *,
        initial_states: Sequence[qt.Qobj | np.ndarray] | None = None,
        observables: Sequence[qt.Qobj | np.ndarray] | None = None,
        target_expectations: Sequence[Sequence[complex | float]] | Sequence[complex | float] | np.ndarray | None = None,
        initial_state: qt.Qobj | np.ndarray | None = None,
        observable: qt.Qobj | np.ndarray | None = None,
        target_expectation: complex | float | None = None,
        state_weights: Sequence[float] | None = None,
        observable_weights: Sequence[float] | None = None,
    ) -> None:
        if initial_state is not None or observable is not None or target_expectation is not None:
            if initial_states is not None or observables is not None or target_expectations is not None:
                raise ValueError("Use either singular or plural observable-target arguments, not both.")
            initial_states = [] if initial_state is None else [initial_state]
            observables = [] if observable is None else [observable]
            target_expectations = [] if target_expectation is None else [[target_expectation]]

        if initial_states is None or observables is None or target_expectations is None:
            raise ValueError("ObservableTarget requires initial_states, observables, and target_expectations.")
        if len(initial_states) == 0 or len(observables) == 0:
            raise ValueError("ObservableTarget requires at least one initial state and one observable.")

        expectations = np.asarray(target_expectations, dtype=np.complex128)
        n_states = len(initial_states)
        n_observables = len(observables)
        if expectations.ndim == 1:
            if n_states == 1 and expectations.size == n_observables:
                expectations = expectations.reshape(1, n_observables)
            elif n_observables == 1 and expectations.size == n_states:
                expectations = expectations.reshape(n_states, 1)
            else:
                raise ValueError(
                    "ObservableTarget target_expectations must have shape (n_states, n_observables), "
                    "or reduce to a single-state/single-observable special case."
                )
        if expectations.shape != (n_states, n_observables):
            raise ValueError(
                f"ObservableTarget target_expectations has shape {expectations.shape}, expected {(n_states, n_observables)}."
            )

        self.initial_states = tuple(initial_states)
        self.observables = tuple(observables)
        self.target_expectations = expectations
        self.state_weights = tuple(float(w) for w in state_weights) if state_weights is not None else None
        self.observable_weights = tuple(float(w) for w in observable_weights) if observable_weights is not None else None

    def infer_dimension(self) -> int:
        return _infer_dimension_from_object(self.initial_states[0])

    def resolved_data(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> tuple[list[qt.Qobj], list[qt.Qobj], np.ndarray, np.ndarray, np.ndarray]:
        initial = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace) for state in self.initial_states]
        observables = [_as_operator_qobj(op, full_dim=full_dim, subspace=subspace) for op in self.observables]
        state_weights = _normalize_weights(self.state_weights, len(initial), name="ObservableTarget")
        observable_weights = _normalize_weights(self.observable_weights, len(observables), name="ObservableTarget")
        return initial, observables, np.asarray(self.target_expectations, dtype=np.complex128), state_weights, observable_weights


@dataclass(frozen=True)
class TrajectoryCheckpoint:
    step: int
    target_states: tuple[qt.Qobj | np.ndarray, ...] = field(default_factory=tuple)
    observables: tuple[qt.Qobj | np.ndarray, ...] = field(default_factory=tuple)
    target_expectations: np.ndarray | Sequence[Sequence[complex | float]] | Sequence[complex | float] = field(default_factory=tuple)
    weight: float = 1.0
    state_weights: tuple[float, ...] = field(default_factory=tuple)
    observable_weights: tuple[float, ...] = field(default_factory=tuple)
    label: str | None = None

    def __post_init__(self) -> None:
        if int(self.step) < 0:
            raise ValueError("TrajectoryCheckpoint.step must be non-negative.")
        if float(self.weight) < 0.0:
            raise ValueError("TrajectoryCheckpoint.weight must be non-negative.")
        if not self.target_states and not self.observables:
            raise ValueError("TrajectoryCheckpoint requires target_states, observables, or both.")
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "target_states", tuple(self.target_states))
        object.__setattr__(self, "observables", tuple(self.observables))
        object.__setattr__(self, "state_weights", tuple(float(w) for w in self.state_weights))
        object.__setattr__(self, "observable_weights", tuple(float(w) for w in self.observable_weights))
        object.__setattr__(self, "target_expectations", np.asarray(self.target_expectations, dtype=np.complex128))

    def resolved_data(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None,
        initial_state_count: int,
        default_state_weights: np.ndarray,
    ) -> dict[str, Any]:
        state_targets = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace) for state in self.target_states]
        if state_targets and len(state_targets) != initial_state_count:
            raise ValueError(
                f"TrajectoryCheckpoint at step {self.step} received {len(state_targets)} state targets, expected {initial_state_count}."
            )
        observables = [_as_operator_qobj(op, full_dim=full_dim, subspace=subspace) for op in self.observables]
        expectations = np.asarray(self.target_expectations, dtype=np.complex128)
        if observables:
            if expectations.size == 0:
                raise ValueError(f"TrajectoryCheckpoint at step {self.step} requires target_expectations for observables.")
            if expectations.ndim == 1:
                if initial_state_count == 1 and expectations.size == len(observables):
                    expectations = expectations.reshape(1, len(observables))
                elif len(observables) == 1 and expectations.size == initial_state_count:
                    expectations = expectations.reshape(initial_state_count, 1)
                else:
                    raise ValueError(
                        f"TrajectoryCheckpoint at step {self.step} target_expectations must match "
                        f"(n_states, n_observables)={(initial_state_count, len(observables))}."
                    )
            if expectations.shape != (initial_state_count, len(observables)):
                raise ValueError(
                    f"TrajectoryCheckpoint at step {self.step} target_expectations has shape {expectations.shape}, "
                    f"expected {(initial_state_count, len(observables))}."
                )
        elif expectations.size != 0:
            raise ValueError(f"TrajectoryCheckpoint at step {self.step} provided target_expectations without observables.")

        state_weights = _normalize_weights(
            self.state_weights if self.state_weights else default_state_weights,
            initial_state_count,
            name=f"TrajectoryCheckpoint(step={self.step})",
        )
        observable_weights = _normalize_weights(
            self.observable_weights if self.observable_weights else None,
            len(observables) if observables else 1,
            name=f"TrajectoryCheckpoint(step={self.step})",
        )
        if not observables:
            observable_weights = np.asarray([], dtype=float)
        return {
            "step": int(self.step),
            "label": str(self.label) if self.label is not None else f"after_gate_{int(self.step)}",
            "weight": float(self.weight),
            "target_states": state_targets,
            "observables": observables,
            "target_expectations": expectations,
            "state_weights": state_weights,
            "observable_weights": observable_weights,
        }


class TrajectoryTarget:
    def __init__(
        self,
        *,
        initial_states: Sequence[qt.Qobj | np.ndarray],
        checkpoints: Sequence[TrajectoryCheckpoint],
        state_weights: Sequence[float] | None = None,
    ) -> None:
        if not initial_states:
            raise ValueError("TrajectoryTarget requires at least one initial state.")
        if not checkpoints:
            raise ValueError("TrajectoryTarget requires at least one checkpoint.")
        self.initial_states = tuple(initial_states)
        self.checkpoints = tuple(checkpoints)
        self.state_weights = tuple(float(w) for w in state_weights) if state_weights is not None else None

    def infer_dimension(self) -> int:
        return _infer_dimension_from_object(self.initial_states[0])

    def resolved_data(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> tuple[list[qt.Qobj], np.ndarray, list[dict[str, Any]]]:
        initial = [_as_state_qobj(state, full_dim=full_dim, subspace=subspace) for state in self.initial_states]
        state_weights = _normalize_weights(self.state_weights, len(initial), name="TrajectoryTarget")
        checkpoints = [
            checkpoint.resolved_data(
                full_dim=full_dim,
                subspace=subspace,
                initial_state_count=len(initial),
                default_state_weights=state_weights,
            )
            for checkpoint in self.checkpoints
        ]
        checkpoints.sort(key=lambda row: int(row["step"]))
        return initial, state_weights, checkpoints


SynthesisTarget = (
    TargetUnitary
    | TargetStateMapping
    | TargetReducedStateMapping
    | TargetIsometry
    | TargetChannel
    | ObservableTarget
    | TrajectoryTarget
)


def coerce_target(target: SynthesisTarget | np.ndarray | qt.Qobj) -> SynthesisTarget:
    if isinstance(
        target,
        (
            TargetUnitary,
            TargetStateMapping,
            TargetReducedStateMapping,
            TargetIsometry,
            TargetChannel,
            ObservableTarget,
            TrajectoryTarget,
        ),
    ):
        return target
    if isinstance(target, qt.Qobj):
        return TargetUnitary(np.asarray(target.full(), dtype=np.complex128))
    return TargetUnitary(np.asarray(target, dtype=np.complex128))


def make_easy_target(n_match: int) -> np.ndarray:
    n_cav = n_match + 1
    theta = np.linspace(0.1, 0.7, n_cav)
    phi = np.linspace(-0.4, 0.3, n_cav)
    sqr = np.asarray(qt.Qobj(np.eye(2 * n_cav)).full(), dtype=np.complex128)
    for n in range(n_cav):
        block = np.asarray(qubit_rotation_xy(theta[n], phi[n]).full(), dtype=np.complex128)
        idx = qubit_cavity_block_indices(n_cav, n)
        sqr[np.ix_(idx, idx)] = block
    snap = _diag_phase(np.linspace(0.0, 0.25, n_cav))
    return snap @ sqr


def make_mps_like_target(kind: str, n_match: int, **kwargs: Any) -> np.ndarray:
    n_cav = n_match + 1
    if kind == "ghz":
        return _ghz_reference_target(n_cav)
    if kind == "cluster":
        return _cluster_reference_target(n_cav, which=kwargs.get("which", "u1"))
    raise ValueError(f"Unsupported MPS-like target kind '{kind}'.")


def load_mps_reference(kind: str, n_match: int, base_dir: str | Path | None = None, **kwargs: Any) -> np.ndarray | None:
    search_root = Path(base_dir) if base_dir is not None else _default_reference_root()
    if search_root is None or not search_root.exists():
        return None

    which = str(kwargs.get("which", "u1")).lower()
    patterns = [
        f"**/*{kind}*{which}*{n_match}*.npy",
        f"**/*{kind}*{which}*{n_match}*.npz",
        f"**/*{kind}*{which}*.npy",
        f"**/*{kind}*{which}*.npz",
        f"**/*{kind}*{n_match}*.npy",
        f"**/*{kind}*{n_match}*.npz",
        f"**/*{kind}*.npy",
        f"**/*{kind}*.npz",
    ]
    for pattern in patterns:
        for file_path in search_root.glob(pattern):
            try:
                if file_path.suffix == ".npy":
                    arr = np.load(file_path)
                else:
                    data = np.load(file_path)
                    key = "U" if "U" in data else list(data.keys())[0]
                    arr = data[key]
                arr = np.asarray(arr, dtype=np.complex128)
                if arr.shape == (_full_dim(n_match), _full_dim(n_match)):
                    return arr
            except Exception:
                continue

    if kind in {"ghz", "cluster"}:
        return make_mps_like_target(kind, n_match, **kwargs)
    return None


def make_target(name: str, n_match: int, variant: str = "analytic", **kwargs: Any) -> np.ndarray:
    lowered = name.lower()
    if lowered in {"easy", "easy_realizable"}:
        return make_easy_target(n_match)
    if lowered in {"ghz", "cluster"}:
        if variant == "mps":
            ref = load_mps_reference(lowered, n_match, base_dir=kwargs.get("base_dir"), **kwargs)
            if ref is not None:
                return ref
        return make_mps_like_target(lowered, n_match, **kwargs)
    raise ValueError(f"Unsupported target '{name}'.")
