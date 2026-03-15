from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_block_indices
from cqed_sim.core.ideal_gates import qubit_rotation_xy
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
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root.parent / "noah_repo" / "miscellaneous" / "sequential_simulation",
        Path(
            r"C:\Users\dazzl\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\noah_repo\miscellaneous\sequential_simulation"
        ),
        Path(
            r"C:\Users\jl82323\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\noah_repo\miscellaneous\sequential_simulation"
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _validate_unitary_matrix(matrix: np.ndarray, *, atol: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Target matrices must be square.")
    ident = np.eye(arr.shape[0], dtype=np.complex128)
    if np.linalg.norm(arr.conj().T @ arr - ident, ord="fro") > atol:
        raise ValueError("Target matrix is not unitary within tolerance.")
    return arr


def _as_state_qobj(state: qt.Qobj | np.ndarray, *, full_dim: int, subspace: Subspace | None = None) -> qt.Qobj:
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

    if obj.isket or (obj.shape[1] == 1 and obj.shape[0] > 1):
        vec = np.asarray(obj.full(), dtype=np.complex128).reshape(-1)
        if subspace is not None and vec.size == subspace.dim:
            vec = subspace.embed(vec)
        if vec.size != full_dim:
            raise ValueError(f"State vector has dimension {vec.size}, expected {full_dim}.")
        norm = np.linalg.norm(vec)
        if norm > 0.0:
            vec = vec / norm
        dims = source_dims if source_dims is not None and int(np.prod(source_dims[0])) == full_dim else [[full_dim], [1]]
        return qt.Qobj(vec.reshape(-1), dims=dims)

    matrix = np.asarray(obj.full(), dtype=np.complex128)
    if subspace is not None and matrix.shape == (subspace.dim, subspace.dim):
        embedded = np.zeros((full_dim, full_dim), dtype=np.complex128)
        idx = np.asarray(subspace.indices, dtype=int)
        embedded[np.ix_(idx, idx)] = matrix
        matrix = embedded
    if matrix.shape != (full_dim, full_dim):
        raise ValueError(f"State operator has shape {matrix.shape}, expected {(full_dim, full_dim)}.")
    dims = source_dims if source_dims is not None and int(np.prod(source_dims[0])) == full_dim else [[full_dim], [full_dim]]
    return qt.Qobj(matrix, dims=dims)


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


SynthesisTarget = TargetUnitary | TargetStateMapping


def coerce_target(target: SynthesisTarget | np.ndarray | qt.Qobj) -> SynthesisTarget:
    if isinstance(target, (TargetUnitary, TargetStateMapping)):
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
