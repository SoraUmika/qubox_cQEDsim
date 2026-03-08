from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_block_indices
from cqed_sim.core.ideal_gates import qubit_rotation_xy


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


def _qubit_rot(theta: float, phi: float, n_cav: int) -> np.ndarray:
    return np.asarray(qt.tensor(qubit_rotation_xy(theta, phi), qt.qeye(n_cav)).full(), dtype=np.complex128)


def _expand_unitary_qc(uni: np.ndarray, dim_q: int, dim_b: int, dim_c: int) -> np.ndarray:
    """Embed a small qubit⊗bond unitary into qubit⊗cavity space."""
    new_dim = dim_q * dim_c
    new_indices = [dim_c * iq + jb for iq in range(dim_q) for jb in range(dim_b)]
    expanded = np.eye(new_dim, dtype=np.complex128)
    for old_row, new_row in enumerate(new_indices):
        for old_col, new_col in enumerate(new_indices):
            expanded[new_row, new_col] = uni[old_row, old_col]
    return expanded


def _ghz_reference_target(n_cav: int) -> np.ndarray:
    # Noah reference: cnot(N=2, control=1, target=0), expanded in qubit⊗cavity.
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
    """Noah reference cluster unitary in qubit⊗cavity ordering.

    The reference in `unitary_util.py` is built in qubit⊗cavity order as:
      U1 = SWAP @ CZ @ (H ⊗ I)
      U2 = (Ry(pi/2) ⊗ I) @ U1

    `test_decomp.ipynb` uses `U1` as the cluster decomposition target, so that is the default.
    """
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
    """Construct a consistent ancilla(qubit)+memory(cavity) target.

    Mapping convention used here: cavity Fock level n encodes the discrete memory/bond index
    of an MPS sequential unitary, truncated to n=0..n_match. The qubit is the ancilla.
    """
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
