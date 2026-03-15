"""Compatibility wrappers for the legacy prototype-style holographic API.

New code should prefer the structured package API exposed through
`cqed_sim.quantum_algorithms.holographic_sim`.
"""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .channel import HolographicChannel
from .channel_embedding import PurifiedChannelStep
from .observables import as_observable
from .utils import as_complex_array, basis_vector, coerce_density_matrix, progress_wrapper


Operator = np.ndarray
Unitary = Any
SYS_DIM = 2
TOL_DEFAULT = 1.0e-15


@dataclass(frozen=True)
class MeasBasis:
    measured: bool
    eigvals: Optional[np.ndarray]
    eigvecs: Optional[np.ndarray]


def _asarray_unitary(unitary: Unitary) -> np.ndarray:
    return np.asarray(as_complex_array(unitary), dtype=np.complex128)


def _observable_from_eig_dict(eig_dict: Dict[float, np.ndarray]) -> np.ndarray:
    pairs = [(complex(key), np.asarray(as_complex_array(vec), dtype=np.complex128).reshape(-1)) for key, vec in eig_dict.items()]
    dim = pairs[0][1].size
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for eigval, eigvec in pairs:
        matrix += eigval * np.outer(eigvec, eigvec.conj())
    return matrix


def _observable_from_eigs(eigvals: np.ndarray, eigvecs: np.ndarray) -> np.ndarray:
    values = np.asarray(eigvals, dtype=np.complex128).reshape(-1)
    vecs = np.asarray(as_complex_array(eigvecs), dtype=np.complex128)
    return vecs @ np.diag(values) @ vecs.conj().T


def _initial_bond_state(bond_dim: int) -> np.ndarray:
    return coerce_density_matrix(basis_vector(int(bond_dim), 0), dim=int(bond_dim))


def infer_bond_dim_from_U_list(U_list: Sequence[Unitary], *, d: int = SYS_DIM) -> int:
    if not U_list:
        raise ValueError("U_list is empty; cannot infer bond dimension.")
    resolved_d = int(d)
    if resolved_d <= 0:
        raise ValueError("d must be positive.")
    first = _asarray_unitary(U_list[0])
    full_dim = int(first.shape[0])
    if first.shape[0] != first.shape[1]:
        raise ValueError("All unitaries must be square.")
    if full_dim % resolved_d != 0:
        raise ValueError(f"full_dim={full_dim} is not divisible by d={resolved_d}.")
    for item in U_list[1:]:
        arr = _asarray_unitary(item)
        if arr.shape != (full_dim, full_dim):
            raise ValueError("All unitaries in U_list must have the same square shape.")
    return full_dim // resolved_d


def holographic_step(
    U_i: Unitary,
    rho_bond: np.ndarray,
    O_eigs_dict: Optional[Dict[float, np.ndarray]] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    rho_dm = coerce_density_matrix(rho_bond)
    bond_dim = int(rho_dm.shape[0])
    full_dim = int(_asarray_unitary(U_i).shape[0])
    if full_dim % bond_dim != 0:
        raise ValueError("Unitary dimension is not compatible with the provided bond state.")
    physical_dim = full_dim // bond_dim
    channel = HolographicChannel.from_unitary(U_i, physical_dim=physical_dim, bond_dim=bond_dim)
    step = PurifiedChannelStep(channel)
    if O_eigs_dict is None:
        return step.propagate(rho_dm), None
    observable = _observable_from_eig_dict(O_eigs_dict)
    outcome = step.sample_measurement(rho_dm, observable)
    eigenvalue = None if outcome.eigenvalue is None else float(np.real_if_close(outcome.eigenvalue))
    return outcome.bond_state, eigenvalue


def holographic_step_alloutcomes(
    U_i: Unitary,
    rho_bond: np.ndarray,
    eigvals_meas: np.ndarray,
    eigvecs_meas: np.ndarray,
    *,
    tol: float = TOL_DEFAULT,
) -> List[Tuple[np.ndarray, float, complex, float]]:
    rho_dm = coerce_density_matrix(rho_bond)
    bond_dim = int(rho_dm.shape[0])
    full_dim = int(_asarray_unitary(U_i).shape[0])
    if full_dim % bond_dim != 0:
        raise ValueError("Unitary dimension is not compatible with the provided bond state.")
    physical_dim = full_dim // bond_dim
    channel = HolographicChannel.from_unitary(U_i, physical_dim=physical_dim, bond_dim=bond_dim)
    step = PurifiedChannelStep(channel)
    observable = _observable_from_eigs(eigvals_meas, eigvecs_meas)
    branches = []
    for outcome in step.enumerate_measurement_branches(rho_dm, observable, atol=tol):
        value = 0.0 if outcome.eigenvalue is None else float(np.real_if_close(outcome.eigenvalue))
        branches.append((outcome.bond_state, float(outcome.probability), complex(value), value))
    return branches


def _simulate_shots_serial(
    unitary_arrays: Sequence[np.ndarray],
    observable_arrays: Sequence[np.ndarray | None],
    *,
    shots: int,
    physical_dim: int,
    show_progress: bool,
) -> np.ndarray:
    bond_dim = infer_bond_dim_from_U_list(unitary_arrays, d=physical_dim)
    steps = [
        PurifiedChannelStep(HolographicChannel.from_unitary(unitary, physical_dim=physical_dim, bond_dim=bond_dim))
        for unitary in unitary_arrays
    ]
    observables = [None if matrix is None else as_observable(matrix) for matrix in observable_arrays]
    samples = np.zeros(int(shots), dtype=np.complex128)
    iterator = progress_wrapper(range(int(shots)), enabled=show_progress, desc="holographic_sim shots")
    for shot in iterator:
        rho = _initial_bond_state(bond_dim)
        estimator = 1.0 + 0.0j
        for step, observable in zip(steps, observables):
            if observable is None:
                rho = step.propagate(rho)
                continue
            outcome = step.sample_measurement(rho, observable)
            rho = outcome.bond_state
            estimator *= complex(0.0 if outcome.eigenvalue is None else outcome.eigenvalue)
        samples[int(shot)] = estimator
    return samples


def _simulate_shots_worker(args: Tuple[Sequence[np.ndarray], Sequence[np.ndarray | None], int, int]) -> np.ndarray:
    unitary_arrays, observable_arrays, shots, physical_dim = args
    return _simulate_shots_serial(
        unitary_arrays,
        observable_arrays,
        shots=int(shots),
        physical_dim=int(physical_dim),
        show_progress=False,
    )


def holographic_sim(
    U_list: Sequence[Unitary],
    op_list: Sequence[Optional[Operator]],
    *,
    shot_nums: int = 10_000,
    d: int = SYS_DIM,
    parallel_process: bool = False,
    process_num: int = 5,
    chunk_size: int = 500,
    show_progress: bool = True,
) -> np.ndarray:
    if len(U_list) != len(op_list):
        raise ValueError("U_list and op_list must have the same length.")
    infer_bond_dim_from_U_list(U_list, d=d)
    unitary_arrays = tuple(_asarray_unitary(item) for item in U_list)
    observable_arrays = tuple(None if item is None else np.asarray(as_complex_array(item), dtype=np.complex128) for item in op_list)
    if not parallel_process or int(shot_nums) <= int(chunk_size):
        return _simulate_shots_serial(
            unitary_arrays,
            observable_arrays,
            shots=int(shot_nums),
            physical_dim=int(d),
            show_progress=show_progress,
        )

    tasks: list[Tuple[Sequence[np.ndarray], Sequence[np.ndarray | None], int, int]] = []
    remaining = int(shot_nums)
    while remaining > 0:
        chunk = min(int(chunk_size), remaining)
        tasks.append((unitary_arrays, observable_arrays, chunk, int(d)))
        remaining -= chunk

    outputs: list[np.ndarray] = []
    with Pool(processes=int(process_num)) as pool:
        iterator: Iterable[np.ndarray] = pool.imap_unordered(_simulate_shots_worker, tasks)
        iterator = progress_wrapper(iterator, enabled=show_progress, total=len(tasks), desc="holographic_sim chunks")
        for result in iterator:
            outputs.append(result)
    return np.concatenate(outputs, axis=0)


def _sorted_eigh_desc(operator: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eigvals, eigvecs = np.linalg.eigh(np.asarray(as_complex_array(operator), dtype=np.complex128))
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def holographic_sim_bfs(
    U_list: Sequence[Unitary],
    op_list: Sequence[Optional[Operator]],
    *,
    d: int = SYS_DIM,
    tol: float = TOL_DEFAULT,
) -> List[Dict[str, Any]]:
    if len(U_list) != len(op_list):
        raise ValueError("U_list and op_list must have the same length.")
    bond_dim = infer_bond_dim_from_U_list(U_list, d=d)
    steps = [
        PurifiedChannelStep(HolographicChannel.from_unitary(unitary, physical_dim=int(d), bond_dim=bond_dim))
        for unitary in U_list
    ]
    observables = [None if operator is None else as_observable(operator) for operator in op_list]
    branches: List[Dict[str, Any]] = [
        {
            "rho_bond": _initial_bond_state(bond_dim),
            "prob": 1.0,
            "weight": 1.0 + 0.0j,
            "outcomes": [],
        }
    ]

    for step, observable in zip(steps, observables):
        next_branches: List[Dict[str, Any]] = []
        for branch in branches:
            if branch["prob"] <= tol:
                continue
            if observable is None:
                next_branches.append(
                    {
                        "rho_bond": step.propagate(branch["rho_bond"]),
                        "prob": float(branch["prob"]),
                        "weight": complex(branch["weight"]),
                        "outcomes": list(branch["outcomes"]),
                    }
                )
                continue
            for outcome in step.enumerate_measurement_branches(branch["rho_bond"], observable, atol=tol):
                new_prob = float(branch["prob"]) * float(outcome.probability)
                if new_prob <= tol:
                    continue
                value = complex(0.0 if outcome.eigenvalue is None else outcome.eigenvalue)
                next_branches.append(
                    {
                        "rho_bond": outcome.bond_state,
                        "prob": new_prob,
                        "weight": complex(branch["weight"]) * value,
                        "outcomes": list(branch["outcomes"]) + [float(np.real_if_close(value))],
                    }
                )
        branches = next_branches
    return [branch for branch in branches if branch["prob"] > tol]


def branches_to_dataframe(
    final_branches: List[Dict[str, Any]],
    op_list: Sequence[Optional[Operator]],
    normalized_probs: np.ndarray | None = None,
):
    try:
        import pandas as pd
    except Exception as exc:  # pragma: no cover - optional dependency shim
        raise ImportError("branches_to_dataframe requires pandas.") from exc

    measured_sites = [idx for idx, operator in enumerate(op_list) if operator is not None]
    if not final_branches:
        cols = [f"s{idx}" for idx in measured_sites] + ["prob", "weight"]
        return pd.DataFrame(columns=cols)

    if normalized_probs is None:
        probs = np.asarray([branch["prob"] for branch in final_branches], dtype=float)
        probs = probs / probs.sum()
    else:
        probs = np.asarray(normalized_probs, dtype=float)

    rows = []
    for branch, prob in zip(final_branches, probs):
        row = {f"s{site}": branch["outcomes"][outcome_idx] for outcome_idx, site in enumerate(measured_sites)}
        row["prob"] = float(prob)
        row["weight"] = complex(branch["weight"])
        rows.append(row)
    site_cols = [f"s{idx}" for idx in measured_sites]
    return pd.DataFrame(rows)[site_cols + ["prob", "weight"]]


def holographic_sim_cached(
    U_list: Sequence[Unitary],
    op_list: Sequence[Optional[Operator]],
    *,
    shot_nums: int = 10_000,
    d: int = SYS_DIM,
    return_df: bool = False,
) -> np.ndarray | Tuple[np.ndarray, Any]:
    branches = holographic_sim_bfs(U_list, op_list, d=d)
    if not branches:
        raise ValueError("Total branch probability is zero.")
    probs = np.asarray([branch["prob"] for branch in branches], dtype=float)
    probs = probs / probs.sum()
    weights = np.asarray([branch["weight"] for branch in branches], dtype=np.complex128)
    rng = np.random.default_rng()
    indices = rng.choice(len(branches), size=int(shot_nums), p=probs)
    samples = weights[indices]
    if not return_df:
        return samples
    return samples, branches_to_dataframe(branches, op_list, normalized_probs=probs)


__all__ = [
    "Operator",
    "Unitary",
    "SYS_DIM",
    "TOL_DEFAULT",
    "MeasBasis",
    "infer_bond_dim_from_U_list",
    "holographic_step",
    "holographic_step_alloutcomes",
    "holographic_sim",
    "holographic_sim_bfs",
    "branches_to_dataframe",
    "holographic_sim_cached",
]
