from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np

from .subspace import Subspace


@dataclass(frozen=True)
class LeakageMetrics:
    average: float
    worst: float
    per_probe: tuple[float, ...]


def _fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def subspace_unitary_fidelity(
    actual: np.ndarray,
    target: np.ndarray,
    gauge: str = "global",
    block_slices: Iterable[slice] | None = None,
) -> float:
    """Return fidelity on subspace with gauge handling.

    - global: invariant to one global phase
    - block: invariant to independent phase per provided block slice
    """
    u = np.asarray(actual, dtype=np.complex128)
    v = np.asarray(target, dtype=np.complex128)
    if u.shape != v.shape or u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("actual and target must be square and shape-matched.")
    d = u.shape[0]
    if gauge == "global":
        return float(np.clip(abs(np.trace(v.conj().T @ u)) / d, 0.0, 1.0))
    if gauge == "block":
        if block_slices is None:
            raise ValueError("block_slices are required when gauge='block'.")
        accum = 0.0
        total_dim = 0
        for sl in block_slices:
            vb = v[sl, :]
            ub = u[sl, :]
            tr = np.trace(vb.conj().T @ ub)
            blk_dim = sl.stop - sl.start
            accum += abs(tr)
            total_dim += blk_dim
        return float(np.clip(accum / max(total_dim, 1), 0.0, 1.0))
    raise ValueError("Unsupported gauge. Use 'global' or 'block'.")


def leakage_metrics(
    full_operator: np.ndarray,
    subspace: Subspace,
    probes: Iterable[np.ndarray] | None = None,
    n_jobs: int = 1,
) -> LeakageMetrics:
    u = np.asarray(full_operator, dtype=np.complex128)
    if u.shape != (subspace.full_dim, subspace.full_dim):
        raise ValueError("full_operator shape mismatch with subspace.full_dim.")
    if probes is None:
        eye = np.eye(subspace.dim, dtype=np.complex128)
        probe_list = [eye[:, k] for k in range(subspace.dim)]
    else:
        probe_list = [np.asarray(p, dtype=np.complex128).reshape(-1) for p in probes]

    def _one(psi_s_raw: np.ndarray) -> float:
        psi_s = np.asarray(psi_s_raw, dtype=np.complex128).reshape(-1)
        psi_s = psi_s / np.linalg.norm(psi_s)
        psi_f = subspace.embed(psi_s)
        out = u @ psi_f
        kept = subspace.extract(out)
        keep_prob = float(np.vdot(kept, kept).real)
        return max(0.0, 1.0 - keep_prob)

    if int(n_jobs) > 1 and len(probe_list) > 1:
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
            values = list(ex.map(_one, probe_list))
    else:
        values = [_one(p) for p in probe_list]
    worst = max(values) if values else 0.0
    avg = float(np.mean(values)) if values else 0.0
    return LeakageMetrics(average=avg, worst=worst, per_probe=tuple(float(v) for v in values))


def objective_breakdown(
    actual_subspace: np.ndarray,
    target_subspace: np.ndarray,
    full_operator: np.ndarray,
    subspace: Subspace,
    leakage_weight: float = 0.0,
    gauge: str = "global",
    leakage_n_jobs: int = 1,
) -> dict[str, float]:
    fidelity = subspace_unitary_fidelity(
        actual_subspace,
        target_subspace,
        gauge=gauge,
        block_slices=subspace.per_fock_blocks() if gauge == "block" else None,
    )
    leak = leakage_metrics(full_operator, subspace, n_jobs=leakage_n_jobs)
    fidelity_loss = 1.0 - fidelity
    leakage_term = float(leakage_weight) * leak.worst
    return {
        "fidelity": fidelity,
        "fidelity_loss": fidelity_loss,
        "leakage_average": leak.average,
        "leakage_worst": leak.worst,
        "leakage_term": leakage_term,
        "objective": fidelity_loss + leakage_term,
    }


def unitarity_error(op: np.ndarray) -> float:
    u = np.asarray(op, dtype=np.complex128)
    ident = np.eye(u.shape[0], dtype=np.complex128)
    return _fro_norm(u.conj().T @ u - ident)
