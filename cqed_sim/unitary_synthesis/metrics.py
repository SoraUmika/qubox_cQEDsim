from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import qutip as qt

from .subspace import Subspace


@dataclass(frozen=True)
class LeakageMetrics:
    average: float
    worst: float
    per_probe: tuple[float, ...]


def _fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def _as_qobj(state: qt.Qobj | np.ndarray) -> qt.Qobj:
    if isinstance(state, qt.Qobj):
        return state
    arr = np.asarray(state, dtype=np.complex128)
    if arr.ndim == 1:
        return qt.Qobj(arr.reshape(-1))
    return qt.Qobj(arr)


def _ket_difference_error(output: qt.Qobj, target: qt.Qobj) -> float:
    psi = np.asarray(output.full(), dtype=np.complex128).reshape(-1)
    phi = np.asarray(target.full(), dtype=np.complex128).reshape(-1)
    return float(np.linalg.norm(psi - phi) ** 2)


def subspace_unitary_fidelity(
    actual: np.ndarray,
    target: np.ndarray,
    gauge: str = "global",
    block_slices: Iterable[slice | Sequence[int] | np.ndarray] | None = None,
) -> float:
    u = np.asarray(actual, dtype=np.complex128)
    v = np.asarray(target, dtype=np.complex128)
    if u.shape != v.shape or u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("actual and target must be square and shape-matched.")
    d = u.shape[0]
    overlap = v.conj().T @ u
    if gauge == "none":
        return float(np.clip(np.real(np.trace(overlap)) / d, 0.0, 1.0))
    if gauge == "global":
        return float(np.clip(abs(np.trace(overlap)) / d, 0.0, 1.0))
    if gauge == "diagonal":
        return float(np.clip(np.mean(np.abs(np.diag(overlap))), 0.0, 1.0))
    if gauge == "block":
        if block_slices is None:
            raise ValueError("block_slices are required when gauge='block'.")
        accum = 0.0
        for block in block_slices:
            if isinstance(block, slice):
                idx = np.arange(block.start, block.stop, dtype=int)
            else:
                idx = np.asarray(block, dtype=int).reshape(-1)
            if idx.size == 0:
                continue
            accum += abs(np.trace(overlap[np.ix_(idx, idx)]))
        return float(np.clip(accum / max(d, 1), 0.0, 1.0))
    raise ValueError("Unsupported gauge. Use 'none', 'global', 'diagonal', or 'block'.")


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


def state_leakage_metrics(states: Sequence[qt.Qobj | np.ndarray], subspace: Subspace) -> LeakageMetrics:
    probe_list = [_as_qobj(state) for state in states]
    values: list[float] = []
    idx = np.asarray(subspace.indices, dtype=int)
    for state in probe_list:
        if state.isoper:
            rho = np.asarray(state.full(), dtype=np.complex128)
            keep_prob = float(np.real_if_close(np.trace(rho[np.ix_(idx, idx)])).real)
        else:
            vec = np.asarray(state.full(), dtype=np.complex128).reshape(-1)
            kept = vec[idx]
            keep_prob = float(np.vdot(kept, kept).real)
        values.append(max(0.0, 1.0 - keep_prob))
    worst = max(values) if values else 0.0
    avg = float(np.mean(values)) if values else 0.0
    return LeakageMetrics(average=avg, worst=worst, per_probe=tuple(float(v) for v in values))


def state_mapping_metrics(
    outputs: Sequence[qt.Qobj | np.ndarray],
    targets: Sequence[qt.Qobj | np.ndarray],
    *,
    weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    if len(outputs) != len(targets):
        raise ValueError("outputs and targets must have the same length.")
    out_states = [_as_qobj(state) for state in outputs]
    tgt_states = [_as_qobj(state) for state in targets]
    if weights is None:
        weight_arr = np.full(len(out_states), 1.0 / max(len(out_states), 1), dtype=float)
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != (len(out_states),):
            raise ValueError("weights must match the number of output states.")
        if np.sum(weight_arr) <= 0.0:
            raise ValueError("weights must sum to a positive value.")
        weight_arr = weight_arr / np.sum(weight_arr)

    errors: list[float] = []
    fidelities: list[float] = []
    for output, target in zip(out_states, tgt_states):
        if output.isoper or target.isoper:
            rho_out = output if output.isoper else output.proj()
            rho_tgt = target if target.isoper else target.proj()
            diff = np.asarray((rho_out - rho_tgt).full(), dtype=np.complex128)
            errors.append(float(np.linalg.norm(diff, ord="fro") ** 2))
            fidelities.append(float(qt.metrics.fidelity(rho_out, rho_tgt)))
        else:
            errors.append(_ket_difference_error(output, target))
            fidelities.append(float(np.abs(target.overlap(output)) ** 2))

    error_arr = np.asarray(errors, dtype=float)
    fidelity_arr = np.asarray(fidelities, dtype=float)
    weighted_error = float(np.sum(weight_arr * error_arr))
    weighted_infidelity = float(np.sum(weight_arr * (1.0 - fidelity_arr)))
    return {
        "state_error_mean": float(np.mean(error_arr)),
        "state_error_max": float(np.max(error_arr)),
        "state_fidelity_mean": float(np.mean(fidelity_arr)),
        "state_fidelity_min": float(np.min(fidelity_arr)),
        "weighted_state_error": weighted_error,
        "weighted_state_infidelity": weighted_infidelity,
        "objective": weighted_error,
    }


def objective_breakdown(
    actual_subspace: np.ndarray,
    target_subspace: np.ndarray,
    full_operator: np.ndarray,
    subspace: Subspace,
    leakage_weight: float = 0.0,
    gauge: str = "global",
    block_slices: Iterable[slice | Sequence[int] | np.ndarray] | None = None,
    leakage_n_jobs: int = 1,
) -> dict[str, float]:
    fidelity = subspace_unitary_fidelity(
        actual_subspace,
        target_subspace,
        gauge=gauge,
        block_slices=block_slices if block_slices is not None else (subspace.per_fock_blocks() if gauge == "block" else None),
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
