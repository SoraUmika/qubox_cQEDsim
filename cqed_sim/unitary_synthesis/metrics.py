from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_index

from .subspace import Subspace


@dataclass(frozen=True)
class LeakageMetrics:
    average: float
    worst: float
    per_probe: tuple[float, ...]


@dataclass(frozen=True)
class LogicalBlockPhaseDiagnostics:
    block_phases_rad: tuple[float, ...]
    relative_block_phases_rad: tuple[float, ...]
    applied_correction_phases_rad: tuple[float, ...]
    best_fit_correction_phases_rad: tuple[float, ...]
    corrected_block_phases_rad: tuple[float, ...]
    corrected_relative_block_phases_rad: tuple[float, ...]
    residual_block_phases_rad: tuple[float, ...]
    rms_block_phase_error_rad: float
    block_gauge_fidelity: float
    corrected_block_gauge_fidelity: float
    best_fit_block_gauge_fidelity: float

    def as_dict(self) -> dict[str, float | list[float]]:
        return {
            "block_phases_rad": [float(value) for value in self.block_phases_rad],
            "relative_block_phases_rad": [float(value) for value in self.relative_block_phases_rad],
            "applied_correction_phases_rad": [float(value) for value in self.applied_correction_phases_rad],
            "best_fit_correction_phases_rad": [float(value) for value in self.best_fit_correction_phases_rad],
            "corrected_block_phases_rad": [float(value) for value in self.corrected_block_phases_rad],
            "corrected_relative_block_phases_rad": [float(value) for value in self.corrected_relative_block_phases_rad],
            "residual_block_phases_rad": [float(value) for value in self.residual_block_phases_rad],
            "rms_block_phase_error_rad": float(self.rms_block_phase_error_rad),
            "block_gauge_fidelity": float(self.block_gauge_fidelity),
            "corrected_block_gauge_fidelity": float(self.corrected_block_gauge_fidelity),
            "best_fit_block_gauge_fidelity": float(self.best_fit_block_gauge_fidelity),
        }


def _fro_norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x, ord="fro"))


def _coerce_projector_matrix(projector: Any, *, full_dim: int | None = None, atol: float = 1.0e-8) -> np.ndarray:
    if isinstance(projector, Subspace):
        arr = projector.projector()
    elif isinstance(projector, qt.Qobj):
        arr = np.asarray(projector.full(), dtype=np.complex128)
    elif isinstance(projector, np.ndarray):
        arr = np.asarray(projector, dtype=np.complex128)
    elif isinstance(projector, Sequence) and projector and all(isinstance(value, (int, np.integer)) for value in projector):
        if full_dim is None:
            raise ValueError("Projector indices require full_dim.")
        arr = np.zeros((int(full_dim), int(full_dim)), dtype=np.complex128)
        idx = [int(value) for value in projector]
        arr[np.ix_(idx, idx)] = np.eye(len(idx), dtype=np.complex128)
    else:
        raise TypeError(f"Could not coerce projector of type {type(projector).__name__}.")
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Projectors must be square matrices.")
    if full_dim is not None and int(arr.shape[0]) != int(full_dim):
        raise ValueError(f"Projector dimension {arr.shape[0]} does not match full_dim={int(full_dim)}.")
    hermitian = 0.5 * (arr + arr.conj().T)
    if _fro_norm(arr - hermitian) > atol:
        raise ValueError("Projector must be Hermitian within tolerance.")
    if _fro_norm(hermitian @ hermitian - hermitian) > max(atol, 1.0e-6):
        raise ValueError("Projector must be idempotent within tolerance.")
    return np.asarray(hermitian, dtype=np.complex128)


def _wrap_phase(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return (arr + np.pi) % (2.0 * np.pi) - np.pi


def projector_population_metrics(
    states: Sequence[qt.Qobj | np.ndarray],
    projector: Any,
    *,
    full_dim: int | None = None,
) -> LeakageMetrics:
    probe_list = [_as_qobj(state) for state in states]
    if not probe_list:
        return LeakageMetrics(average=0.0, worst=0.0, per_probe=())
    inferred_dim = int(probe_list[0].shape[0])
    proj = _coerce_projector_matrix(projector, full_dim=inferred_dim if full_dim is None else int(full_dim))
    values: list[float] = []
    for state in probe_list:
        if int(state.shape[0]) != int(proj.shape[0]):
            raise ValueError("State dimension does not match projector dimension.")
        if state.isoper:
            matrix = np.asarray(state.full(), dtype=np.complex128)
            value = float(np.real(np.trace(proj @ matrix)))
        else:
            vec = np.asarray(state.full(), dtype=np.complex128).reshape(-1)
            value = float(np.real(np.vdot(vec, proj @ vec)))
        values.append(float(np.clip(value, 0.0, 1.0)))
    avg = float(np.mean(values)) if values else 0.0
    worst = float(np.max(values)) if values else 0.0
    return LeakageMetrics(average=avg, worst=worst, per_probe=tuple(float(v) for v in values))


def projected_density_matrix(
    state: qt.Qobj | np.ndarray,
    projector: Any,
    *,
    full_dim: int | None = None,
    normalize: bool = False,
) -> tuple[np.ndarray, float]:
    obj = _as_qobj(state)
    inferred_dim = int(obj.shape[0])
    proj = _coerce_projector_matrix(projector, full_dim=inferred_dim if full_dim is None else int(full_dim))
    if obj.isoper:
        density = np.asarray(obj.full(), dtype=np.complex128)
    else:
        vec = np.asarray(obj.full(), dtype=np.complex128).reshape(-1)
        density = np.outer(vec, np.conjugate(vec))
    projected = np.asarray(proj @ density @ proj, dtype=np.complex128)
    trace_value = float(np.real_if_close(np.trace(projected)).real)
    if normalize and trace_value > 1.0e-18:
        projected = projected / trace_value
    return projected, trace_value


def _block_indices(block: slice | Sequence[int] | np.ndarray) -> np.ndarray:
    if isinstance(block, slice):
        start = 0 if block.start is None else int(block.start)
        stop = int(block.stop)
        step = 1 if block.step is None else int(block.step)
        return np.arange(start, stop, step, dtype=int)
    return np.asarray(block, dtype=int).reshape(-1)


def _canonical_block_phase(block: np.ndarray, *, atol: float = 1.0e-12) -> float:
    matrix = np.asarray(block, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Block phases require square block matrices.")
    if matrix.shape[0] == 0:
        return 0.0
    det = complex(np.linalg.det(matrix))
    if abs(det) > atol:
        return float(np.angle(det) / matrix.shape[0])
    trace = complex(np.trace(matrix))
    if abs(trace) > atol:
        return float(np.angle(trace))
    return 0.0


def _block_phase_layer(dim: int, blocks: Sequence[np.ndarray], phases: Sequence[float]) -> np.ndarray:
    layer = np.eye(int(dim), dtype=np.complex128)
    for idx, phase in zip(blocks, phases, strict=True):
        layer[np.ix_(idx, idx)] *= np.exp(1j * float(phase))
    return layer


def logical_block_phase_diagnostics(
    actual: np.ndarray,
    target: np.ndarray,
    *,
    block_slices: Iterable[slice | Sequence[int] | np.ndarray],
    applied_correction_phases: Sequence[float] | np.ndarray | None = None,
) -> LogicalBlockPhaseDiagnostics:
    u = np.asarray(actual, dtype=np.complex128)
    v = np.asarray(target, dtype=np.complex128)
    if u.shape != v.shape or u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("actual and target must be square and shape-matched.")

    blocks = tuple(_block_indices(block) for block in block_slices)
    overlap = v.conj().T @ u
    block_phases = np.asarray(
        [_canonical_block_phase(overlap[np.ix_(idx, idx)]) for idx in blocks],
        dtype=float,
    )
    if applied_correction_phases is None:
        applied = np.zeros(block_phases.size, dtype=float)
    else:
        applied = np.asarray(applied_correction_phases, dtype=float).reshape(-1)
        if applied.shape != block_phases.shape:
            raise ValueError(
                f"applied_correction_phases has shape {applied.shape}, expected {block_phases.shape}."
            )

    if block_phases.size == 0:
        zeros = tuple()
        fidelity = subspace_unitary_fidelity(u, v, gauge="block", block_slices=blocks)
        return LogicalBlockPhaseDiagnostics(
            block_phases_rad=zeros,
            relative_block_phases_rad=zeros,
            applied_correction_phases_rad=zeros,
            best_fit_correction_phases_rad=zeros,
            corrected_block_phases_rad=zeros,
            corrected_relative_block_phases_rad=zeros,
            residual_block_phases_rad=zeros,
            rms_block_phase_error_rad=0.0,
            block_gauge_fidelity=float(fidelity),
            corrected_block_gauge_fidelity=float(fidelity),
            best_fit_block_gauge_fidelity=float(fidelity),
        )

    relative = _wrap_phase(block_phases - float(block_phases[0]))
    best_fit = _wrap_phase(-block_phases)
    corrected_block_phases = _wrap_phase(block_phases + applied)
    corrected_relative = _wrap_phase(corrected_block_phases - float(corrected_block_phases[0]))
    residual = corrected_relative
    rms = float(np.sqrt(np.mean(np.square(residual))))

    block_fidelity = subspace_unitary_fidelity(u, v, gauge="block", block_slices=blocks)
    corrected_fidelity = subspace_unitary_fidelity(
        _block_phase_layer(u.shape[0], blocks, applied) @ u,
        v,
        gauge="block",
        block_slices=blocks,
    )
    best_fit_fidelity = subspace_unitary_fidelity(
        _block_phase_layer(u.shape[0], blocks, best_fit) @ u,
        v,
        gauge="block",
        block_slices=blocks,
    )
    return LogicalBlockPhaseDiagnostics(
        block_phases_rad=tuple(float(value) for value in block_phases),
        relative_block_phases_rad=tuple(float(value) for value in relative),
        applied_correction_phases_rad=tuple(float(value) for value in applied),
        best_fit_correction_phases_rad=tuple(float(value) for value in best_fit),
        corrected_block_phases_rad=tuple(float(value) for value in corrected_block_phases),
        corrected_relative_block_phases_rad=tuple(float(value) for value in corrected_relative),
        residual_block_phases_rad=tuple(float(value) for value in residual),
        rms_block_phase_error_rad=float(rms),
        block_gauge_fidelity=float(block_fidelity),
        corrected_block_gauge_fidelity=float(corrected_fidelity),
        best_fit_block_gauge_fidelity=float(best_fit_fidelity),
    )


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


def subspace_unitary_process_fidelity(
    actual: np.ndarray,
    target: np.ndarray,
    gauge: str = "global",
    block_slices: Iterable[slice | Sequence[int] | np.ndarray] | None = None,
) -> float:
    overlap = subspace_unitary_fidelity(actual, target, gauge=gauge, block_slices=block_slices)
    return float(np.clip(overlap * overlap, 0.0, 1.0))


def _matrix_sqrt_psd(matrix: np.ndarray, *, atol: float = 1.0e-12) -> np.ndarray:
    hermitian = 0.5 * (np.asarray(matrix, dtype=np.complex128) + np.asarray(matrix, dtype=np.complex128).conj().T)
    eigvals, eigvecs = np.linalg.eigh(hermitian)
    clipped = np.clip(np.real(eigvals), 0.0, None)
    clipped[clipped < atol] = 0.0
    return (eigvecs * np.sqrt(clipped)) @ eigvecs.conj().T


def density_matrix_fidelity(actual: np.ndarray, target: np.ndarray) -> float:
    rho = np.asarray(actual, dtype=np.complex128)
    sigma = np.asarray(target, dtype=np.complex128)
    if rho.shape != sigma.shape or rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("density_matrix_fidelity requires square, shape-matched density matrices.")
    sqrt_rho = _matrix_sqrt_psd(rho)
    middle = sqrt_rho @ sigma @ sqrt_rho
    sqrt_middle = _matrix_sqrt_psd(middle)
    value = float(np.real(np.trace(sqrt_middle)) ** 2)
    return float(np.clip(value, 0.0, 1.0))


def _normalized_choi_state(choi: np.ndarray, *, input_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(choi, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Choi matrices must be square.")
    trace_value = float(np.real_if_close(np.trace(arr)).real)
    if trace_value <= 0.0:
        if input_dim is None:
            raise ValueError("Could not normalize Choi matrix with non-positive trace without input_dim.")
        trace_value = float(input_dim)
    return np.asarray(arr / trace_value, dtype=np.complex128)


def channel_process_fidelity(
    actual_choi: np.ndarray,
    target_choi: np.ndarray,
    *,
    input_dim: int | None = None,
) -> float:
    rho_actual = _normalized_choi_state(actual_choi, input_dim=input_dim)
    rho_target = _normalized_choi_state(target_choi, input_dim=input_dim)
    return density_matrix_fidelity(rho_actual, rho_target)


def channel_entanglement_fidelity(
    actual_choi: np.ndarray,
    target_choi: np.ndarray,
    *,
    input_dim: int,
) -> float:
    rho_actual = _normalized_choi_state(actual_choi, input_dim=input_dim)
    rho_target = _normalized_choi_state(target_choi, input_dim=input_dim)
    target_purity = float(np.real(np.trace(rho_target @ rho_target)))
    if not np.isclose(target_purity, 1.0, atol=1.0e-8):
        return float("nan")
    overlap = float(np.real(np.trace(rho_target @ rho_actual)))
    return float(np.clip(overlap, 0.0, 1.0))


def channel_average_gate_fidelity(
    actual_choi: np.ndarray,
    target_choi: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
) -> float:
    if int(input_dim) != int(output_dim):
        return float("nan")
    entanglement = channel_entanglement_fidelity(actual_choi, target_choi, input_dim=input_dim)
    if not np.isfinite(entanglement):
        return float("nan")
    dim = float(input_dim)
    return float(np.clip((dim * entanglement + 1.0) / (dim + 1.0), 0.0, 1.0))


def _validate_basis_matrix(basis: np.ndarray, *, expected_rows: int | None = None, expected_cols: int | None = None) -> np.ndarray:
    arr = np.asarray(basis, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError("Basis matrices must be two-dimensional.")
    if expected_rows is not None and int(arr.shape[0]) != int(expected_rows):
        raise ValueError(f"Basis matrix row count {arr.shape[0]} does not match expected {int(expected_rows)}.")
    if expected_cols is not None and int(arr.shape[1]) != int(expected_cols):
        raise ValueError(f"Basis matrix column count {arr.shape[1]} does not match expected {int(expected_cols)}.")
    ident = np.eye(arr.shape[1], dtype=np.complex128)
    if np.linalg.norm(arr.conj().T @ arr - ident, ord="fro") > 1.0e-8:
        raise ValueError("Basis matrix columns must be orthonormal.")
    return arr


def _resolve_input_basis_matrix(
    *,
    actual_operator: np.ndarray,
    target_isometry: np.ndarray,
    input_basis: np.ndarray | None,
) -> np.ndarray:
    full_dim = int(np.asarray(actual_operator, dtype=np.complex128).shape[0])
    input_dim = int(np.asarray(target_isometry, dtype=np.complex128).shape[1])
    if input_basis is None:
        if input_dim > full_dim:
            raise ValueError("Target input dimension exceeds the available Hilbert-space dimension.")
        return _validate_basis_matrix(np.eye(full_dim, dtype=np.complex128)[:, :input_dim], expected_rows=full_dim, expected_cols=input_dim)
    return _validate_basis_matrix(np.asarray(input_basis, dtype=np.complex128), expected_rows=full_dim, expected_cols=input_dim)


def isometry_coherent_fidelity(
    actual_operator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_basis: np.ndarray | None = None,
) -> float:
    u = np.asarray(actual_operator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError("actual_operator must be a square matrix.")
    if v.ndim != 2 or v.shape[0] != u.shape[0]:
        raise ValueError("target_isometry must have shape (full_dim, input_dim).")
    basis = _resolve_input_basis_matrix(actual_operator=u, target_isometry=v, input_basis=input_basis)
    overlap = v.conj().T @ (u @ basis)
    input_dim = int(v.shape[1])
    return float(np.clip(abs(np.trace(overlap)) ** 2 / max(input_dim * input_dim, 1), 0.0, 1.0))


def isometry_basis_fidelity(
    actual_operator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_basis: np.ndarray | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    u = np.asarray(actual_operator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    basis = _resolve_input_basis_matrix(actual_operator=u, target_isometry=v, input_basis=input_basis)
    input_dim = int(v.shape[1])
    if weights is None:
        weight_arr = np.full(input_dim, 1.0 / max(input_dim, 1), dtype=float)
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != (input_dim,):
            raise ValueError("weights must match the isometry input dimension.")
        total = float(np.sum(weight_arr))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value.")
        weight_arr = weight_arr / total
    overlaps = []
    for column in range(input_dim):
        overlaps.append(float(abs(np.vdot(v[:, column], u @ basis[:, column])) ** 2))
    return float(np.clip(np.dot(weight_arr, overlaps), 0.0, 1.0))


def isometry_retention(
    actual_operator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_basis: np.ndarray | None = None,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    u = np.asarray(actual_operator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    basis = _resolve_input_basis_matrix(actual_operator=u, target_isometry=v, input_basis=input_basis)
    input_dim = int(v.shape[1])
    if weights is None:
        weight_arr = np.full(input_dim, 1.0 / max(input_dim, 1), dtype=float)
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != (input_dim,):
            raise ValueError("weights must match the isometry input dimension.")
        total = float(np.sum(weight_arr))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value.")
        weight_arr = weight_arr / total
    projector_out = np.asarray(v @ v.conj().T, dtype=np.complex128)
    retention = []
    for column in range(input_dim):
        out = u @ basis[:, column]
        retention.append(float(np.real(np.vdot(out, projector_out @ out))))
    return float(np.clip(np.dot(weight_arr, retention), 0.0, 1.0))


def isometry_random_superposition_metrics(
    actual_operator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_basis: np.ndarray | None = None,
    sample_count: int = 16,
    seed: int = 0,
) -> dict[str, float]:
    u = np.asarray(actual_operator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    basis = _resolve_input_basis_matrix(actual_operator=u, target_isometry=v, input_basis=input_basis)
    input_dim = int(v.shape[1])
    rng = np.random.default_rng(int(seed))
    fidelities: list[float] = []
    for _ in range(int(sample_count)):
        coeff = rng.standard_normal(input_dim) + 1j * rng.standard_normal(input_dim)
        coeff = coeff / np.linalg.norm(coeff)
        actual = u @ (basis @ coeff)
        target = v @ coeff
        fidelities.append(float(abs(np.vdot(target, actual)) ** 2))
    return {
        "isometry_random_state_fidelity_mean": float(np.mean(fidelities)) if fidelities else float("nan"),
        "isometry_random_state_fidelity_min": float(np.min(fidelities)) if fidelities else float("nan"),
    }


def channel_isometry_basis_fidelity(
    actual_superoperator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    superop = np.asarray(actual_superoperator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    if v.shape != (int(output_dim), int(input_dim)):
        raise ValueError("target_isometry shape does not match the requested channel dimensions.")
    if weights is None:
        weight_arr = np.full(int(input_dim), 1.0 / max(int(input_dim), 1), dtype=float)
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != (int(input_dim),):
            raise ValueError("weights must match input_dim.")
        total = float(np.sum(weight_arr))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value.")
        weight_arr = weight_arr / total
    fidelities: list[float] = []
    for index in range(int(input_dim)):
        rho_out = unvectorize_operator(superop[:, index + int(input_dim) * index], output_dim)
        target_density = np.outer(v[:, index], np.conjugate(v[:, index]))
        fidelities.append(density_matrix_fidelity(rho_out, target_density))
    return float(np.clip(np.dot(weight_arr, fidelities), 0.0, 1.0))


def channel_isometry_retention(
    actual_superoperator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
    weights: Sequence[float] | np.ndarray | None = None,
) -> float:
    superop = np.asarray(actual_superoperator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    if weights is None:
        weight_arr = np.full(int(input_dim), 1.0 / max(int(input_dim), 1), dtype=float)
    else:
        weight_arr = np.asarray(weights, dtype=float)
        if weight_arr.shape != (int(input_dim),):
            raise ValueError("weights must match input_dim.")
        total = float(np.sum(weight_arr))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value.")
        weight_arr = weight_arr / total
    projector_out = np.asarray(v @ v.conj().T, dtype=np.complex128)
    retention: list[float] = []
    for index in range(int(input_dim)):
        rho_out = unvectorize_operator(superop[:, index + int(input_dim) * index], output_dim)
        retention.append(float(np.real(np.trace(projector_out @ rho_out))))
    return float(np.clip(np.dot(weight_arr, retention), 0.0, 1.0))


def channel_isometry_random_superposition_metrics(
    actual_superoperator: np.ndarray,
    target_isometry: np.ndarray,
    *,
    input_dim: int,
    output_dim: int,
    sample_count: int = 16,
    seed: int = 0,
) -> dict[str, float]:
    superop = np.asarray(actual_superoperator, dtype=np.complex128)
    v = np.asarray(target_isometry, dtype=np.complex128)
    rng = np.random.default_rng(int(seed))
    fidelities: list[float] = []
    for _ in range(int(sample_count)):
        coeff = rng.standard_normal(int(input_dim)) + 1j * rng.standard_normal(int(input_dim))
        coeff = coeff / np.linalg.norm(coeff)
        rho_in = np.outer(coeff, np.conjugate(coeff))
        rho_out = unvectorize_operator(superop @ vectorize_operator(rho_in), output_dim)
        target_density = np.outer(v @ coeff, np.conjugate(v @ coeff))
        fidelities.append(density_matrix_fidelity(rho_out, target_density))
    return {
        "isometry_random_state_fidelity_mean": float(np.mean(fidelities)) if fidelities else float("nan"),
        "isometry_random_state_fidelity_min": float(np.min(fidelities)) if fidelities else float("nan"),
    }


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


def observable_expectation_metrics(
    outputs: Sequence[qt.Qobj | np.ndarray],
    observables: Sequence[qt.Qobj | np.ndarray],
    target_expectations: Sequence[Sequence[complex | float]] | np.ndarray,
    *,
    state_weights: Sequence[float] | np.ndarray | None = None,
    observable_weights: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float | np.ndarray]:
    out_states = [_as_qobj(state) for state in outputs]
    obs_ops = [_as_qobj(op) for op in observables]
    if not out_states:
        raise ValueError("observable_expectation_metrics requires at least one propagated state.")
    if not obs_ops:
        raise ValueError("observable_expectation_metrics requires at least one observable.")

    expectations = np.asarray(target_expectations, dtype=np.complex128)
    if expectations.shape != (len(out_states), len(obs_ops)):
        raise ValueError(
            f"target_expectations has shape {expectations.shape}, expected {(len(out_states), len(obs_ops))}."
        )

    if state_weights is None:
        state_weight_arr = np.full(len(out_states), 1.0 / len(out_states), dtype=float)
    else:
        state_weight_arr = np.asarray(state_weights, dtype=float)
        if state_weight_arr.shape != (len(out_states),):
            raise ValueError("state_weights must match the number of output states.")
        total = float(np.sum(state_weight_arr))
        if total <= 0.0:
            raise ValueError("state_weights must sum to a positive value.")
        state_weight_arr = state_weight_arr / total

    if observable_weights is None:
        observable_weight_arr = np.full(len(obs_ops), 1.0 / len(obs_ops), dtype=float)
    else:
        observable_weight_arr = np.asarray(observable_weights, dtype=float)
        if observable_weight_arr.shape != (len(obs_ops),):
            raise ValueError("observable_weights must match the number of observables.")
        total = float(np.sum(observable_weight_arr))
        if total <= 0.0:
            raise ValueError("observable_weights must sum to a positive value.")
        observable_weight_arr = observable_weight_arr / total

    actual = np.zeros((len(out_states), len(obs_ops)), dtype=np.complex128)
    for row, state in enumerate(out_states):
        for col, observable in enumerate(obs_ops):
            if state.isoper:
                actual[row, col] = complex((observable * state).tr())
            else:
                actual[row, col] = complex(qt.expect(observable, state))

    errors = np.abs(actual - expectations) ** 2
    weights = np.outer(state_weight_arr, observable_weight_arr)
    weighted_error = float(np.sum(weights * errors))
    return {
        "observable_error_mean": float(np.mean(errors)),
        "observable_error_max": float(np.max(errors)),
        "weighted_observable_error": weighted_error,
        "objective": weighted_error,
        "actual_expectations": actual,
    }


def vectorize_operator(operator: qt.Qobj | np.ndarray) -> np.ndarray:
    obj = _as_qobj(operator)
    matrix = np.asarray(obj.full(), dtype=np.complex128)
    if matrix.ndim != 2:
        raise ValueError("Operators must be matrices.")
    return matrix.reshape(-1, order="F")


def unvectorize_operator(vector: np.ndarray, dim: int, n_cols: int | None = None) -> np.ndarray:
    arr = np.asarray(vector, dtype=np.complex128).reshape(-1)
    rows = int(dim)
    cols = int(dim if n_cols is None else n_cols)
    if arr.size != rows * cols:
        raise ValueError(f"Vectorized operator has length {arr.size}, expected {rows * cols}.")
    return arr.reshape((rows, cols), order="F")


def matrix_unit(dim: int, row: int, col: int) -> np.ndarray:
    out = np.zeros((int(dim), int(dim)), dtype=np.complex128)
    out[int(row), int(col)] = 1.0
    return out


def superoperator_from_kraus(kraus_ops: Sequence[qt.Qobj | np.ndarray]) -> np.ndarray:
    if not kraus_ops:
        raise ValueError("superoperator_from_kraus requires at least one Kraus operator.")
    matrices = [np.asarray(_as_qobj(op).full(), dtype=np.complex128) for op in kraus_ops]
    out_dim = int(matrices[0].shape[0])
    in_dim = int(matrices[0].shape[1])
    for matrix in matrices:
        if matrix.shape != (out_dim, in_dim):
            raise ValueError(f"All Kraus operators must have shape {(out_dim, in_dim)}, received {matrix.shape}.")
    columns: list[np.ndarray] = []
    for col in range(in_dim):
        for row in range(in_dim):
            basis = matrix_unit(in_dim, row, col)
            evolved = np.zeros((out_dim, out_dim), dtype=np.complex128)
            for kraus in matrices:
                evolved = evolved + kraus @ basis @ kraus.conj().T
            columns.append(vectorize_operator(evolved))
    return np.column_stack(columns)


def superoperator_from_unitary(unitary: qt.Qobj | np.ndarray) -> np.ndarray:
    return superoperator_from_kraus([unitary])


def choi_from_superoperator(
    superoperator: np.ndarray,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> np.ndarray:
    sup = np.asarray(superoperator, dtype=np.complex128)
    if sup.ndim != 2:
        raise ValueError("Superoperators must be matrices.")
    if input_dim is None or output_dim is None:
        if sup.shape[0] != sup.shape[1]:
            raise ValueError("Rectangular superoperators require explicit input_dim and output_dim.")
        dim_float = np.sqrt(float(sup.shape[0]))
        dim = int(round(dim_float))
        if dim * dim != sup.shape[0]:
            raise ValueError("Superoperator dimension must be a perfect square.")
        input_dim = dim
        output_dim = dim
    if int(output_dim) * int(output_dim) != sup.shape[0]:
        raise ValueError("Superoperator row dimension does not match output_dim^2.")
    if int(input_dim) * int(input_dim) != sup.shape[1]:
        raise ValueError("Superoperator column dimension does not match input_dim^2.")
    tensor = np.zeros((int(output_dim), int(input_dim), int(output_dim), int(input_dim)), dtype=np.complex128)
    for col in range(int(input_dim)):
        for row in range(int(input_dim)):
            tensor[:, row, :, col] = unvectorize_operator(sup[:, row + int(input_dim) * col], int(output_dim))
    return tensor.reshape(int(output_dim) * int(input_dim), int(output_dim) * int(input_dim))


def superoperator_from_choi(
    choi: np.ndarray,
    *,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> np.ndarray:
    arr = np.asarray(choi, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Choi matrices must be square.")
    if input_dim is None or output_dim is None:
        dim_float = np.sqrt(float(arr.shape[0]))
        dim = int(round(dim_float))
        if dim * dim != arr.shape[0]:
            raise ValueError("Rectangular Choi matrices require explicit input_dim and output_dim.")
        input_dim = dim
        output_dim = dim
    if int(input_dim) * int(output_dim) != arr.shape[0]:
        raise ValueError("Choi dimension does not match input_dim * output_dim.")
    tensor = arr.reshape(int(output_dim), int(input_dim), int(output_dim), int(input_dim))
    superoperator = np.zeros((int(output_dim) * int(output_dim), int(input_dim) * int(input_dim)), dtype=np.complex128)
    for col in range(int(input_dim)):
        for row in range(int(input_dim)):
            superoperator[:, row + int(input_dim) * col] = vectorize_operator(tensor[:, row, :, col])
    return superoperator


def channel_representation_from_outputs(
    outputs: Sequence[qt.Qobj | np.ndarray],
    *,
    input_dim: int,
    output_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_dim = int(input_dim)
    out_dim = int(in_dim if output_dim is None else output_dim)
    expected = in_dim * in_dim
    if len(outputs) != expected:
        raise ValueError(f"Channel reconstruction requires {expected} outputs, received {len(outputs)}.")
    columns: list[np.ndarray] = []
    traces: list[complex] = []
    for output in outputs:
        matrix = np.asarray(_as_qobj(output).full(), dtype=np.complex128)
        if matrix.shape != (out_dim, out_dim):
            raise ValueError(f"Channel output operator has shape {matrix.shape}, expected {(out_dim, out_dim)}.")
        columns.append(vectorize_operator(matrix))
        traces.append(complex(np.trace(matrix)))
    superoperator = np.column_stack(columns)
    return (
        superoperator,
        choi_from_superoperator(superoperator, input_dim=in_dim, output_dim=out_dim),
        np.asarray(traces, dtype=np.complex128),
    )


def channel_action_metrics(
    actual_superoperator: np.ndarray,
    actual_choi: np.ndarray,
    *,
    target_choi: np.ndarray,
    target_superoperator: np.ndarray | None = None,
    trace_values: np.ndarray | None = None,
    input_dim: int | None = None,
    output_dim: int | None = None,
) -> dict[str, float]:
    target_choi_arr = np.asarray(target_choi, dtype=np.complex128)
    actual_choi_arr = np.asarray(actual_choi, dtype=np.complex128)
    if target_choi_arr.shape != actual_choi_arr.shape:
        raise ValueError("actual_choi and target_choi must have the same shape.")
    if input_dim is None or output_dim is None:
        dim_float = np.sqrt(float(actual_choi_arr.shape[0]))
        dim = int(round(dim_float))
        if dim * dim != actual_choi_arr.shape[0]:
            raise ValueError("Rectangular channel Choi comparisons require explicit input_dim and output_dim.")
        input_dim = dim
        output_dim = dim

    choi_norm = max(_fro_norm(target_choi_arr), 1.0e-18)
    choi_error = _fro_norm(actual_choi_arr - target_choi_arr) ** 2 / (choi_norm**2)
    actual_norm = max(_fro_norm(actual_choi_arr), 1.0e-18)
    choi_overlap = abs(np.trace(target_choi_arr.conj().T @ actual_choi_arr)) / (choi_norm * actual_norm)
    process_fid = channel_process_fidelity(actual_choi_arr, target_choi_arr, input_dim=int(input_dim))
    entanglement_fid = channel_entanglement_fidelity(actual_choi_arr, target_choi_arr, input_dim=int(input_dim))
    average_gate_fid = channel_average_gate_fidelity(
        actual_choi_arr,
        target_choi_arr,
        input_dim=int(input_dim),
        output_dim=int(output_dim),
    )

    if target_superoperator is not None:
        target_super = np.asarray(target_superoperator, dtype=np.complex128)
        actual_super = np.asarray(actual_superoperator, dtype=np.complex128)
        if target_super.shape != actual_super.shape:
            raise ValueError("actual_superoperator and target_superoperator must have the same shape.")
        target_super_norm = max(_fro_norm(target_super), 1.0e-18)
        super_error = _fro_norm(actual_super - target_super) ** 2 / (target_super_norm**2)
    else:
        super_error = float("nan")

    hermitian_choi = 0.5 * (actual_choi_arr + actual_choi_arr.conj().T)
    choi_hermiticity_error = _fro_norm(actual_choi_arr - actual_choi_arr.conj().T)
    choi_min_eig = float(np.min(np.linalg.eigvalsh(hermitian_choi)).real)
    complete_positivity_violation = max(0.0, -choi_min_eig)

    trace_preservation_error_mean = 0.0
    trace_preservation_error_max = 0.0
    trace_loss_mean = 0.0
    trace_loss_worst = 0.0
    if trace_values is not None:
        expected = np.zeros(int(input_dim) * int(input_dim), dtype=np.complex128)
        trace_losses: list[float] = []
        diagonal_indices: list[int] = []
        for col in range(int(input_dim)):
            for row in range(int(input_dim)):
                index = row + int(input_dim) * col
                if row == col:
                    expected[index] = 1.0
                    diagonal_indices.append(index)
        trace_delta = np.asarray(trace_values, dtype=np.complex128).reshape(-1) - expected
        trace_preservation_error_mean = float(np.mean(np.abs(trace_delta)))
        trace_preservation_error_max = float(np.max(np.abs(trace_delta)))
        for index in diagonal_indices:
            trace_losses.append(max(0.0, 1.0 - float(np.real(trace_values[index]))))
        if trace_losses:
            trace_loss_mean = float(np.mean(trace_losses))
            trace_loss_worst = float(np.max(trace_losses))

    objective = float(choi_error)
    return {
        "channel_error_mean": float(choi_error),
        "channel_error_max": float(choi_error),
        "channel_overlap": float(np.clip(choi_overlap, 0.0, 1.0)),
        "channel_process_fidelity": float(np.clip(process_fid, 0.0, 1.0)),
        "channel_entanglement_fidelity": float(np.clip(entanglement_fid, 0.0, 1.0)) if np.isfinite(entanglement_fid) else float("nan"),
        "channel_average_gate_fidelity": float(np.clip(average_gate_fid, 0.0, 1.0)) if np.isfinite(average_gate_fid) else float("nan"),
        "channel_choi_error": float(choi_error),
        "channel_superoperator_error": float(super_error),
        "trace_preservation_error_mean": float(trace_preservation_error_mean),
        "trace_preservation_error_max": float(trace_preservation_error_max),
        "trace_loss_mean": float(trace_loss_mean),
        "trace_loss_worst": float(trace_loss_worst),
        "choi_hermiticity_error": float(choi_hermiticity_error),
        "choi_min_eig": float(choi_min_eig),
        "complete_positivity_violation": float(complete_positivity_violation),
        "objective": objective,
    }


def state_population_distribution(state: qt.Qobj | np.ndarray) -> np.ndarray:
    obj = _as_qobj(state)
    if obj.isoper:
        matrix = np.asarray(obj.full(), dtype=np.complex128)
        return np.clip(np.real(np.diag(matrix)), 0.0, None)
    vec = np.asarray(obj.full(), dtype=np.complex128).reshape(-1)
    return np.abs(vec) ** 2


def _truncation_groups(subspace: Subspace) -> tuple[list[int], list[tuple[str, tuple[int, ...]]]]:
    edge_indices = [int(subspace.indices[-1])]
    outside_groups: list[tuple[str, tuple[int, ...]]] = []
    metadata = {} if subspace.metadata is None else dict(subspace.metadata)
    if subspace.kind == "qubit_cavity_block" and "n_match" in metadata and "n_cav" in metadata:
        n_match = int(metadata["n_match"])
        n_cav = int(metadata["n_cav"])
        edge_indices = [qubit_cavity_index(n_cav, q, n_match) for q in range(2)]
        for level in range(n_match + 1, n_cav):
            outside_groups.append(
                (
                    f"n={level}",
                    tuple(qubit_cavity_index(n_cav, q, level) for q in range(2)),
                )
            )
        return edge_indices, outside_groups
    if subspace.kind == "cavity_only" and "n_match" in metadata and "n_cav" in metadata:
        n_match = int(metadata["n_match"])
        n_cav = int(metadata["n_cav"])
        qubit = 0 if str(metadata.get("qubit", "g")) == "g" else 1
        edge_indices = [qubit_cavity_index(n_cav, qubit, n_match)]
        for level in range(n_match + 1, n_cav):
            outside_groups.append((f"n={level}", (qubit_cavity_index(n_cav, qubit, level),)))
        return edge_indices, outside_groups
    outside = [index for index in range(subspace.full_dim) if index not in set(subspace.indices)]
    outside_groups = [(f"index={int(index)}", (int(index),)) for index in outside]
    return edge_indices, outside_groups


def truncation_sanity_metrics(states: Sequence[qt.Qobj | np.ndarray], subspace: Subspace) -> dict[str, float | list[dict[str, float | str]]]:
    probe_list = [_as_qobj(state) for state in states]
    if not probe_list:
        return {
            "retained_edge_population_average": 0.0,
            "retained_edge_population_worst": 0.0,
            "outside_tail_population_average": 0.0,
            "outside_tail_population_worst": 0.0,
            "outside_population_profile": [],
        }

    edge_indices, outside_groups = _truncation_groups(subspace)
    edge_values: list[float] = []
    outside_values: list[float] = []
    outside_group_values: dict[str, list[float]] = {label: [] for label, _ in outside_groups}
    for state in probe_list:
        population = state_population_distribution(state)
        if population.size != int(subspace.full_dim):
            raise ValueError(
                f"Truncation sanity metrics expected population length {int(subspace.full_dim)}, received {population.size}."
            )
        edge_values.append(float(np.sum(population[np.asarray(edge_indices, dtype=int)])))
        outside_total = 0.0
        for label, indices in outside_groups:
            value = float(np.sum(population[np.asarray(indices, dtype=int)]))
            outside_group_values[label].append(value)
            outside_total += value
        outside_values.append(float(outside_total))

    profile = [
        {
            "label": label,
            "average": float(np.mean(values)) if values else 0.0,
            "worst": float(np.max(values)) if values else 0.0,
        }
        for label, values in outside_group_values.items()
    ]
    return {
        "retained_edge_population_average": float(np.mean(edge_values)),
        "retained_edge_population_worst": float(np.max(edge_values)),
        "outside_tail_population_average": float(np.mean(outside_values)),
        "outside_tail_population_worst": float(np.max(outside_values)),
        "outside_population_profile": profile,
    }


def operator_truncation_sanity_metrics(full_operator: np.ndarray, subspace: Subspace) -> dict[str, float | list[dict[str, float | str]]]:
    operator = np.asarray(full_operator, dtype=np.complex128)
    if operator.shape != (subspace.full_dim, subspace.full_dim):
        raise ValueError("full_operator shape mismatch with subspace.full_dim.")
    probe_states: list[np.ndarray] = []
    eye = np.eye(subspace.dim, dtype=np.complex128)
    for column in range(subspace.dim):
        psi_full = subspace.embed(eye[:, column])
        probe_states.append(operator @ psi_full)
    return truncation_sanity_metrics(probe_states, subspace)


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
