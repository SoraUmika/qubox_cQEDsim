from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any, Mapping, Sequence

import numpy as np
import qutip as qt
from scipy.linalg import expm

from cqed_sim.core.conventions import qubit_cavity_block_indices

from .config import ExecutionOptions, LeakagePenalty
from .metrics import observable_expectation_metrics, state_leakage_metrics, state_mapping_metrics, subspace_unitary_fidelity
from .sequence import (
    BlueSidebandExchange,
    CavityBlockPhase,
    ConditionalDisplacement,
    Displacement,
    FreeEvolveCondPhase,
    GateSequence,
    JaynesCummingsExchange,
    PrimitiveGate,
    QubitRotation,
    SNAP,
    SQR,
    drift_phase_table,
)
from .subspace import Subspace


def _has_open_system(settings: Mapping[str, Any]) -> bool:
    return settings.get("c_ops") is not None or settings.get("noise") is not None


def _state_to_array(state: qt.Qobj | np.ndarray) -> tuple[np.ndarray, bool, Any]:
    if isinstance(state, qt.Qobj):
        arr = np.asarray(state.full(), dtype=np.complex128)
        return (arr.reshape(-1) if not state.isoper else arr, bool(state.isoper), state.dims)
    arr = np.asarray(state, dtype=np.complex128)
    if arr.ndim == 1:
        return arr.reshape(-1), False, None
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr.reshape(-1), False, None
    if arr.ndim == 2:
        return arr, True, None
    raise ValueError("States must be vectors or density matrices.")


def _array_to_state(array: np.ndarray, *, is_operator: bool, dims: Any, full_dim: int) -> qt.Qobj:
    arr = np.asarray(array, dtype=np.complex128)
    if is_operator:
        return qt.Qobj(arr.reshape(full_dim, full_dim), dims=dims if dims is not None else [[full_dim], [full_dim]])
    return qt.Qobj(arr.reshape(-1), dims=dims if dims is not None else [[full_dim], [1]])


def _rotation_xy_matrix(theta: float, phi: float) -> np.ndarray:
    c = float(np.cos(theta / 2.0))
    s = float(np.sin(theta / 2.0))
    phase = np.exp(-1j * float(phi))
    return np.asarray(
        [
            [c, -1j * s * phase],
            [-1j * s * np.conjugate(phase), c],
        ],
        dtype=np.complex128,
    )


def _block_diag_rotations(theta: Sequence[float], phi: Sequence[float]) -> np.ndarray:
    n_cav = len(theta)
    out = np.zeros((2 * n_cav, 2 * n_cav), dtype=np.complex128)
    for level, (theta_level, phi_level) in enumerate(zip(theta, phi)):
        idx = qubit_cavity_block_indices(n_cav, level)
        out[np.ix_(idx, idx)] = _rotation_xy_matrix(float(theta_level), float(phi_level))
    return out


def _diag_cavity_block_phase(n_cav: int, levels: Sequence[int], phases: Sequence[float]) -> np.ndarray:
    diag = np.ones(2 * int(n_cav), dtype=np.complex128)
    phase_array = np.asarray(phases, dtype=float)
    for level, phase in zip(levels, phase_array, strict=True):
        idx = qubit_cavity_block_indices(int(n_cav), int(level))
        diag[idx[0]] = np.exp(1j * float(phase))
        diag[idx[1]] = np.exp(1j * float(phase))
    return np.diag(diag)


def _displacement_matrix(n_cav: int, alpha: complex) -> np.ndarray:
    a = np.zeros((n_cav, n_cav), dtype=np.complex128)
    for level in range(1, n_cav):
        a[level - 1, level] = np.sqrt(level)
    adag = a.conj().T
    displacement = expm(alpha * adag - np.conjugate(alpha) * a)
    return np.kron(np.eye(2, dtype=np.complex128), displacement)


def _drift_phase_matrix(n_cav: int, duration: float, gate: FreeEvolveCondPhase | SQR) -> np.ndarray:
    table = drift_phase_table(n_cav=n_cav, duration=duration, model=gate.drift_model)
    diag = np.zeros(2 * n_cav, dtype=np.complex128)
    for level in range(n_cav):
        idx = qubit_cavity_block_indices(n_cav, level)
        diag[idx[0]] = np.exp(-1j * table.phase_g[level])
        diag[idx[1]] = np.exp(-1j * table.phase_e[level])
    return np.diag(diag)


@dataclass(frozen=True)
class FastPathSelection:
    requested_engine: str
    selected_engine: str
    device: str
    fast_path_used: bool
    reason: str

    def to_record(self) -> dict[str, Any]:
        return {
            "requested_engine": str(self.requested_engine),
            "selected_engine": str(self.selected_engine),
            "device": str(self.device),
            "fast_path_used": bool(self.fast_path_used),
            "reason": str(self.reason),
        }


@dataclass
class FastEvaluationResult:
    metrics: dict[str, float]
    full_operator: np.ndarray | None = None
    subspace_operator: np.ndarray | None = None
    state_outputs: list[qt.Qobj] | None = None
    selection: FastPathSelection | None = None


class FastObjectiveEvaluator:
    def __init__(
        self,
        sequence_template: GateSequence,
        payload: Mapping[str, Any],
        execution: ExecutionOptions,
    ) -> None:
        self.sequence_template = sequence_template
        self.payload = payload
        self.execution = execution
        self.full_dim = int(payload["subspace"].full_dim)
        self.n_cav = int(sequence_template.resolve_n_cav(system=payload.get("system"))) if any(
            not isinstance(gate, PrimitiveGate) for gate in sequence_template.gates
        ) else -1
        self.leakage_subspace: Subspace | None = payload.get("leakage_subspace")
        self.selection = self._resolve_selection()
        self._jax = None
        self._jnp = None
        if self.selection.selected_engine == "jax":
            import jax  # type: ignore
            import jax.numpy as jnp  # type: ignore

            self._jax = jax
            self._jnp = jnp

    def _resolve_selection(self) -> FastPathSelection:
        requested = str(self.execution.engine)
        if not bool(self.execution.use_fast_path) or requested == "legacy":
            return FastPathSelection(requested, "legacy", "cpu", False, "Fast path disabled by execution settings.")
        if str(self.payload.get("backend")) != "ideal":
            return FastPathSelection(requested, "legacy", "cpu", False, "Fast path currently supports backend='ideal' only.")
        if _has_open_system(self.payload.get("backend_settings", {})):
            return FastPathSelection(requested, "legacy", "cpu", False, "Fast path currently supports closed-system objectives only.")
        supported_targets = {"unitary", "state_mapping", "isometry", "observable", "trajectory"}
        target_type = str(self.payload.get("target_type", "unitary"))
        if target_type not in supported_targets:
            return FastPathSelection(
                requested,
                "legacy",
                "cpu",
                False,
                f"Fast path does not yet support target_type='{target_type}'.",
            )
        for gate in self.sequence_template.gates:
            if isinstance(gate, PrimitiveGate) and gate.mode == "waveform":
                return FastPathSelection(requested, "legacy", "cpu", False, "Waveform primitives currently use the legacy simulation path.")
        if requested in {"auto", "jax"}:
            if find_spec("jax") is None:
                if requested == "jax":
                    if self.execution.fallback_engine == "numpy":
                        return FastPathSelection(requested, "numpy", "cpu", True, "JAX was requested but is not installed; using NumPy fast path.")
                    return FastPathSelection(requested, "legacy", "cpu", False, "JAX was requested but is not installed.")
                return FastPathSelection(requested, "numpy", "cpu", True, "Using NumPy fast path; optional JAX dependency is not installed.")
            device = str(self.execution.device)
            if device == "gpu":
                import jax  # type: ignore

                gpu_devices = [row for row in jax.devices() if row.platform == "gpu"]
                if not gpu_devices:
                    if self.execution.fallback_engine == "numpy":
                        return FastPathSelection(requested, "numpy", "cpu", True, "GPU was requested but no JAX GPU device is available; using NumPy fast path.")
                    return FastPathSelection(requested, "legacy", "cpu", False, "GPU was requested but no JAX GPU device is available.")
                return FastPathSelection(requested, "jax", "gpu", True, "Using JAX fast path on GPU.")
            return FastPathSelection(requested, "jax", "cpu", True, "Using JAX fast path.")
        return FastPathSelection(requested, "numpy", "cpu", True, "Using NumPy fast path.")

    def _gate_matrix(self, gate: Any) -> np.ndarray:
        if isinstance(gate, QubitRotation):
            return np.kron(_rotation_xy_matrix(gate.theta, gate.phi), np.eye(self.n_cav, dtype=np.complex128))
        if isinstance(gate, SQR):
            theta, phi = gate._padded(self.n_cav)
            drive = _block_diag_rotations(theta, phi)
            if not gate.include_conditional_phase:
                return drive
            return _drift_phase_matrix(self.n_cav, gate.duration, gate) @ drive
        if isinstance(gate, CavityBlockPhase):
            return _diag_cavity_block_phase(self.n_cav, gate._resolved_levels(self.n_cav), gate.get_parameters(self.n_cav))
        if isinstance(gate, Displacement):
            return _displacement_matrix(self.n_cav, gate.alpha)
        if isinstance(gate, ConditionalDisplacement):
            return np.asarray(gate.ideal_unitary(self.n_cav).full(), dtype=np.complex128)
        if isinstance(gate, JaynesCummingsExchange):
            return np.asarray(gate.ideal_unitary(self.n_cav).full(), dtype=np.complex128)
        if isinstance(gate, BlueSidebandExchange):
            return np.asarray(gate.ideal_unitary(self.n_cav).full(), dtype=np.complex128)
        if isinstance(gate, FreeEvolveCondPhase):
            return _drift_phase_matrix(self.n_cav, gate.wait_time, gate)
        if isinstance(gate, PrimitiveGate):
            return np.asarray(
                gate._matrix_operator(
                    model=self.payload.get("system").runtime_model() if self.payload.get("system") is not None else None,
                    system=self.payload.get("system"),
                ),
                dtype=np.complex128,
            )
        raise TypeError(f"Unsupported gate type for fast evaluation: {gate.__class__.__name__}")

    def _compose_operator(self, gate_matrices: Sequence[np.ndarray]) -> np.ndarray:
        if self.selection.selected_engine == "jax" and self._jnp is not None:
            matrices = [self._jnp.asarray(matrix) for matrix in gate_matrices]
            acc = self._jnp.eye(self.full_dim, dtype=self._jnp.complex64)
            for matrix in matrices:
                acc = matrix @ acc
            return np.asarray(acc, dtype=np.complex128)
        acc = np.eye(self.full_dim, dtype=np.complex128)
        for matrix in gate_matrices:
            acc = matrix @ acc
        return acc

    def _propagate_states(
        self,
        gate_matrices: Sequence[np.ndarray],
        initial_states: Sequence[qt.Qobj | np.ndarray],
        *,
        checkpoints: Sequence[int] | None = None,
    ) -> tuple[list[qt.Qobj], dict[int, list[qt.Qobj]]]:
        raw_states = [_state_to_array(state) for state in initial_states]
        vectors = [(index, array, dims) for index, (array, is_oper, dims) in enumerate(raw_states) if not is_oper]
        densities = [(index, array, dims) for index, (array, is_oper, dims) in enumerate(raw_states) if is_oper]
        selected = {int(step) for step in (checkpoints or ())}
        history_arrays: dict[int, list[np.ndarray]] = {}

        vector_batch = None
        vector_meta: list[tuple[int, Any]] = []
        vector_index_map: dict[int, int] = {}    # original_index -> batch_index
        if vectors:
            vector_meta = [(index, dims) for index, _, dims in vectors]
            vector_index_map = {index: batch_idx for batch_idx, (index, _) in enumerate(vector_meta)}
            vector_batch = np.stack([array.reshape(-1) for _, array, _ in vectors], axis=1)

        density_batch = None
        density_meta: list[tuple[int, Any]] = []
        density_index_map: dict[int, int] = {}   # original_index -> batch_index
        if densities:
            density_meta = [(index, dims) for index, _, dims in densities]
            density_index_map = {index: batch_idx for batch_idx, (index, _) in enumerate(density_meta)}
            density_batch = np.stack([array.reshape(self.full_dim, self.full_dim) for _, array, _ in densities], axis=0)

        def _snapshot(step: int) -> None:
            rows: list[np.ndarray] = [None] * len(raw_states)  # type: ignore[list-item]
            if vector_batch is not None:
                for batch_index, (original_index, _) in enumerate(vector_meta):
                    rows[original_index] = np.asarray(vector_batch[:, batch_index], dtype=np.complex128)
            if density_batch is not None:
                for batch_index, (original_index, _) in enumerate(density_meta):
                    rows[original_index] = np.asarray(density_batch[batch_index], dtype=np.complex128)
            history_arrays[int(step)] = rows

        if 0 in selected:
            _snapshot(0)

        for step, matrix in enumerate(gate_matrices, start=1):
            if vector_batch is not None:
                vector_batch = matrix @ vector_batch
            if density_batch is not None:
                density_batch = np.einsum("ab,nbd,cd->nac", matrix, density_batch, np.conjugate(matrix), optimize=True)
            if step in selected:
                _snapshot(step)

        final_states: list[qt.Qobj] = []
        for index, (_, is_operator, dims) in enumerate(raw_states):
            if not is_operator:
                assert vector_batch is not None
                batch_index = vector_index_map[index]
                final_states.append(_array_to_state(vector_batch[:, batch_index], is_operator=False, dims=dims, full_dim=self.full_dim))
            else:
                assert density_batch is not None
                batch_index = density_index_map[index]
                final_states.append(_array_to_state(density_batch[batch_index], is_operator=True, dims=dims, full_dim=self.full_dim))

        history: dict[int, list[qt.Qobj]] = {}
        for step, arrays in history_arrays.items():
            rows: list[qt.Qobj] = []
            for array, (_, is_operator, dims) in zip(arrays, raw_states):
                rows.append(_array_to_state(array, is_operator=is_operator, dims=dims, full_dim=self.full_dim))
            history[int(step)] = rows
        return final_states, history

    def _operator_leakage_metrics(self, operator: np.ndarray, subspace: Subspace) -> dict[str, float]:
        idx = np.asarray(subspace.indices, dtype=int)
        columns = operator[:, idx]
        kept = columns[idx, :]
        keep_prob = np.sum(np.abs(kept) ** 2, axis=0)
        leakage = np.maximum(0.0, 1.0 - np.real_if_close(keep_prob))
        return {
            "leakage_average": float(np.mean(leakage)) if leakage.size else 0.0,
            "leakage_worst": float(np.max(leakage)) if leakage.size else 0.0,
        }

    def _checkpoint_leakage(
        self,
        gate_matrices: Sequence[np.ndarray],
        penalty: LeakagePenalty | None,
        subspace: Subspace | None,
        *,
        initial_states: Sequence[qt.Qobj | np.ndarray] | None = None,
    ) -> dict[str, float]:
        if penalty is None or float(penalty.checkpoint_weight) <= 0.0 or subspace is None:
            return {
                "checkpoint_leakage_average": 0.0,
                "checkpoint_leakage_worst": 0.0,
                "checkpoint_leakage_term": 0.0,
            }
        steps = list(penalty.checkpoints) if penalty.checkpoints else list(range(1, len(gate_matrices) + 1))
        if initial_states is None:
            eye = np.eye(subspace.dim, dtype=np.complex128)
            initial_states = [subspace.embed(eye[:, col]) for col in range(subspace.dim)]
        _, history = self._propagate_states(gate_matrices, initial_states, checkpoints=steps)
        leakage_rows = []
        for step in steps:
            outputs = history.get(int(step), [])
            if not outputs:
                continue
            leak = state_leakage_metrics(outputs, subspace)
            leakage_rows.append((float(leak.average), float(leak.worst)))
        if not leakage_rows:
            return {
                "checkpoint_leakage_average": 0.0,
                "checkpoint_leakage_worst": 0.0,
                "checkpoint_leakage_term": 0.0,
            }
        avg = float(np.mean([row[0] for row in leakage_rows]))
        worst = float(np.max([row[1] for row in leakage_rows]))
        raw = avg if str(penalty.metric) == "average" else worst
        return {
            "checkpoint_leakage_average": avg,
            "checkpoint_leakage_worst": worst,
            "checkpoint_leakage_term": float(penalty.checkpoint_weight) * raw,
        }

    def evaluate(self, sequence: GateSequence, *, return_artifacts: bool = False) -> FastEvaluationResult:
        if not self.selection.fast_path_used:
            raise RuntimeError("FastObjectiveEvaluator.evaluate called without a usable fast path.")

        gate_matrices = [self._gate_matrix(gate) for gate in sequence.gates]
        target_type = str(self.payload["target_type"])
        checkpoint_leak = self._checkpoint_leakage(
            gate_matrices,
            self.payload.get("leakage_penalty"),
            self.leakage_subspace,
            initial_states=(
                self.payload.get("state_mapping", {}).get("initial_states")
                if target_type == "state_mapping"
                else self.payload.get("observable_target", {}).get("initial_states")
                if target_type == "observable"
                else self.payload.get("trajectory_target", {}).get("initial_states")
                if target_type == "trajectory"
                else None
            ),
        )

        if target_type == "unitary":
            full_operator = self._compose_operator(gate_matrices)
            subspace = self.payload["subspace"]
            sub_operator = subspace.restrict_operator(full_operator)
            fidelity = subspace_unitary_fidelity(
                sub_operator,
                np.asarray(self.payload["target_subspace"], dtype=np.complex128),
                gauge=str(self.payload["gauge"]),
                block_slices=self.payload.get("target_blocks"),
            )
            if self.leakage_subspace is not None:
                leakage = self._operator_leakage_metrics(full_operator, self.leakage_subspace)
            else:
                leakage = {"leakage_average": 0.0, "leakage_worst": 0.0}
            metrics = {
                "fidelity": float(fidelity),
                "fidelity_loss": float(1.0 - fidelity),
                "leakage_average": float(leakage["leakage_average"]),
                "leakage_worst": float(leakage["leakage_worst"]),
                "leakage_term": 0.0,
                "objective": float(1.0 - fidelity + checkpoint_leak["checkpoint_leakage_term"]),
                **checkpoint_leak,
            }
            return FastEvaluationResult(
                metrics=metrics,
                full_operator=full_operator if return_artifacts else None,
                subspace_operator=sub_operator if return_artifacts else None,
                selection=self.selection,
            )

        if target_type in {"state_mapping", "isometry"}:
            mapping = self.payload["state_mapping"]
            final_states, _ = self._propagate_states(gate_matrices, mapping["initial_states"])
            sim_metrics = state_mapping_metrics(final_states, mapping["target_states"], weights=mapping["weights"])
            if self.leakage_subspace is not None:
                leak = state_leakage_metrics(final_states, self.leakage_subspace)
                sim_metrics.update(
                    {
                        "leakage_average": float(leak.average),
                        "leakage_worst": float(leak.worst),
                    }
                )
            else:
                sim_metrics.update({"leakage_average": 0.0, "leakage_worst": 0.0})
            sim_metrics.update(
                {
                    "fidelity": float(sim_metrics.get("state_fidelity_mean", np.nan)),
                    "fidelity_loss": float(sim_metrics.get("weighted_state_infidelity", np.nan)),
                    "leakage_term": 0.0,
                    "objective": float(sim_metrics.get("weighted_state_error", np.nan) + checkpoint_leak["checkpoint_leakage_term"]),
                    **checkpoint_leak,
                }
            )
            return FastEvaluationResult(
                metrics={str(key): float(value) for key, value in sim_metrics.items() if np.isscalar(value)},
                state_outputs=final_states if return_artifacts else None,
                selection=self.selection,
            )

        if target_type == "observable":
            observable_target = self.payload["observable_target"]
            final_states, _ = self._propagate_states(gate_matrices, observable_target["initial_states"])
            observable_metrics = observable_expectation_metrics(
                final_states,
                observable_target["observables"],
                observable_target["target_expectations"],
                state_weights=observable_target["state_weights"],
                observable_weights=observable_target["observable_weights"],
            )
            if self.leakage_subspace is not None:
                leak = state_leakage_metrics(final_states, self.leakage_subspace)
                leakage = {"leakage_average": float(leak.average), "leakage_worst": float(leak.worst)}
            else:
                leakage = {"leakage_average": 0.0, "leakage_worst": 0.0}
            metrics = {
                "fidelity": float("nan"),
                "fidelity_loss": float(observable_metrics["weighted_observable_error"]),
                "observable_error_mean": float(observable_metrics["observable_error_mean"]),
                "observable_error_max": float(observable_metrics["observable_error_max"]),
                "weighted_observable_error": float(observable_metrics["weighted_observable_error"]),
                "leakage_average": float(leakage["leakage_average"]),
                "leakage_worst": float(leakage["leakage_worst"]),
                "leakage_term": 0.0,
                "objective": float(observable_metrics["weighted_observable_error"] + checkpoint_leak["checkpoint_leakage_term"]),
                **checkpoint_leak,
            }
            return FastEvaluationResult(
                metrics=metrics,
                state_outputs=final_states if return_artifacts else None,
                selection=self.selection,
            )

        if target_type == "trajectory":
            trajectory_target = self.payload["trajectory_target"]
            checkpoint_steps = [int(row["step"]) for row in trajectory_target["checkpoints"]]
            final_states, history = self._propagate_states(
                gate_matrices,
                trajectory_target["initial_states"],
                checkpoints=checkpoint_steps,
            )
            checkpoint_losses: list[float] = []
            checkpoint_weights: list[float] = []
            state_losses: list[float] = []
            observable_losses: list[float] = []
            for checkpoint in trajectory_target["checkpoints"]:
                outputs = history[int(checkpoint["step"])]
                checkpoint_loss = 0.0
                if checkpoint["target_states"]:
                    state_metrics = state_mapping_metrics(
                        outputs,
                        checkpoint["target_states"],
                        weights=checkpoint["state_weights"],
                    )
                    loss = float(state_metrics["weighted_state_error"])
                    checkpoint_loss += loss
                    state_losses.append(loss)
                if checkpoint["observables"]:
                    observable_metrics = observable_expectation_metrics(
                        outputs,
                        checkpoint["observables"],
                        checkpoint["target_expectations"],
                        state_weights=checkpoint["state_weights"],
                        observable_weights=checkpoint["observable_weights"],
                    )
                    loss = float(observable_metrics["weighted_observable_error"])
                    checkpoint_loss += loss
                    observable_losses.append(loss)
                checkpoint_losses.append(float(checkpoint_loss))
                checkpoint_weights.append(float(checkpoint["weight"]))
            total_weight = float(np.sum(checkpoint_weights)) if checkpoint_weights else 1.0
            task_loss = float(np.dot(checkpoint_losses, checkpoint_weights) / max(total_weight, 1.0e-18)) if checkpoint_losses else 0.0
            if self.leakage_subspace is not None:
                leak = state_leakage_metrics(final_states, self.leakage_subspace)
                leakage = {"leakage_average": float(leak.average), "leakage_worst": float(leak.worst)}
            else:
                leakage = {"leakage_average": 0.0, "leakage_worst": 0.0}
            metrics = {
                "fidelity": float("nan"),
                "fidelity_loss": float(task_loss),
                "trajectory_task_loss": float(task_loss),
                "trajectory_state_error_mean": float(np.mean(state_losses)) if state_losses else 0.0,
                "trajectory_observable_error_mean": float(np.mean(observable_losses)) if observable_losses else 0.0,
                "checkpoint_count": float(len(checkpoint_steps)),
                "leakage_average": float(leakage["leakage_average"]),
                "leakage_worst": float(leakage["leakage_worst"]),
                "leakage_term": 0.0,
                "objective": float(task_loss + checkpoint_leak["checkpoint_leakage_term"]),
                **checkpoint_leak,
            }
            return FastEvaluationResult(
                metrics=metrics,
                state_outputs=final_states if return_artifacts else None,
                selection=self.selection,
            )

        raise ValueError(f"Unsupported fast-evaluation target type '{target_type}'.")
