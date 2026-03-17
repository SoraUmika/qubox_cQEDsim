from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.sim import reduced_cavity_state, reduced_qubit_state, subsystem_level_population
from cqed_sim.unitary_synthesis.metrics import state_leakage_metrics, state_mapping_metrics, subspace_unitary_fidelity
from cqed_sim.unitary_synthesis.subspace import Subspace


def _as_density_matrix(state: qt.Qobj) -> qt.Qobj:
    return state if state.isoper else state.proj()


def state_fidelity(actual: qt.Qobj, target: qt.Qobj) -> float:
    rho_actual = _as_density_matrix(actual)
    rho_target = _as_density_matrix(target)
    return float(np.clip(qt.fidelity(rho_actual, rho_target) ** 2, 0.0, 1.0))


def ancilla_return_metric(state: qt.Qobj, *, target_level: int = 0) -> float:
    return float(np.clip(subsystem_level_population(state, "transmon", int(target_level)), 0.0, 1.0))


def photon_number_distribution(state: qt.Qobj) -> np.ndarray:
    rho_c = reduced_cavity_state(state)
    distribution = np.real(np.diag(rho_c.full()))
    return np.asarray(np.clip(distribution, 0.0, 1.0), dtype=float)


def parity_expectation(state: qt.Qobj, *, displacement: complex = 0.0j) -> float:
    rho_c = reduced_cavity_state(state)
    if displacement != 0.0j:
        operator = qt.displace(int(rho_c.shape[0]), complex(displacement))
        rho_c = operator * rho_c * operator.dag()
    parity = np.diag([1.0 if index % 2 == 0 else -1.0 for index in range(int(rho_c.shape[0]))])
    return float(np.real(np.trace(rho_c.full() @ parity)))


def sparse_wigner_samples(state: qt.Qobj, points: Sequence[complex]) -> np.ndarray:
    rho_c = reduced_cavity_state(state)
    values: list[float] = []
    for point in points:
        x_value = float(np.real(point))
        p_value = float(np.imag(point))
        sample = qt.wigner(rho_c, [x_value], [p_value])
        values.append(float(np.asarray(sample, dtype=float)[0, 0]))
    return np.asarray(values, dtype=float)


def reconstruct_subspace_operator(outputs: Sequence[qt.Qobj], subspace: Subspace) -> np.ndarray:
    columns: list[np.ndarray] = []
    for state in outputs:
        if state.isoper:
            raise ValueError("Subspace operator reconstruction requires pure-state outputs.")
        columns.append(subspace.extract(np.asarray(state.full(), dtype=np.complex128).reshape(-1)))
    return np.column_stack(columns)


def summarize_distribution(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=float)
    if data.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p05": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
        }
    return {
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "p05": float(np.percentile(data, 5.0)),
        "p50": float(np.percentile(data, 50.0)),
        "p95": float(np.percentile(data, 95.0)),
    }


def evaluate_state_task_metrics(task: Any, model: Any, state: qt.Qobj) -> dict[str, float | dict[str, float]]:
    target_state = task.build_target_state(model)
    metrics: dict[str, float | dict[str, float]] = {
        "state_fidelity": float(state_fidelity(state, target_state)),
        "ancilla_return": float(ancilla_return_metric(state, target_level=task.target_ancilla_level)),
    }
    subspace = task.build_subspace(model)
    if subspace is not None:
        leakage = state_leakage_metrics([state], subspace)
        metrics["leakage_average"] = float(leakage.average)
        metrics["leakage_worst"] = float(leakage.worst)
    else:
        metrics["leakage_average"] = float(max(0.0, 1.0 - ancilla_return_metric(state)))
        metrics["leakage_worst"] = float(metrics["leakage_average"])
    metrics["success"] = float(metrics["state_fidelity"] >= float(task.success_threshold))
    return metrics


def evaluate_unitary_task_metrics(task: Any, model: Any, outputs: Sequence[qt.Qobj]) -> dict[str, float]:
    subspace = task.build_subspace(model)
    if subspace is None:
        raise ValueError("Unitary-synthesis metrics require a task subspace.")
    target_outputs = task.build_target_probe_states(model)
    metrics = {
        key: float(value)
        for key, value in state_mapping_metrics(outputs, target_outputs).items()
        if isinstance(value, (int, float))
    }
    leakage = state_leakage_metrics(outputs, subspace)
    metrics["leakage_average"] = float(leakage.average)
    metrics["leakage_worst"] = float(leakage.worst)
    metrics["ancilla_return_mean"] = float(np.mean([ancilla_return_metric(state) for state in outputs]))
    target_operator = task.build_target_operator(model)
    if target_operator is not None and all(not state.isoper for state in outputs):
        actual_operator = reconstruct_subspace_operator(outputs, subspace)
        target_array = np.asarray(target_operator, dtype=np.complex128)
        metrics["process_fidelity"] = float(
            subspace_unitary_fidelity(
                actual_operator,
                target_array,
                gauge=task.gauge,
                block_slices=subspace.per_fock_blocks() if task.gauge == "block" else None,
            )
        )
    else:
        metrics["process_fidelity"] = float("nan")
    metrics["success"] = float(metrics.get("process_fidelity", metrics["state_fidelity_mean"]) >= float(task.success_threshold))
    return metrics


__all__ = [
    "state_fidelity",
    "ancilla_return_metric",
    "photon_number_distribution",
    "parity_expectation",
    "sparse_wigner_samples",
    "reconstruct_subspace_operator",
    "summarize_distribution",
    "evaluate_state_task_metrics",
    "evaluate_unitary_task_metrics",
]