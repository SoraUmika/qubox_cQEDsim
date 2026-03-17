from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import qutip as qt

from cqed_sim.sim import cavity_moments, reduced_cavity_state, reduced_qubit_state

from .metrics import ancilla_return_metric, parity_expectation, photon_number_distribution
from .runtime import ClassicalProcessor


def _to_real_vector(values: Sequence[complex | float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.complex128).astype(np.complex128).view(np.float64)


def _padded_density_block(rho: qt.Qobj, dimension: int) -> np.ndarray:
    array = np.asarray(rho.full(), dtype=np.complex128)
    out = np.zeros((int(dimension), int(dimension)), dtype=np.complex128)
    rows = min(out.shape[0], array.shape[0])
    cols = min(out.shape[1], array.shape[1])
    out[:rows, :cols] = array[:rows, :cols]
    return out.reshape(-1)


class IdealSummaryObservation:
    requires_measurement = False

    def __init__(self, *, photon_cutoff: int = 6):
        self.photon_cutoff = int(photon_cutoff)

    def encode(
        self,
        *,
        state: qt.Qobj | None,
        model: Any,
        task: Any,
        metrics: dict[str, Any] | None = None,
        measurement: Any = None,
        **_: Any,
    ) -> np.ndarray:
        del model, task, measurement
        if state is None:
            raise ValueError("IdealSummaryObservation requires a state.")
        rho_q = np.asarray(reduced_qubit_state(state).full(), dtype=np.complex128)
        p0 = float(np.real(rho_q[0, 0])) if rho_q.shape[0] >= 1 else 0.0
        p1 = float(np.real(rho_q[1, 1])) if rho_q.shape[0] >= 2 else 0.0
        p_rest = float(max(0.0, 1.0 - p0 - p1))
        coherence01 = complex(rho_q[0, 1]) if rho_q.shape[0] >= 2 else 0.0j
        moments = cavity_moments(state)
        photon_distribution = photon_number_distribution(state)
        padded_distribution = np.zeros(self.photon_cutoff, dtype=float)
        count = min(self.photon_cutoff, photon_distribution.size)
        padded_distribution[:count] = photon_distribution[:count]
        metrics = {} if metrics is None else metrics
        vector = np.concatenate(
            [
                np.asarray(
                    [
                        p0,
                        p1,
                        p_rest,
                        coherence01.real,
                        coherence01.imag,
                        float(np.real(moments["a"])),
                        float(np.imag(moments["a"])),
                        float(np.real(moments["n"])),
                        float(parity_expectation(state)),
                        float(metrics.get("state_fidelity", metrics.get("state_fidelity_mean", 0.0))),
                        float(metrics.get("process_fidelity", 0.0) if np.isfinite(metrics.get("process_fidelity", 0.0)) else 0.0),
                        float(metrics.get("leakage_average", 0.0)),
                        float(metrics.get("ancilla_return", metrics.get("ancilla_return_mean", ancilla_return_metric(state)))),
                    ],
                    dtype=float,
                ),
                padded_distribution,
            ]
        )
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


class ReducedDensityObservation:
    requires_measurement = False

    def __init__(self, *, qubit_levels: int = 3, cavity_levels: int = 8):
        self.qubit_levels = int(qubit_levels)
        self.cavity_levels = int(cavity_levels)

    def encode(self, *, state: qt.Qobj | None, **_: Any) -> np.ndarray:
        if state is None:
            raise ValueError("ReducedDensityObservation requires a state.")
        rho_q = reduced_qubit_state(state)
        rho_c = reduced_cavity_state(state)
        vector = np.concatenate(
            [
                _to_real_vector(_padded_density_block(rho_q, self.qubit_levels)),
                _to_real_vector(_padded_density_block(rho_c, self.cavity_levels)),
            ]
        )
        return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


class GateMetricObservation:
    requires_measurement = False

    def __init__(self, *, keys: Sequence[str] = ("process_fidelity", "state_fidelity_mean", "leakage_average", "ancilla_return_mean", "success")):
        self.keys = tuple(str(key) for key in keys)

    def encode(self, *, metrics: dict[str, Any] | None = None, **_: Any) -> np.ndarray:
        metrics = {} if metrics is None else metrics
        values = []
        for key in self.keys:
            value = metrics.get(key, 0.0)
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                value = 0.0
            values.append(float(value))
        return np.asarray(values, dtype=float)


class MeasurementIQObservation:
    requires_measurement = True

    def __init__(self, *, mode: str = "iq_mean"):
        self.mode = str(mode)

    def encode(self, *, measurement: Any = None, metrics: dict[str, Any] | None = None, **_: Any) -> np.ndarray:
        metrics = {} if metrics is None else metrics
        measurement_vec = ClassicalProcessor.measurement_vector(measurement, mode=self.mode)
        trailing = np.asarray(
            [
                float(metrics.get("state_fidelity", metrics.get("state_fidelity_mean", 0.0))),
                float(metrics.get("leakage_average", 0.0)),
            ],
            dtype=float,
        )
        return np.nan_to_num(np.concatenate([measurement_vec, trailing]), nan=0.0, posinf=0.0, neginf=0.0)


class HistoryObservationWrapper:
    def __init__(self, base_model: Any, *, history_length: int = 2, action_dim: int = 0):
        self.base_model = base_model
        self.history_length = int(history_length)
        self.action_dim = int(action_dim)
        self.requires_measurement = bool(getattr(base_model, "requires_measurement", False))

    def encode(
        self,
        *,
        observation_history: Sequence[np.ndarray] | None = None,
        action_history: Sequence[np.ndarray] | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        current = np.asarray(self.base_model.encode(observation_history=observation_history, action_history=action_history, **kwargs), dtype=float)
        observation_history = [] if observation_history is None else [np.asarray(obs, dtype=float) for obs in observation_history]
        previous = observation_history[-max(self.history_length - 1, 0) :]
        zero_obs = np.zeros_like(current)
        previous = [
            np.pad(vector.reshape(-1), (0, max(0, current.size - vector.size)), mode="constant")[: current.size]
            for vector in previous
        ]
        padded_observations = [zero_obs.copy() for _ in range(max(self.history_length - 1 - len(previous), 0))]
        padded_observations.extend(previous)
        stacked_observations = padded_observations + [current]

        action_vectors: list[np.ndarray] = []
        if action_history:
            action_vectors = [np.asarray(action, dtype=float).reshape(-1) for action in action_history[-self.history_length :]]
        action_dim = self.action_dim or (0 if not action_vectors else int(action_vectors[-1].size))
        zero_action = np.zeros(action_dim, dtype=float)
        padded_actions = [zero_action.copy() for _ in range(max(self.history_length - len(action_vectors), 0))]
        if action_dim > 0:
            padded_actions.extend(
                [
                    np.pad(vector, (0, max(0, action_dim - vector.size)), mode="constant")[:action_dim]
                    for vector in action_vectors
                ]
            )
        return np.concatenate(stacked_observations + padded_actions)


def build_observation_model(observation_mode: str, *, action_dim: int = 0, history_length: int = 1, **kwargs: Any) -> Any:
    mode_key = str(observation_mode).strip().lower()
    if mode_key == "ideal_summary":
        model = IdealSummaryObservation(**kwargs)
    elif mode_key == "reduced_density":
        model = ReducedDensityObservation(**kwargs)
    elif mode_key in {"gate_metrics", "process_metrics"}:
        model = GateMetricObservation(**kwargs)
    elif mode_key in {"measurement_iq", "iq"}:
        model = MeasurementIQObservation(**kwargs)
    elif mode_key in {"measurement_counts", "counts"}:
        model = MeasurementIQObservation(mode="counts")
    elif mode_key in {"measurement_classifier_probs", "classifier_probs"}:
        model = MeasurementIQObservation(mode="classifier_probs")
    elif mode_key in {"measurement_classifier_logits", "classifier_logits", "measurement_logits"}:
        model = MeasurementIQObservation(mode="classifier_logits")
    elif mode_key in {"measurement_outcome", "measurement_outcome_onehot", "outcome_onehot", "outcome"}:
        model = MeasurementIQObservation(mode="outcome_onehot")
    else:
        raise ValueError(f"Unsupported observation mode '{observation_mode}'.")
    if history_length > 1:
        return HistoryObservationWrapper(model, history_length=int(history_length), action_dim=int(action_dim))
    return model


__all__ = [
    "IdealSummaryObservation",
    "ReducedDensityObservation",
    "GateMetricObservation",
    "MeasurementIQObservation",
    "HistoryObservationWrapper",
    "build_observation_model",
]