from __future__ import annotations

import copy
import json
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import qutip as qt
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit

from .backends import SimulationResult
from .config import LeakagePenalty, MultiObjective, ParameterDistribution, SynthesisConstraints
from .constraints import (
    TimeGridResult,
    enforce_slew_limit,
    evaluate_tone_spacing,
    piecewise_constant_samples,
    snap_times_to_grid,
)
from .metrics import state_leakage_metrics, state_mapping_metrics
from .progress import (
    CompositeReporter,
    HistoryReporter,
    JupyterLiveReporter,
    NullReporter,
    PROGRESS_SCHEMA_VERSION,
    ProgressEvent,
    ProgressReporter,
    save_history_csv,
    save_history_json,
)
from .reporting import make_run_report
from .sequence import (
    ConditionalPhaseSQR,
    Displacement,
    DriftPhaseModel,
    FreeEvolveCondPhase,
    GateSequence,
    PrimitiveGate,
    QubitRotation,
    SNAP,
    SQR,
)
from .subspace import Subspace
from .systems import QuantumSystem, resolve_quantum_system
from .targets import TargetStateMapping, TargetUnitary, coerce_target


@dataclass(frozen=True)
class TimeMapper:
    t_min: float
    t_max: float

    def map(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return self.t_min + (self.t_max - self.t_min) * expit(x)

    def inverse(self, t: np.ndarray) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        eps = 1e-15
        y = np.clip((t - self.t_min) / (self.t_max - self.t_min), eps, 1.0 - eps)
        return np.log(y / (1.0 - y))

    def grad(self, x: np.ndarray) -> np.ndarray:
        s = expit(np.asarray(x, dtype=float))
        return (self.t_max - self.t_min) * s * (1.0 - s)


@dataclass
class SynthesisResult:
    success: bool
    objective: float
    sequence: GateSequence
    simulation: SimulationResult
    report: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    history_by_run: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    progress_schema_version: int = PROGRESS_SCHEMA_VERSION

    def save_history(self, path: str | Path) -> Path:
        return save_history_json(self.history, path)

    def save_history_csv(self, path: str | Path) -> Path:
        return save_history_csv(self.history, path)

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): SynthesisResult._json_ready(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [SynthesisResult._json_ready(v) for v in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if hasattr(value, "to_record") and callable(value.to_record):
            try:
                return SynthesisResult._json_ready(value.to_record())
            except Exception:
                return str(value)
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                return value
        try:
            json.dumps(value)
            return value
        except Exception:
            return str(value)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "objective": float(self.objective),
            "success": bool(self.success),
            "sequence": self.sequence.serialize(),
            "time_parameters": self.sequence.serialize_time_parameters(),
            "report": self.report,
            "history": list(self.history),
            "history_by_run": dict(self.history_by_run),
        }

    def to_payload(self, *, include_history: bool = True) -> dict[str, Any]:
        self.sequence.sync_time_params_from_gates()
        payload = {
            "schema_version": 2,
            "success": bool(self.success),
            "objective": float(self.objective),
            "sequence": self.sequence.serialize(),
            "time_parameters": self.sequence.serialize_time_parameters(),
            "parameter_vector": self.sequence.get_parameter_vector().tolist(),
            "time_raw_vector": self.sequence.get_time_raw_vector(active_only=True).tolist(),
            "report": self.report,
            "backend": str(self.simulation.backend),
            "backend_settings": dict(self.simulation.settings),
            "metrics": dict(self.simulation.metrics),
        }
        if include_history:
            payload["history"] = list(self.history)
            payload["history_by_run"] = dict(self.history_by_run)
        return self._json_ready(payload)

    def save(self, path: str | Path, *, include_history: bool = True) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_payload(include_history=include_history), indent=2), encoding="utf-8")
        return path

    def plot_convergence(self, what: str = "objective_total", fidelity_what: str = "metrics.fidelity_subspace") -> Any:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        from .progress import plot_history

        plot_history(self.history_by_run or self.history, what=what, ax=axes[0], title="Objective")
        plot_history(self.history_by_run or self.history, what=fidelity_what, ax=axes[1], title="Fidelity")
        fig.tight_layout()
        return fig


@dataclass
class ParetoFrontResult:
    results: list[SynthesisResult]
    nondominated_indices: list[int]

    def nondominated(self) -> list[SynthesisResult]:
        return [self.results[idx] for idx in self.nondominated_indices]


def _default_hardware_limits() -> dict[str, Any]:
    inf = float("inf")
    return {
        "dt": 1e-9,
        "channels": {
            "qubit_drive": {"A_max": inf, "delta_max": inf, "slew_max": inf, "bw_max": inf},
            "cavity_drive": {"A_max": inf, "delta_max": inf, "slew_max": inf, "bw_max": inf},
        },
        "gate_type_overrides": {},
    }


def _default_constraints() -> dict[str, Any]:
    return {
        "slew": {"enabled": False, "S_max": float("inf"), "mode": "penalty", "lambda": 1.0},
        "tone_spacing": {
            "enabled": False,
            "domega_min": 0.0,
            "Ntones_max": None,
            "projection": False,
            "lambda": 1.0,
        },
        "forbidden_bands": [],
        "amplitude_detuning": {"enabled": True, "lambda": 1.0, "project": True},
    }


def _default_time_grid() -> dict[str, Any]:
    return {"dt": 1e-9, "mode": "hard", "lambda_grid": 0.0}


def _default_parallel() -> dict[str, Any]:
    return {"enabled": False, "n_jobs": 1, "backend": "multiprocessing"}


def _default_optimizer_options() -> dict[str, Any]:
    return {
        "xatol": 1.0e-4,
        "fatol": 1.0e-6,
        "popsize": 8,
        "tol": 1.0e-3,
        "polish": True,
    }


def _default_progress() -> dict[str, Any]:
    return {"enabled": False, "every": 1, "live": False, "print_every": 10}


def _coerce_synthesis_constraints(value: SynthesisConstraints | Mapping[str, Any] | None) -> SynthesisConstraints | None:
    if value is None:
        return None
    if isinstance(value, SynthesisConstraints):
        return value
    return SynthesisConstraints(**dict(value))


def _coerce_leakage_penalty(value: LeakagePenalty | Mapping[str, Any] | None) -> LeakagePenalty | None:
    if value is None:
        return None
    if isinstance(value, LeakagePenalty):
        return value
    return LeakagePenalty(**dict(value))


def _coerce_multi_objective(value: MultiObjective | Mapping[str, Any] | None) -> MultiObjective | None:
    if value is None:
        return None
    if isinstance(value, MultiObjective):
        return value
    return MultiObjective(**dict(value))


def _deep_update(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in update.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _gate_channel_name(gate: Any) -> str | None:
    gate_type = gate.type
    if gate_type in {"QubitRotation", "SQR", "ConditionalPhaseSQR"}:
        return "qubit_drive"
    if gate_type in {"Displacement"}:
        return "cavity_drive"
    return None


def _gate_amplitude_proxy(gate: Any, n_cav: int) -> float:
    gate_type = gate.type
    if gate_type == "QubitRotation":
        return float(abs(getattr(gate, "theta", 0.0)))
    if gate_type == "SQR":
        vals = np.asarray(getattr(gate, "theta_n", []), dtype=float)
        return float(np.max(np.abs(vals[:n_cav]))) if vals.size else 0.0
    if gate_type == "Displacement":
        return float(abs(getattr(gate, "alpha", 0.0 + 0.0j)))
    if gate_type == "ConditionalPhaseSQR":
        vals = np.asarray(getattr(gate, "phases_n", []), dtype=float)
        return float(np.max(np.abs(vals[:n_cav]))) if vals.size else 0.0
    return 0.0


def _gate_detuning_proxy(gate: Any, n_cav: int) -> float:
    if gate.type == "SQR":
        freqs = np.asarray(getattr(gate, "tone_freqs", []), dtype=float)
        if freqs.size:
            return float(np.max(np.abs(freqs)))
    return 0.0


def _clip_gate_amplitude(gate: Any, a_max: float, n_cav: int) -> None:
    if not np.isfinite(a_max):
        return
    if gate.type == "QubitRotation":
        gate.theta = float(np.clip(gate.theta, -a_max, a_max))
    elif gate.type == "SQR":
        vals = np.asarray(getattr(gate, "theta_n", []), dtype=float)
        if vals.size:
            vals[:n_cav] = np.clip(vals[:n_cav], -a_max, a_max)
            gate.theta_n = [float(x) for x in vals]
    elif gate.type == "Displacement":
        alpha = complex(getattr(gate, "alpha", 0.0 + 0.0j))
        mag = abs(alpha)
        if mag > a_max and mag > 0.0:
            gate.alpha = alpha * (a_max / mag)
    elif gate.type == "ConditionalPhaseSQR":
        vals = np.asarray(getattr(gate, "phases_n", []), dtype=float)
        if vals.size:
            vals[:n_cav] = np.clip(vals[:n_cav], -a_max, a_max)
            gate.phases_n = [float(x) for x in vals]


def _clip_gate_detuning(gate: Any, delta_max: float) -> None:
    if not np.isfinite(delta_max):
        return
    if gate.type == "SQR":
        freqs = np.asarray(getattr(gate, "tone_freqs", []), dtype=float)
        if freqs.size:
            gate.tone_freqs = [float(x) for x in np.clip(freqs, -delta_max, delta_max)]


def _channel_limit(hardware_limits: Mapping[str, Any], gate_type: str, channel: str | None, key: str) -> float:
    inf = float("inf")
    value = inf
    if channel is not None:
        value = float(hardware_limits.get("channels", {}).get(channel, {}).get(key, inf))
    value = float(hardware_limits.get("gate_type_overrides", {}).get(gate_type, {}).get(key, value))
    return value


def _times_summary(sequence: GateSequence) -> dict[str, Any]:
    summary: dict[str, list[float]] = {}
    for gate in sequence.gates:
        summary.setdefault(gate.type, []).append(float(gate.duration))
    out: dict[str, Any] = {}
    for gate_type, values in summary.items():
        rounded = np.unique(np.round(np.asarray(values, dtype=float), 15))
        out[gate_type] = float(values[0]) if rounded.size == 1 else [float(v) for v in values]
    return out


def _with_phase2_amplitude_limit(
    hardware_limits: Mapping[str, Any],
    constraints_profile: SynthesisConstraints | None,
) -> dict[str, Any]:
    out = copy.deepcopy(dict(hardware_limits))
    if constraints_profile is None or constraints_profile.max_amplitude is None:
        return out
    a_max = float(constraints_profile.max_amplitude)
    for channel_cfg in out.get("channels", {}).values():
        current = float(channel_cfg.get("A_max", np.inf))
        channel_cfg["A_max"] = float(min(current, a_max))
    return out


def _has_open_system(settings: Mapping[str, Any]) -> bool:
    return settings.get("c_ops") is not None or settings.get("noise") is not None


def _resolve_dynamic_simulation_setting(value: Any, model: Any) -> Any:
    if callable(value):
        return value(model)
    return value


def _runtime_model(system: QuantumSystem | None) -> Any | None:
    if system is None:
        return None
    return system.runtime_model()


def _apply_system_dims_to_states(states: Sequence[qt.Qobj], system: QuantumSystem | None) -> list[qt.Qobj]:
    model = _runtime_model(system)
    if model is None or not hasattr(model, "subsystem_dims"):
        return list(states)
    dims = [int(dim) for dim in getattr(model, "subsystem_dims")]
    out: list[qt.Qobj] = []
    for state in states:
        arr = np.asarray(state.full(), dtype=np.complex128)
        if state.isoper:
            out.append(qt.Qobj(arr, dims=[dims, dims]))
        else:
            out.append(qt.Qobj(arr.reshape(-1), dims=[dims, [1] * len(dims)]))
    return out


def _sequence_amplitude_series(sequence: GateSequence, channel: str) -> np.ndarray:
    amps: list[float] = []
    n_cav = sequence.n_cav if sequence.n_cav is not None else -1
    for gate in sequence.gates:
        amps.append(_gate_amplitude_proxy(gate, n_cav) if _gate_channel_name(gate) == channel else 0.0)
    return np.asarray(amps, dtype=float)


def _sequence_duration_metric(sequence: GateSequence, t0_ref: np.ndarray, dt: float) -> float:
    denom = max(float(np.sum(t0_ref)), float(dt), 1.0e-18)
    return float(sequence.total_duration() / denom)


def _sequence_power_metric(sequence: GateSequence) -> float:
    n_cav = sequence.n_cav if sequence.n_cav is not None else -1
    values = []
    for gate in sequence.gates:
        amp = _gate_amplitude_proxy(gate, n_cav)
        values.append((float(amp) ** 2) * float(gate.duration))
    if not values:
        return 0.0
    total_duration = max(float(sequence.total_duration()), 1.0e-18)
    return float(np.sum(values) / total_duration)


def _sequence_smoothness_metric(sequence: GateSequence) -> float:
    penalties: list[float] = []
    for channel in ("qubit_drive", "cavity_drive"):
        amps = _sequence_amplitude_series(sequence, channel)
        if amps.size > 1:
            penalties.append(float(np.mean(np.diff(amps) ** 2)))
    return float(np.mean(penalties)) if penalties else 0.0


def _bandwidth_penalty(samples: np.ndarray, dt: float, bw_max: float) -> float:
    if samples.size < 2 or not np.isfinite(float(bw_max)):
        return 0.0
    centered = np.asarray(samples, dtype=float) - float(np.mean(samples))
    freqs = np.fft.rfftfreq(centered.size, d=float(dt))
    spec = np.abs(np.fft.rfft(centered)) ** 2
    mask = freqs > float(bw_max)
    if not np.any(mask):
        return 0.0
    return float(np.sum(spec[mask]) / max(centered.size, 1))


def _project_total_duration(sequence: GateSequence, max_duration: float) -> None:
    total = float(sequence.total_duration())
    if total <= max_duration or total <= 0.0:
        return
    times = sequence.get_time_vector(active_only=False)
    scaled = times * (float(max_duration) / total)
    sequence.set_time_vector(scaled, active_only=False)


def _phase2_constraint_penalties(
    sequence: GateSequence,
    constraints_profile: SynthesisConstraints | None,
    *,
    dt: float,
) -> dict[str, Any]:
    if constraints_profile is None:
        return {
            "total_penalty": 0.0,
            "duration_penalty": 0.0,
            "bandwidth_penalty": 0.0,
            "smoothness_penalty": 0.0,
            "forbidden_parameter_penalty": 0.0,
            "duration_violation": 0.0,
            "forbidden_parameter_violations": 0,
            "bandwidth_proxy": 0.0,
        }

    if constraints_profile.max_duration is not None and constraints_profile.duration_mode == "hard":
        _project_total_duration(sequence, float(constraints_profile.max_duration))

    total_penalty = 0.0
    duration_penalty = 0.0
    bandwidth_penalty = 0.0
    smoothness_penalty = 0.0
    forbidden_penalty = 0.0
    duration_violation = 0.0
    forbidden_violations = 0
    bandwidth_proxy = 0.0

    total_duration = float(sequence.total_duration())
    if constraints_profile.max_duration is not None:
        duration_violation = max(0.0, total_duration - float(constraints_profile.max_duration))
        duration_penalty = duration_violation**2
        total_penalty += duration_penalty

    if constraints_profile.smoothness_penalty:
        smoothness_penalty = float(constraints_profile.smoothness_weight) * _sequence_smoothness_metric(sequence)
        total_penalty += smoothness_penalty

    if constraints_profile.max_bandwidth is not None:
        for channel in ("qubit_drive", "cavity_drive"):
            samples = piecewise_constant_samples(_sequence_amplitude_series(sequence, channel), sequence.gate_durations(), dt=dt)
            penalty = _bandwidth_penalty(samples, dt=dt, bw_max=float(constraints_profile.max_bandwidth))
            bandwidth_penalty += penalty
            bandwidth_proxy = max(bandwidth_proxy, float(penalty))
        bandwidth_penalty *= float(constraints_profile.bandwidth_weight)
        total_penalty += bandwidth_penalty

    if constraints_profile.forbidden_parameter_ranges:
        values = sequence.get_parameter_vector()
        layout = sequence.parameter_layout()
        for (gate_idx, param_name, _), value in zip(layout, values):
            gate = sequence.gates[gate_idx]
            candidates = (
                str(param_name),
                f"{gate.name}.{param_name}",
                f"{gate.type}.{param_name}",
            )
            for pattern, ranges in constraints_profile.forbidden_parameter_ranges.items():
                if not any(constraints_profile.matches_parameter(pattern, candidate) for candidate in candidates):
                    continue
                for lo, hi in ranges:
                    if float(lo) <= float(value) <= float(hi):
                        depth = min(float(value) - float(lo), float(hi) - float(value))
                        forbidden_penalty += max(depth, 1.0e-12) ** 2
                        forbidden_violations += 1
        forbidden_penalty *= float(constraints_profile.forbidden_range_weight)
        total_penalty += forbidden_penalty

    return {
        "total_penalty": float(total_penalty),
        "duration_penalty": float(duration_penalty),
        "bandwidth_penalty": float(bandwidth_penalty),
        "smoothness_penalty": float(smoothness_penalty),
        "forbidden_parameter_penalty": float(forbidden_penalty),
        "duration_violation": float(duration_violation),
        "forbidden_parameter_violations": int(forbidden_violations),
        "bandwidth_proxy": float(bandwidth_proxy),
    }


def _leakage_cost(metrics: Mapping[str, Any], penalty: LeakagePenalty | None, fallback_weight: float) -> tuple[float, float]:
    weight = float(fallback_weight if penalty is None else penalty.weight)
    mode = "worst" if penalty is None else str(penalty.metric)
    if mode == "average":
        raw = float(metrics.get("leakage_average", 0.0))
    else:
        raw = float(metrics.get("leakage_worst", 0.0))
    return raw, weight * raw


def _target_gauge(target: Any, fallback: str) -> str:
    if isinstance(target, TargetUnitary):
        return target.resolved_gauge(fallback=fallback)
    return str(fallback)


def _target_blocks(target: Any, subspace: Subspace | None, fallback: str) -> tuple[tuple[int, ...], ...] | None:
    if isinstance(target, TargetUnitary):
        return target.resolved_blocks(subspace=subspace, fallback=fallback)
    return None


def _evaluate_target_simulation(
    sequence: GateSequence,
    payload: Mapping[str, Any],
    *,
    model_override: Any | None = None,
    return_simulation: bool = False,
) -> tuple[dict[str, float], SimulationResult | None]:
    target_type = str(payload["target_type"])
    target_object = payload.get("target_object")
    system: QuantumSystem = payload["system"]
    eval_system = system if model_override is None else system.with_model(model_override)
    sim_settings = dict(payload.get("backend_settings", {}))
    sim_settings.pop("system", None)
    runtime_model = _runtime_model(eval_system)
    if "c_ops" in sim_settings:
        sim_settings["c_ops"] = _resolve_dynamic_simulation_setting(sim_settings.get("c_ops"), runtime_model)
    if "noise" in sim_settings:
        sim_settings["noise"] = _resolve_dynamic_simulation_setting(sim_settings.get("noise"), runtime_model)
    sim_settings.setdefault("dt", float(payload["time_grid"]["dt"]))

    open_system = _has_open_system(sim_settings)
    leakage_subspace = payload.get("leakage_subspace")

    if target_type == "state_mapping":
        mapping = payload["state_mapping"]
        initial_states = _apply_system_dims_to_states(list(mapping["initial_states"]), eval_system)
        target_states = _apply_system_dims_to_states(list(mapping["target_states"]), eval_system)
        simulation = eval_system.simulate_sequence(
            sequence=sequence,
            subspace=payload["subspace"],
            backend=payload["backend"],
            state_inputs=initial_states,
            need_operator=False,
            leakage_weight=float(payload["leakage_weight"]),
            gauge=str(payload["gauge"]),
            block_slices=payload.get("target_blocks"),
            **sim_settings,
        )
        if simulation.state_outputs is None:
            raise ValueError("State-mapping optimization requires propagated output states.")
        sim_metrics = state_mapping_metrics(
            simulation.state_outputs,
            target_states,
            weights=mapping["weights"],
        )
        leak_term = 0.0
        if leakage_subspace is not None:
            leak = state_leakage_metrics(simulation.state_outputs, leakage_subspace)
            leak_term = float(payload["leakage_weight"]) * float(leak.worst)
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
                "leakage_term": float(leak_term),
                "objective": float(sim_metrics.get("weighted_state_error", np.nan) + leak_term),
            }
        )
        simulation.metrics = dict(sim_metrics)
        return dict(sim_metrics), simulation if return_simulation else None

    target = target_object
    if not isinstance(target, TargetUnitary):
        raise ValueError("Unitary-target simulation requires TargetUnitary metadata.")

    if open_system:
        probe_states, target_states = target.resolved_probe_pairs(
            full_dim=payload["subspace"].full_dim,
            subspace=payload["subspace"],
        )
        probe_states = _apply_system_dims_to_states(probe_states, eval_system)
        target_states = _apply_system_dims_to_states(target_states, eval_system)
        simulation = eval_system.simulate_sequence(
            sequence=sequence,
            subspace=payload["subspace"],
            backend=payload["backend"],
            state_inputs=probe_states,
            need_operator=False,
            leakage_weight=float(payload["leakage_weight"]),
            gauge=str(payload["gauge"]),
            block_slices=payload.get("target_blocks"),
            **sim_settings,
        )
        if simulation.state_outputs is None:
            raise ValueError("Open-system unitary targets require propagated probe states.")
        sim_metrics = state_mapping_metrics(
            simulation.state_outputs,
            target_states,
        )
        leak_term = 0.0
        if leakage_subspace is not None:
            leak = state_leakage_metrics(simulation.state_outputs, leakage_subspace)
            leak_term = float(payload["leakage_weight"]) * float(leak.worst)
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
                "leakage_term": float(leak_term),
                "objective": float(sim_metrics.get("weighted_state_error", np.nan) + leak_term),
            }
        )
        simulation.metrics = dict(sim_metrics)
        return dict(sim_metrics), simulation if return_simulation else None

    simulation = eval_system.simulate_sequence(
        sequence=sequence,
        subspace=payload["subspace"],
        backend=payload["backend"],
        target_subspace=payload.get("target_subspace"),
        leakage_weight=float(payload["leakage_weight"]),
        gauge=str(payload["gauge"]),
        block_slices=payload.get("target_blocks"),
        need_operator=True,
        **sim_settings,
    )
    return dict(simulation.metrics), simulation if return_simulation else None


def _nondominated_indices(points: Sequence[Mapping[str, float]], keys: Sequence[str]) -> list[int]:
    rows = [{str(key): float(point[key]) for key in keys} for point in points]
    keep: list[int] = []
    for i, left in enumerate(rows):
        dominated = False
        for j, right in enumerate(rows):
            if i == j:
                continue
            no_worse = all(right[key] <= left[key] for key in keys)
            strictly_better = any(right[key] < left[key] for key in keys)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def _max_gate_amplitude(sequence: GateSequence) -> float:
    if not sequence.gates:
        return 0.0
    return float(max(_gate_amplitude_proxy(gate, sequence.n_cav) for gate in sequence.gates))


def _max_gate_detuning(sequence: GateSequence) -> float:
    if not sequence.gates:
        return 0.0
    return float(max(_gate_detuning_proxy(gate, sequence.n_cav) for gate in sequence.gates))


def _make_progress_event(
    *,
    run_id: str,
    iteration: int,
    objective_total: float,
    details: Mapping[str, Any],
    sequence: GateSequence,
    backend: str,
    best_objective: float,
    best_iteration: int,
) -> dict[str, Any]:
    sim_metrics = dict(details.get("simulation_metrics", {}))
    constraints = dict(details.get("constraints", {}))
    phase2_constraints = dict(details.get("phase2_constraints", {}))
    event = ProgressEvent(
        run_id=str(run_id),
        iteration=int(iteration),
        timestamp=float(time.time()),
        objective_total=float(objective_total),
        objective_terms={
            "infidelity": float(sim_metrics.get("fidelity_loss", 0.0)),
            "leakage": float(sim_metrics.get("leakage_term", 0.0)),
            "duration": float(sim_metrics.get("duration_metric", 0.0)),
            "pulse_power": float(sim_metrics.get("pulse_power_metric", 0.0)),
            "robustness": float(sim_metrics.get("robustness_penalty", 0.0)),
            "time_reg": float(details.get("time_reg_term", 0.0)),
            "grid_penalty": float(details.get("grid_term", 0.0)),
            "constraint_penalty": float(constraints.get("total_penalty", 0.0) + phase2_constraints.get("total_penalty", 0.0)),
        },
        metrics={
            "fidelity_subspace": float(sim_metrics.get("fidelity", np.nan)),
            "leakage_avg": float(sim_metrics.get("leakage_average", np.nan)),
            "leakage_worst": float(sim_metrics.get("leakage_worst", np.nan)),
        },
        best_so_far={
            "objective_total": float(best_objective),
            "iteration": int(best_iteration),
        },
        params_summary={
            "max_amp": _max_gate_amplitude(sequence),
            "max_detuning": _max_gate_detuning(sequence),
            "times": _times_summary(sequence),
        },
        backend=str(backend),
        solver_stats=dict(details.get("solver_stats", {})),
    )
    return event.to_dict()


def _apply_hardware_constraints(
    sequence: GateSequence,
    hardware_limits: Mapping[str, Any],
    constraints: Mapping[str, Any],
    dt: float,
) -> dict[str, Any]:
    amp_cfg = dict(constraints.get("amplitude_detuning", {}))
    amp_enabled = bool(amp_cfg.get("enabled", True))
    amp_lambda = float(amp_cfg.get("lambda", 1.0))
    amp_project = bool(amp_cfg.get("project", True))

    amp_penalty = 0.0
    det_penalty = 0.0
    amp_violations = 0
    det_violations = 0

    for gate in sequence.gates:
        channel = _gate_channel_name(gate)
        a_max = _channel_limit(hardware_limits, gate.type, channel, "A_max")
        d_max = _channel_limit(hardware_limits, gate.type, channel, "delta_max")

        amp = _gate_amplitude_proxy(gate, sequence.n_cav)
        if np.isfinite(a_max):
            v = max(0.0, amp - a_max)
            if v > 0.0:
                amp_violations += 1
                amp_penalty += v**2
                if amp_enabled and amp_project:
                    _clip_gate_amplitude(gate, a_max, sequence.n_cav)

        det = _gate_detuning_proxy(gate, sequence.n_cav)
        if np.isfinite(d_max):
            v = max(0.0, det - d_max)
            if v > 0.0:
                det_violations += 1
                det_penalty += v**2
                if amp_enabled and amp_project:
                    _clip_gate_detuning(gate, d_max)

    # Slew constraints on piecewise-constant per-channel amplitudes.
    slew_cfg = dict(constraints.get("slew", {}))
    slew_enabled = bool(slew_cfg.get("enabled", False))
    slew_lambda = float(slew_cfg.get("lambda", 1.0))
    slew_mode = str(slew_cfg.get("mode", "penalty"))
    slew_penalty = 0.0
    slew_max_violation = 0.0

    if slew_enabled:
        durations = sequence.gate_durations()
        for channel in ["qubit_drive", "cavity_drive"]:
            amps = []
            for gate in sequence.gates:
                gch = _gate_channel_name(gate)
                if gch == channel:
                    amps.append(_gate_amplitude_proxy(gate, sequence.n_cav))
                else:
                    amps.append(0.0)
            samples = piecewise_constant_samples(amps, durations, dt=dt)
            s_max = float(hardware_limits.get("channels", {}).get(channel, {}).get("slew_max", np.inf))
            s_max = float(constraints.get("gate_type_overrides", {}).get(channel, {}).get("slew_max", s_max))
            if np.isfinite(s_max):
                sr = enforce_slew_limit(samples, dt=dt, s_max=s_max, mode=slew_mode)
                slew_penalty += sr.penalty
                slew_max_violation = max(slew_max_violation, sr.max_violation)

    # Tone spacing constraints on SQR gates.
    tone_cfg = dict(constraints.get("tone_spacing", {}))
    tone_enabled = bool(tone_cfg.get("enabled", False))
    tone_lambda = float(tone_cfg.get("lambda", 1.0))
    tone_penalty = 0.0
    tone_min_spacing = float("inf")
    tone_violations = 0

    if tone_enabled:
        domega_min = float(tone_cfg.get("domega_min", 0.0))
        ntones_max = tone_cfg.get("Ntones_max", None)
        projection = bool(tone_cfg.get("projection", False))
        forbidden = constraints.get("forbidden_bands", [])
        for gate in sequence.gates:
            if gate.type != "SQR":
                continue
            freqs = np.asarray(getattr(gate, "tone_freqs", []), dtype=float)
            if freqs.size == 0:
                continue
            tr = evaluate_tone_spacing(
                freqs=freqs,
                domega_min=domega_min,
                ntones_max=ntones_max,
                forbidden_bands=forbidden,
                project=projection,
            )
            tone_penalty += tr.total_penalty
            tone_min_spacing = min(tone_min_spacing, tr.min_spacing)
            if tr.total_penalty > 0.0:
                tone_violations += 1
            if projection:
                gate.tone_freqs = [float(x) for x in tr.freqs_projected]

    total_penalty = 0.0
    if amp_enabled:
        total_penalty += amp_lambda * (amp_penalty + det_penalty)
    if slew_enabled:
        total_penalty += slew_lambda * slew_penalty
    if tone_enabled:
        total_penalty += tone_lambda * tone_penalty

    return {
        "total_penalty": float(total_penalty),
        "amplitude_penalty": float(amp_lambda * amp_penalty if amp_enabled else 0.0),
        "detuning_penalty": float(amp_lambda * det_penalty if amp_enabled else 0.0),
        "slew_penalty": float(slew_lambda * slew_penalty if slew_enabled else 0.0),
        "tone_penalty": float(tone_lambda * tone_penalty if tone_enabled else 0.0),
        "amplitude_violations": int(amp_violations),
        "detuning_violations": int(det_violations),
        "slew_max_violation": float(slew_max_violation),
        "tone_min_spacing": float(tone_min_spacing),
        "tone_violations": int(tone_violations),
    }


def _evaluate_objective_vector(
    x: np.ndarray,
    sequence: GateSequence,
    payload: dict[str, Any],
    *,
    return_simulation: bool = False,
) -> tuple[float, dict[str, Any], SimulationResult | None]:
    x = np.asarray(x, dtype=float)
    param_size = int(payload["param_size"])
    sequence.set_parameter_vector(x[:param_size])
    if x.size > param_size:
        sequence.set_time_raw_vector(x[param_size:], active_only=True)

    grid_cfg = payload["time_grid"]
    grid: TimeGridResult = snap_times_to_grid(sequence.get_time_vector(active_only=False), dt=float(grid_cfg["dt"]), mode=str(grid_cfg["mode"]))
    sequence.set_time_vector(grid.snapped, active_only=False)

    grid_lambda = float(grid_cfg.get("lambda_grid", 0.0))
    grid_term = 0.0
    if str(grid_cfg.get("mode", "hard")) == "soft":
        grid_term = grid_lambda * float(np.sum(grid.grid_residual**2))

    c_eval = _apply_hardware_constraints(
        sequence=sequence,
        hardware_limits=payload["hardware_limits"],
        constraints=payload["constraints"],
        dt=float(grid_cfg["dt"]),
    )
    phase2_eval = _phase2_constraint_penalties(
        sequence=sequence,
        constraints_profile=payload.get("synthesis_constraints"),
        dt=float(grid_cfg["dt"]),
    )

    sim_t0 = time.perf_counter()
    simulation_metrics, simulation = _evaluate_target_simulation(
        sequence,
        payload,
        return_simulation=return_simulation,
    )
    robust_samples: list[dict[str, Any]] = []
    robust_penalty = 0.0
    robust_aggregate = float(simulation_metrics.get("objective", np.nan))
    parameter_distribution = payload.get("parameter_distribution")
    robust_assignments = list(payload.get("robust_assignments", []))
    robust_base_model = payload.get("robust_base_model")
    if parameter_distribution is not None and robust_base_model is not None and robust_assignments:
        sample_values: list[float] = []
        for assignment in robust_assignments:
            sampled_model = parameter_distribution.apply(robust_base_model, assignment)
            sample_metrics, _ = _evaluate_target_simulation(
                sequence,
                payload,
                model_override=sampled_model,
                return_simulation=False,
            )
            sample_objective = float(sample_metrics.get("objective", sample_metrics.get("fidelity_loss", np.inf)))
            sample_values.append(sample_objective)
            robust_samples.append(
                {
                    "assignment": {str(k): float(v) for k, v in assignment.items()},
                    "objective": sample_objective,
                    "fidelity": float(sample_metrics.get("fidelity", np.nan)),
                    "fidelity_loss": float(sample_metrics.get("fidelity_loss", np.nan)),
                    "leakage_average": float(sample_metrics.get("leakage_average", 0.0)),
                    "leakage_worst": float(sample_metrics.get("leakage_worst", 0.0)),
                }
            )
        if sample_values:
            if str(parameter_distribution.aggregate) == "worst":
                robust_aggregate = float(np.max(sample_values))
            else:
                robust_aggregate = float(np.mean(sample_values))
            nominal_objective = float(simulation_metrics.get("objective", simulation_metrics.get("fidelity_loss", np.inf)))
            robust_penalty = max(0.0, robust_aggregate - nominal_objective)

    sim_elapsed = float(time.perf_counter() - sim_t0)

    raw_leakage, leakage_term = _leakage_cost(
        simulation_metrics,
        payload.get("leakage_penalty"),
        float(payload["leakage_weight"]),
    )

    active_times = sequence.get_time_vector(active_only=True)
    t0_ref = np.asarray(payload["t0_ref"], dtype=float)
    reg_term = 0.0
    if active_times.size:
        reg_term = float(payload["time_reg_weight"]) * float(np.sum((active_times - t0_ref) ** 2))

    smooth_term = 0.0
    durations = sequence.gate_durations()
    if float(payload["time_smooth_weight"]) > 0.0 and durations.size > 1:
        smooth_term = float(payload["time_smooth_weight"]) * float(np.sum(np.diff(durations) ** 2))

    duration_metric = _sequence_duration_metric(sequence, t0_ref=t0_ref, dt=float(grid_cfg["dt"]))
    pulse_power_metric = _sequence_power_metric(sequence)
    amplitude_smoothness_metric = _sequence_smoothness_metric(sequence)

    phase2_active = bool(payload.get("phase2_active", False))
    if phase2_active:
        objectives: MultiObjective = payload["objectives"]
        total = (
            float(objectives.fidelity_weight) * float(simulation_metrics.get("fidelity_loss", np.inf))
            + float(leakage_term)
            + float(objectives.duration_weight) * float(duration_metric)
            + float(objectives.pulse_power_weight) * float(pulse_power_metric)
            + float(objectives.smoothness_weight) * float(amplitude_smoothness_metric)
            + float(objectives.robustness_weight) * float(robust_penalty)
            + reg_term
            + smooth_term
            + grid_term
            + float(objectives.hardware_penalty_weight) * float(c_eval["total_penalty"] + phase2_eval["total_penalty"])
        )
    else:
        total = (
            float(simulation_metrics.get("objective", np.inf))
            + reg_term
            + smooth_term
            + grid_term
            + float(c_eval["total_penalty"])
        )
        if float(phase2_eval["total_penalty"]) > 0.0:
            total += float(phase2_eval["total_penalty"])

    details = {
        "grid": grid,
        "grid_term": float(grid_term),
        "constraints": c_eval,
        "phase2_constraints": phase2_eval,
        "time_reg_term": float(reg_term),
        "time_smooth_term": float(smooth_term),
        "simulation_metrics": {
            **dict(simulation_metrics),
            "leakage_selected": float(raw_leakage),
            "leakage_term": float(leakage_term),
            "duration_metric": float(duration_metric),
            "pulse_power_metric": float(pulse_power_metric),
            "amplitude_smoothness_metric": float(amplitude_smoothness_metric),
            "robustness_penalty": float(robust_penalty),
            "robustness_objective": float(robust_aggregate),
        },
        "robustness": {
            "enabled": bool(parameter_distribution is not None and robust_assignments),
            "aggregate": None if parameter_distribution is None else str(parameter_distribution.aggregate),
            "samples": robust_samples,
            "objective": float(robust_aggregate),
            "penalty": float(robust_penalty),
        },
        "solver_stats": {
            "n_steps": int(max(1, round(float(np.sum(sequence.gate_durations())) / max(float(grid_cfg["dt"]), 1e-18))))
            if str(payload["backend"]) == "pulse"
            else 0,
            "dt": float(grid_cfg["dt"]),
            "solver_time_sec": sim_elapsed,
        }
        if str(payload["backend"]) == "pulse"
        else {},
    }
    if simulation is not None:
        simulation.metrics = dict(details["simulation_metrics"])
    return float(total), details, simulation if return_simulation else None


def _normalize_optimizer_name(name: str) -> str:
    lowered = str(name).strip().lower().replace("-", "_")
    aliases = {
        "lbfgsb": "l_bfgs_b",
        "l-bfgs-b": "l_bfgs_b",
        "nm": "nelder_mead",
        "de": "differential_evolution",
        "cmaes": "cma_es",
    }
    return aliases.get(lowered, lowered)


def _scipy_minimize_method(name: str, grid_mode: str) -> str:
    normalized = _normalize_optimizer_name(name)
    if normalized == "auto":
        return "Powell" if grid_mode == "hard" else "L-BFGS-B"
    mapping = {
        "nelder_mead": "Nelder-Mead",
        "powell": "Powell",
        "bfgs": "BFGS",
        "l_bfgs_b": "L-BFGS-B",
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported local optimizer '{name}'.")
    return mapping[normalized]


def _run_single_start(payload: dict[str, Any], reporter: ProgressReporter | None = None) -> dict[str, Any]:
    worker_seed = int(payload["worker_seed"])
    np.random.seed(worker_seed % (2**32 - 1))

    sequence = copy.deepcopy(payload["sequence_template"])
    x_start = np.asarray(payload["x_start"], dtype=float)
    maxiter = int(payload["maxiter"])
    progress_cfg = dict(payload.get("progress", {}))
    every = max(1, int(progress_cfg.get("every", 1)))
    run_id = str(payload.get("run_id", f"run_{int(payload['start_index']):03d}"))

    history = HistoryReporter()
    run_reporter = CompositeReporter([history, reporter or NullReporter()])
    best_objective = float("inf")
    best_iteration = 0
    last_emitted_iteration: int | None = None

    def _objective(x: np.ndarray) -> float:
        val, _, _ = _evaluate_objective_vector(x, sequence, payload, return_simulation=False)
        return float(val)

    def _emit(iteration: int, x_vec: np.ndarray, *, value: float | None = None, details: dict[str, Any] | None = None) -> None:
        nonlocal best_objective, best_iteration, last_emitted_iteration
        if details is None or value is None:
            value, details, _ = _evaluate_objective_vector(x_vec, sequence, payload, return_simulation=False)
        assert details is not None
        if float(value) <= best_objective:
            best_objective = float(value)
            best_iteration = int(iteration)
        event = _make_progress_event(
            run_id=run_id,
            iteration=iteration,
            objective_total=float(value),
            details=details,
            sequence=sequence,
            backend=str(payload["backend"]),
            best_objective=best_objective,
            best_iteration=best_iteration,
        )
        run_reporter.on_event(event)
        last_emitted_iteration = int(iteration)

    run_reporter.on_start(
        {
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "run_id": run_id,
            "start_index": int(payload["start_index"]),
            "worker_seed": worker_seed,
            "backend": str(payload["backend"]),
            "maxiter": maxiter,
            "emit_every": every,
            "timestamp": float(time.time()),
        }
    )
    _emit(0, x_start)

    grid_mode = str(payload.get("time_grid", {}).get("mode", "hard"))
    optimizer_name = _normalize_optimizer_name(str(payload.get("optimizer", "auto")))
    optimizer_options = dict(payload.get("optimizer_options", {}))
    callback_state = {"count": 0}

    def _callback(xk: np.ndarray) -> None:
        callback_state["count"] += 1
        if callback_state["count"] % every != 0:
            return
        _emit(int(callback_state["count"]), np.asarray(xk, dtype=float))

    if optimizer_name == "differential_evolution":
        bounds = payload.get("search_bounds")
        if not bounds or any((not np.isfinite(low)) or (not np.isfinite(high)) for low, high in bounds):
            raise ValueError("differential_evolution requires finite parameter bounds.")
        workers = 1
        if bool(payload.get("parallel", {}).get("enabled", False)):
            workers = max(1, int(payload.get("parallel", {}).get("n_jobs", 1)))
        res = differential_evolution(
            _objective,
            bounds=bounds,
            maxiter=maxiter,
            seed=worker_seed,
            polish=bool(optimizer_options.get("polish", True)),
            popsize=int(optimizer_options.get("popsize", 8)),
            tol=float(optimizer_options.get("tol", 1.0e-3)),
            workers=workers,
            updating="deferred" if workers > 1 else "immediate",
            callback=lambda xk, convergence=0.0: (_callback(np.asarray(xk, dtype=float)), False)[1],
        )
    elif optimizer_name == "cma_es":
        try:
            import cma  # type: ignore
        except Exception as exc:
            raise ValueError("optimizer='cma_es' requires the optional 'cma' package to be installed.") from exc
        sigma0 = float(optimizer_options.get("sigma0", 0.2))
        opts = {"maxiter": maxiter, "seed": worker_seed, "verbose": -9}
        bounds = payload.get("search_bounds")
        if bounds and all(np.isfinite(low) and np.isfinite(high) for low, high in bounds):
            opts["bounds"] = [[float(low) for low, _ in bounds], [float(high) for _, high in bounds]]
        es = cma.CMAEvolutionStrategy(x_start.tolist(), sigma0, opts)
        while not es.stop():
            candidates = es.ask()
            values = [_objective(np.asarray(candidate, dtype=float)) for candidate in candidates]
            es.tell(candidates, values)
            callback_state["count"] += 1
            if callback_state["count"] % every == 0:
                best_x = np.asarray(es.result.xbest, dtype=float)
                _emit(int(callback_state["count"]), best_x)
            if callback_state["count"] >= maxiter:
                break

        class _CMAResult:
            pass

        res = _CMAResult()
        res.x = np.asarray(es.result.xbest, dtype=float)
        res.fun = float(es.result.fbest)
        res.success = True
        res.message = "cma_es completed"
        res.nit = int(callback_state["count"])
    else:
        method = _scipy_minimize_method(optimizer_name, grid_mode)
        local_options: dict[str, Any] = {"maxiter": maxiter}
        if method == "Nelder-Mead":
            if "xatol" in optimizer_options:
                local_options["xatol"] = float(optimizer_options["xatol"])
            if "fatol" in optimizer_options:
                local_options["fatol"] = float(optimizer_options["fatol"])
        elif method == "Powell":
            if "xatol" in optimizer_options:
                local_options["xtol"] = float(optimizer_options["xatol"])
            if "fatol" in optimizer_options:
                local_options["ftol"] = float(optimizer_options["fatol"])
        else:
            for key in ("gtol", "ftol", "eps", "maxfun"):
                if key in optimizer_options:
                    local_options[key] = optimizer_options[key]
        res = minimize(
            _objective,
            x_start,
            method=method,
            callback=_callback,
            options=local_options,
        )
    final_val, final_details, _ = _evaluate_objective_vector(res.x, sequence, payload, return_simulation=False)
    final_iteration = max(int(getattr(res, "nit", 0)), int(callback_state["count"]))
    if last_emitted_iteration != final_iteration:
        _emit(final_iteration, np.asarray(res.x, dtype=float), value=float(final_val), details=final_details)

    run_reporter.on_end(
        {
            "progress_schema_version": PROGRESS_SCHEMA_VERSION,
            "run_id": run_id,
            "start_index": int(payload["start_index"]),
            "worker_seed": worker_seed,
            "success": bool(res.success),
            "message": str(res.message),
            "nit": int(getattr(res, "nit", 0)),
            "best_objective_total": float(best_objective),
            "best_iteration": int(best_iteration),
            "timestamp": float(time.time()),
        }
    )

    return {
        "start_index": int(payload["start_index"]),
        "run_id": run_id,
        "worker_seed": worker_seed,
        "x": np.asarray(res.x, dtype=float),
        "fun": float(final_val),
        "success": bool(res.success),
        "message": str(res.message),
        "nit": int(res.nit),
        "details": final_details,
        "history": history.events,
        "history_start": history.starts,
        "history_end": history.ends,
    }


class UnitarySynthesizer:
    def __init__(
        self,
        subspace: Subspace | None = None,
        backend: str = "ideal",
        gateset: list[str] | None = None,
        primitives: list[Any] | None = None,
        target: Any | None = None,
        model: Any | None = None,
        system: QuantumSystem | None = None,
        optimizer: str = "auto",
        optimizer_options: Mapping[str, Any] | None = None,
        simulation_options: Mapping[str, Any] | None = None,
        optimize_times: bool = True,
        time_bounds: dict[str, tuple[float, float]] | None = None,
        time_policy: dict[str, dict[str, Any]] | None = None,
        time_mode: str = "per-instance",
        time_groups: dict[str, str] | None = None,
        leakage_weight: float = 0.0,
        time_reg_weight: float = 0.0,
        time_smooth_weight: float = 0.0,
        gauge: str = "global",
        drift_config: Mapping[str, Any] | None = None,
        include_conditional_phase_in_sqr: bool = False,
        hardware_limits: Mapping[str, Any] | None = None,
        time_grid: Mapping[str, Any] | None = None,
        constraints: Mapping[str, Any] | None = None,
        synthesis_constraints: SynthesisConstraints | Mapping[str, Any] | None = None,
        leakage_penalty: LeakagePenalty | Mapping[str, Any] | None = None,
        objectives: MultiObjective | Mapping[str, Any] | None = None,
        parameter_distribution: ParameterDistribution | None = None,
        warm_start: str | Path | Mapping[str, Any] | SynthesisResult | None = None,
        parallel: Mapping[str, Any] | None = None,
        progress: Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self.target = None if target is None else coerce_target(target)
        self.backend = backend
        self.gateset = gateset or ["QubitRotation", "SQR", "SNAP"]
        self.primitives = list(primitives) if primitives is not None else None
        self.system = resolve_quantum_system(
            system=system,
            model=model,
            subspace=subspace,
            primitives=self.primitives,
            gateset=self.gateset if primitives is None else None,
        )
        self.model = self.system.runtime_model()
        self.optimizer = _normalize_optimizer_name(optimizer)
        self.optimizer_options = _deep_update(_default_optimizer_options(), dict(optimizer_options or {}))
        self.backend_settings = dict(simulation_options or {})
        self.backend_settings["system"] = self.system
        self.optimize_times = bool(optimize_times)
        self.time_bounds = time_bounds or {"default": (20e-9, 2000e-9)}
        self.time_policy = dict(time_policy or {})
        self.time_mode = time_mode
        self.time_groups = dict(time_groups or {})
        self.synthesis_constraints = _coerce_synthesis_constraints(synthesis_constraints)
        self.leakage_penalty = _coerce_leakage_penalty(leakage_penalty)
        self.objectives = _coerce_multi_objective(objectives)
        self.parameter_distribution = parameter_distribution
        resolved_leakage_weight = float(leakage_weight)
        if self.objectives is not None:
            resolved_leakage_weight = float(self.objectives.leakage_weight)
        if self.leakage_penalty is not None:
            resolved_leakage_weight = float(self.leakage_penalty.weight)
        self.leakage_weight = float(resolved_leakage_weight)
        self.time_reg_weight = float(time_reg_weight)
        self.time_smooth_weight = float(time_smooth_weight)
        self.gauge = gauge
        self.warm_start = warm_start
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.drift_model = self._make_drift_model(drift_config)
        self.include_conditional_phase_in_sqr = bool(include_conditional_phase_in_sqr)

        self.hardware_limits = _with_phase2_amplitude_limit(
            _deep_update(_default_hardware_limits(), dict(hardware_limits or {})),
            self.synthesis_constraints,
        )
        self.time_grid = _deep_update(_default_time_grid(), dict(time_grid or {}))
        self.constraints = _deep_update(_default_constraints(), dict(constraints or {}))
        self.parallel = _deep_update(_default_parallel(), dict(parallel or {}))
        self.progress = _deep_update(_default_progress(), dict(progress or {}))

        if "dt" not in dict(time_grid or {}) and "dt" in self.hardware_limits:
            self.time_grid["dt"] = float(self.hardware_limits["dt"])

        self.subspace = self._resolve_default_subspace(subspace)

        self.sequence = self._build_initial_sequence()
        if self.synthesis_constraints is not None:
            self.synthesis_constraints.validate_sequence_length(len(self.sequence.gates))
        self._apply_warm_start(self.warm_start)

    @staticmethod
    def _normalize_gate_name(name: str) -> str:
        aliases = {
            "CondPhaseSQR": "ConditionalPhaseSQR",
            "ConditionalPhase": "ConditionalPhaseSQR",
            "FreeCondPhaseWait": "FreeEvolveCondPhase",
        }
        return aliases.get(name, name)

    @staticmethod
    def _make_drift_model(drift_config: Mapping[str, Any] | None) -> DriftPhaseModel:
        cfg = dict(drift_config or {})
        return DriftPhaseModel(
            chi=float(cfg.get("chi", 0.0)),
            chi2=float(cfg.get("chi2", 0.0)),
            kerr=float(cfg.get("kerr", cfg.get("K", 0.0))),
            kerr2=float(cfg.get("kerr2", cfg.get("K2", 0.0))),
            delta_c=float(cfg.get("delta_c", 0.0)),
            delta_q=float(cfg.get("delta_q", 0.0)),
            frame=str(cfg.get("frame", "rotating_omega_c_omega_q")),
        )

    @staticmethod
    def _infer_state_mapping_dim(target: TargetStateMapping) -> int:
        for state in list(target.initial_states) + list(target.target_states):
            if hasattr(state, "shape"):
                shape = getattr(state, "shape")
                if len(shape) == 1:
                    return int(shape[0])
                if len(shape) == 2:
                    return int(shape[0])
            arr = np.asarray(state, dtype=np.complex128)
            if arr.ndim == 1:
                return int(arr.size)
            if arr.ndim == 2:
                return int(arr.shape[0])
        raise ValueError("Could not infer a Hilbert-space dimension from the target state mapping.")

    def _system_hilbert_dimension(
        self,
        *,
        subspace: Subspace | None = None,
        target: TargetUnitary | TargetStateMapping | None = None,
        sequence: GateSequence | None = None,
    ) -> int | None:
        dim = self.system.hilbert_dimension(sequence=sequence, subspace=subspace, target=target)
        return None if dim is None else int(dim)

    def _resolve_default_subspace(self, subspace: Subspace | None) -> Subspace:
        if subspace is not None:
            return subspace
        if isinstance(self.target, TargetUnitary):
            return Subspace.custom(self.target.dim, range(self.target.dim))
        if isinstance(self.target, TargetStateMapping):
            dim = self._system_hilbert_dimension(target=self.target)
            if dim is None:
                dim = self._infer_state_mapping_dim(self.target)
            return Subspace.custom(dim, range(dim))
        dim = self._system_hilbert_dimension(target=self.target)
        if dim is not None:
            return Subspace.custom(dim, range(dim))
        if self.primitives:
            seq = GateSequence(gates=copy.deepcopy(self.primitives))
            self.system.configure_sequence(seq, subspace=None)
            dim = seq.resolve_full_dim(system=self.system)
            return Subspace.custom(dim, range(dim))
        raise ValueError("UnitarySynthesizer requires subspace or enough context to infer the full Hilbert dimension.")

    def _warm_start_payload(self, warm_start: str | Path | Mapping[str, Any] | SynthesisResult | None) -> dict[str, Any] | None:
        if warm_start is None:
            return None
        if isinstance(warm_start, SynthesisResult):
            return warm_start.to_payload(include_history=False)
        if isinstance(warm_start, Mapping):
            return dict(warm_start)
        path = Path(warm_start)
        return json.loads(path.read_text(encoding="utf-8"))

    def _apply_warm_start(self, warm_start: str | Path | Mapping[str, Any] | SynthesisResult | None) -> None:
        payload = self._warm_start_payload(warm_start)
        if payload is None:
            return
        parameter_vector = np.asarray(payload.get("parameter_vector", []), dtype=float)
        time_raw_vector = np.asarray(payload.get("time_raw_vector", []), dtype=float)
        current_params = self.sequence.get_parameter_vector()
        current_times = self.sequence.get_time_raw_vector(active_only=True)

        if parameter_vector.size:
            if parameter_vector.size != current_params.size:
                raise ValueError(
                    f"Warm-start parameter vector has length {parameter_vector.size}, expected {current_params.size}."
                )
            self.sequence.set_parameter_vector(parameter_vector)
        if time_raw_vector.size:
            if time_raw_vector.size != current_times.size:
                raise ValueError(f"Warm-start time vector has length {time_raw_vector.size}, expected {current_times.size}.")
            self.sequence.set_time_raw_vector(time_raw_vector, active_only=True)

    def _default_bounds(self) -> tuple[float, float]:
        low, high = self.time_bounds.get("default", (20e-9, 2000e-9))
        return float(low), float(high)

    def _default_duration(self) -> float:
        low, high = self._default_bounds()
        return float(0.5 * (low + high))

    def _time_bounds_for(self, gate_name: str) -> tuple[float, float]:
        norm = self._normalize_gate_name(gate_name)
        if norm in self.time_bounds:
            low, high = self.time_bounds[norm]
            return float(low), float(high)
        low, high = self._default_bounds()
        return float(low), float(high)

    def _merged_time_policy(self) -> dict[str, dict[str, Any]]:
        low, high = self._default_bounds()
        policy: dict[str, dict[str, Any]] = {
            "default": {
                "optimize": self.optimize_times,
                "bounds": (low, high),
                "init": 0.5 * (low + high),
            }
        }

        for gate_name, bounds in self.time_bounds.items():
            if gate_name == "default":
                continue
            norm = self._normalize_gate_name(gate_name)
            gate_low, gate_high = float(bounds[0]), float(bounds[1])
            policy[norm] = {
                "optimize": self.optimize_times,
                "bounds": (gate_low, gate_high),
                "init": 0.5 * (gate_low + gate_high),
            }

        for gate_name, override in self.time_policy.items():
            key = gate_name if gate_name == "default" else self._normalize_gate_name(gate_name)
            merged = dict(policy.get(key, {}))
            merged.update(dict(override))
            policy[key] = merged

        return policy

    def _build_initial_sequence(self) -> GateSequence:
        if self.primitives is not None:
            gates = copy.deepcopy(self.primitives)
            original_durations = [float(gate.duration) for gate in gates]
            seq = GateSequence(gates=gates, full_dim=self.subspace.full_dim)
            self.system.configure_sequence(seq, subspace=self.subspace)
            seq.configure_time_parameters(
                time_policy=self._merged_time_policy(),
                mode=self.time_mode,
                shared_groups=self.time_groups,
            )
            for gate, duration in zip(seq.gates, original_durations):
                gate.duration = float(duration)
            seq.sync_time_params_from_gates()
            grid = snap_times_to_grid(seq.get_time_vector(active_only=False), dt=float(self.time_grid["dt"]), mode=str(self.time_grid["mode"]))
            seq.set_time_vector(grid.snapped, active_only=False)
            return seq

        seq = self.system.build_sequence_from_gateset(
            self.gateset,
            subspace=self.subspace,
            default_duration=self._default_duration(),
            optimize_times=self.optimize_times,
            time_bounds_for=self._time_bounds_for,
            include_conditional_phase_in_sqr=self.include_conditional_phase_in_sqr,
            drift_model=self.drift_model,
        )
        self.system.configure_sequence(seq, subspace=self.subspace)
        seq.configure_time_parameters(
            time_policy=self._merged_time_policy(),
            mode=self.time_mode,
            shared_groups=self.time_groups,
        )
        # Ensure initial times honor configured grid.
        grid = snap_times_to_grid(seq.get_time_vector(active_only=False), dt=float(self.time_grid["dt"]), mode=str(self.time_grid["mode"]))
        seq.set_time_vector(grid.snapped, active_only=False)
        return seq

    def _target_subspace(self, target: np.ndarray) -> np.ndarray:
        tgt = np.asarray(target, dtype=np.complex128)
        if tgt.shape == (self.subspace.full_dim, self.subspace.full_dim):
            return self.subspace.restrict_operator(tgt)
        if tgt.shape == (self.subspace.dim, self.subspace.dim):
            return tgt
        raise ValueError("target must be full-space or subspace matrix with matching dimensions.")

    def _resolve_target(self, target: Any | None) -> TargetUnitary | TargetStateMapping:
        resolved = self.target if target is None else coerce_target(target)
        if resolved is None:
            raise ValueError("A target must be provided either to the constructor or to fit().")
        return resolved

    def _search_bounds(self) -> list[tuple[float, float]]:
        bounds = list(self.sequence.parameter_bounds_vector())
        bounds.extend([(-6.0, 6.0) for _ in self.sequence.active_time_params()])
        return bounds

    def _build_start_vectors(self, x0: np.ndarray, starts: int, init_guess: str) -> tuple[list[np.ndarray], list[int]]:
        vecs: list[np.ndarray] = []
        seeds: list[int] = []
        for k in range(starts):
            seed_k = int(self.seed + 1000 + k)
            seeds.append(seed_k)
            if k == 0 or init_guess != "random":
                vecs.append(x0.copy())
            else:
                rng_k = np.random.default_rng(seed_k)
                vecs.append(x0 + 0.2 * rng_k.standard_normal(x0.size))
        return vecs, seeds

    def _evaluate_best_solution(
        self,
        best_x: np.ndarray,
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], SimulationResult, float]:
        eval_payload = dict(payload)
        eval_payload["sequence_template"] = copy.deepcopy(self.sequence)
        value, details, sim = _evaluate_objective_vector(best_x, self.sequence, eval_payload, return_simulation=True)
        assert sim is not None
        return details, sim, float(value)

    def fit(
        self,
        target: Any | None = None,
        init_guess: str = "heuristic",
        multistart: int = 1,
        maxiter: int = 300,
    ) -> SynthesisResult:
        resolved_target = self._resolve_target(target)
        target_type = "unitary" if isinstance(resolved_target, TargetUnitary) else "state_mapping"
        if self.parameter_distribution is not None and self.system.runtime_model() is None:
            raise ValueError("ParameterDistribution requires a system with an underlying model so sampled parameters can be applied.")

        target_sub: np.ndarray | None = None
        state_mapping_payload: dict[str, Any] | None = None
        if isinstance(resolved_target, TargetUnitary):
            target_sub = self._target_subspace(resolved_target.matrix)
        else:
            initial_states, target_states, weights = resolved_target.resolved_pairs(
                full_dim=self.subspace.full_dim,
                subspace=self.subspace,
            )
            state_mapping_payload = {
                "initial_states": initial_states,
                "target_states": target_states,
                "weights": weights,
            }

        self.sequence.sync_time_params_from_gates()
        p0 = self.sequence.get_parameter_vector()
        raw_t0 = self.sequence.get_time_raw_vector(active_only=True)

        # Reference times use snapped values under the active grid policy.
        grid0 = snap_times_to_grid(self.sequence.get_time_vector(active_only=False), dt=float(self.time_grid["dt"]), mode=str(self.time_grid["mode"]))
        self.sequence.set_time_vector(grid0.snapped, active_only=False)
        t0_ref = self.sequence.get_time_vector(active_only=True)

        x0 = np.concatenate([p0, raw_t0]) if raw_t0.size else p0.copy()
        starts = max(1, int(multistart))
        x_starts, worker_seeds = self._build_start_vectors(x0=x0, starts=starts, init_guess=init_guess)

        par_enabled = bool(self.parallel.get("enabled", False))
        n_jobs = max(1, int(self.parallel.get("n_jobs", 1)))
        par_backend = str(self.parallel.get("backend", "multiprocessing"))

        # Avoid nested parallel oversubscription: keep probe parallelization serial inside parallel multistart workers.
        leakage_n_jobs = n_jobs if not (par_enabled and n_jobs > 1 and starts > 1) else 1
        state_workers = n_jobs if not (par_enabled and n_jobs > 1 and starts > 1) else 1
        target_gauge = _target_gauge(resolved_target, self.gauge)
        target_blocks = _target_blocks(resolved_target, self.subspace, target_gauge)
        leakage_subspace = self.leakage_penalty.resolve_subspace(self.subspace) if self.leakage_penalty is not None else self.subspace
        open_system_unitary = isinstance(resolved_target, TargetUnitary) and _has_open_system(self.backend_settings)
        phase2_active = any(
            value is not None
            for value in (self.synthesis_constraints, self.leakage_penalty, self.objectives, self.parameter_distribution)
        ) or open_system_unitary
        objective_cfg = self.objectives if self.objectives is not None else (MultiObjective(leakage_weight=self.leakage_weight) if phase2_active else None)
        robust_assignments = []
        if self.parameter_distribution is not None:
            robust_rng = np.random.default_rng(self.seed + 4242)
            robust_assignments = self.parameter_distribution.sample_assignments(robust_rng)

        common_payload = {
            "sequence_template": copy.deepcopy(self.sequence),
            "param_size": int(p0.size),
            "subspace": self.subspace,
            "backend": self.backend,
            "target_type": target_type,
            "target_object": resolved_target,
            "target_subspace": target_sub,
            "target_blocks": target_blocks,
            "state_mapping": state_mapping_payload,
            "system": self.system,
            "leakage_weight": self.leakage_weight,
            "leakage_penalty": self.leakage_penalty,
            "leakage_subspace": leakage_subspace,
            "gauge": target_gauge,
            "t0_ref": np.asarray(t0_ref, dtype=float),
            "time_reg_weight": self.time_reg_weight,
            "time_smooth_weight": self.time_smooth_weight,
            "time_grid": self.time_grid,
            "hardware_limits": self.hardware_limits,
            "constraints": self.constraints,
            "synthesis_constraints": self.synthesis_constraints,
            "phase2_active": bool(phase2_active),
            "objectives": objective_cfg,
            "parameter_distribution": self.parameter_distribution,
            "robust_base_model": self.system.runtime_model(),
            "robust_assignments": robust_assignments,
            "leakage_n_jobs": int(leakage_n_jobs),
            "state_workers": int(state_workers),
            "backend_settings": dict(self.backend_settings),
            "maxiter": int(maxiter),
            "progress": self.progress,
            "optimizer": self.optimizer,
            "optimizer_options": dict(self.optimizer_options),
            "parallel": copy.deepcopy(self.parallel),
            "search_bounds": self._search_bounds(),
        }

        payloads = []
        for k in range(starts):
            p = dict(common_payload)
            p["x_start"] = x_starts[k]
            p["start_index"] = int(k)
            p["worker_seed"] = int(worker_seeds[k])
            p["run_id"] = f"run_{k:03d}"
            payloads.append(p)

        live_enabled = bool(self.progress.get("enabled", False)) and bool(self.progress.get("live", False))
        serial_live = live_enabled and not (par_enabled and n_jobs > 1 and starts > 1 and par_backend == "multiprocessing")
        serial_reporter: ProgressReporter | None = None
        if serial_live:
            serial_reporter = JupyterLiveReporter(print_every=int(self.progress.get("print_every", 10)))

        if par_enabled and n_jobs > 1 and starts > 1 and par_backend == "multiprocessing":
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=n_jobs) as pool:
                start_results = pool.map(_run_single_start, payloads)
        else:
            start_results = [_run_single_start(p, reporter=serial_reporter) for p in payloads]

        best_result = sorted(start_results, key=lambda r: (float(r["fun"]), int(r["start_index"])))[0]

        final_details, simulation, final_objective = self._evaluate_best_solution(
            best_x=np.asarray(best_result["x"], dtype=float),
            payload=common_payload,
        )

        n_match = None
        if isinstance(self.subspace.metadata, dict) and "n_match" in self.subspace.metadata:
            n_match = int(self.subspace.metadata["n_match"])

        grid_final: TimeGridResult = final_details["grid"]
        c_eval = final_details["constraints"]
        phase2_c_eval = final_details.get("phase2_constraints", {})
        sim_metrics = final_details["simulation_metrics"]
        robustness_info = final_details.get("robustness", {})
        history_by_run = {
            str(row["run_id"]): sorted(list(row.get("history", [])), key=lambda event: int(event["iteration"]))
            for row in sorted(start_results, key=lambda item: int(item["start_index"]))
        }
        history = [event for run_id in history_by_run for event in history_by_run[run_id]]
        history.sort(key=lambda event: (str(event["run_id"]), int(event["iteration"]), float(event["timestamp"])))

        report = {
            "system": self.system.to_record(),
            "target": {
                "type": target_type,
                "dimension": int(self.subspace.dim if target_type == "unitary" else self.subspace.full_dim),
                "pairs": 0 if state_mapping_payload is None else len(state_mapping_payload["initial_states"]),
                "gauge": str(target_gauge),
                "block_gauge": None if target_blocks is None else [list(block) for block in target_blocks],
                "open_system_probe_strategy": getattr(resolved_target, "open_system_probe_strategy", None),
            },
            "subspace": {
                "kind": self.subspace.kind,
                "indices": list(self.subspace.indices),
                "dim": self.subspace.dim,
                "labels": list(self.subspace.labels),
            },
            "backend": {"type": self.backend, "settings": simulation.settings},
            "time_policy": {
                "mode": self.time_mode,
                "policy": self._merged_time_policy(),
                "shared_groups": dict(self.time_groups),
            },
            "time_grid": {
                "mode": str(grid_final.mode),
                "dt": float(grid_final.dt),
                "raw": grid_final.raw.tolist(),
                "snapped": grid_final.snapped.tolist(),
                "ticks": grid_final.ticks.astype(int).tolist(),
                "grid_residual": grid_final.grid_residual.tolist(),
            },
            "hardware_limits": copy.deepcopy(self.hardware_limits),
            "constraints": copy.deepcopy(self.constraints),
            "synthesis_constraints": None if self.synthesis_constraints is None else self.synthesis_constraints.to_record(),
            "leakage_penalty": None if self.leakage_penalty is None else self.leakage_penalty.to_record(),
            "multi_objective": None if objective_cfg is None else objective_cfg.to_record(),
            "parameter_distribution": None if self.parameter_distribution is None else self.parameter_distribution.to_record(),
            "parallel": {
                "enabled": bool(par_enabled),
                "n_jobs": int(n_jobs),
                "backend": str(par_backend),
                "worker_seeds": [int(r["worker_seed"]) for r in start_results],
            },
            "drift_model": {
                "frame": self.drift_model.frame,
                "chi": float(self.drift_model.chi),
                "chi2": float(self.drift_model.chi2),
                "kerr": float(self.drift_model.kerr),
                "kerr2": float(self.drift_model.kerr2),
                "delta_c": float(self.drift_model.delta_c),
                "delta_q": float(self.drift_model.delta_q),
            },
            "objective": {
                "total": float(final_objective),
                "fidelity_term": float(sim_metrics.get("fidelity_loss", 0.0)),
                "leakage_term": float(sim_metrics.get("leakage_term", 0.0)),
                "duration_term": float(sim_metrics.get("duration_metric", 0.0)),
                "pulse_power_term": float(sim_metrics.get("pulse_power_metric", 0.0)),
                "amplitude_smoothness_term": float(sim_metrics.get("amplitude_smoothness_metric", 0.0)),
                "robustness_term": float(sim_metrics.get("robustness_penalty", 0.0)),
                "time_reg_term": float(final_details.get("time_reg_term", 0.0)),
                "time_smooth_term": float(final_details.get("time_smooth_term", 0.0)),
                "grid_term": float(final_details.get("grid_term", 0.0)),
                "hardware_penalty_term": float(c_eval.get("total_penalty", 0.0)),
                "phase2_constraint_term": float(phase2_c_eval.get("total_penalty", 0.0)),
                "hardware_breakdown": {
                    "amplitude_penalty": float(c_eval.get("amplitude_penalty", 0.0)),
                    "detuning_penalty": float(c_eval.get("detuning_penalty", 0.0)),
                    "slew_penalty": float(c_eval.get("slew_penalty", 0.0)),
                    "tone_penalty": float(c_eval.get("tone_penalty", 0.0)),
                },
                "phase2_constraint_breakdown": {
                    "duration_penalty": float(phase2_c_eval.get("duration_penalty", 0.0)),
                    "bandwidth_penalty": float(phase2_c_eval.get("bandwidth_penalty", 0.0)),
                    "smoothness_penalty": float(phase2_c_eval.get("smoothness_penalty", 0.0)),
                    "forbidden_parameter_penalty": float(phase2_c_eval.get("forbidden_parameter_penalty", 0.0)),
                },
            },
            "constraint_violations": {
                "amplitude_violations": int(c_eval.get("amplitude_violations", 0)),
                "detuning_violations": int(c_eval.get("detuning_violations", 0)),
                "slew_max_violation": float(c_eval.get("slew_max_violation", 0.0)),
                "tone_min_spacing": float(c_eval.get("tone_min_spacing", np.inf)),
                "tone_violations": int(c_eval.get("tone_violations", 0)),
                "duration_violation": float(phase2_c_eval.get("duration_violation", 0.0)),
                "forbidden_parameter_violations": int(phase2_c_eval.get("forbidden_parameter_violations", 0)),
            },
            "metrics": {
                "fidelity": float(sim_metrics.get("fidelity", np.nan)),
                "leakage_average": float(sim_metrics.get("leakage_average", np.nan)),
                "leakage_worst": float(sim_metrics.get("leakage_worst", np.nan)),
                "leakage_selected": float(sim_metrics.get("leakage_selected", np.nan)),
                "state_error_mean": float(sim_metrics.get("state_error_mean", np.nan)),
                "state_error_max": float(sim_metrics.get("state_error_max", np.nan)),
                "state_fidelity_mean": float(sim_metrics.get("state_fidelity_mean", np.nan)),
                "state_fidelity_min": float(sim_metrics.get("state_fidelity_min", np.nan)),
                "duration_metric": float(sim_metrics.get("duration_metric", np.nan)),
                "pulse_power_metric": float(sim_metrics.get("pulse_power_metric", np.nan)),
                "amplitude_smoothness_metric": float(sim_metrics.get("amplitude_smoothness_metric", np.nan)),
                "robustness_objective": float(sim_metrics.get("robustness_objective", np.nan)),
            },
            "robustness": dict(robustness_info),
            "parameters": {
                "gates": self.sequence.serialize(),
                "time_parameters": self.sequence.serialize_time_parameters(),
                "durations": [float(g.duration) for g in self.sequence.gates],
                "time_grid_per_param": [
                    {
                        "param_id": p.param_id,
                        "group": p.group,
                        "raw": float(grid_final.raw[i]),
                        "snapped": float(grid_final.snapped[i]),
                        "ticks": int(grid_final.ticks[i]),
                    }
                    for i, p in enumerate(self.sequence.time_params)
                ],
            },
            "phase_decomposition": self.sequence.phase_decomposition(n_match=n_match),
            "optimizer": {
                "success": bool(best_result["success"]),
                "name": str(self.optimizer),
                "message": str(best_result["message"]),
                "nit": int(best_result["nit"]),
                "multistart": starts,
                "n_time_params_active": len(self.sequence.active_time_params()),
                "selected_start_index": int(best_result["start_index"]),
                "start_objectives": [float(r["fun"]) for r in sorted(start_results, key=lambda r: int(r["start_index"]))],
                "progress": {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "every": int(self.progress.get("every", 1)),
                    "live": bool(serial_live),
                    "enabled": bool(self.progress.get("enabled", False)),
                    "history_events": len(history),
                    "history_runs": len(history_by_run),
                    "global_best": {
                        "run_id": str(best_result["run_id"]),
                        "objective_total": float(best_result["fun"]),
                        "iteration": int(
                            min(
                                (
                                    int(event["iteration"])
                                    for event in history_by_run.get(str(best_result["run_id"]), [])
                                    if float(event["best_so_far"]["objective_total"]) <= float(best_result["fun"]) + 1e-15
                                ),
                                default=0,
                            )
                        ),
                    },
                },
            },
            "warm_start": None if self.warm_start is None else str(self.warm_start) if isinstance(self.warm_start, (str, Path)) else "inline",
        }

        if simulation.subspace_operator is not None:
            report = make_run_report(report, simulation.subspace_operator)

        return SynthesisResult(
            success=bool(best_result["success"]),
            objective=float(final_objective),
            sequence=self.sequence,
            simulation=simulation,
            report=report,
            history=history,
            history_by_run=history_by_run,
        )

    def explore_pareto(
        self,
        weight_sets: Sequence[MultiObjective | Mapping[str, Any]],
        *,
        target: Any | None = None,
        init_guess: str = "heuristic",
        multistart: int = 1,
        maxiter: int = 300,
    ) -> ParetoFrontResult:
        if not weight_sets:
            raise ValueError("explore_pareto requires at least one weight set.")

        original_objectives = self.objectives
        original_leakage_weight = self.leakage_weight
        original_sequence = copy.deepcopy(self.sequence)
        results: list[SynthesisResult] = []
        try:
            for weight_set in weight_sets:
                self.sequence = copy.deepcopy(original_sequence)
                self._apply_warm_start(self.warm_start)
                objective_cfg = _coerce_multi_objective(weight_set)
                assert objective_cfg is not None
                self.objectives = objective_cfg
                self.leakage_weight = float(self.leakage_penalty.weight) if self.leakage_penalty is not None else float(objective_cfg.leakage_weight)
                results.append(
                    self.fit(
                        target=target,
                        init_guess=init_guess,
                        multistart=multistart,
                        maxiter=maxiter,
                    )
                )
        finally:
            self.objectives = original_objectives
            self.leakage_weight = float(original_leakage_weight)
            self.sequence = original_sequence

        points = [
            {
                "fidelity_loss": float(result.report["objective"]["fidelity_term"]) + float(result.report["objective"]["leakage_term"]),
                "duration_metric": float(result.report["metrics"].get("duration_metric", 0.0)),
                "pulse_power_metric": float(result.report["metrics"].get("pulse_power_metric", 0.0)),
                "robustness_objective": float(result.report["metrics"].get("robustness_objective", 0.0)),
            }
            for result in results
        ]
        active_keys = ["fidelity_loss"]
        if any(float(point["duration_metric"]) > 0.0 for point in points):
            active_keys.append("duration_metric")
        if any(float(point["pulse_power_metric"]) > 0.0 for point in points):
            active_keys.append("pulse_power_metric")
        if any(np.isfinite(float(point["robustness_objective"])) and float(point["robustness_objective"]) > 0.0 for point in points):
            active_keys.append("robustness_objective")
        keep = _nondominated_indices(points, active_keys)
        return ParetoFrontResult(results=results, nondominated_indices=keep)

