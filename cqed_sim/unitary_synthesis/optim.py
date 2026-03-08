from __future__ import annotations

import copy
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .backends import SimulationResult, simulate_sequence
from .constraints import (
    TimeGridResult,
    enforce_slew_limit,
    evaluate_tone_spacing,
    piecewise_constant_samples,
    snap_times_to_grid,
)
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
from .sequence import (
    ConditionalPhaseSQR,
    Displacement,
    DriftPhaseModel,
    FreeEvolveCondPhase,
    GateSequence,
    QubitRotation,
    SNAP,
    SQR,
)
from .subspace import Subspace


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


def _default_progress() -> dict[str, Any]:
    return {"enabled": False, "every": 1, "live": False, "print_every": 10}


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
    event = ProgressEvent(
        run_id=str(run_id),
        iteration=int(iteration),
        timestamp=float(time.time()),
        objective_total=float(objective_total),
        objective_terms={
            "infidelity": float(sim_metrics.get("fidelity_loss", 0.0)),
            "leakage": float(sim_metrics.get("leakage_term", 0.0)),
            "time_reg": float(details.get("time_reg_term", 0.0)),
            "grid_penalty": float(details.get("grid_term", 0.0)),
            "constraint_penalty": float(constraints.get("total_penalty", 0.0)),
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

    sim_t0 = time.perf_counter()
    simulation = simulate_sequence(
        sequence=sequence,
        subspace=payload["subspace"],
        backend=payload["backend"],
        target_subspace=payload["target_subspace"],
        leakage_weight=float(payload["leakage_weight"]),
        gauge=str(payload["gauge"]),
        leakage_n_jobs=int(payload.get("leakage_n_jobs", 1)),
    )
    sim_elapsed = float(time.perf_counter() - sim_t0)

    active_times = sequence.get_time_vector(active_only=True)
    t0_ref = np.asarray(payload["t0_ref"], dtype=float)
    reg_term = 0.0
    if active_times.size:
        reg_term = float(payload["time_reg_weight"]) * float(np.sum((active_times - t0_ref) ** 2))

    smooth_term = 0.0
    durations = sequence.gate_durations()
    if float(payload["time_smooth_weight"]) > 0.0 and durations.size > 1:
        smooth_term = float(payload["time_smooth_weight"]) * float(np.sum(np.diff(durations) ** 2))

    total = (
        float(simulation.metrics.get("objective", np.inf))
        + reg_term
        + smooth_term
        + grid_term
        + float(c_eval["total_penalty"])
    )

    details = {
        "grid": grid,
        "grid_term": float(grid_term),
        "constraints": c_eval,
        "time_reg_term": float(reg_term),
        "time_smooth_term": float(smooth_term),
        "simulation_metrics": dict(simulation.metrics),
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
    return float(total), details, simulation if return_simulation else None


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
    method = "Powell" if grid_mode == "hard" else "L-BFGS-B"
    callback_state = {"count": 0}

    def _callback(xk: np.ndarray) -> None:
        callback_state["count"] += 1
        if callback_state["count"] % every != 0:
            return
        _emit(int(callback_state["count"]), np.asarray(xk, dtype=float))

    res = minimize(_objective, x_start, method=method, callback=_callback, options={"maxiter": maxiter})
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
        subspace: Subspace,
        backend: str = "ideal",
        gateset: list[str] | None = None,
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
        parallel: Mapping[str, Any] | None = None,
        progress: Mapping[str, Any] | None = None,
        seed: int = 0,
    ):
        self.subspace = subspace
        self.backend = backend
        self.gateset = gateset or ["QubitRotation", "SQR", "SNAP"]
        self.optimize_times = bool(optimize_times)
        self.time_bounds = time_bounds or {"default": (20e-9, 2000e-9)}
        self.time_policy = dict(time_policy or {})
        self.time_mode = time_mode
        self.time_groups = dict(time_groups or {})
        self.leakage_weight = float(leakage_weight)
        self.time_reg_weight = float(time_reg_weight)
        self.time_smooth_weight = float(time_smooth_weight)
        self.gauge = gauge
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.drift_model = self._make_drift_model(drift_config)
        self.include_conditional_phase_in_sqr = bool(include_conditional_phase_in_sqr)

        self.hardware_limits = _deep_update(_default_hardware_limits(), dict(hardware_limits or {}))
        self.time_grid = _deep_update(_default_time_grid(), dict(time_grid or {}))
        self.constraints = _deep_update(_default_constraints(), dict(constraints or {}))
        self.parallel = _deep_update(_default_parallel(), dict(parallel or {}))
        self.progress = _deep_update(_default_progress(), dict(progress or {}))

        if "dt" not in dict(time_grid or {}) and "dt" in self.hardware_limits:
            self.time_grid["dt"] = float(self.hardware_limits["dt"])

        self.sequence = self._build_initial_sequence()

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
        n_cav = self.subspace.full_dim // 2
        dur = self._default_duration()
        gates = []
        for i, raw_name in enumerate(self.gateset):
            name = self._normalize_gate_name(raw_name)
            bounds = self._time_bounds_for(name)
            kw = {
                "name": f"{name}_{i}",
                "duration": dur,
                "optimize_time": self.optimize_times,
                "time_bounds": bounds,
                "duration_ref": dur,
            }
            if name == "QubitRotation":
                gates.append(QubitRotation(theta=0.3, phi=0.0, **kw))
            elif name == "SQR":
                gates.append(
                    SQR(
                        theta_n=[0.1] * n_cav,
                        phi_n=[0.0] * n_cav,
                        tones=n_cav,
                        tone_freqs=[] ,
                        include_conditional_phase=self.include_conditional_phase_in_sqr,
                        drift_model=self.drift_model,
                        **kw,
                    )
                )
            elif name == "SNAP":
                gates.append(SNAP(phases=[0.0] * n_cav, **kw))
            elif name == "Displacement":
                gates.append(Displacement(alpha=0.0 + 0.0j, **kw))
            elif name == "ConditionalPhaseSQR":
                gates.append(ConditionalPhaseSQR(phases_n=[0.0] * n_cav, drift_model=self.drift_model, **kw))
            elif name == "FreeEvolveCondPhase":
                gates.append(FreeEvolveCondPhase(drift_model=self.drift_model, **kw))
            else:
                raise ValueError(f"Unsupported gate in gateset: {raw_name}")

        seq = GateSequence(gates=gates, n_cav=n_cav)
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
        target_sub: np.ndarray,
        t0_ref: np.ndarray,
        leakage_n_jobs: int,
    ) -> tuple[dict[str, Any], SimulationResult, float]:
        payload = {
            "param_size": int(self.sequence.get_parameter_vector().size),
            "subspace": self.subspace,
            "backend": self.backend,
            "target_subspace": target_sub,
            "leakage_weight": self.leakage_weight,
            "gauge": self.gauge,
            "t0_ref": np.asarray(t0_ref, dtype=float),
            "time_reg_weight": self.time_reg_weight,
            "time_smooth_weight": self.time_smooth_weight,
            "time_grid": self.time_grid,
            "hardware_limits": self.hardware_limits,
            "constraints": self.constraints,
            "leakage_n_jobs": int(leakage_n_jobs),
        }
        value, details, sim = _evaluate_objective_vector(best_x, self.sequence, payload, return_simulation=True)
        assert sim is not None
        return details, sim, float(value)

    def fit(
        self,
        target: np.ndarray,
        init_guess: str = "heuristic",
        multistart: int = 1,
        maxiter: int = 300,
    ) -> SynthesisResult:
        target_sub = self._target_subspace(target)

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

        common_payload = {
            "sequence_template": copy.deepcopy(self.sequence),
            "param_size": int(p0.size),
            "subspace": self.subspace,
            "backend": self.backend,
            "target_subspace": target_sub,
            "leakage_weight": self.leakage_weight,
            "gauge": self.gauge,
            "t0_ref": np.asarray(t0_ref, dtype=float),
            "time_reg_weight": self.time_reg_weight,
            "time_smooth_weight": self.time_smooth_weight,
            "time_grid": self.time_grid,
            "hardware_limits": self.hardware_limits,
            "constraints": self.constraints,
            "leakage_n_jobs": int(leakage_n_jobs),
            "maxiter": int(maxiter),
            "progress": self.progress,
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
            target_sub=target_sub,
            t0_ref=t0_ref,
            leakage_n_jobs=leakage_n_jobs,
        )

        n_match = None
        if isinstance(self.subspace.metadata, dict) and "n_match" in self.subspace.metadata:
            n_match = int(self.subspace.metadata["n_match"])

        grid_final: TimeGridResult = final_details["grid"]
        c_eval = final_details["constraints"]
        sim_metrics = final_details["simulation_metrics"]
        history_by_run = {
            str(row["run_id"]): sorted(list(row.get("history", [])), key=lambda event: int(event["iteration"]))
            for row in sorted(start_results, key=lambda item: int(item["start_index"]))
        }
        history = [event for run_id in history_by_run for event in history_by_run[run_id]]
        history.sort(key=lambda event: (str(event["run_id"]), int(event["iteration"]), float(event["timestamp"])))

        report = {
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
                "time_reg_term": float(final_details.get("time_reg_term", 0.0)),
                "time_smooth_term": float(final_details.get("time_smooth_term", 0.0)),
                "grid_term": float(final_details.get("grid_term", 0.0)),
                "hardware_penalty_term": float(c_eval.get("total_penalty", 0.0)),
                "hardware_breakdown": {
                    "amplitude_penalty": float(c_eval.get("amplitude_penalty", 0.0)),
                    "detuning_penalty": float(c_eval.get("detuning_penalty", 0.0)),
                    "slew_penalty": float(c_eval.get("slew_penalty", 0.0)),
                    "tone_penalty": float(c_eval.get("tone_penalty", 0.0)),
                },
            },
            "constraint_violations": {
                "amplitude_violations": int(c_eval.get("amplitude_violations", 0)),
                "detuning_violations": int(c_eval.get("detuning_violations", 0)),
                "slew_max_violation": float(c_eval.get("slew_max_violation", 0.0)),
                "tone_min_spacing": float(c_eval.get("tone_min_spacing", np.inf)),
                "tone_violations": int(c_eval.get("tone_violations", 0)),
            },
            "metrics": {
                "fidelity": float(sim_metrics.get("fidelity", np.nan)),
                "leakage_average": float(sim_metrics.get("leakage_average", np.nan)),
                "leakage_worst": float(sim_metrics.get("leakage_worst", np.nan)),
            },
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
        }

        return SynthesisResult(
            success=bool(best_result["success"]),
            objective=float(final_objective),
            sequence=self.sequence,
            simulation=simulation,
            report=report,
            history=history,
            history_by_run=history_by_run,
        )

