from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_block_indices, qubit_cavity_dims
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.ideal_gates import displacement_op, qubit_rotation_xy, sqr_op
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence import SequenceCompiler as RuntimeSequenceCompiler
from cqed_sim.sim import SimulationConfig as RuntimeSimulationConfig
from cqed_sim.sim import prepare_simulation, simulate_batch
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import hamiltonian_time_slices


def _sigmoid_stable(x: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(arr, -60.0, 60.0)))


def _as_complex_array(value: qt.Qobj | np.ndarray) -> np.ndarray:
    if isinstance(value, qt.Qobj):
        return np.asarray(value.full(), dtype=np.complex128)
    return np.asarray(value, dtype=np.complex128)


def _validate_unitary_matrix(matrix: np.ndarray, *, atol: float = 1.0e-8) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Primitive matrices must be square.")
    ident = np.eye(arr.shape[0], dtype=np.complex128)
    if np.linalg.norm(arr.conj().T @ arr - ident, ord="fro") > atol:
        raise ValueError("Primitive matrix is not unitary within tolerance.")
    return arr


def _coerce_frame_spec(frame: FrameSpec | Mapping[str, Any] | None) -> FrameSpec:
    if frame is None:
        return FrameSpec()
    if isinstance(frame, FrameSpec):
        return frame
    return FrameSpec(
        omega_c_frame=float(frame.get("omega_c_frame", frame.get("omega_s_frame", 0.0))),
        omega_q_frame=float(frame.get("omega_q_frame", 0.0)),
        omega_r_frame=float(frame.get("omega_r_frame", 0.0)),
    )


def _runtime_simulation_config(settings: Mapping[str, Any]) -> RuntimeSimulationConfig:
    raw = settings.get("simulation_config")
    if isinstance(raw, RuntimeSimulationConfig):
        return raw

    frame = _coerce_frame_spec(settings.get("frame"))
    if isinstance(raw, Mapping):
        data = dict(raw)
        if "frame" in data:
            frame = _coerce_frame_spec(data.pop("frame"))
        return RuntimeSimulationConfig(
            frame=frame,
            atol=float(data.get("atol", 1.0e-8)),
            rtol=float(data.get("rtol", 1.0e-7)),
            max_step=None if data.get("max_step") is None else float(data["max_step"]),
            store_states=bool(data.get("store_states", False)),
            backend=data.get("backend"),
        )

    return RuntimeSimulationConfig(
        frame=frame,
        atol=float(settings.get("atol", 1.0e-8)),
        rtol=float(settings.get("rtol", 1.0e-7)),
        max_step=None if settings.get("max_step") is None else float(settings["max_step"]),
        store_states=bool(settings.get("store_states", False)),
        backend=settings.get("runtime_backend"),
    )


def _solver_options(cfg: RuntimeSimulationConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "atol": float(cfg.atol),
        "rtol": float(cfg.rtol),
        "store_final_state": True,
    }
    if cfg.max_step is not None:
        options["max_step"] = float(cfg.max_step)
    return options


def _coerce_state_qobj(state: qt.Qobj | np.ndarray, *, subsystem_dims: list[int] | None = None) -> qt.Qobj:
    if isinstance(state, qt.Qobj):
        return state
    arr = np.asarray(state, dtype=np.complex128)
    if arr.ndim == 1:
        if subsystem_dims is None:
            return qt.Qobj(arr.reshape(-1))
        return qt.Qobj(arr.reshape(-1), dims=[list(subsystem_dims), [1] * len(subsystem_dims)])
    if arr.ndim == 2:
        if subsystem_dims is None:
            return qt.Qobj(arr)
        return qt.Qobj(arr, dims=[list(subsystem_dims), list(subsystem_dims)])
    raise ValueError("States must be vectors or density matrices.")


def _apply_operator_to_state(operator: np.ndarray, state: qt.Qobj) -> qt.Qobj:
    dims = state.dims
    op = qt.Qobj(np.asarray(operator, dtype=np.complex128), dims=[dims[0], dims[0]])
    if state.isoper:
        return op * state * op.dag()
    return op * state


def _waveform_payload(result: Any) -> tuple[list[Pulse], dict[str, Any], dict[str, Any]]:
    if isinstance(result, dict):
        pulses = list(result.get("pulses", []))
        drive_ops = dict(result.get("drive_ops", {}))
        meta = dict(result.get("meta", {}))
    elif isinstance(result, tuple):
        if len(result) == 2:
            pulses, drive_ops = result
            meta = {}
        elif len(result) == 3:
            pulses, drive_ops, meta = result
        else:
            raise ValueError("Waveform primitives must return pulses, (pulses, drive_ops), or (pulses, drive_ops, meta).")
        pulses = list(pulses)
        drive_ops = dict(drive_ops)
        meta = dict(meta)
    else:
        pulses = list(result)
        drive_ops = {}
        meta = {}
    if not pulses:
        raise ValueError("Waveform primitives must return at least one Pulse.")
    if not drive_ops:
        drive_ops = {pulse.channel: pulse.channel for pulse in pulses}
    return pulses, drive_ops, meta


def _compile_waveform(
    pulses: list[Pulse],
    *,
    dt: float,
    hardware: Mapping[str, Any] | None,
    crosstalk: Mapping[str, Any] | None,
    t_end: float | None,
) -> Any:
    compiler = RuntimeSequenceCompiler(
        dt=float(dt),
        hardware=None if hardware is None else dict(hardware),
        crosstalk_matrix=None if crosstalk is None else dict(crosstalk),
        enable_cache=True,
    )
    if t_end is None:
        t_end = max(float(pulse.t1) for pulse in pulses) + float(dt)
    return compiler.compile(pulses, t_end=float(t_end))


def _final_unitary_from_compiled(
    model: Any,
    compiled: Any,
    drive_ops: Mapping[str, Any],
    *,
    frame: FrameSpec,
    config: RuntimeSimulationConfig,
) -> np.ndarray:
    hamiltonian = hamiltonian_time_slices(model, compiled, dict(drive_ops), frame=frame)
    propagators = qt.propagator(
        hamiltonian,
        compiled.tlist,
        options=_solver_options(config),
        tlist=compiled.tlist,
    )
    final = propagators[-1] if isinstance(propagators, list) else propagators
    return np.asarray(final.full(), dtype=np.complex128)


def _simulate_waveform_states(
    model: Any,
    compiled: Any,
    drive_ops: Mapping[str, Any],
    states: list[qt.Qobj],
    *,
    config: RuntimeSimulationConfig,
    c_ops: list[qt.Qobj] | None,
    noise: NoiseSpec | None,
    max_workers: int,
) -> list[qt.Qobj]:
    session = prepare_simulation(
        model,
        compiled,
        dict(drive_ops),
        config=config,
        c_ops=c_ops,
        noise=noise,
        e_ops={},
    )
    results = simulate_batch(session, states, max_workers=max(1, int(max_workers)))
    return [row.final_state for row in results]


@dataclass
class GateTimeParam:
    """Bounded, optionally shared gate-time parameter.

    The physical duration is constrained to [t_min, t_max] by a stable sigmoid map.
    Parameters can be grouped (shared across gate instances) and frozen.
    """

    param_id: str
    gate_type: str
    group: str
    t_min: float
    t_max: float
    t_init: float
    value: float
    optimize: bool = True
    frozen: bool = False
    gate_names: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.t_min <= 0.0 or self.t_max <= 0.0:
            raise ValueError("Time bounds must be positive.")
        if self.t_min >= self.t_max:
            raise ValueError("Time bounds must satisfy t_min < t_max.")
        self.t_init = self._clip(self.t_init)
        self.value = self._clip(self.value)

    @property
    def active(self) -> bool:
        return bool(self.optimize) and not bool(self.frozen)

    def _clip(self, t: float) -> float:
        return float(np.clip(float(t), self.t_min, self.t_max))

    def map_raw(self, raw: np.ndarray | float) -> np.ndarray:
        s = _sigmoid_stable(raw)
        return self.t_min + (self.t_max - self.t_min) * s

    def inverse_map(self, t: np.ndarray | float) -> np.ndarray:
        vals = np.asarray(t, dtype=float)
        eps = 1e-15
        y = np.clip((vals - self.t_min) / (self.t_max - self.t_min), eps, 1.0 - eps)
        return np.log(y / (1.0 - y))

    def grad_raw(self, raw: np.ndarray | float) -> np.ndarray:
        s = _sigmoid_stable(raw)
        return (self.t_max - self.t_min) * s * (1.0 - s)

    def set_from_raw(self, raw: float) -> float:
        self.value = self._clip(float(self.map_raw(raw)))
        return self.value

    def set_value(self, t: float) -> float:
        self.value = self._clip(t)
        return self.value

    def raw_value(self) -> float:
        return float(self.inverse_map(self.value))

    def to_record(self) -> dict[str, Any]:
        return {
            "param_id": self.param_id,
            "gate_type": self.gate_type,
            "group": self.group,
            "bounds": (self.t_min, self.t_max),
            "init": float(self.t_init),
            "value": float(self.value),
            "optimize": bool(self.optimize),
            "frozen": bool(self.frozen),
            "active": bool(self.active),
            "gate_names": list(self.gate_names),
        }


@dataclass(frozen=True)
class DriftPhaseModel:
    """Diagonal drift model for qubit+cavity in the rotating frame.

    Convention:
    - Joint basis ordering is |q>_qubit tensor |n>_cavity with q in {g,e}.
    - Default frame is rotating at (omega_c, omega_q), so bare terms are
      removed and only residual offsets (delta_c, delta_q) are retained.
    - Dispersive terms use the excitation-projector convention
      ``+chi * n * |e><e|`` and ``+chi2 * n(n-1) * |e><e|``.

    Energies used for phase accumulation:
            E_g,n = delta_c*n + Kerr(n)
            E_e,n = delta_c*n + delta_q + (chi*n + chi2*n(n-1)) + Kerr(n)

      Kerr(n) = 0.5*K*n(n-1) + (K2/6)*n(n-1)(n-2)

    All coefficients are angular frequencies in rad/s and duration is in seconds.
    """

    chi: float = 0.0
    chi2: float = 0.0
    kerr: float = 0.0
    kerr2: float = 0.0
    delta_c: float = 0.0
    delta_q: float = 0.0
    frame: str = "rotating_omega_c_omega_q"


@dataclass(frozen=True)
class DriftPhaseTable:
    n: np.ndarray
    e_g: np.ndarray
    e_e: np.ndarray
    disp: np.ndarray
    kerr: np.ndarray
    frame_offset_n: np.ndarray
    frame_offset_q_g: np.ndarray
    frame_offset_q_e: np.ndarray
    phase_g: np.ndarray
    phase_e: np.ndarray
    phase_delta: np.ndarray
    disp_phase_g: np.ndarray
    disp_phase_e: np.ndarray
    kerr_phase: np.ndarray
    frame_offset_phase_n: np.ndarray
    frame_offset_phase_q_g: np.ndarray
    frame_offset_phase_q_e: np.ndarray


def _falling_factorial_array(n: np.ndarray, order: int) -> np.ndarray:
    out = np.ones_like(n, dtype=float)
    for k in range(order):
        out *= n - float(k)
    return out


def drift_phase_table(n_cav: int, duration: float, model: DriftPhaseModel) -> DriftPhaseTable:
    n = np.arange(int(n_cav), dtype=float)
    disp = model.chi * n + model.chi2 * _falling_factorial_array(n, 2)
    kerr = 0.5 * model.kerr * n * (n - 1.0) + (model.kerr2 / 6.0) * n * (n - 1.0) * (n - 2.0)
    frame_offset_n = model.delta_c * n
    frame_offset_q_g = np.zeros(n.shape, dtype=float)
    frame_offset_q_e = np.full(n.shape, model.delta_q, dtype=float)
    e_g = frame_offset_n + frame_offset_q_g + kerr
    e_e = frame_offset_n + frame_offset_q_e + disp + kerr

    t = float(duration)
    phase_g = e_g * t
    phase_e = e_e * t
    return DriftPhaseTable(
        n=n,
        e_g=e_g,
        e_e=e_e,
        disp=disp,
        kerr=kerr,
        frame_offset_n=frame_offset_n,
        frame_offset_q_g=frame_offset_q_g,
        frame_offset_q_e=frame_offset_q_e,
        phase_g=phase_g,
        phase_e=phase_e,
        phase_delta=phase_e - phase_g,
        disp_phase_g=np.zeros(n.shape, dtype=float),
        disp_phase_e=(disp) * t,
        kerr_phase=kerr * t,
        frame_offset_phase_n=frame_offset_n * t,
        frame_offset_phase_q_g=frame_offset_q_g * t,
        frame_offset_phase_q_e=frame_offset_q_e * t,
    )


def drift_phase_unitary(n_cav: int, duration: float, model: DriftPhaseModel) -> qt.Qobj:
    table = drift_phase_table(n_cav=n_cav, duration=duration, model=model)
    diag = np.zeros(2 * n_cav, dtype=np.complex128)
    for i in range(n_cav):
        g_idx, e_idx = qubit_cavity_block_indices(n_cav, i)
        diag[g_idx] = np.exp(-1j * table.phase_g[i])
        diag[e_idx] = np.exp(-1j * table.phase_e[i])
    return qt.Qobj(np.diag(diag), dims=qubit_cavity_dims(2, n_cav))


def drift_hamiltonian_qobj(n_cav: int, model: DriftPhaseModel) -> qt.Qobj:
    """Return the diagonal drift Hamiltonian in the synthesis basis."""
    table = drift_phase_table(n_cav=n_cav, duration=1.0, model=model)
    diag = np.zeros(2 * n_cav, dtype=np.complex128)
    for i in range(n_cav):
        g_idx, e_idx = qubit_cavity_block_indices(n_cav, i)
        diag[g_idx] = table.e_g[i]
        diag[e_idx] = table.e_e[i]
    return qt.Qobj(np.diag(diag), dims=qubit_cavity_dims(2, n_cav))


def drift_phase_from_hamiltonian(n_cav: int, duration: float, model: DriftPhaseModel) -> qt.Qobj:
    """Pulse-backend drift-only propagation under exp(-i H0 t)."""
    h0 = drift_hamiltonian_qobj(n_cav=n_cav, model=model)
    return (-1j * float(duration) * h0).expm()


def drift_phase_report_row(
    *,
    gate_name: str,
    gate_type: str,
    enabled: bool,
    duration: float,
    model: DriftPhaseModel,
    table: DriftPhaseTable,
    drive_relative_phase: list[float] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "gate_name": gate_name,
        "gate_type": gate_type,
        "enabled": bool(enabled),
        "duration": float(duration),
        "frame": model.frame,
        "n": table.n.astype(int).tolist(),
        "phi_g": table.phase_g.tolist(),
        "phi_e": table.phase_e.tolist(),
        "delta_phi": table.phase_delta.tolist(),
        "contributions": {
            "chi_chi2": {
                "g": table.disp_phase_g.tolist(),
                "e": table.disp_phase_e.tolist(),
            },
            "kerr_kerr2": {
                "common": table.kerr_phase.tolist(),
            },
            "frame_offsets": {
                "cavity_common": table.frame_offset_phase_n.tolist(),
                "qubit_g": table.frame_offset_phase_q_g.tolist(),
                "qubit_e": table.frame_offset_phase_q_e.tolist(),
            },
        },
    }
    if drive_relative_phase is not None:
        row["drive_relative_phase"] = list(drive_relative_phase)
    return row


@dataclass
class GateBase:
    name: str
    duration: float
    optimize_time: bool = True
    time_bounds: tuple[float, float] | None = None
    duration_ref: float | None = None
    time_group: str | None = None
    time_policy_locked: bool = False

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError("Gate duration must be positive.")
        if self.duration_ref is None:
            self.duration_ref = float(self.duration)

    @property
    def type(self) -> str:
        return self.__class__.__name__

    def to_record(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "name": self.name,
            "duration": self.duration,
            "optimize_time": self.optimize_time,
            "time_bounds": self.time_bounds,
            "time_group": self.time_group,
            "time_policy_locked": self.time_policy_locked,
        }

    def parameter_names(self, n_cav: int) -> list[str]:
        return []

    def get_parameters(self, n_cav: int) -> np.ndarray:
        return np.asarray([], dtype=float)

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        if params.size != 0:
            raise ValueError("No parameters expected for this gate.")

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        return [(-np.inf, np.inf) for _ in self.parameter_names(n_cav)]

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        raise NotImplementedError

    def pulse_unitary(self, n_cav: int, **kwargs: Any) -> qt.Qobj:
        return self.ideal_unitary(n_cav, scale_by_time=True, **kwargs)

    def phase_decomposition(self, n_cav: int, n_match: int | None = None) -> dict[str, Any] | None:
        return None

    def propagate_states(
        self,
        states: list[qt.Qobj],
        n_cav: int,
        *,
        backend: str = "ideal",
        **kwargs: Any,
    ) -> list[qt.Qobj]:
        if backend == "pulse":
            op = self.pulse_unitary(n_cav, **kwargs)
        elif backend == "ideal":
            op = self.ideal_unitary(n_cav, scale_by_time=False, **kwargs)
        else:
            raise ValueError("backend must be 'ideal' or 'pulse'.")
        op_arr = _as_complex_array(op)
        return [_apply_operator_to_state(op_arr, state) for state in states]


@dataclass
class QubitRotation(GateBase):
    theta: float = 0.0
    phi: float = 0.0

    def parameter_names(self, n_cav: int) -> list[str]:
        return ["theta", "phi"]

    def get_parameters(self, n_cav: int) -> np.ndarray:
        return np.asarray([self.theta, self.phi], dtype=float)

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        self.theta = float(params[0])
        self.phi = float(params[1])

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        return [(-2.0 * np.pi, 2.0 * np.pi), (-np.pi, np.pi)]

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        theta = self.theta
        if scale_by_time:
            theta = theta * float(self.duration / self.duration_ref)
        return qt.tensor(qubit_rotation_xy(theta, self.phi), qt.qeye(n_cav))


@dataclass
class SQR(GateBase):
    theta_n: list[float] = field(default_factory=list)
    phi_n: list[float] = field(default_factory=list)
    tones: int | None = None
    tone_freqs: list[float] = field(default_factory=list)
    include_conditional_phase: bool = False
    drift_model: DriftPhaseModel = field(default_factory=DriftPhaseModel)

    def to_record(self) -> dict[str, Any]:
        row = super().to_record()
        row["tone_freqs"] = [float(x) for x in self.tone_freqs]
        row["tones"] = self.tones
        return row

    def parameter_names(self, n_cav: int) -> list[str]:
        return [f"theta_{i}" for i in range(n_cav)] + [f"phi_{i}" for i in range(n_cav)]

    def _padded(self, n_cav: int) -> tuple[np.ndarray, np.ndarray]:
        theta = np.zeros(n_cav, dtype=float)
        phi = np.zeros(n_cav, dtype=float)
        for i, val in enumerate(self.theta_n[:n_cav]):
            theta[i] = float(val)
        for i, val in enumerate(self.phi_n[:n_cav]):
            phi[i] = float(val)
        return theta, phi

    def get_parameters(self, n_cav: int) -> np.ndarray:
        theta, phi = self._padded(n_cav)
        return np.concatenate([theta, phi])

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        theta = params[:n_cav]
        phi = params[n_cav : 2 * n_cav]
        self.theta_n = [float(x) for x in theta]
        self.phi_n = [float(x) for x in phi]

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        return [(-2.0 * np.pi, 2.0 * np.pi) for _ in range(n_cav)] + [(-np.pi, np.pi) for _ in range(n_cav)]

    def _drive_unitary(self, n_cav: int, scale_by_time: bool) -> qt.Qobj:
        theta, phi = self._padded(n_cav)
        if scale_by_time:
            theta = theta * float(self.duration / self.duration_ref)
        return sqr_op(theta, phi)

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        drive = self._drive_unitary(n_cav, scale_by_time=scale_by_time)
        if not self.include_conditional_phase:
            return drive
        return drift_phase_unitary(n_cav=n_cav, duration=self.duration, model=self.drift_model) * drive

    def pulse_unitary(self, n_cav: int, **kwargs: Any) -> qt.Qobj:
        drive = self._drive_unitary(n_cav=n_cav, scale_by_time=True)
        if not self.include_conditional_phase:
            return drive
        return drift_phase_from_hamiltonian(n_cav=n_cav, duration=self.duration, model=self.drift_model) * drive

    def phase_decomposition(self, n_cav: int, n_match: int | None = None) -> dict[str, Any] | None:
        max_n = n_cav - 1 if n_match is None else min(int(n_match), n_cav - 1)
        table = drift_phase_table(n_cav=max_n + 1, duration=self.duration, model=self.drift_model)
        return drift_phase_report_row(
            gate_name=self.name,
            gate_type=self.type,
            enabled=self.include_conditional_phase,
            duration=self.duration,
            model=self.drift_model,
            table=table,
        )


@dataclass
class SNAP(GateBase):
    phases: list[float] = field(default_factory=list)

    def parameter_names(self, n_cav: int) -> list[str]:
        return [f"phase_{i}" for i in range(n_cav)]

    def get_parameters(self, n_cav: int) -> np.ndarray:
        out = np.zeros(n_cav, dtype=float)
        for i, val in enumerate(self.phases[:n_cav]):
            out[i] = float(val)
        return out

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        self.phases = [float(x) for x in params[:n_cav]]

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        return [(-np.pi, np.pi) for _ in range(n_cav)]

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        phases = self.get_parameters(n_cav)
        if scale_by_time:
            phases = phases * float(self.duration / self.duration_ref)
        op_c = qt.Qobj(np.diag(np.exp(1j * phases)), dims=[[n_cav], [n_cav]])
        return qt.tensor(qt.qeye(2), op_c)


@dataclass
class Displacement(GateBase):
    alpha: complex = 0.0 + 0.0j

    def parameter_names(self, n_cav: int) -> list[str]:
        return ["alpha_re", "alpha_im"]

    def get_parameters(self, n_cav: int) -> np.ndarray:
        return np.asarray([self.alpha.real, self.alpha.imag], dtype=float)

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        self.alpha = complex(float(params[0]), float(params[1]))

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        return [(-2.0, 2.0), (-2.0, 2.0)]

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        alpha = self.alpha
        if scale_by_time:
            alpha = alpha * float(self.duration / self.duration_ref)
        return qt.tensor(qt.qeye(2), displacement_op(n_cav, alpha))


@dataclass
class ConditionalPhaseSQR(GateBase):
    phases_n: list[float] = field(default_factory=list)
    drift_model: DriftPhaseModel = field(default_factory=DriftPhaseModel)
    include_drift: bool = True

    def parameter_names(self, n_cav: int) -> list[str]:
        return [f"cp_phase_{i}" for i in range(n_cav)]

    def get_parameters(self, n_cav: int) -> np.ndarray:
        out = np.zeros(n_cav, dtype=float)
        for i, val in enumerate(self.phases_n[:n_cav]):
            out[i] = float(val)
        return out

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        self.phases_n = [float(x) for x in params[:n_cav]]

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        return [(-2.0 * np.pi, 2.0 * np.pi) for _ in range(n_cav)]

    def _drive_unitary(self, n_cav: int, scale_by_time: bool) -> qt.Qobj:
        phases = self.get_parameters(n_cav)
        if scale_by_time:
            phases = phases * float(self.duration / self.duration_ref)
        full = np.zeros((2 * n_cav, 2 * n_cav), dtype=np.complex128)
        for n in range(n_cav):
            p = phases[n]
            block = np.diag([np.exp(-0.5j * p), np.exp(0.5j * p)])
            idx = qubit_cavity_block_indices(n_cav, n)
            full[np.ix_(idx, idx)] = block
        return qt.Qobj(full, dims=qubit_cavity_dims(2, n_cav))

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        drive = self._drive_unitary(n_cav=n_cav, scale_by_time=scale_by_time)
        if not self.include_drift:
            return drive
        drift = drift_phase_unitary(n_cav=n_cav, duration=self.duration, model=self.drift_model)
        return drift * drive

    def pulse_unitary(self, n_cav: int, **kwargs: Any) -> qt.Qobj:
        drive = self._drive_unitary(n_cav=n_cav, scale_by_time=True)
        if not self.include_drift:
            return drive
        drift = drift_phase_from_hamiltonian(n_cav=n_cav, duration=self.duration, model=self.drift_model)
        return drift * drive

    def phase_decomposition(self, n_cav: int, n_match: int | None = None) -> dict[str, Any] | None:
        max_n = n_cav - 1 if n_match is None else min(int(n_match), n_cav - 1)
        table = drift_phase_table(n_cav=max_n + 1, duration=self.duration, model=self.drift_model)
        phases = self.get_parameters(max_n + 1)
        return drift_phase_report_row(
            gate_name=self.name,
            gate_type=self.type,
            enabled=self.include_drift,
            duration=self.duration,
            model=self.drift_model,
            table=table,
            drive_relative_phase=phases.tolist(),
        )


@dataclass
class FreeEvolveCondPhase(GateBase):
    """Wait-only free evolution under the shared drift Hamiltonian."""

    drift_model: DriftPhaseModel = field(default_factory=DriftPhaseModel)

    @property
    def wait_time(self) -> float:
        return float(self.duration)

    @wait_time.setter
    def wait_time(self, value: float) -> None:
        self.duration = float(value)

    def to_record(self) -> dict[str, Any]:
        row = super().to_record()
        row["wait_time"] = float(self.wait_time)
        return row

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **_: Any) -> qt.Qobj:
        wait_time = self.wait_time
        if scale_by_time:
            wait_time = wait_time * float(self.duration / self.duration_ref)
        return drift_phase_unitary(n_cav=n_cav, duration=wait_time, model=self.drift_model)

    def pulse_unitary(self, n_cav: int, **kwargs: Any) -> qt.Qobj:
        return drift_phase_from_hamiltonian(n_cav=n_cav, duration=self.wait_time, model=self.drift_model)

    def phase_decomposition(self, n_cav: int, n_match: int | None = None) -> dict[str, Any] | None:
        max_n = n_cav - 1 if n_match is None else min(int(n_match), n_cav - 1)
        table = drift_phase_table(n_cav=max_n + 1, duration=self.wait_time, model=self.drift_model)
        row = drift_phase_report_row(
            gate_name=self.name,
            gate_type=self.type,
            enabled=True,
            duration=self.wait_time,
            model=self.drift_model,
            table=table,
        )
        row["wait_time"] = float(self.wait_time)
        return row


@dataclass
class PrimitiveGate(GateBase):
    """Generic synthesis primitive backed by either a matrix or a waveform generator."""

    matrix: np.ndarray | Callable[[dict[str, Any], Any | None], np.ndarray] | None = None
    waveform: Callable[[dict[str, Any], Any | None], Any] | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    parameter_bounds: dict[str, Any] = field(default_factory=dict)
    hilbert_dim: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _operator_cache: dict[tuple[Any, ...], np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        if (self.matrix is None) == (self.waveform is None):
            raise ValueError("PrimitiveGate requires exactly one of matrix or waveform.")
        if "duration" in self.parameters:
            self.duration = float(self.parameters["duration"])
            if self.duration_ref is None:
                self.duration_ref = float(self.duration)
        if self.time_bounds is None and "duration" in self.parameter_bounds:
            bounds = self.parameter_bounds["duration"]
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                self.time_bounds = (float(bounds[0]), float(bounds[1]))
        if isinstance(self.matrix, np.ndarray):
            validated = _validate_unitary_matrix(self.matrix)
            self.matrix = validated
            if self.hilbert_dim is None:
                self.hilbert_dim = int(validated.shape[0])

    @property
    def mode(self) -> str:
        return "waveform" if self.waveform is not None else "matrix"

    def to_record(self) -> dict[str, Any]:
        row = super().to_record()
        row.update(
            {
                "mode": self.mode,
                "hilbert_dim": self.hilbert_dim,
                "metadata": dict(self.metadata),
            }
        )
        return row

    def _restore_parameter_container(self, raw: Any, data: np.ndarray) -> Any:
        if isinstance(raw, np.ndarray):
            return np.asarray(data, dtype=data.dtype)
        if isinstance(raw, tuple):
            return tuple(data.tolist())
        if isinstance(raw, list):
            return data.tolist()
        return data.reshape(()).item()

    def parameter_names(self, n_cav: int) -> list[str]:
        names: list[str] = []
        for key, raw in self.parameters.items():
            if key == "duration":
                continue
            arr = np.asarray(raw)
            if arr.ndim == 0:
                if np.iscomplexobj(arr):
                    names.extend([f"{key}_re", f"{key}_im"])
                else:
                    names.append(str(key))
                continue
            if arr.ndim != 1:
                raise ValueError("PrimitiveGate parameters only support scalars and 1D arrays.")
            for idx in range(arr.shape[0]):
                if np.iscomplexobj(arr):
                    names.extend([f"{key}_{idx}_re", f"{key}_{idx}_im"])
                else:
                    names.append(f"{key}_{idx}")
        return names

    def get_parameters(self, n_cav: int) -> np.ndarray:
        values: list[float] = []
        for key, raw in self.parameters.items():
            if key == "duration":
                continue
            arr = np.asarray(raw)
            if arr.ndim == 0:
                if np.iscomplexobj(arr):
                    values.extend([float(arr.real), float(arr.imag)])
                else:
                    values.append(float(arr))
                continue
            flat = arr.reshape(-1)
            if np.iscomplexobj(arr):
                for val in flat:
                    values.extend([float(np.real(val)), float(np.imag(val))])
            else:
                values.extend(float(val) for val in flat)
        return np.asarray(values, dtype=float)

    def set_parameters(self, params: np.ndarray, n_cav: int) -> None:
        params = np.asarray(params, dtype=float)
        offset = 0
        updated: dict[str, Any] = {}
        for key, raw in self.parameters.items():
            if key == "duration":
                updated[key] = float(self.duration)
                continue
            arr = np.asarray(raw)
            if arr.ndim == 0:
                if np.iscomplexobj(arr):
                    updated[key] = complex(float(params[offset]), float(params[offset + 1]))
                    offset += 2
                else:
                    updated[key] = float(params[offset])
                    offset += 1
                continue
            if arr.ndim != 1:
                raise ValueError("PrimitiveGate parameters only support scalars and 1D arrays.")
            if np.iscomplexobj(arr):
                data = np.zeros(arr.shape, dtype=np.complex128)
                for idx in range(arr.shape[0]):
                    data[idx] = complex(float(params[offset]), float(params[offset + 1]))
                    offset += 2
            else:
                data = np.asarray(params[offset : offset + arr.shape[0]], dtype=float)
                offset += arr.shape[0]
            updated[key] = self._restore_parameter_container(raw, np.asarray(data))
        if offset != params.size:
            raise ValueError("Parameter vector length mismatch.")
        self.parameters.update(updated)
        self.parameters["duration"] = float(self.duration)

    def parameter_bounds_vector(self, n_cav: int) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for key, raw in self.parameters.items():
            if key == "duration":
                continue
            arr = np.asarray(raw)
            count = int(arr.size) * (2 if np.iscomplexobj(arr) else 1)
            if arr.ndim == 0:
                count = 2 if np.iscomplexobj(arr) else 1
            spec = self.parameter_bounds.get(key, (-np.inf, np.inf))
            if isinstance(spec, (list, tuple)) and len(spec) == count and count > 1 and isinstance(spec[0], (list, tuple)):
                bounds.extend([(float(row[0]), float(row[1])) for row in spec])
            else:
                if not isinstance(spec, (list, tuple)) or len(spec) != 2:
                    raise ValueError(f"PrimitiveGate parameter_bounds['{key}'] must be a 2-tuple or a per-entry tuple list.")
                bounds.extend([(float(spec[0]), float(spec[1])) for _ in range(count)])
        return bounds

    def resolved_dimension(self, model: Any | None = None, system: Any | None = None) -> int:
        runtime_model = model if model is not None else (None if system is None else system.runtime_model())
        if self.hilbert_dim is not None:
            return int(self.hilbert_dim)
        if isinstance(self.matrix, np.ndarray):
            self.hilbert_dim = int(self.matrix.shape[0])
            return int(self.hilbert_dim)
        if self.matrix is not None and callable(self.matrix):
            matrix = _validate_unitary_matrix(self.matrix(self.runtime_parameters(), runtime_model))
            self.hilbert_dim = int(matrix.shape[0])
            return int(self.hilbert_dim)
        if system is not None:
            dim = system.hilbert_dimension(primitive=self)
            if dim is not None:
                self.hilbert_dim = int(dim)
                return int(self.hilbert_dim)
        if runtime_model is not None and hasattr(runtime_model, "subsystem_dims"):
            self.hilbert_dim = int(np.prod(tuple(int(dim) for dim in runtime_model.subsystem_dims)))
            return int(self.hilbert_dim)
        raise ValueError("Could not infer PrimitiveGate Hilbert dimension. Provide hilbert_dim or a model.")

    def runtime_parameters(self) -> dict[str, Any]:
        out = dict(self.parameters)
        out["duration"] = float(self.duration)
        return out

    def _operator_cache_key(self, model: Any | None, settings: Mapping[str, Any]) -> tuple[Any, ...]:
        frame = _coerce_frame_spec(settings.get("frame"))
        dt = float(settings.get("dt", 1.0e-9))
        params = tuple(np.round(self.get_parameters(-1), 12).tolist())
        return (
            self.mode,
            id(model),
            params,
            round(float(self.duration), 15),
            round(float(frame.omega_c_frame), 9),
            round(float(frame.omega_q_frame), 9),
            round(float(frame.omega_r_frame), 9),
            round(dt, 15),
        )

    def _matrix_operator(self, model: Any | None = None, system: Any | None = None) -> np.ndarray:
        runtime_model = model if model is not None else (None if system is None else system.runtime_model())
        if callable(self.matrix):
            return _validate_unitary_matrix(self.matrix(self.runtime_parameters(), runtime_model))
        if self.matrix is None:
            raise ValueError("PrimitiveGate is not matrix defined.")
        return _validate_unitary_matrix(self.matrix)

    def _waveform_unitary(self, *, model: Any | None, settings: Mapping[str, Any]) -> np.ndarray:
        if self.waveform is None:
            raise ValueError("PrimitiveGate is not waveform defined.")
        system = settings.get("system")
        if system is not None:
            return np.asarray(system.simulate_primitive_unitary(self, settings=settings), dtype=np.complex128)
        if model is None:
            raise ValueError("Waveform-defined primitives require a model.")
        cache_key = self._operator_cache_key(model, settings)
        cached = self._operator_cache.get(cache_key)
        if cached is not None:
            return cached

        pulses, drive_ops, meta = _waveform_payload(self.waveform(self.runtime_parameters(), model))
        dt = float(settings.get("dt", 1.0e-9))
        compiled = _compile_waveform(
            pulses,
            dt=dt,
            hardware=settings.get("compiler_hardware"),
            crosstalk=settings.get("compiler_crosstalk"),
            t_end=meta.get("t_end"),
        )
        cfg = _runtime_simulation_config(settings)
        if settings.get("c_ops") or settings.get("noise"):
            raise ValueError("Waveform primitive unitary extraction only supports closed-system simulation. Use state mappings for dissipative synthesis.")
        operator = _final_unitary_from_compiled(
            model,
            compiled,
            drive_ops,
            frame=cfg.frame,
            config=cfg,
        )
        self._operator_cache[cache_key] = operator
        if self.hilbert_dim is None:
            self.hilbert_dim = int(operator.shape[0])
        return operator

    def ideal_unitary(self, n_cav: int, scale_by_time: bool = False, **kwargs: Any) -> qt.Qobj:
        model = kwargs.get("model")
        system = kwargs.get("system")
        operator = self._matrix_operator(model=model, system=system) if self.mode == "matrix" else self._waveform_unitary(model=model, settings=kwargs)
        dim = int(operator.shape[0])
        return qt.Qobj(operator, dims=[[dim], [dim]])

    def pulse_unitary(self, n_cav: int, **kwargs: Any) -> qt.Qobj:
        return self.ideal_unitary(n_cav=n_cav, scale_by_time=False, **kwargs)

    def propagate_states(
        self,
        states: list[qt.Qobj],
        n_cav: int,
        *,
        backend: str = "ideal",
        **kwargs: Any,
    ) -> list[qt.Qobj]:
        if self.mode == "matrix":
            operator = self._matrix_operator(model=kwargs.get("model"), system=kwargs.get("system"))
            return [_apply_operator_to_state(operator, state) for state in states]

        system = kwargs.get("system")
        if system is not None:
            return list(system.simulate_primitive_states(self, states, settings=kwargs))

        model = kwargs.get("model")
        if model is None:
            raise ValueError("Waveform-defined primitives require a model.")
        if not kwargs.get("c_ops") and not kwargs.get("noise"):
            operator = self._waveform_unitary(model=model, settings=kwargs)
            return [_apply_operator_to_state(operator, state) for state in states]

        pulses, drive_ops, meta = _waveform_payload(self.waveform(self.runtime_parameters(), model))
        dt = float(kwargs.get("dt", 1.0e-9))
        compiled = _compile_waveform(
            pulses,
            dt=dt,
            hardware=kwargs.get("compiler_hardware"),
            crosstalk=kwargs.get("compiler_crosstalk"),
            t_end=meta.get("t_end"),
        )
        cfg = _runtime_simulation_config(kwargs)
        return _simulate_waveform_states(
            model,
            compiled,
            drive_ops,
            states,
            config=cfg,
            c_ops=None if kwargs.get("c_ops") is None else list(kwargs["c_ops"]),
            noise=kwargs.get("noise"),
            max_workers=int(kwargs.get("state_workers", 1)),
        )


GatePrimitive = QubitRotation | SQR | SNAP | Displacement | ConditionalPhaseSQR | FreeEvolveCondPhase | PrimitiveGate


@dataclass
class GateSequence:
    gates: list[GatePrimitive]
    n_cav: int | None = None
    full_dim: int | None = None
    time_params: list[GateTimeParam] = field(default_factory=list, init=False)
    _gate_time_param_indices: list[int] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self.configure_time_parameters(time_policy=None, mode="per-instance", shared_groups=None)

    @staticmethod
    def _resolve_policy_for_gate(gate_type: str, time_policy: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        default = time_policy.get("default", {})
        out.update(default)

        aliases = [gate_type]
        if gate_type == "ConditionalPhaseSQR":
            aliases.append("CondPhaseSQR")
        if gate_type == "CondPhaseSQR":
            aliases.append("ConditionalPhaseSQR")

        for key in aliases:
            if key in time_policy:
                out.update(time_policy[key])
        return out

    def resolve_full_dim(self, model: Any | None = None, system: Any | None = None) -> int:
        if self.full_dim is not None:
            return int(self.full_dim)
        if self.n_cav is not None:
            return int(2 * self.n_cav)
        if system is not None:
            dim = system.hilbert_dimension(sequence=self)
            if dim is not None:
                return int(dim)
        if model is not None and hasattr(model, "subsystem_dims"):
            return int(np.prod(tuple(int(dim) for dim in model.subsystem_dims)))
        for gate in self.gates:
            if isinstance(gate, PrimitiveGate):
                return int(gate.resolved_dimension(model=model, system=system))
        raise ValueError("Could not infer GateSequence full Hilbert dimension.")

    def resolve_n_cav(self, model: Any | None = None, system: Any | None = None) -> int:
        if self.n_cav is not None:
            return int(self.n_cav)
        if system is not None:
            n_cav = system.infer_n_cav(sequence=self, full_dim=self.full_dim)
            if n_cav is not None:
                return int(n_cav)
        if model is not None and hasattr(model, "subsystem_dims"):
            dims = tuple(int(dim) for dim in model.subsystem_dims)
            if len(dims) >= 2 and dims[0] == 2:
                return int(dims[1])
        if self.full_dim is not None and int(self.full_dim) % 2 == 0:
            return int(self.full_dim // 2)
        raise ValueError("This gate sequence needs n_cav, but it could not be inferred.")

    def configure_time_parameters(
        self,
        time_policy: Mapping[str, Mapping[str, Any]] | None,
        mode: str = "per-instance",
        shared_groups: Mapping[str, str] | None = None,
    ) -> None:
        if mode not in {"per-instance", "per-type", "hybrid"}:
            raise ValueError("mode must be 'per-instance', 'per-type', or 'hybrid'.")

        policy: Mapping[str, Mapping[str, Any]] = time_policy or {}
        groups = dict(shared_groups or {})

        params: list[GateTimeParam] = []
        gate_param_idx: list[int] = []
        group_to_index: dict[str, int] = {}

        for gate_index, gate in enumerate(self.gates):
            gate_type = gate.type
            gate_policy = self._resolve_policy_for_gate(gate_type, policy)
            default_bounds = gate.time_bounds if gate.time_bounds is not None else (1.0e-9, 10.0e-6)

            optimize = bool(gate_policy.get("optimize", gate.optimize_time))
            if gate.time_policy_locked:
                optimize = bool(gate.optimize_time)

            bounds_raw = gate_policy.get("bounds", default_bounds)
            if gate.time_policy_locked and gate.time_bounds is not None:
                bounds_raw = gate.time_bounds
            if bounds_raw is None:
                bounds_raw = (1.0e-9, 10.0e-6)
            t_min, t_max = float(bounds_raw[0]), float(bounds_raw[1])

            init = float(gate_policy.get("init", gate.duration))
            if gate.time_policy_locked:
                init = float(gate.duration)

            if gate.time_policy_locked and gate.time_group is not None:
                group = str(gate.time_group)
            elif mode == "per-type":
                group = f"type:{gate_type}"
            elif mode == "per-instance":
                group = f"instance:{gate_index}:{gate.name}"
            else:
                if gate.name in groups:
                    group = str(groups[gate.name])
                elif "group" in gate_policy:
                    group = str(gate_policy["group"])
                else:
                    group = f"instance:{gate_index}:{gate.name}"

            frozen = bool(gate_policy.get("frozen", False))
            if not optimize:
                frozen = True

            if group not in group_to_index:
                param = GateTimeParam(
                    param_id=f"t_{len(params)}",
                    gate_type=gate_type,
                    group=group,
                    t_min=t_min,
                    t_max=t_max,
                    t_init=init,
                    value=init,
                    optimize=optimize,
                    frozen=frozen,
                    gate_names=[gate.name],
                )
                params.append(param)
                group_to_index[group] = len(params) - 1
            else:
                idx = group_to_index[group]
                existing = params[idx]
                if not (np.isclose(existing.t_min, t_min) and np.isclose(existing.t_max, t_max)):
                    raise ValueError(
                        f"Time group '{group}' has conflicting bounds: {(existing.t_min, existing.t_max)} vs {(t_min, t_max)}"
                    )
                if existing.gate_type != gate_type and mode == "per-type":
                    raise ValueError(f"Per-type mode cannot share group '{group}' across different gate types.")
                if existing.optimize != optimize:
                    raise ValueError(f"Time group '{group}' has conflicting optimize flags.")
                if existing.frozen != frozen:
                    raise ValueError(f"Time group '{group}' has conflicting frozen flags.")
                existing.gate_names.append(gate.name)

            gate_idx = group_to_index[group]
            gate_param_idx.append(gate_idx)
            gate.time_group = group
            gate.optimize_time = optimize
            gate.time_bounds = (t_min, t_max)

        self.time_params = params
        self._gate_time_param_indices = gate_param_idx
        self._apply_time_params_to_gates()

    def _apply_time_params_to_gates(self) -> None:
        for gate_idx, gate in enumerate(self.gates):
            param = self.time_params[self._gate_time_param_indices[gate_idx]]
            gate.duration = float(param.value)
            if isinstance(gate, PrimitiveGate):
                gate.parameters["duration"] = float(param.value)

    def sync_time_params_from_gates(self) -> None:
        if not self.time_params:
            return
        seen: set[int] = set()
        for gate_idx, gate in enumerate(self.gates):
            pidx = self._gate_time_param_indices[gate_idx]
            if pidx in seen:
                continue
            seen.add(pidx)
            self.time_params[pidx].set_value(gate.duration)
        self._apply_time_params_to_gates()

    def total_duration(self) -> float:
        self.sync_time_params_from_gates()
        return float(sum(g.duration for g in self.gates))

    def unitary(self, backend: str = "ideal", backend_settings: Mapping[str, Any] | None = None) -> np.ndarray:
        settings = dict(backend_settings or {})
        system = settings.get("system")
        model = settings.get("model")
        full_dim = self.resolve_full_dim(model=model, system=system)
        n_cav = self.resolve_n_cav(model=model, system=system) if any(not isinstance(gate, PrimitiveGate) for gate in self.gates) else -1
        self.sync_time_params_from_gates()
        u = np.eye(full_dim, dtype=np.complex128)
        for gate in self.gates:
            if backend == "pulse":
                gate_u = gate.pulse_unitary(n_cav, **settings)
            elif backend == "ideal":
                gate_u = gate.ideal_unitary(n_cav, scale_by_time=False, **settings)
            else:
                raise ValueError("backend must be 'ideal' or 'pulse'.")
            u = _as_complex_array(gate_u) @ u
        return np.asarray(u, dtype=np.complex128)

    def propagate_states(
        self,
        states: list[qt.Qobj | np.ndarray],
        *,
        backend: str = "ideal",
        backend_settings: Mapping[str, Any] | None = None,
    ) -> list[qt.Qobj]:
        settings = dict(backend_settings or {})
        system = settings.get("system")
        model = settings.get("model")
        full_dim = self.resolve_full_dim(model=model, system=system)
        if system is not None:
            dims_raw = list(system.subsystem_dimensions(sequence=self, full_dim=full_dim))
        else:
            dims_raw = list(getattr(model, "subsystem_dims", (full_dim,)))
        n_cav = self.resolve_n_cav(model=model, system=system) if any(not isinstance(gate, PrimitiveGate) for gate in self.gates) else -1
        outputs = [_coerce_state_qobj(state, subsystem_dims=dims_raw) for state in states]
        for gate in self.gates:
            outputs = gate.propagate_states(outputs, n_cav, backend=backend, **settings)
        return outputs

    def serialize(self) -> list[dict[str, Any]]:
        self.sync_time_params_from_gates()
        rows = []
        n_cav = self.n_cav if self.n_cav is not None else -1
        for gate_idx, gate in enumerate(self.gates):
            row = gate.to_record()
            row["parameters"] = gate.get_parameters(n_cav).tolist()
            row["time_param_id"] = self.time_params[self._gate_time_param_indices[gate_idx]].param_id
            rows.append(row)
        return rows

    def serialize_time_parameters(self) -> list[dict[str, Any]]:
        return [param.to_record() for param in self.time_params]

    def phase_decomposition(self, n_match: int | None = None) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if self.n_cav is None:
            return rows
        for gate in self.gates:
            row = gate.phase_decomposition(self.n_cav, n_match=n_match)
            if row is not None:
                rows.append(row)
        return rows

    def parameter_layout(self) -> list[tuple[int, str, int]]:
        layout: list[tuple[int, str, int]] = []
        n_cav = self.n_cav if self.n_cav is not None else -1
        for i, gate in enumerate(self.gates):
            for j, name in enumerate(gate.parameter_names(n_cav)):
                layout.append((i, name, j))
        return layout

    def get_parameter_vector(self) -> np.ndarray:
        n_cav = self.n_cav if self.n_cav is not None else -1
        chunks = [gate.get_parameters(n_cav) for gate in self.gates]
        if not chunks:
            return np.asarray([], dtype=float)
        return np.concatenate(chunks)

    def set_parameter_vector(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=float)
        n_cav = self.n_cav if self.n_cav is not None else -1
        offset = 0
        for gate in self.gates:
            count = gate.get_parameters(n_cav).size
            gate.set_parameters(values[offset : offset + count], n_cav)
            offset += count
        if offset != values.size:
            raise ValueError("Parameter vector length mismatch.")

    def parameter_bounds_vector(self) -> list[tuple[float, float]]:
        n_cav = self.n_cav if self.n_cav is not None else -1
        bounds: list[tuple[float, float]] = []
        for gate in self.gates:
            bounds.extend(gate.parameter_bounds_vector(n_cav))
        return bounds

    def active_time_params(self) -> list[GateTimeParam]:
        return [p for p in self.time_params if p.active]

    def get_time_vector(self, active_only: bool = True) -> np.ndarray:
        params = self.active_time_params() if active_only else self.time_params
        return np.asarray([p.value for p in params], dtype=float)

    def set_time_vector(self, times: np.ndarray, active_only: bool = True) -> None:
        times = np.asarray(times, dtype=float)
        params = self.active_time_params() if active_only else self.time_params
        if times.size != len(params):
            raise ValueError("Time vector length mismatch.")
        for p, t in zip(params, times):
            p.set_value(float(t))
        self._apply_time_params_to_gates()

    def get_time_raw_vector(self, active_only: bool = True) -> np.ndarray:
        params = self.active_time_params() if active_only else self.time_params
        return np.asarray([p.raw_value() for p in params], dtype=float)

    def set_time_raw_vector(self, raw_values: np.ndarray, active_only: bool = True) -> np.ndarray:
        raw_values = np.asarray(raw_values, dtype=float)
        params = self.active_time_params() if active_only else self.time_params
        if raw_values.size != len(params):
            raise ValueError("Raw time vector length mismatch.")
        for p, raw in zip(params, raw_values):
            p.set_from_raw(float(raw))
        self._apply_time_params_to_gates()
        return self.get_time_vector(active_only=active_only)

    def time_grad_vector(self, raw_values: np.ndarray, active_only: bool = True) -> np.ndarray:
        raw_values = np.asarray(raw_values, dtype=float)
        params = self.active_time_params() if active_only else self.time_params
        if raw_values.size != len(params):
            raise ValueError("Raw time vector length mismatch.")
        return np.asarray([p.grad_raw(raw) for p, raw in zip(params, raw_values)], dtype=float)

    def time_bounds(self, active_only: bool = True) -> list[tuple[float, float]]:
        params = self.active_time_params() if active_only else self.time_params
        return [(p.t_min, p.t_max) for p in params]

    def gate_durations(self) -> np.ndarray:
        self.sync_time_params_from_gates()
        return np.asarray([gate.duration for gate in self.gates], dtype=float)
