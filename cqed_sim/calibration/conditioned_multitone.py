from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import manifold_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.calibration import build_sqr_tone_specs, pad_parameter_array
from cqed_sim.pulses.envelopes import MultitoneTone, multitone_envelope, normalized_gaussian
from cqed_sim.pulses.hardware import HardwareConfig
from cqed_sim.pulses.pulse import EnvelopeFunc, Pulse
from cqed_sim.sequence.scheduler import CompiledSequence, SequenceCompiler
from cqed_sim.sim.extractors import bloch_xyz_from_qubit_state, conditioned_qubit_state
from cqed_sim.sim.runner import SimulationConfig, prepare_simulation


_SIGMA_PLUS = qt.create(2)
_SIGMA_MINUS = qt.destroy(2)
_N_Q = qt.num(2)


def _wrap_pi(value: float) -> float:
    return float((float(value) + np.pi) % (2.0 * np.pi) - np.pi)


def _normalize_weights(weights: Sequence[float]) -> tuple[float, ...]:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("weights must not be empty.")
    if np.any(arr < 0.0):
        raise ValueError("weights must be non-negative.")
    total = float(np.sum(arr))
    if total <= 0.0:
        raise ValueError("weights must contain at least one positive value.")
    arr = arr / total
    return tuple(float(x) for x in arr)


def _as_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _as_serializable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_serializable(inner) for inner in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def bloch_vector_from_angles(theta: float, phi: float) -> tuple[float, float, float]:
    theta_val = float(theta)
    phi_val = float(phi)
    return (
        float(np.sin(theta_val) * np.cos(phi_val)),
        float(np.sin(theta_val) * np.sin(phi_val)),
        float(np.cos(theta_val)),
    )


def qubit_state_from_angles(theta: float, phi: float) -> qt.Qobj:
    theta_val = float(theta)
    phi_val = float(phi)
    data = np.asarray(
        [
            np.cos(theta_val / 2.0),
            np.exp(1j * phi_val) * np.sin(theta_val / 2.0),
        ],
        dtype=np.complex128,
    )
    return qt.Qobj(data.reshape((2, 1)), dims=[[2], [1]])


def qubit_density_matrix_from_angles(theta: float, phi: float) -> qt.Qobj:
    ket = qubit_state_from_angles(theta, phi)
    return ket * ket.dag()


def bloch_angles_from_density_matrix(
    rho_q: qt.Qobj,
    *,
    radius_threshold: float = 1.0e-10,
    xy_threshold: float = 1.0e-10,
) -> tuple[float, float, float]:
    x, y, z = bloch_xyz_from_qubit_state(rho_q)
    radius = float(np.sqrt(x * x + y * y + z * z))
    if radius <= float(radius_threshold):
        return float("nan"), float("nan"), radius
    theta = float(np.arccos(np.clip(z / radius, -1.0, 1.0)))
    xy = float(np.sqrt(x * x + y * y))
    if xy <= float(xy_threshold):
        return theta, float("nan"), radius
    phi = float(np.mod(np.arctan2(y, x), 2.0 * np.pi))
    return theta, phi, radius


def _coerce_weights(
    weights: Mapping[int, float] | Sequence[float] | None,
    n_levels: int,
) -> tuple[float, ...]:
    if weights is None:
        return tuple(float(1.0 / max(int(n_levels), 1)) for _ in range(int(n_levels)))
    if isinstance(weights, Mapping):
        arr = np.zeros(int(n_levels), dtype=float)
        for level, value in weights.items():
            idx = int(level)
            if idx < 0 or idx >= int(n_levels):
                raise IndexError(f"Weight index {idx} out of range for n_levels={n_levels}.")
            arr[idx] = float(value)
        return _normalize_weights(arr)
    arr = np.asarray(weights, dtype=float).reshape(-1)
    if arr.size != int(n_levels):
        raise ValueError(f"weights must have length {n_levels}, received {arr.size}.")
    return _normalize_weights(arr)


@dataclass(frozen=True)
class ConditionedQubitTargets:
    theta: tuple[float, ...]
    phi: tuple[float, ...]
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        theta = tuple(float(value) for value in self.theta)
        phi = tuple(float(value) for value in self.phi)
        if len(theta) != len(phi):
            raise ValueError("theta and phi must have the same length.")
        object.__setattr__(self, "theta", theta)
        object.__setattr__(self, "phi", phi)
        if not self.weights:
            weights = tuple(float(1.0 / max(len(theta), 1)) for _ in range(len(theta)))
        else:
            if len(self.weights) != len(theta):
                raise ValueError("weights must have the same length as theta and phi.")
            weights = _normalize_weights(self.weights)
        object.__setattr__(self, "weights", weights)

    @classmethod
    def from_spec(
        cls,
        targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
        *,
        n_levels: int | None = None,
        weights: Mapping[int, float] | Sequence[float] | None = None,
    ) -> ConditionedQubitTargets:
        if isinstance(targets, cls):
            if weights is None and n_levels is None:
                return targets
            total_levels = targets.n_levels if n_levels is None else int(n_levels)
            theta = np.zeros(total_levels, dtype=float)
            phi = np.zeros(total_levels, dtype=float)
            count = min(total_levels, targets.n_levels)
            theta[:count] = np.asarray(targets.theta[:count], dtype=float)
            phi[:count] = np.asarray(targets.phi[:count], dtype=float)
            if weights is None:
                base_weights = np.zeros(total_levels, dtype=float)
                if count > 0:
                    base_weights[:count] = np.asarray(targets.weights[:count], dtype=float)
                weight_values = _normalize_weights(base_weights if np.any(base_weights > 0.0) else np.ones(total_levels, dtype=float))
            else:
                weight_values = _coerce_weights(weights, total_levels)
            return cls(
                tuple(float(x) for x in theta),
                tuple(float(x) for x in phi),
                tuple(float(x) for x in weight_values),
            )

        if isinstance(targets, Mapping):
            inferred = 0 if not targets else max(int(level) for level in targets) + 1
            total_levels = inferred if n_levels is None else int(n_levels)
            theta = np.zeros(total_levels, dtype=float)
            phi = np.zeros(total_levels, dtype=float)
            for level, values in targets.items():
                idx = int(level)
                if idx < 0 or idx >= total_levels:
                    raise IndexError(f"Target index {idx} out of range for n_levels={total_levels}.")
                theta[idx] = float(values[0])
                phi[idx] = float(values[1])
            weight_values = _coerce_weights(weights, total_levels)
            return cls(tuple(float(x) for x in theta), tuple(float(x) for x in phi), weight_values)

        pairs = list(targets)
        total_levels = len(pairs) if n_levels is None else int(n_levels)
        theta = np.zeros(total_levels, dtype=float)
        phi = np.zeros(total_levels, dtype=float)
        for idx, values in enumerate(pairs[:total_levels]):
            theta[idx] = float(values[0])
            phi[idx] = float(values[1])
        weight_values = _coerce_weights(weights, total_levels)
        return cls(tuple(float(x) for x in theta), tuple(float(x) for x in phi), weight_values)

    @property
    def n_levels(self) -> int:
        return len(self.theta)

    def target_ket(self, n: int) -> qt.Qobj:
        return qubit_state_from_angles(self.theta[int(n)], self.phi[int(n)])

    def target_density_matrix(self, n: int) -> qt.Qobj:
        return qubit_density_matrix_from_angles(self.theta[int(n)], self.phi[int(n)])

    def target_bloch_vector(self, n: int) -> tuple[float, float, float]:
        return bloch_vector_from_angles(self.theta[int(n)], self.phi[int(n)])

    def as_rows(self) -> list[dict[str, float]]:
        return [
            {
                "n": int(n),
                "theta_target_rad": float(self.theta[n]),
                "phi_target_rad": float(self.phi[n]),
                "weight": float(self.weights[n]),
            }
            for n in range(self.n_levels)
        ]


@dataclass(frozen=True)
class ConditionedMultitoneCorrections:
    d_lambda: tuple[float, ...] = ()
    d_alpha: tuple[float, ...] = ()
    d_omega_rad_s: tuple[float, ...] = ()

    @classmethod
    def zeros(cls, n_levels: int) -> ConditionedMultitoneCorrections:
        zeros = tuple(0.0 for _ in range(int(n_levels)))
        return cls(d_lambda=zeros, d_alpha=zeros, d_omega_rad_s=zeros)

    def padded(self, n_levels: int) -> ConditionedMultitoneCorrections:
        n = int(n_levels)
        return ConditionedMultitoneCorrections(
            d_lambda=tuple(float(x) for x in pad_parameter_array(list(self.d_lambda), n)),
            d_alpha=tuple(float(x) for x in pad_parameter_array(list(self.d_alpha), n)),
            d_omega_rad_s=tuple(float(x) for x in pad_parameter_array(list(self.d_omega_rad_s), n)),
        )

    def correction_for_n(self, n: int) -> tuple[float, float, float]:
        idx = int(n)
        return (
            float(self.d_lambda[idx]) if idx < len(self.d_lambda) else 0.0,
            float(self.d_alpha[idx]) if idx < len(self.d_alpha) else 0.0,
            float(self.d_omega_rad_s[idx]) if idx < len(self.d_omega_rad_s) else 0.0,
        )


@dataclass(frozen=True)
class ConditionedMultitoneRunConfig:
    frame: FrameSpec = FrameSpec()
    duration_s: float = 1.0e-6
    dt_s: float = 4.0e-9
    sigma_fraction: float = 1.0 / 6.0
    tone_cutoff: float = 1.0e-10
    include_all_levels: bool = False
    max_step_s: float | None = None
    fock_fqs_hz: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if float(self.duration_s) <= 0.0:
            raise ValueError("duration_s must be positive.")
        if float(self.dt_s) <= 0.0:
            raise ValueError("dt_s must be positive.")
        if float(self.sigma_fraction) <= 0.0:
            raise ValueError("sigma_fraction must be positive.")
        if self.max_step_s is not None and float(self.max_step_s) <= 0.0:
            raise ValueError("max_step_s must be positive when provided.")
        if self.fock_fqs_hz is not None:
            object.__setattr__(self, "fock_fqs_hz", tuple(float(value) for value in self.fock_fqs_hz))


@dataclass(frozen=True)
class ConditionedOptimizationConfig:
    active_levels: tuple[int, ...] = ()
    parameters: tuple[str, ...] = ("d_lambda", "d_alpha", "d_omega")
    method_stage1: str = "Powell"
    method_stage2: str | None = "L-BFGS-B"
    maxiter_stage1: int = 40
    maxiter_stage2: int = 60
    d_lambda_bounds: tuple[float, float] = (-0.5, 0.5)
    d_alpha_bounds: tuple[float, float] = (-np.pi, np.pi)
    d_omega_hz_bounds: tuple[float, float] = (-2.0e6, 2.0e6)
    regularization_lambda: float = 0.0
    regularization_alpha: float = 0.0
    regularization_omega: float = 0.0

    def __post_init__(self) -> None:
        allowed = {"d_lambda", "d_alpha", "d_omega"}
        parameters = tuple(str(name) for name in self.parameters)
        if not parameters:
            raise ValueError("parameters must contain at least one optimization variable.")
        invalid = sorted(set(parameters) - allowed)
        if invalid:
            raise ValueError(f"Unsupported optimization parameters: {invalid}.")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "active_levels", tuple(int(level) for level in self.active_levels))


@dataclass(frozen=True)
class ConditionedMultitoneWaveform:
    pulse: Pulse
    tone_specs: tuple[MultitoneTone, ...]
    drive_channel: str = "qubit"
    drive_target: str = "qubit"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return float(self.pulse.duration)

    @property
    def drive_ops(self) -> dict[str, str]:
        return {str(self.drive_channel): str(self.drive_target)}

    def tone_rows(self) -> list[dict[str, float | int]]:
        return [tone.as_dict() for tone in self.tone_specs]

    def sample(self, tlist: np.ndarray) -> np.ndarray:
        return self.pulse.sample(np.asarray(tlist, dtype=float))


@dataclass(frozen=True)
class ConditionedSectorMetrics:
    n: int
    weight: float
    fidelity: float
    target_theta_rad: float
    target_phi_rad: float
    target_bloch_x: float
    target_bloch_y: float
    target_bloch_z: float
    simulated_bloch_x: float
    simulated_bloch_y: float
    simulated_bloch_z: float
    bloch_radius: float
    purity: float
    theta_simulated_rad: float
    phi_simulated_rad: float
    theta_error_rad: float
    phi_error_rad: float
    bloch_distance: float
    sector_population: float
    dominant_error: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": int(self.n),
            "weight": float(self.weight),
            "fidelity": float(self.fidelity),
            "target_theta_rad": float(self.target_theta_rad),
            "target_phi_rad": float(self.target_phi_rad),
            "target_bloch_x": float(self.target_bloch_x),
            "target_bloch_y": float(self.target_bloch_y),
            "target_bloch_z": float(self.target_bloch_z),
            "simulated_bloch_x": float(self.simulated_bloch_x),
            "simulated_bloch_y": float(self.simulated_bloch_y),
            "simulated_bloch_z": float(self.simulated_bloch_z),
            "bloch_radius": float(self.bloch_radius),
            "purity": float(self.purity),
            "theta_simulated_rad": float(self.theta_simulated_rad),
            "phi_simulated_rad": float(self.phi_simulated_rad),
            "theta_error_rad": float(self.theta_error_rad),
            "phi_error_rad": float(self.phi_error_rad),
            "bloch_distance": float(self.bloch_distance),
            "sector_population": float(self.sector_population),
            "dominant_error": str(self.dominant_error),
        }


@dataclass
class ConditionedValidationResult:
    targets: ConditionedQubitTargets
    corrections: ConditionedMultitoneCorrections
    waveform: ConditionedMultitoneWaveform
    compiled: CompiledSequence = field(repr=False)
    sector_metrics: tuple[ConditionedSectorMetrics, ...] = ()
    aggregate_cost: float = 0.0
    weighted_mean_fidelity: float = 0.0
    simulation_mode: str = "reduced"
    final_qubit_states: tuple[qt.Qobj, ...] = field(default_factory=tuple, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def sector_rows(self) -> list[dict[str, Any]]:
        return [metric.as_dict() for metric in self.sector_metrics]

    def as_dict(self) -> dict[str, Any]:
        return {
            "simulation_mode": str(self.simulation_mode),
            "aggregate_cost": float(self.aggregate_cost),
            "weighted_mean_fidelity": float(self.weighted_mean_fidelity),
            "targets": self.targets.as_rows(),
            "corrections": {
                "d_lambda": [float(x) for x in self.corrections.d_lambda],
                "d_alpha": [float(x) for x in self.corrections.d_alpha],
                "d_omega_rad_s": [float(x) for x in self.corrections.d_omega_rad_s],
                "d_omega_hz": [float(x / (2.0 * np.pi)) for x in self.corrections.d_omega_rad_s],
            },
            "tone_specs": self.waveform.tone_rows(),
            "sector_metrics": self.sector_rows(),
            "metadata": _as_serializable(self.metadata),
        }


@dataclass
class ConditionedOptimizationResult:
    initial_result: ConditionedValidationResult
    optimized_result: ConditionedValidationResult
    optimized_corrections: ConditionedMultitoneCorrections
    active_levels: tuple[int, ...]
    parameters: tuple[str, ...]
    history: list[dict[str, float]] = field(default_factory=list)
    success_stage1: bool = False
    success_stage2: bool = False
    message_stage1: str = ""
    message_stage2: str = ""

    def improvement_summary(self) -> dict[str, Any]:
        return {
            "initial_cost": float(self.initial_result.aggregate_cost),
            "optimized_cost": float(self.optimized_result.aggregate_cost),
            "cost_reduction": float(self.initial_result.aggregate_cost - self.optimized_result.aggregate_cost),
            "initial_weighted_mean_fidelity": float(self.initial_result.weighted_mean_fidelity),
            "optimized_weighted_mean_fidelity": float(self.optimized_result.weighted_mean_fidelity),
            "active_levels": [int(level) for level in self.active_levels],
            "parameters": [str(name) for name in self.parameters],
        }


def _ensure_target_object(
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
) -> ConditionedQubitTargets:
    if isinstance(targets, ConditionedQubitTargets):
        return targets
    return ConditionedQubitTargets.from_spec(targets)


def _classify_error(theta_error: float, phi_error: float, bloch_radius: float) -> str:
    if np.isnan(theta_error) and np.isnan(phi_error):
        return "undefined"
    if float(bloch_radius) < 0.98:
        return "mixed-or-decohered"
    theta_abs = float(abs(theta_error)) if np.isfinite(theta_error) else 0.0
    phi_abs = float(abs(phi_error)) if np.isfinite(phi_error) else 0.0
    if theta_abs < 0.03 and phi_abs < 0.03:
        return "small"
    if theta_abs > 2.0 * phi_abs and theta_abs > 0.05:
        return "amplitude-like"
    if phi_abs > 2.0 * theta_abs and phi_abs > 0.05:
        return "phase-or-detuning-like"
    return "mixed-or-crosstalk-like"


def build_conditioned_multitone_tones(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    run_config: ConditionedMultitoneRunConfig,
    *,
    corrections: ConditionedMultitoneCorrections | None = None,
) -> list[MultitoneTone]:
    target_obj = _ensure_target_object(targets)
    if int(target_obj.n_levels) > int(model.n_cav):
        raise ValueError(f"Target uses {target_obj.n_levels} Fock levels but model.n_cav={model.n_cav}.")
    corr = ConditionedMultitoneCorrections.zeros(target_obj.n_levels) if corrections is None else corrections.padded(target_obj.n_levels)
    raw_tones = build_sqr_tone_specs(
        model=model,
        frame=run_config.frame,
        theta_values=list(target_obj.theta),
        phi_values=list(target_obj.phi),
        duration_s=float(run_config.duration_s),
        d_lambda_values=list(corr.d_lambda),
        fock_fqs_hz=None if run_config.fock_fqs_hz is None else list(run_config.fock_fqs_hz),
        include_all_levels=bool(run_config.include_all_levels),
        tone_cutoff=float(run_config.tone_cutoff),
    )
    tones: list[MultitoneTone] = []
    for tone in raw_tones:
        _d_lambda, d_alpha, d_omega = corr.correction_for_n(int(tone.manifold))
        tones.append(
            MultitoneTone(
                manifold=int(tone.manifold),
                omega_rad_s=float(tone.omega_rad_s + d_omega),
                amp_rad_s=float(tone.amp_rad_s),
                phase_rad=float(tone.phase_rad + d_alpha),
            )
        )
    return tones


def build_conditioned_multitone_waveform(
    tone_specs: Sequence[MultitoneTone],
    run_config: ConditionedMultitoneRunConfig,
    *,
    base_envelope: EnvelopeFunc | None = None,
    base_samples: np.ndarray | None = None,
    sample_rate: float | None = None,
    channel: str = "qubit",
    drive_target: str = "qubit",
    label: str | None = None,
) -> ConditionedMultitoneWaveform:
    if base_envelope is not None and base_samples is not None:
        raise ValueError("Use either base_envelope or base_samples, not both.")
    tone_tuple = tuple(tone_specs)
    duration_s = float(run_config.duration_s)
    metadata: dict[str, Any] = {
        "duration_s": duration_s,
        "sigma_fraction": float(run_config.sigma_fraction),
        "tone_cutoff": float(run_config.tone_cutoff),
    }

    if base_samples is not None:
        if sample_rate is None or float(sample_rate) <= 0.0:
            raise ValueError("sample_rate must be positive when base_samples are provided.")
        base = np.asarray(base_samples, dtype=np.complex128).reshape(-1)
        if base.size == 0:
            raise ValueError("base_samples must contain at least one sample.")
        time_grid = np.arange(base.size, dtype=float) / float(sample_rate)
        coeff = np.zeros_like(time_grid, dtype=np.complex128)
        for tone in tone_tuple:
            coeff += tone.amp_rad_s * np.exp(1j * tone.phase_rad) * np.exp(1j * tone.omega_rad_s * time_grid)
        envelope = base * coeff
        pulse = Pulse(
            channel=str(channel),
            t0=0.0,
            duration=duration_s,
            envelope=envelope,
            sample_rate=float(sample_rate),
            amp=1.0,
            phase=0.0,
            label=label,
        )
        metadata["base_envelope"] = "samples"
        metadata["sample_rate"] = float(sample_rate)
        metadata["base_sample_count"] = int(base.size)
    else:
        if base_envelope is None:
            sigma_fraction = float(run_config.sigma_fraction)

            def default_base_envelope(t_rel: np.ndarray) -> np.ndarray:
                return normalized_gaussian(t_rel, sigma_fraction=sigma_fraction)

            base_envelope = default_base_envelope
            metadata["base_envelope"] = "normalized_gaussian"
        else:
            metadata["base_envelope"] = getattr(base_envelope, "__name__", "callable")

        def envelope(t_rel: np.ndarray) -> np.ndarray:
            return multitone_envelope(
                np.asarray(t_rel, dtype=float),
                duration_s=duration_s,
                tone_specs=tone_tuple,
                base_envelope=base_envelope,
            )

        pulse = Pulse(
            channel=str(channel),
            t0=0.0,
            duration=duration_s,
            envelope=envelope,
            amp=1.0,
            phase=0.0,
            label=label,
        )

    return ConditionedMultitoneWaveform(
        pulse=pulse,
        tone_specs=tone_tuple,
        drive_channel=str(channel),
        drive_target=str(drive_target),
        metadata=metadata,
    )


def compile_conditioned_multitone_waveform(
    waveform: ConditionedMultitoneWaveform,
    run_config: ConditionedMultitoneRunConfig,
    *,
    hardware: dict[str, HardwareConfig] | None = None,
    crosstalk_matrix: Mapping[str, Mapping[str, float]] | None = None,
) -> CompiledSequence:
    compiler = SequenceCompiler(
        dt=float(run_config.dt_s),
        hardware=hardware,
        crosstalk_matrix=None
        if crosstalk_matrix is None
        else {
            str(src): {str(dst): float(value) for dst, value in mapping.items()}
            for src, mapping in crosstalk_matrix.items()
        },
    )
    return compiler.compile([waveform.pulse], t_end=float(run_config.duration_s + run_config.dt_s))


def sample_conditioned_multitone_waveform(
    waveform: ConditionedMultitoneWaveform,
    run_config: ConditionedMultitoneRunConfig,
    *,
    hardware: dict[str, HardwareConfig] | None = None,
    crosstalk_matrix: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, np.ndarray]:
    compiled = compile_conditioned_multitone_waveform(
        waveform,
        run_config,
        hardware=hardware,
        crosstalk_matrix=crosstalk_matrix,
    )
    channel = compiled.channels[waveform.drive_channel]
    return {
        "t_s": np.asarray(compiled.tlist, dtype=float),
        "baseband": np.asarray(channel.baseband, dtype=np.complex128),
        "distorted": np.asarray(channel.distorted, dtype=np.complex128),
        "i": np.asarray(np.real(channel.distorted), dtype=float),
        "q": np.asarray(np.imag(channel.distorted), dtype=float),
    }


def _reduced_sector_qubit_state(
    model: DispersiveTransmonCavityModel,
    compiled: CompiledSequence,
    waveform: ConditionedMultitoneWaveform,
    run_config: ConditionedMultitoneRunConfig,
    n: int,
) -> qt.Qobj:
    coeff = np.asarray(compiled.channels[waveform.drive_channel].distorted, dtype=np.complex128)
    detuning = float(manifold_transition_frequency(model, int(n), frame=run_config.frame))
    hamiltonian = [
        detuning * _N_Q,
        [_SIGMA_PLUS, coeff],
        [_SIGMA_MINUS, np.conj(coeff)],
    ]
    options: dict[str, Any] = {
        "atol": 1.0e-8,
        "rtol": 1.0e-7,
        "store_states": False,
        "store_final_state": True,
    }
    if run_config.max_step_s is not None:
        options["max_step"] = float(run_config.max_step_s)
    result = qt.sesolve(
        hamiltonian,
        qt.basis(2, 0),
        compiled.tlist,
        e_ops=[],
        options=options,
    )
    final_state = getattr(result, "final_state", None)
    ket = result.states[-1] if final_state is None else final_state
    return ket.proj() if not ket.isoper else ket


def _sector_metrics(
    targets: ConditionedQubitTargets,
    n: int,
    rho_q: qt.Qobj,
    *,
    sector_population: float,
) -> ConditionedSectorMetrics:
    target_bloch = targets.target_bloch_vector(int(n))
    x, y, z = bloch_xyz_from_qubit_state(rho_q)
    theta_sim, phi_sim, bloch_radius = bloch_angles_from_density_matrix(rho_q)
    theta_target = float(targets.theta[int(n)])
    phi_target = float(np.mod(targets.phi[int(n)], 2.0 * np.pi))
    theta_error = float("nan") if np.isnan(theta_sim) else _wrap_pi(theta_sim - theta_target)
    phi_error = float("nan") if np.isnan(phi_sim) else _wrap_pi(phi_sim - phi_target)
    target_dm = targets.target_density_matrix(int(n))
    fidelity = float(np.clip(np.real((target_dm * rho_q).tr()), 0.0, 1.0))
    purity = float(np.real((rho_q * rho_q).tr()))
    bloch_distance = float(
        np.sqrt(
            (x - target_bloch[0]) ** 2
            + (y - target_bloch[1]) ** 2
            + (z - target_bloch[2]) ** 2
        )
    )
    return ConditionedSectorMetrics(
        n=int(n),
        weight=float(targets.weights[int(n)]),
        fidelity=float(fidelity),
        target_theta_rad=theta_target,
        target_phi_rad=phi_target,
        target_bloch_x=float(target_bloch[0]),
        target_bloch_y=float(target_bloch[1]),
        target_bloch_z=float(target_bloch[2]),
        simulated_bloch_x=float(x),
        simulated_bloch_y=float(y),
        simulated_bloch_z=float(z),
        bloch_radius=float(bloch_radius),
        purity=float(purity),
        theta_simulated_rad=float(theta_sim),
        phi_simulated_rad=float(phi_sim),
        theta_error_rad=float(theta_error),
        phi_error_rad=float(phi_error),
        bloch_distance=float(bloch_distance),
        sector_population=float(sector_population),
        dominant_error=_classify_error(theta_error, phi_error, bloch_radius),
    )


def evaluate_conditioned_multitone(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    waveform: ConditionedMultitoneWaveform,
    run_config: ConditionedMultitoneRunConfig,
    *,
    corrections: ConditionedMultitoneCorrections | None = None,
    simulation_mode: str = "reduced",
    hardware: dict[str, HardwareConfig] | None = None,
    crosstalk_matrix: Mapping[str, Mapping[str, float]] | None = None,
) -> ConditionedValidationResult:
    target_obj = _ensure_target_object(targets)
    if target_obj.n_levels > int(model.n_cav):
        raise ValueError(f"Target uses {target_obj.n_levels} Fock levels but model.n_cav={model.n_cav}.")
    corr = ConditionedMultitoneCorrections.zeros(target_obj.n_levels) if corrections is None else corrections.padded(target_obj.n_levels)
    compiled = compile_conditioned_multitone_waveform(
        waveform,
        run_config,
        hardware=hardware,
        crosstalk_matrix=crosstalk_matrix,
    )
    mode = str(simulation_mode).strip().lower()
    if mode not in {"reduced", "full"}:
        raise ValueError("simulation_mode must be either 'reduced' or 'full'.")

    sector_metrics: list[ConditionedSectorMetrics] = []
    final_qubit_states: list[qt.Qobj] = []
    if mode == "reduced":
        for n in range(target_obj.n_levels):
            rho_q = _reduced_sector_qubit_state(model, compiled, waveform, run_config, n)
            final_qubit_states.append(rho_q)
            sector_metrics.append(_sector_metrics(target_obj, n, rho_q, sector_population=1.0))
    else:
        session = prepare_simulation(
            model,
            compiled,
            waveform.drive_ops,
            config=SimulationConfig(frame=run_config.frame, max_step=run_config.max_step_s),
        )
        initial_states = [model.basis_state(0, n) for n in range(target_obj.n_levels)]
        results = session.run_many(initial_states)
        for n, result in enumerate(results):
            rho_q, population, valid = conditioned_qubit_state(result.final_state, n=n, fallback="zero")
            if not valid:
                rho_q = qt.Qobj(np.zeros((2, 2), dtype=np.complex128), dims=[[2], [2]])
            final_qubit_states.append(rho_q)
            sector_metrics.append(_sector_metrics(target_obj, n, rho_q, sector_population=population))

    weights = np.asarray(target_obj.weights, dtype=float)
    fidelities = np.asarray([metric.fidelity for metric in sector_metrics], dtype=float)
    aggregate_cost = float(np.sum(weights * (1.0 - fidelities)))
    metadata = {
        "t_s": np.asarray(compiled.tlist, dtype=float),
        "tone_specs": waveform.tone_rows(),
        "waveform_metadata": dict(waveform.metadata),
        "weights": [float(x) for x in target_obj.weights],
    }
    return ConditionedValidationResult(
        targets=target_obj,
        corrections=corr,
        waveform=waveform,
        compiled=compiled,
        sector_metrics=tuple(sector_metrics),
        aggregate_cost=float(aggregate_cost),
        weighted_mean_fidelity=float(1.0 - aggregate_cost),
        simulation_mode=mode,
        final_qubit_states=tuple(final_qubit_states),
        metadata=metadata,
    )


def run_conditioned_multitone_validation(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    run_config: ConditionedMultitoneRunConfig,
    *,
    corrections: ConditionedMultitoneCorrections | None = None,
    simulation_mode: str = "reduced",
    base_envelope: EnvelopeFunc | None = None,
    base_samples: np.ndarray | None = None,
    sample_rate: float | None = None,
    hardware: dict[str, HardwareConfig] | None = None,
    crosstalk_matrix: Mapping[str, Mapping[str, float]] | None = None,
    channel: str = "qubit",
    drive_target: str = "qubit",
    label: str | None = None,
) -> ConditionedValidationResult:
    target_obj = _ensure_target_object(targets)
    corr = ConditionedMultitoneCorrections.zeros(target_obj.n_levels) if corrections is None else corrections.padded(target_obj.n_levels)
    tones = build_conditioned_multitone_tones(model, target_obj, run_config, corrections=corr)
    waveform = build_conditioned_multitone_waveform(
        tones,
        run_config,
        base_envelope=base_envelope,
        base_samples=base_samples,
        sample_rate=sample_rate,
        channel=channel,
        drive_target=drive_target,
        label=label,
    )
    return evaluate_conditioned_multitone(
        model,
        target_obj,
        waveform,
        run_config,
        corrections=corr,
        simulation_mode=simulation_mode,
        hardware=hardware,
        crosstalk_matrix=crosstalk_matrix,
    )


def _active_levels(targets: ConditionedQubitTargets, config: ConditionedOptimizationConfig) -> tuple[int, ...]:
    if config.active_levels:
        return tuple(int(level) for level in config.active_levels)
    return tuple(int(level) for level in range(targets.n_levels) if float(targets.weights[level]) > 0.0)


def _vector_from_corrections(
    corrections: ConditionedMultitoneCorrections,
    active_levels: Sequence[int],
    parameters: Sequence[str],
) -> np.ndarray:
    max_level = int(max(active_levels, default=-1)) + 1
    corr = corrections.padded(max(max_level, len(corrections.d_lambda), len(corrections.d_alpha), len(corrections.d_omega_rad_s)))
    vector: list[float] = []
    for level in active_levels:
        for name in parameters:
            if name == "d_lambda":
                vector.append(float(corr.d_lambda[int(level)]))
            elif name == "d_alpha":
                vector.append(float(corr.d_alpha[int(level)]))
            elif name == "d_omega":
                vector.append(float(corr.d_omega_rad_s[int(level)]))
    return np.asarray(vector, dtype=float)


def _corrections_from_vector(
    base: ConditionedMultitoneCorrections,
    vector: np.ndarray,
    n_levels: int,
    active_levels: Sequence[int],
    parameters: Sequence[str],
) -> ConditionedMultitoneCorrections:
    corr = base.padded(n_levels)
    d_lambda = np.asarray(corr.d_lambda, dtype=float)
    d_alpha = np.asarray(corr.d_alpha, dtype=float)
    d_omega = np.asarray(corr.d_omega_rad_s, dtype=float)
    data = np.asarray(vector, dtype=float).reshape(-1)
    expected = len(active_levels) * len(parameters)
    if data.size != expected:
        raise ValueError(f"Expected optimization vector of length {expected}, received {data.size}.")
    offset = 0
    for level in active_levels:
        for name in parameters:
            if name == "d_lambda":
                d_lambda[int(level)] = float(data[offset])
            elif name == "d_alpha":
                d_alpha[int(level)] = float(data[offset])
            elif name == "d_omega":
                d_omega[int(level)] = float(data[offset])
            offset += 1
    return ConditionedMultitoneCorrections(
        d_lambda=tuple(float(x) for x in d_lambda),
        d_alpha=tuple(float(x) for x in d_alpha),
        d_omega_rad_s=tuple(float(x) for x in d_omega),
    )


def _optimization_bounds(
    config: ConditionedOptimizationConfig,
    active_levels: Sequence[int],
) -> Bounds:
    lower: list[float] = []
    upper: list[float] = []
    for _level in active_levels:
        for name in config.parameters:
            if name == "d_lambda":
                lower.append(float(config.d_lambda_bounds[0]))
                upper.append(float(config.d_lambda_bounds[1]))
            elif name == "d_alpha":
                lower.append(float(config.d_alpha_bounds[0]))
                upper.append(float(config.d_alpha_bounds[1]))
            elif name == "d_omega":
                lower.append(float(2.0 * np.pi * config.d_omega_hz_bounds[0]))
                upper.append(float(2.0 * np.pi * config.d_omega_hz_bounds[1]))
    return Bounds(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float))


def _regularization_cost(
    corrections: ConditionedMultitoneCorrections,
    active_levels: Sequence[int],
    parameters: Sequence[str],
    config: ConditionedOptimizationConfig,
) -> float:
    value = 0.0
    for level in active_levels:
        d_lambda, d_alpha, d_omega = corrections.correction_for_n(int(level))
        for name in parameters:
            if name == "d_lambda":
                value += float(config.regularization_lambda) * float(d_lambda**2)
            elif name == "d_alpha":
                value += float(config.regularization_alpha) * float(d_alpha**2)
            elif name == "d_omega":
                value += float(config.regularization_omega) * float(d_omega**2)
    return float(value)


def optimize_conditioned_multitone(
    model: DispersiveTransmonCavityModel,
    targets: ConditionedQubitTargets | Mapping[int, tuple[float, float]] | Sequence[tuple[float, float]],
    run_config: ConditionedMultitoneRunConfig,
    *,
    initial_corrections: ConditionedMultitoneCorrections | None = None,
    optimization_config: ConditionedOptimizationConfig | None = None,
    simulation_mode: str = "reduced",
    base_envelope: EnvelopeFunc | None = None,
    base_samples: np.ndarray | None = None,
    sample_rate: float | None = None,
    hardware: dict[str, HardwareConfig] | None = None,
    crosstalk_matrix: Mapping[str, Mapping[str, float]] | None = None,
    channel: str = "qubit",
    drive_target: str = "qubit",
    label: str | None = None,
) -> ConditionedOptimizationResult:
    target_obj = _ensure_target_object(targets)
    opt_cfg = ConditionedOptimizationConfig() if optimization_config is None else optimization_config
    base_corr = ConditionedMultitoneCorrections.zeros(target_obj.n_levels) if initial_corrections is None else initial_corrections.padded(target_obj.n_levels)
    active_levels = _active_levels(target_obj, opt_cfg)
    x0 = _vector_from_corrections(base_corr, active_levels, opt_cfg.parameters)
    bounds = _optimization_bounds(opt_cfg, active_levels)
    history: list[dict[str, float]] = []

    initial_result = run_conditioned_multitone_validation(
        model,
        target_obj,
        run_config,
        corrections=base_corr,
        simulation_mode=simulation_mode,
        base_envelope=base_envelope,
        base_samples=base_samples,
        sample_rate=sample_rate,
        hardware=hardware,
        crosstalk_matrix=crosstalk_matrix,
        channel=channel,
        drive_target=drive_target,
        label=label,
    )

    def objective(vector: np.ndarray) -> float:
        corr = _corrections_from_vector(base_corr, vector, target_obj.n_levels, active_levels, opt_cfg.parameters)
        validation = run_conditioned_multitone_validation(
            model,
            target_obj,
            run_config,
            corrections=corr,
            simulation_mode=simulation_mode,
            base_envelope=base_envelope,
            base_samples=base_samples,
            sample_rate=sample_rate,
            hardware=hardware,
            crosstalk_matrix=crosstalk_matrix,
            channel=channel,
            drive_target=drive_target,
            label=label,
        )
        reg = _regularization_cost(corr, active_levels, opt_cfg.parameters, opt_cfg)
        objective_value = float(validation.aggregate_cost + reg)
        history.append(
            {
                "evaluation": float(len(history)),
                "aggregate_cost": float(validation.aggregate_cost),
                "regularization": float(reg),
                "objective": float(objective_value),
                "weighted_mean_fidelity": float(validation.weighted_mean_fidelity),
            }
        )
        return objective_value

    stage1 = minimize(
        objective,
        x0=x0,
        method=str(opt_cfg.method_stage1),
        bounds=bounds,
        options={"maxiter": int(opt_cfg.maxiter_stage1), "disp": False},
    )

    stage2 = None
    if opt_cfg.method_stage2:
        stage2 = minimize(
            objective,
            x0=np.asarray(stage1.x, dtype=float),
            method=str(opt_cfg.method_stage2),
            bounds=bounds,
            options={"maxiter": int(opt_cfg.maxiter_stage2)},
        )

    candidates = [np.asarray(stage1.x, dtype=float)]
    candidate_scores = [float(stage1.fun)]
    if stage2 is not None:
        candidates.append(np.asarray(stage2.x, dtype=float))
        candidate_scores.append(float(stage2.fun))
    best_vector = candidates[int(np.argmin(candidate_scores))]
    optimized_corrections = _corrections_from_vector(base_corr, best_vector, target_obj.n_levels, active_levels, opt_cfg.parameters)
    optimized_result = run_conditioned_multitone_validation(
        model,
        target_obj,
        run_config,
        corrections=optimized_corrections,
        simulation_mode=simulation_mode,
        base_envelope=base_envelope,
        base_samples=base_samples,
        sample_rate=sample_rate,
        hardware=hardware,
        crosstalk_matrix=crosstalk_matrix,
        channel=channel,
        drive_target=drive_target,
        label=label,
    )
    return ConditionedOptimizationResult(
        initial_result=initial_result,
        optimized_result=optimized_result,
        optimized_corrections=optimized_corrections,
        active_levels=tuple(int(level) for level in active_levels),
        parameters=tuple(str(name) for name in opt_cfg.parameters),
        history=history,
        success_stage1=bool(stage1.success),
        success_stage2=False if stage2 is None else bool(stage2.success),
        message_stage1=str(stage1.message),
        message_stage2="" if stage2 is None else str(stage2.message),
    )


__all__ = [
    "ConditionedMultitoneCorrections",
    "ConditionedMultitoneRunConfig",
    "ConditionedMultitoneWaveform",
    "ConditionedOptimizationConfig",
    "ConditionedOptimizationResult",
    "ConditionedQubitTargets",
    "ConditionedSectorMetrics",
    "ConditionedValidationResult",
    "bloch_angles_from_density_matrix",
    "bloch_vector_from_angles",
    "build_conditioned_multitone_tones",
    "build_conditioned_multitone_waveform",
    "compile_conditioned_multitone_waveform",
    "evaluate_conditioned_multitone",
    "optimize_conditioned_multitone",
    "qubit_density_matrix_from_angles",
    "qubit_state_from_angles",
    "run_conditioned_multitone_validation",
    "sample_conditioned_multitone_waveform",
]