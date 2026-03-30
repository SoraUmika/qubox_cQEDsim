from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.core import (
    BosonicModeSpec,
    CrossKerrSpec,
    DispersiveTransmonCavityModel,
    ExchangeSpec,
    FrameSpec,
    SidebandDriveSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
    carrier_for_transition_frequency,
    drive_frequency_for_transition_frequency,
    internal_carrier_from_drive_frequency,
    sideband_transition_frequency,
    transmon_transition_frequency,
)
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit
from cqed_sim.pulses import HardwareConfig, Pulse
from cqed_sim.pulses.calibration import displacement_square_amplitude, rotation_gaussian_amplitude
from cqed_sim.pulses.envelopes import normalized_gaussian, square_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import NoiseSpec, SimulationConfig, prepare_simulation, simulate_sequence

from .action_spaces import (
    CavityDisplacementAction,
    HybridBlockAction,
    MeasurementAction,
    PrimitiveAction,
    QubitGaussianAction,
    ResetAction,
    SidebandAction,
    WaitAction,
    WaveformAction,
)
from .configs import FullPulseModelConfig, HybridSystemConfig, ReducedDispersiveModelConfig
from .domain_randomization import RandomizationSample


def _shift_pulses(pulses: list[Pulse], offset: float) -> list[Pulse]:
    return [replace(pulse, t0=float(pulse.t0 + offset)) for pulse in pulses]


def _gaussian_envelope(sigma_fraction: float = 0.18):
    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return normalized_gaussian(t_rel, sigma_fraction=sigma_fraction)

    return envelope


def _mode_transition_frequency(model: Any, *, mode: str, frame: FrameSpec) -> float:
    if hasattr(model, "mode_transition_frequency"):
        return float(model.mode_transition_frequency(mode, transmon_level=0, frame=frame))
    if hasattr(model, "as_universal_model"):
        return float(model.as_universal_model().mode_transition_frequency(mode, transmon_level=0, frame=frame))
    return 0.0


def _apply_overrides_dataclass(base: Any, overrides: dict[str, Any]) -> Any:
    if base is None:
        if not overrides:
            return None
        raise ValueError("Cannot apply overrides to a missing base dataclass instance.")
    return replace(base, **overrides)


def _apply_hardware_overrides(
    base: dict[str, HardwareConfig],
    overrides: dict[str, dict[str, Any]],
) -> dict[str, HardwareConfig]:
    resolved = {channel: config for channel, config in base.items()}
    for channel, values in overrides.items():
        resolved[channel] = replace(resolved.get(channel, HardwareConfig()), **values)
    return resolved


def _drift_value(drift_state: dict[str, Any], specific_key: str, default_key: str, default: float) -> float:
    if specific_key in drift_state:
        return float(drift_state[specific_key])
    if default_key in drift_state:
        return float(drift_state[default_key])
    return float(default)


@dataclass
class ControlSegment:
    pulses: list[Pulse]
    drive_ops: dict[str, Any]
    duration: float
    measurement_requested: bool = False
    reset_requested: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeModelBundle:
    model: Any
    regime: str
    frame: FrameSpec
    noise: NoiseSpec | None
    hardware: dict[str, HardwareConfig]
    crosstalk_matrix: dict[str, dict[str, float]]
    dt: float
    max_step: float | None
    measurement_spec: QubitMeasurementSpec | None
    randomization: RandomizationSample | None = None

    @property
    def drift_state(self) -> dict[str, Any]:
        if self.randomization is None:
            return {}
        return dict(self.randomization.drift_state)


class HamiltonianModelFactory:
    @staticmethod
    def _resolve_frame(model: Any, *, configured_frame: FrameSpec, use_model_rotating_frame: bool) -> FrameSpec:
        frame = configured_frame
        if not use_model_rotating_frame:
            return frame
        if any(abs(value) > 0.0 for value in (frame.omega_c_frame, frame.omega_q_frame, frame.omega_r_frame)):
            return frame
        omega_q = 0.0
        omega_c = 0.0
        if hasattr(model, "omega_q"):
            omega_q = float(model.omega_q)
        elif getattr(model, "transmon", None) is not None:
            omega_q = float(model.transmon.omega)
        if hasattr(model, "omega_c"):
            omega_c = float(model.omega_c)
        elif getattr(model, "bosonic_modes", None):
            omega_c = float(model.bosonic_modes[0].omega)
        return FrameSpec(omega_c_frame=omega_c, omega_q_frame=omega_q, omega_r_frame=float(frame.omega_r_frame))

    @staticmethod
    def _build_reduced_model(config: ReducedDispersiveModelConfig) -> DispersiveTransmonCavityModel:
        return DispersiveTransmonCavityModel(
            omega_c=float(config.omega_c),
            omega_q=float(config.omega_q),
            alpha=float(config.alpha),
            chi=float(config.chi),
            kerr=float(config.kerr),
            chi_higher=tuple(float(value) for value in config.chi_higher),
            kerr_higher=tuple(float(value) for value in config.kerr_higher),
            n_cav=int(config.n_cav),
            n_tr=int(config.n_tr),
        )

    @staticmethod
    def _build_full_model(config: FullPulseModelConfig) -> UniversalCQEDModel:
        cross_kerr_terms = ()
        if abs(float(config.cross_kerr)) > 0.0:
            cross_kerr_terms = (CrossKerrSpec("storage", "qubit", float(config.cross_kerr)),)
        exchange_terms = ()
        if abs(float(config.exchange_g)) > 0.0:
            exchange_terms = (ExchangeSpec("storage", "qubit", float(config.exchange_g)),)
        return UniversalCQEDModel(
            transmon=TransmonModeSpec(
                omega=float(config.omega_q),
                dim=int(config.n_tr),
                alpha=float(config.alpha),
                label="qubit",
                aliases=("qubit", "transmon"),
                frame_channel="q",
            ),
            bosonic_modes=(
                BosonicModeSpec(
                    label="storage",
                    omega=float(config.omega_c),
                    dim=int(config.n_cav),
                    kerr=float(config.kerr),
                    aliases=("storage", "cavity"),
                    frame_channel="c",
                ),
            ),
            cross_kerr_terms=cross_kerr_terms,
            exchange_terms=exchange_terms,
        )

    @classmethod
    def build(
        cls,
        system_config: HybridSystemConfig,
        *,
        randomization: RandomizationSample | None = None,
        measurement_spec: QubitMeasurementSpec | None = None,
    ) -> EpisodeModelBundle:
        randomization = randomization or RandomizationSample()
        if system_config.regime == "reduced_dispersive":
            reduced_model = _apply_overrides_dataclass(system_config.reduced_model, randomization.model_overrides)
            model = cls._build_reduced_model(reduced_model)
        elif system_config.regime == "full_pulse":
            full_model = _apply_overrides_dataclass(system_config.full_model, randomization.model_overrides)
            model = cls._build_full_model(full_model)
        else:
            raise ValueError(f"Unsupported system regime '{system_config.regime}'.")

        frame = cls._resolve_frame(
            model,
            configured_frame=system_config.frame,
            use_model_rotating_frame=bool(system_config.use_model_rotating_frame),
        )

        noise = None if system_config.noise is None and not randomization.noise_overrides else _apply_overrides_dataclass(system_config.noise or NoiseSpec(), randomization.noise_overrides)
        hardware = _apply_hardware_overrides(system_config.hardware, randomization.hardware_overrides)
        resolved_measurement_spec = None
        if measurement_spec is not None or randomization.measurement_overrides:
            resolved_measurement_spec = replace(measurement_spec or QubitMeasurementSpec(), **randomization.measurement_overrides)
        return EpisodeModelBundle(
            model=model,
            regime=str(system_config.regime),
            frame=frame,
            noise=noise,
            hardware=hardware,
            crosstalk_matrix={channel: dict(values) for channel, values in system_config.crosstalk_matrix.items()},
            dt=float(system_config.dt),
            max_step=system_config.max_step,
            measurement_spec=resolved_measurement_spec,
            randomization=randomization,
        )


class PulseGenerator:
    def generate(self, action: Any, bundle: EpisodeModelBundle) -> ControlSegment:
        if isinstance(action, PrimitiveAction):
            return self.generate(self._primitive_to_action(action), bundle)
        if isinstance(action, QubitGaussianAction):
            return self._qubit_segment(action, bundle)
        if isinstance(action, CavityDisplacementAction):
            return self._cavity_segment(action, bundle)
        if isinstance(action, SidebandAction):
            return self._sideband_segment(action, bundle)
        if isinstance(action, WaitAction):
            return ControlSegment(pulses=[], drive_ops={}, duration=float(action.duration), metadata={"action_type": "wait", "duration": float(action.duration), "max_abs_amp": 0.0})
        if isinstance(action, MeasurementAction):
            return ControlSegment(pulses=[], drive_ops={}, duration=0.0, measurement_requested=True, metadata={"action_type": "measure", "duration": 0.0, "max_abs_amp": 0.0, "collapse": bool(action.collapse)})
        if isinstance(action, ResetAction):
            return ControlSegment(pulses=[], drive_ops={}, duration=0.0, reset_requested=True, metadata={"action_type": "reset", "duration": 0.0, "max_abs_amp": 0.0})
        if isinstance(action, HybridBlockAction):
            return self._hybrid_block_segment(action, bundle)
        if isinstance(action, WaveformAction):
            return self._waveform_segment(action)
        raise TypeError(f"Unsupported action type '{type(action).__name__}'.")

    def _primitive_to_action(self, action: PrimitiveAction) -> Any:
        primitive = str(action.primitive)
        if primitive == "qubit_gaussian":
            return QubitGaussianAction(
                theta=float(action.amplitude),
                phi=float(action.phase),
                detuning=float(action.detuning),
                duration=float(action.duration),
                drag=float(action.drag),
            )
        if primitive == "cavity_displacement":
            return CavityDisplacementAction(alpha=complex(action.alpha), duration=float(action.duration), detuning=float(action.detuning))
        if primitive == "sideband":
            return SidebandAction(
                amplitude=float(action.amplitude),
                detuning=float(action.detuning),
                duration=float(action.duration),
                lower_level=int(action.lower_level),
                upper_level=int(action.upper_level),
                mode=str(action.mode),
                sideband=str(action.sideband),
                phase=float(action.phase),
            )
        if primitive == "measure":
            return MeasurementAction(collapse=bool(action.collapse))
        if primitive == "reset":
            return ResetAction(ideal=True)
        return WaitAction(duration=float(action.duration))

    def _qubit_segment(self, action: QubitGaussianAction, bundle: EpisodeModelBundle) -> ControlSegment:
        transition = transmon_transition_frequency(bundle.model, cavity_level=0, lower_level=0, upper_level=1, frame=bundle.frame)
        detuning = float(action.detuning) + _drift_value(bundle.drift_state, "qubit_detuning_offset", "detuning_offset", 0.0)
        amplitude_scale = _drift_value(bundle.drift_state, "qubit_amplitude_scale", "amplitude_scale", 1.0)
        drive_frequency = drive_frequency_for_transition_frequency(transition + detuning, bundle.frame.omega_q_frame)
        carrier = internal_carrier_from_drive_frequency(drive_frequency, bundle.frame.omega_q_frame)
        pulse = Pulse(
            "qubit",
            0.0,
            float(action.duration),
            _gaussian_envelope(),
            carrier=carrier,
            phase=float(action.phi),
            amp=float(amplitude_scale * rotation_gaussian_amplitude(float(action.theta), float(action.duration))),
            drag=float(action.drag),
            label="rl_qubit_gaussian",
        )
        return ControlSegment(
            pulses=[pulse],
            drive_ops={"qubit": "qubit"},
            duration=float(action.duration),
            metadata={
                "action_type": "qubit_gaussian",
                "duration": float(action.duration),
                "max_abs_amp": float(abs(pulse.amp)),
                "drive_frequency": float(drive_frequency),
                "internal_carrier": float(carrier),
            },
        )

    def _cavity_segment(self, action: CavityDisplacementAction, bundle: EpisodeModelBundle) -> ControlSegment:
        transition = _mode_transition_frequency(bundle.model, mode="storage", frame=bundle.frame)
        detuning = float(action.detuning) + _drift_value(bundle.drift_state, "storage_detuning_offset", "detuning_offset", 0.0)
        amplitude_scale = _drift_value(bundle.drift_state, "storage_amplitude_scale", "amplitude_scale", 1.0)
        epsilon = amplitude_scale * displacement_square_amplitude(complex(action.alpha), float(action.duration))
        drive_frequency = drive_frequency_for_transition_frequency(transition + detuning, bundle.frame.omega_c_frame)
        carrier = internal_carrier_from_drive_frequency(drive_frequency, bundle.frame.omega_c_frame)
        pulse = Pulse(
            "storage",
            0.0,
            float(action.duration),
            square_envelope,
            carrier=carrier,
            phase=float(np.angle(epsilon)) if abs(epsilon) > 0.0 else 0.0,
            amp=float(abs(epsilon)),
            label="rl_storage_displacement",
        )
        return ControlSegment(
            pulses=[pulse],
            drive_ops={"storage": "storage"},
            duration=float(action.duration),
            metadata={
                "action_type": "cavity_displacement",
                "duration": float(action.duration),
                "max_abs_amp": float(abs(pulse.amp)),
                "target_alpha": complex(action.alpha),
                "drive_frequency": float(drive_frequency),
                "internal_carrier": float(carrier),
            },
        )

    def _sideband_segment(self, action: SidebandAction, bundle: EpisodeModelBundle) -> ControlSegment:
        transition = sideband_transition_frequency(
            bundle.model,
            cavity_level=0,
            lower_level=int(action.lower_level),
            upper_level=int(action.upper_level),
            mode=str(action.mode),
            sideband=str(action.sideband),
            frame=bundle.frame,
        )
        detuning = float(action.detuning) + _drift_value(bundle.drift_state, "sideband_detuning_offset", "detuning_offset", 0.0)
        amplitude_scale = _drift_value(bundle.drift_state, "sideband_amplitude_scale", "amplitude_scale", 1.0)
        modulation_frequency = float(transition + detuning)
        carrier = carrier_for_transition_frequency(modulation_frequency)
        pulse = Pulse(
            "sideband",
            0.0,
            float(action.duration),
            _gaussian_envelope(),
            carrier=carrier,
            phase=float(action.phase),
            amp=float(amplitude_scale * action.amplitude),
            label="rl_sideband",
        )
        drive_ops = {
            "sideband": SidebandDriveSpec(
                mode=str(action.mode),
                lower_level=int(action.lower_level),
                upper_level=int(action.upper_level),
                sideband=str(action.sideband),
            )
        }
        return ControlSegment(
            pulses=[pulse],
            drive_ops=drive_ops,
            duration=float(action.duration),
            metadata={
                "action_type": "sideband",
                "duration": float(action.duration),
                "max_abs_amp": float(abs(pulse.amp)),
                "mode": str(action.mode),
                "sideband": str(action.sideband),
                "modulation_frequency": modulation_frequency,
                "internal_carrier": float(carrier),
            },
        )

    def _hybrid_block_segment(self, action: HybridBlockAction, bundle: EpisodeModelBundle) -> ControlSegment:
        pulses: list[Pulse] = []
        drive_ops: dict[str, Any] = {}
        cursor = 0.0
        if action.cavity_duration > 0.0 and abs(action.cavity_alpha) > 0.0:
            segment = self._cavity_segment(CavityDisplacementAction(alpha=action.cavity_alpha, duration=action.cavity_duration, detuning=action.cavity_detuning), bundle)
            pulses.extend(_shift_pulses(segment.pulses, cursor))
            drive_ops.update(segment.drive_ops)
            cursor += float(segment.duration)
        if action.qubit_duration > 0.0 and abs(action.qubit_theta) > 0.0:
            segment = self._qubit_segment(
                QubitGaussianAction(
                    theta=action.qubit_theta,
                    phi=action.qubit_phi,
                    detuning=action.qubit_detuning,
                    duration=action.qubit_duration,
                    drag=action.qubit_drag,
                ),
                bundle,
            )
            pulses.extend(_shift_pulses(segment.pulses, cursor))
            drive_ops.update(segment.drive_ops)
            cursor += float(segment.duration)
        if action.sideband_duration > 0.0 and abs(action.sideband_amplitude) > 0.0:
            segment = self._sideband_segment(
                SidebandAction(
                    amplitude=action.sideband_amplitude,
                    detuning=action.sideband_detuning,
                    duration=action.sideband_duration,
                    phase=action.sideband_phase,
                ),
                bundle,
            )
            pulses.extend(_shift_pulses(segment.pulses, cursor))
            drive_ops.update(segment.drive_ops)
            cursor += float(segment.duration)
        cursor += float(max(0.0, action.wait_duration))
        max_abs_amp = 0.0 if not pulses else float(max(abs(pulse.amp) for pulse in pulses))
        return ControlSegment(
            pulses=pulses,
            drive_ops=drive_ops,
            duration=float(cursor),
            measurement_requested=bool(action.measurement_requested),
            metadata={
                "action_type": "hybrid_block",
                "duration": float(cursor),
                "max_abs_amp": max_abs_amp,
                "num_pulses": len(pulses),
                "measurement_requested": bool(action.measurement_requested),
            },
        )

    def _waveform_segment(self, action: WaveformAction) -> ControlSegment:
        pulses: list[Pulse] = []
        drive_ops: dict[str, Any] = {}
        for channel, samples in action.channel_samples.items():
            drive_ops[str(channel)] = "qubit" if channel == "qubit" else "storage"
            pulses.append(
                Pulse(
                    str(channel),
                    0.0,
                    float(action.duration),
                    np.asarray(samples, dtype=np.complex128),
                    sample_rate=1.0 / float(action.dt),
                    amp=1.0,
                    label=f"waveform_{channel}",
                )
            )
        max_abs_amp = 0.0 if not pulses else float(max(np.max(np.abs(np.asarray(pulse.envelope))) for pulse in pulses if not callable(pulse.envelope)))
        return ControlSegment(
            pulses=pulses,
            drive_ops=drive_ops,
            duration=float(action.duration),
            metadata={"action_type": "waveform", "duration": float(action.duration), "max_abs_amp": max_abs_amp},
        )


class DistortionModel:
    def __init__(self, bundle: EpisodeModelBundle):
        self.compiler = SequenceCompiler(
            dt=float(bundle.dt),
            hardware=bundle.hardware,
            crosstalk_matrix=bundle.crosstalk_matrix,
            enable_cache=False,
        )

    def compile(self, segment: ControlSegment):
        t_end = None if segment.pulses else float(segment.duration)
        if segment.pulses:
            t_end = max(float(segment.duration), max(float(pulse.t1) for pulse in segment.pulses))
        return self.compiler.compile(segment.pulses, t_end=t_end)


class OpenSystemEngine:
    def __init__(self, bundle: EpisodeModelBundle):
        self.bundle = bundle

    def _simulation_config(self, *, store_states: bool = False) -> SimulationConfig:
        return SimulationConfig(frame=self.bundle.frame, max_step=self.bundle.max_step, store_states=store_states)

    def propagate_state(self, initial_state: qt.Qobj, compiled: Any, drive_ops: dict[str, Any], *, store_states: bool = False):
        return simulate_sequence(
            self.bundle.model,
            compiled,
            initial_state,
            drive_ops,
            config=self._simulation_config(store_states=store_states),
            noise=self.bundle.noise,
            e_ops={} if not store_states else None,
        )

    def propagate_states(self, initial_states: list[qt.Qobj], compiled: Any, drive_ops: dict[str, Any]) -> list[Any]:
        session = prepare_simulation(
            self.bundle.model,
            compiled,
            drive_ops,
            config=self._simulation_config(store_states=False),
            noise=self.bundle.noise,
            e_ops={},
        )
        return session.run_many(initial_states, max_workers=1)


class MeasurementModel:
    def __init__(self, spec: QubitMeasurementSpec | None = None, *, collapse_on_measurement: bool = False):
        self.spec = spec
        self.collapse_on_measurement = bool(collapse_on_measurement)

    def observe(self, state: qt.Qobj, *, bundle: EpisodeModelBundle, seed: int | None = None) -> Any:
        base_spec = bundle.measurement_spec or self.spec or QubitMeasurementSpec()
        spec = replace(base_spec, seed=seed) if seed is not None else base_spec
        return measure_qubit(state, spec)

    def collapse_joint_state(self, state: qt.Qobj, measurement: Any, *, seed: int | None = None) -> qt.Qobj:
        rng = np.random.default_rng(seed)
        if measurement.samples is not None and len(measurement.samples) > 0:
            outcome = int(measurement.samples[0])
        else:
            outcome = int(rng.choice(np.array([0, 1], dtype=int), p=[measurement.observed_probabilities["g"], measurement.observed_probabilities["e"]]))
        dims = [int(dim) for dim in state.dims[0]]
        projector = qt.basis(dims[0], outcome) * qt.basis(dims[0], outcome).dag()
        full_projector = qt.tensor(projector, *(qt.qeye(dim) for dim in dims[1:]))
        if state.isoper:
            collapsed = full_projector * state * full_projector
            norm = float(np.real(collapsed.tr()))
            return state if norm <= 1.0e-15 else collapsed / norm
        collapsed = full_projector * state
        norm = float(collapsed.norm())
        return state if norm <= 1.0e-15 else collapsed / norm


class ClassicalProcessor:
    @staticmethod
    def measurement_vector(measurement: Any, *, mode: str = "iq_mean") -> np.ndarray:
        if measurement is None:
            if mode in {"iq_mean", "counts", "classifier_probs", "classifier_logits", "outcome_onehot"}:
                return np.zeros(2, dtype=float)
            return np.zeros(2, dtype=float)
        if mode == "iq_mean":
            if measurement.iq_samples is None or len(measurement.iq_samples) == 0:
                return np.zeros(2, dtype=float)
            return np.asarray(np.mean(measurement.iq_samples, axis=0), dtype=float)
        if mode == "counts":
            return np.asarray([measurement.observed_probabilities["g"], measurement.observed_probabilities["e"]], dtype=float)
        if mode == "classifier_probs":
            return np.asarray([measurement.observed_probabilities["g"], measurement.observed_probabilities["e"]], dtype=float)
        if mode == "classifier_logits":
            probabilities = np.clip(
                np.asarray([measurement.observed_probabilities["g"], measurement.observed_probabilities["e"]], dtype=float),
                1.0e-12,
                1.0,
            )
            return np.log(probabilities)
        if mode == "outcome_onehot":
            if measurement.samples is not None and len(measurement.samples) > 0:
                outcome = int(measurement.samples[0])
            else:
                outcome = 0 if measurement.observed_probabilities["g"] >= measurement.observed_probabilities["e"] else 1
            vector = np.zeros(2, dtype=float)
            vector[int(np.clip(outcome, 0, 1))] = 1.0
            return vector
        raise ValueError(f"Unsupported classical processing mode '{mode}'.")


__all__ = [
    "ControlSegment",
    "EpisodeModelBundle",
    "HamiltonianModelFactory",
    "PulseGenerator",
    "DistortionModel",
    "OpenSystemEngine",
    "MeasurementModel",
    "ClassicalProcessor",
]