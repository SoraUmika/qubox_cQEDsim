from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import qutip as qt

from cqed_sim.io.gates import DisplacementGate, Gate, RotationGate, SQRGate
from cqed_sim.pulses.calibration import (
    build_sqr_tone_specs,
    displacement_square_amplitude,
    rotation_gaussian_amplitude,
)
from cqed_sim.pulses.envelopes import multitone_gaussian_envelope, normalized_gaussian, square_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from cqed_sim.simulators.common import (
    build_frame,
    build_initial_state,
    build_model,
    build_noise_spec,
    finalize_track,
    snapshot_from_state,
)


def build_displacement_pulse(
    gate: DisplacementGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    duration_s = float(config["duration_displacement_s"])
    eps = displacement_square_amplitude(gate.alpha, duration_s)
    pulse = Pulse(
        "storage",
        0.0,
        duration_s,
        square_envelope,
        amp=float(abs(eps)),
        phase=float(np.angle(eps)) if abs(eps) > 0 else 0.0,
        label=gate.name,
    )
    return [pulse], {"storage": "cavity"}, {
        "mapping": "Square cavity drive with analytic rotating-frame calibration alpha = -i * integral epsilon(t) dt.",
        "duration_s": duration_s,
        "drive_amp": pulse.amp,
        "drive_phase": pulse.phase,
        "target_alpha": gate.alpha,
    }


def build_rotation_pulse(
    gate: RotationGate,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    duration_s = float(config["duration_rotation_s"])
    sigma_fraction = float(config["rotation_sigma_fraction"])

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return normalized_gaussian(t_rel, sigma_fraction=sigma_fraction)

    pulse = Pulse(
        "qubit",
        0.0,
        duration_s,
        envelope,
        amp=rotation_gaussian_amplitude(gate.theta, duration_s),
        phase=gate.phi,
        label=gate.name,
    )
    return [pulse], {"qubit": "qubit"}, {
        "mapping": "Gaussian qubit drive with analytic RWA calibration theta = 2 * integral Omega(t) dt.",
        "duration_s": duration_s,
        "drive_amp": pulse.amp,
        "drive_phase": pulse.phase,
        "sigma_fraction": sigma_fraction,
    }


def build_sqr_multitone_pulse(
    gate: SQRGate,
    model,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    duration_s = float(config["duration_sqr_s"])
    sigma_fraction = float(config["sqr_sigma_fraction"])
    frame = build_frame(model, config)
    tone_specs = build_sqr_tone_specs(
        model=model,
        frame=frame,
        theta_values=list(gate.theta),
        phi_values=list(gate.phi),
        duration_s=duration_s,
        tone_cutoff=float(config["sqr_theta_cutoff"]),
    )

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return multitone_gaussian_envelope(
            t_rel,
            duration_s=duration_s,
            sigma_fraction=sigma_fraction,
            tone_specs=tone_specs,
        )

    pulse = Pulse("qubit", 0.0, duration_s, envelope, amp=1.0, phase=0.0, label=gate.name)
    return [pulse], {"qubit": "qubit"}, {
        "mapping": "Simplified multitone Gaussian SQR using cqed_sim manifold_transition_frequency(...) and per-tone RWA area calibration.",
        "duration_s": duration_s,
        "drive_amp": None,
        "drive_phase": None,
        "sigma_fraction": sigma_fraction,
        "active_tones": [spec.as_dict() for spec in tone_specs],
    }


def build_gate_segment(
    gate: Gate,
    model,
    config: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    if isinstance(gate, DisplacementGate):
        return build_displacement_pulse(gate, config)
    if isinstance(gate, RotationGate):
        return build_rotation_pulse(gate, config)
    if isinstance(gate, SQRGate):
        return build_sqr_multitone_pulse(gate, model, config)
    raise TypeError(f"Unsupported gate type '{type(gate).__name__}'.")


def evolve_segment(
    model,
    state: qt.Qobj,
    pulses: list[Pulse],
    drive_ops: dict[str, str],
    config: Mapping[str, Any],
    noise: NoiseSpec | None,
) -> qt.Qobj:
    duration_s = max((pulse.t1 for pulse in pulses), default=0.0)
    compiled = SequenceCompiler(dt=float(config["dt_s"])).compile(
        pulses,
        t_end=duration_s + float(config["dt_s"]),
    )
    result = simulate_sequence(
        model,
        compiled,
        state,
        drive_ops,
        config=SimulationConfig(
            frame=build_frame(model, config),
            max_step=float(config["max_step_s"]),
            store_states=False,
        ),
        noise=noise,
    )
    return result.final_state


def run_pulse_case(
    gates: list[Gate],
    config: Mapping[str, Any],
    include_dissipation: bool,
    case_label: str,
) -> dict[str, Any]:
    model = build_model(config)
    noise = build_noise_spec(config, enabled=include_dissipation)
    state = build_initial_state(config, n_cav_dim=model.n_cav)
    snapshots = [snapshot_from_state(state, 0, None, config, case_label=case_label)]
    mapping_rows: list[dict[str, Any]] = []
    for step_index, gate in enumerate(gates, start=1):
        pulses, drive_ops, meta = build_gate_segment(gate, model, config)
        state = evolve_segment(model, state, pulses, drive_ops, config, noise)
        snapshots.append(snapshot_from_state(state, step_index, gate, config, case_label=case_label, extra=meta))
        mapping_rows.append({"index": step_index, "type": gate.type, "name": gate.name, **meta})
    solver_name = "mesolve" if include_dissipation else "sesolve"
    return finalize_track(
        case_label,
        snapshots,
        metadata={"solver": solver_name, "mapping_rows": mapping_rows, "model": model, "noise": noise},
    )


def run_case_b(gates: list[Gate], config: Mapping[str, Any], case_label: str = "Case B") -> dict[str, Any]:
    return run_pulse_case(gates, config, include_dissipation=False, case_label=case_label)
