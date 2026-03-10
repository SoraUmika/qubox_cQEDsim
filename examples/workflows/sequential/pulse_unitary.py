from __future__ import annotations

from typing import Any, Mapping

import qutip as qt

from cqed_sim.calibration.sqr import SQRCalibrationResult
from cqed_sim.io.gates import DisplacementGate, Gate, RotationGate, SQRGate
from cqed_sim.pulses.builders import build_displacement_pulse, build_rotation_pulse, build_sqr_multitone_pulse
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from .common import (
    build_frame,
    build_initial_state,
    build_model,
    build_noise_spec,
    finalize_track,
    snapshot_from_state,
)


def build_gate_segment(
    gate: Gate,
    model,
    config: Mapping[str, Any],
    sqr_calibration_map: Mapping[str, SQRCalibrationResult] | None = None,
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    if isinstance(gate, DisplacementGate):
        return build_displacement_pulse(gate, config)
    if isinstance(gate, RotationGate):
        return build_rotation_pulse(gate, config)
    if isinstance(gate, SQRGate):
        calibration = None if sqr_calibration_map is None else sqr_calibration_map.get(gate.name)
        return build_sqr_multitone_pulse(gate, model, config, calibration=calibration)
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
    sqr_calibration_map: Mapping[str, SQRCalibrationResult] | None = None,
) -> dict[str, Any]:
    model = build_model(config)
    noise = build_noise_spec(config, enabled=include_dissipation)
    state = build_initial_state(config, n_cav_dim=model.n_cav)
    snapshots = [snapshot_from_state(state, 0, None, config, case_label=case_label)]
    mapping_rows: list[dict[str, Any]] = []
    for step_index, gate in enumerate(gates, start=1):
        pulses, drive_ops, meta = build_gate_segment(gate, model, config, sqr_calibration_map=sqr_calibration_map)
        state = evolve_segment(model, state, pulses, drive_ops, config, noise)
        snapshots.append(snapshot_from_state(state, step_index, gate, config, case_label=case_label, extra=meta))
        mapping_rows.append({"index": step_index, "type": gate.type, "name": gate.name, **meta})
    solver_name = "mesolve" if include_dissipation else "sesolve"
    return finalize_track(
        case_label,
        snapshots,
        metadata={
            "solver": solver_name,
            "mapping_rows": mapping_rows,
            "model": model,
            "noise": noise,
            "sqr_calibration_map": None if sqr_calibration_map is None else dict(sqr_calibration_map),
        },
    )


def run_case_b(gates: list[Gate], config: Mapping[str, Any], case_label: str = "Case B") -> dict[str, Any]:
    return run_pulse_case(gates, config, include_dissipation=False, case_label=case_label)
