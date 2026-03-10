from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from cqed_sim.core.ideal_gates import displacement_op, qubit_rotation_xy, sqr_op
from cqed_sim.io.gates import DisplacementGate, Gate, RotationGate, SQRGate
from cqed_sim.observables.trajectories import bloch_trajectory_from_states
from cqed_sim.operators.basic import embed_cavity_op, embed_qubit_op
from cqed_sim.pulses.calibration import pad_sqr_angles
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from .common import build_frame, build_model, build_noise_spec
from .pulse_unitary import build_gate_segment


def simulate_gate_bloch_trajectory(
    track: dict[str, Any],
    gates: list[Gate],
    config: Mapping[str, Any],
    gate_index: int,
    conditioned_n_levels: list[int] | tuple[int, ...] | None = None,
    probability_threshold: float = 1.0e-8,
    include_dissipation: bool | None = None,
) -> dict[str, Any]:
    if gate_index < 1 or gate_index > len(gates):
        raise IndexError(f"gate_index must be in [1, {len(gates)}], got {gate_index}.")

    pre_gate_state = track["snapshots"][gate_index - 1]["state"]
    gate = gates[gate_index - 1]
    model = track["metadata"].get("model") or build_model(config)
    noise = track["metadata"].get("noise")
    if include_dissipation is not None:
        noise = build_noise_spec(config, enabled=include_dissipation)

    pulses, drive_ops, meta = build_gate_segment(
        gate,
        model,
        config,
        sqr_calibration_map=track["metadata"].get("sqr_calibration_map"),
    )
    duration_s = max((pulse.t1 for pulse in pulses), default=0.0)
    compiled = SequenceCompiler(dt=float(config["dt_s"])).compile(
        pulses,
        t_end=duration_s + float(config["dt_s"]),
    )
    result = simulate_sequence(
        model,
        compiled,
        pre_gate_state,
        drive_ops,
        config=SimulationConfig(
            frame=build_frame(model, config),
            max_step=float(config["max_step_s"]),
            store_states=True,
        ),
        noise=noise,
    )
    if result.states is None:
        raise RuntimeError("Expected stored states for gate-trajectory simulation.")

    trajectory = bloch_trajectory_from_states(
        result.states,
        conditioned_n_levels=conditioned_n_levels,
        probability_threshold=probability_threshold,
    )
    trajectory.update(
        {
            "case": track["case"],
            "gate_index": int(gate_index),
            "gate_type": gate.type,
            "gate_name": gate.name,
            "times_s": np.asarray(compiled.tlist, dtype=float),
            "times_ns": np.asarray(compiled.tlist, dtype=float) * 1.0e9,
            "mapping": meta.get("mapping"),
            "include_dissipation": noise is not None,
            "conditioned_n_levels": [] if conditioned_n_levels is None else [int(n) for n in conditioned_n_levels],
        }
    )
    return trajectory


def ideal_gate_partial_unitary(gate: Gate, n_cav_dim: int, fraction: float):
    frac = float(np.clip(fraction, 0.0, 1.0))
    if isinstance(gate, DisplacementGate):
        return embed_cavity_op(displacement_op(n_cav_dim, frac * gate.alpha), n_tr=2)
    if isinstance(gate, RotationGate):
        return embed_qubit_op(qubit_rotation_xy(frac * gate.theta, gate.phi), n_cav_dim)
    if isinstance(gate, SQRGate):
        theta, phi = pad_sqr_angles(gate.theta, gate.phi, n_cav_dim)
        return sqr_op([frac * value for value in theta], phi)
    raise TypeError(f"Unsupported gate type '{type(gate).__name__}'.")


def ideal_gate_bloch_trajectory(
    ideal_track: dict[str, Any],
    gates: list[Gate],
    config: Mapping[str, Any],
    gate_index: int,
    times_s: np.ndarray,
    conditioned_n_levels: list[int] | tuple[int, ...] | None = None,
    probability_threshold: float = 1.0e-8,
) -> dict[str, Any]:
    if gate_index < 1 or gate_index > len(gates):
        raise IndexError(f"gate_index must be in [1, {len(gates)}], got {gate_index}.")

    gate = gates[gate_index - 1]
    pre_gate_state = ideal_track["snapshots"][gate_index - 1]["state"]
    times = np.asarray(times_s, dtype=float)
    duration_s = max(float(times[-1]), 1.0e-18)
    n_cav_dim = int(config["n_cav_dim"])
    states = []
    for t in times:
        unitary = ideal_gate_partial_unitary(gate, n_cav_dim=n_cav_dim, fraction=float(t / duration_s))
        states.append(unitary * pre_gate_state * unitary.dag())

    trajectory = bloch_trajectory_from_states(
        states,
        conditioned_n_levels=conditioned_n_levels,
        probability_threshold=probability_threshold,
    )
    trajectory.update(
        {
            "case": ideal_track["case"],
            "gate_index": int(gate_index),
            "gate_type": gate.type,
            "gate_name": gate.name,
            "times_s": times,
            "times_ns": times * 1.0e9,
            "mapping": "Ideal fractional gate path on the same time grid as the simulated segment.",
            "include_dissipation": False,
            "conditioned_n_levels": [] if conditioned_n_levels is None else [int(n) for n in conditioned_n_levels],
            "trajectory_kind": "ideal_fractional_gate",
        }
    )
    return trajectory
