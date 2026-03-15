from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from cqed_sim.core import FrameSpec
from cqed_sim.io.gates import DisplacementGate, RotationGate, SQRGate
from cqed_sim.pulses.builders import build_displacement_pulse, build_rotation_pulse, build_sqr_multitone_pulse

from .sequence import Displacement, GateSequence, PrimitiveGate, QubitRotation, SQR


_SUPPORTED_GATE_TYPES = (QubitRotation, Displacement, SQR)


def _default_time_bounds(duration: float) -> tuple[float, float]:
    duration = float(duration)
    lower = max(duration * 0.25, 1.0e-9)
    upper = max(duration * 4.0, lower + 1.0e-9)
    return lower, upper


def _resolve_duration_bounds(gate: Any) -> tuple[float, float]:
    bounds = getattr(gate, "time_bounds", None)
    if bounds is None:
        return _default_time_bounds(getattr(gate, "duration", 40.0e-9))
    return float(bounds[0]), float(bounds[1])


def _base_waveform_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = {
        "rotation_sigma_fraction": 0.18,
        "sqr_sigma_fraction": 0.18,
        "sqr_theta_cutoff": 1.0e-8,
        "use_rotating_frame": False,
        "fock_fqs_hz": None,
    }
    if config is not None:
        merged.update(dict(config))
    return merged


def waveform_primitive_from_gate(
    gate: QubitRotation | Displacement | SQR,
    *,
    index: int = 0,
    frame: FrameSpec | None = None,
    calibration: Any | None = None,
    config: Mapping[str, Any] | None = None,
    hilbert_dim: int | None = None,
) -> PrimitiveGate:
    if not isinstance(gate, _SUPPORTED_GATE_TYPES):
        raise TypeError(
            "waveform_primitive_from_gate supports only QubitRotation, Displacement, and SQR gates."
        )

    base_config = _base_waveform_config(config)
    time_bounds = _resolve_duration_bounds(gate)
    duration = float(gate.duration)
    duration_ref = float(getattr(gate, "duration_ref", duration))
    common = {
        "name": str(gate.name),
        "duration": duration,
        "optimize_time": bool(getattr(gate, "optimize_time", True)),
        "time_bounds": time_bounds,
        "duration_ref": duration_ref,
        "time_group": getattr(gate, "time_group", None),
        "time_policy_locked": bool(getattr(gate, "time_policy_locked", False)),
        "hilbert_dim": hilbert_dim,
    }

    if isinstance(gate, QubitRotation):
        def waveform(params: dict[str, Any], model: Any) -> Any:
            io_gate = RotationGate(
                index=int(index),
                name=str(gate.name),
                theta=float(params["theta"]),
                phi=float(params["phi"]),
            )
            local_config = dict(base_config)
            local_config["duration_rotation_s"] = float(params["duration"])
            return build_rotation_pulse(io_gate, local_config)

        return PrimitiveGate(
            waveform=waveform,
            parameters={"theta": float(gate.theta), "phi": float(gate.phi), "duration": duration},
            parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "phi": (-np.pi, np.pi), "duration": time_bounds},
            metadata={"source_gate_type": gate.type, "waveform_family": "rotation_gaussian", "frame": None if frame is None else frame.__dict__},
            **common,
        )

    if isinstance(gate, Displacement):
        def waveform(params: dict[str, Any], model: Any) -> Any:
            alpha = complex(params["alpha"])
            io_gate = DisplacementGate(
                index=int(index),
                name=str(gate.name),
                re=float(alpha.real),
                im=float(alpha.imag),
            )
            local_config = dict(base_config)
            local_config["duration_displacement_s"] = float(params["duration"])
            return build_displacement_pulse(io_gate, local_config)

        return PrimitiveGate(
            waveform=waveform,
            parameters={"alpha": complex(gate.alpha), "duration": duration},
            parameter_bounds={"alpha": (-2.0, 2.0), "duration": time_bounds},
            metadata={"source_gate_type": gate.type, "waveform_family": "displacement_square", "frame": None if frame is None else frame.__dict__},
            **common,
        )

    def waveform(params: dict[str, Any], model: Any) -> Any:
        theta = tuple(np.asarray(params["theta"], dtype=float).tolist())
        phi = tuple(np.asarray(params["phi"], dtype=float).tolist())
        io_gate = SQRGate(
            index=int(index),
            name=str(gate.name),
            theta=theta,
            phi=phi,
        )
        local_config = dict(base_config)
        local_config["duration_sqr_s"] = float(params["duration"])
        return build_sqr_multitone_pulse(io_gate, model, local_config, frame=frame, calibration=calibration)

    return PrimitiveGate(
        waveform=waveform,
        parameters={
            "theta": np.asarray(gate.theta_n, dtype=float),
            "phi": np.asarray(gate.phi_n, dtype=float),
            "duration": duration,
        },
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "phi": (-np.pi, np.pi), "duration": time_bounds},
        metadata={"source_gate_type": gate.type, "waveform_family": "sqr_multitone_gaussian", "frame": None if frame is None else frame.__dict__},
        **common,
    )


def waveform_sequence_from_gates(
    sequence: GateSequence,
    *,
    frame: FrameSpec | None = None,
    calibration: Any | None = None,
    config: Mapping[str, Any] | None = None,
    hilbert_dim: int | None = None,
) -> GateSequence:
    converted: list[PrimitiveGate] = []
    resolved_dim = hilbert_dim if hilbert_dim is not None else sequence.full_dim
    for index, gate in enumerate(sequence.gates):
        converted.append(
            waveform_primitive_from_gate(
                gate,
                index=index,
                frame=frame,
                calibration=calibration,
                config=config,
                hilbert_dim=resolved_dim,
            )
        )
    return GateSequence(gates=converted, n_cav=sequence.n_cav, full_dim=sequence.full_dim)


__all__ = ["waveform_primitive_from_gate", "waveform_sequence_from_gates"]