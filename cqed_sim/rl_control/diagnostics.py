from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.sim import cavity_wigner, reduced_cavity_state, reduced_qubit_state, transmon_level_populations

from .metrics import ancilla_return_metric, parity_expectation, photon_number_distribution


def _state_payload(state: qt.Qobj) -> dict[str, Any]:
    return {
        "is_density_matrix": bool(state.isoper),
        "array": np.asarray(state.full(), dtype=np.complex128),
    }


def _channel_payload(compiled: Any) -> dict[str, dict[str, np.ndarray]]:
    if compiled is None:
        return {}
    payload: dict[str, dict[str, np.ndarray]] = {}
    for name, channel in compiled.channels.items():
        payload[name] = {
            "baseband": np.asarray(channel.baseband, dtype=np.complex128),
            "distorted": np.asarray(channel.distorted, dtype=np.complex128),
            "rf": np.asarray(channel.rf, dtype=float),
        }
    return payload


def _pulse_summary(segment: Any) -> list[dict[str, Any]]:
    if segment is None:
        return []
    payload: list[dict[str, Any]] = []
    for pulse in getattr(segment, "pulses", []):
        payload.append(
            {
                "channel": str(pulse.channel),
                "t0": float(pulse.t0),
                "duration": float(pulse.duration),
                "carrier": float(pulse.carrier),
                "phase": float(pulse.phase),
                "amp": float(pulse.amp),
                "drag": float(pulse.drag),
                "label": pulse.label,
            }
        )
    return payload


def build_rollout_diagnostics(
    *,
    model: Any,
    state: qt.Qobj | None,
    probe_states: list[qt.Qobj] | None,
    compiled: Any = None,
    segment: Any = None,
    measurement: Any = None,
    metrics: dict[str, Any] | None = None,
    randomization: dict[str, Any] | None = None,
    regime: str | None = None,
    frame: Any = None,
    include_wigner: bool = True,
    wigner_points: int = 41,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "model_class": model.__class__.__name__,
        "regime": None if regime is None else str(regime),
        "subsystem_dims": tuple(int(dim) for dim in getattr(model, "subsystem_dims")),
        "frame": None if frame is None else {
            "omega_c_frame": float(frame.omega_c_frame),
            "omega_q_frame": float(frame.omega_q_frame),
            "omega_r_frame": float(frame.omega_r_frame),
        },
        "compiled_tlist": None if compiled is None else np.asarray(compiled.tlist, dtype=float),
        "channels": _channel_payload(compiled),
        "segment_metadata": {} if segment is None else dict(getattr(segment, "metadata", {})),
        "pulse_summary": _pulse_summary(segment),
        "measurement": None if measurement is None else {
            "probabilities": dict(measurement.probabilities),
            "observed_probabilities": dict(measurement.observed_probabilities),
            "counts": None if measurement.counts is None else dict(measurement.counts),
            "iq_samples": None if measurement.iq_samples is None else np.asarray(measurement.iq_samples, dtype=float),
            "readout_metadata": measurement.readout_metadata,
        },
        "metrics": {} if metrics is None else dict(metrics),
        "randomization": {} if randomization is None else dict(randomization),
    }
    if state is not None:
        diagnostics["joint_state"] = _state_payload(state)
        diagnostics["reduced_qubit_state"] = _state_payload(reduced_qubit_state(state))
        diagnostics["reduced_cavity_state"] = _state_payload(reduced_cavity_state(state))
        diagnostics["ancilla_populations"] = {int(level): float(value) for level, value in transmon_level_populations(state).items()}
        diagnostics["photon_number_distribution"] = photon_number_distribution(state)
        diagnostics["ancilla_return"] = float(ancilla_return_metric(state))
        diagnostics["parity"] = float(parity_expectation(state))
        if include_wigner:
            x_values, y_values, wigner = cavity_wigner(reduced_cavity_state(state), n_points=int(wigner_points))
            diagnostics["wigner"] = {
                "x": x_values,
                "y": y_values,
                "values": wigner,
            }
    if probe_states is not None:
        diagnostics["probe_states"] = [_state_payload(probe_state) for probe_state in probe_states]
    return diagnostics


__all__ = ["build_rollout_diagnostics"]