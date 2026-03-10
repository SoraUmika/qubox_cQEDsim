from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from cqed_sim.calibration.sqr import SQRCalibrationResult
from cqed_sim.io.gates import Gate, SQRGate, load_gate_sequence
from cqed_sim.pulses.calibration import build_sqr_tone_specs, pad_sqr_angles
from cqed_sim.pulses.pulse import Pulse
from .sequential.common import build_frame, build_initial_state, build_model, build_noise_spec, finalize_track, snapshot_from_state
from .sequential.pulse_unitary import build_gate_segment, build_sqr_multitone_pulse, evolve_segment


SQR_TRANSFER_SCHEMA_VERSION = "cqed_sim.sqr_transfer.v1"


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonify(value.item())
    if isinstance(value, complex):
        return {"re": float(value.real), "im": float(value.imag)}
    return value


def _coerce_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    arr = np.asarray(values, dtype=float).reshape(-1)
    return [float(item) for item in arr.tolist()]


def _pad_float_list(values: Sequence[float], size: int) -> list[float]:
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size < int(size):
        arr = np.pad(arr, (0, int(size) - arr.size), constant_values=0.0)
    return [float(item) for item in arr[: int(size)].tolist()]


def _sample_complex_envelope(duration_s: float, dt_s: float, envelope: Pulse) -> tuple[np.ndarray, np.ndarray]:
    n_samples = max(1, int(round(float(duration_s) / float(dt_s))))
    tlist = np.arange(n_samples, dtype=float) * float(dt_s)
    coeff = np.asarray(envelope.sample(tlist), dtype=np.complex128)
    return tlist, coeff


def gate_calibration_from_experiment_entry(
    entry: Mapping[str, Any],
    gate: SQRGate | None = None,
) -> SQRCalibrationResult | None:
    params = dict(entry.get("params", {}))
    d_lambda = _coerce_float_list(params.get("d_lambda"))
    d_alpha = _coerce_float_list(params.get("d_alpha"))
    d_omega = _coerce_float_list(params.get("d_omega"))
    if not d_lambda and not d_alpha and not d_omega:
        return None

    theta_target = [] if gate is None else [float(value) for value in gate.theta]
    phi_target = [] if gate is None else [float(value) for value in gate.phi]
    n_levels = max(len(theta_target), len(phi_target), len(d_lambda), len(d_alpha), len(d_omega), 1)
    d_omega_is_hz = bool(params.get("d_omega_is_hz", False))
    d_omega_rad_s = [
        float(2.0 * np.pi * value) if d_omega_is_hz else float(value)
        for value in _pad_float_list(d_omega, n_levels)
    ]

    return SQRCalibrationResult(
        sqr_name=str(entry.get("name", gate.name if gate is not None else "unnamed_sqr")),
        max_n=int(n_levels - 1),
        d_lambda=_pad_float_list(d_lambda, n_levels),
        d_alpha=_pad_float_list(d_alpha, n_levels),
        d_omega_rad_s=d_omega_rad_s,
        theta_target=_pad_float_list(theta_target, n_levels),
        phi_target=_pad_float_list(phi_target, n_levels),
        initial_loss=[0.0] * int(n_levels),
        optimized_loss=[0.0] * int(n_levels),
        levels=[],
        metadata={
            "source": "experiment_decomposition",
            "d_omega_is_hz": bool(d_omega_is_hz),
            "ref_sel_pulse": params.get("ref_sel_pulse"),
            "ref_sel_x180_pulse": params.get("ref_sel_x180_pulse"),
        },
    )


def load_gate_sequence_with_sqr_corrections(
    path_like: str | Path,
) -> tuple[Path, list[Gate], dict[str, SQRCalibrationResult], list[dict[str, Any]]]:
    chosen, gates = load_gate_sequence(path_like)
    raw = json.loads(chosen.read_text(encoding="utf-8"))
    calibration_map: dict[str, SQRCalibrationResult] = {}
    for gate, entry in zip(gates, raw):
        if not isinstance(gate, SQRGate):
            continue
        calibration = gate_calibration_from_experiment_entry(entry, gate=gate)
        if calibration is not None:
            calibration_map[gate.name] = calibration
    return chosen, list(gates), calibration_map, list(raw)


def build_sqr_transfer_artifact(
    gate: SQRGate,
    config: Mapping[str, Any],
    *,
    calibration: SQRCalibrationResult | None = None,
    reference_pulse: Mapping[str, Any] | None = None,
    sample_dt_s: float | None = None,
    source_path: str | Path | None = None,
    source_notebook: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(gate, SQRGate):
        raise TypeError(f"build_sqr_transfer_artifact expects SQRGate, got {type(gate).__name__}.")

    cfg = dict(config)
    duration_s = float(cfg["duration_sqr_s"])
    sample_dt = float(cfg.get("dt_s", 1.0e-9) if sample_dt_s is None else sample_dt_s)
    model = build_model(cfg)
    frame = build_frame(model, cfg)
    pulses, _drive_ops, pulse_meta = build_sqr_multitone_pulse(gate, model, cfg, calibration=calibration)
    pulse = pulses[0]
    tlist, coeff = _sample_complex_envelope(duration_s=float(pulse.duration), dt_s=sample_dt, envelope=pulse)

    theta_target, phi_target = pad_sqr_angles(gate.theta, gate.phi, int(cfg["n_cav_dim"]))
    d_lambda_values = None if calibration is None else list(calibration.d_lambda)
    raw_specs = build_sqr_tone_specs(
        model=model,
        frame=frame,
        theta_values=list(gate.theta),
        phi_values=list(gate.phi),
        duration_s=duration_s,
        d_lambda_values=d_lambda_values,
        fock_fqs_hz=cfg.get("fock_fqs_hz"),
        include_all_levels=True,
        tone_cutoff=float(cfg.get("sqr_theta_cutoff", 1.0e-10)),
    )
    spec_by_n = {int(spec.manifold): spec for spec in raw_specs}

    tone_rows: list[dict[str, Any]] = []
    for n in range(int(cfg["n_cav_dim"])):
        spec = spec_by_n.get(int(n))
        d_lambda, d_alpha, d_omega_rad_s = (0.0, 0.0, 0.0) if calibration is None else calibration.correction_for_n(int(n))
        amp_rad_s = 0.0 if spec is None else float(spec.amp_rad_s)
        base_phase = float(phi_target[n]) if spec is None else float(spec.phase_rad)
        omega_waveform = None if spec is None else float(spec.omega_rad_s + d_omega_rad_s)
        tone_rows.append(
            {
                "n": int(n),
                "theta_rad": float(theta_target[n]),
                "phi_rad": float(phi_target[n]),
                "d_lambda_norm": float(d_lambda),
                "d_alpha_rad": float(d_alpha),
                "d_omega_rad_s": float(d_omega_rad_s),
                "d_omega_hz": float(d_omega_rad_s / (2.0 * np.pi)),
                "amp_rad_s": float(amp_rad_s),
                "phase_rad": float(base_phase + d_alpha),
                "omega_waveform_rad_s": None if omega_waveform is None else float(omega_waveform),
                "omega_waveform_hz": None if omega_waveform is None else float(omega_waveform / (2.0 * np.pi)),
                "active": bool(abs(amp_rad_s) > 1.0e-15),
            }
        )

    ref_payload = None if reference_pulse is None else _jsonify(dict(reference_pulse))
    ref_pulse_name = None if ref_payload is None else ref_payload.get("pulse_name")
    frequency_source = "direct_fock_fqs_hz" if cfg.get("fock_fqs_hz") is not None else "chi_polynomial"
    artifact = {
        "schema_version": SQR_TRANSFER_SCHEMA_VERSION,
        "artifact_kind": "sqr_transfer",
        "gate": {
            "index": int(gate.index),
            "name": str(gate.name),
            "type": "SQR",
            "target": "qubit",
            "theta": [float(value) for value in gate.theta],
            "phi": [float(value) for value in gate.phi],
        },
        "pulse_family": {
            "duration_s": float(duration_s),
            "duration_ns": int(round(duration_s * 1.0e9)),
            "sigma_fraction": float(cfg["sqr_sigma_fraction"]),
            "envelope_type": "multitone_gaussian",
            "waveform_convention": "I+iQ with exp(+i phase) and exp(+i omega t)",
        },
        "reference_pulse": ref_payload,
        "tones": tone_rows,
        "sampled_waveform": {
            "dt_s": float(sample_dt),
            "sample_rate_hz": float(1.0 / sample_dt),
            "n_samples": int(coeff.size),
            "time_s": [float(value) for value in tlist.tolist()],
            "I": [float(value) for value in np.real(coeff).tolist()],
            "Q": [float(value) for value in np.imag(coeff).tolist()],
        },
        "qubox_legacy": {
            "type": "SQR",
            "params": {
                "theta": [float(value) for value in gate.theta],
                "phi": [float(value) for value in gate.phi],
                "d_lambda": [float(row["d_lambda_norm"]) for row in tone_rows],
                "d_alpha": [float(row["d_alpha_rad"]) for row in tone_rows],
                "d_omega": [float(row["d_omega_rad_s"]) for row in tone_rows],
                "d_omega_is_hz": False,
                "ref_sel_pulse": ref_pulse_name,
                "ref_sel_x180_pulse": ref_pulse_name,
            },
        },
        "simulator_metadata": {
            "config_snapshot": _jsonify(cfg),
            "device_parameter_snapshot": _jsonify(cfg.get("device_parameter_snapshot")),
            "active_tones": _jsonify(pulse_meta.get("active_tones", [])),
            "calibration_applied": bool(calibration is not None),
            "calibration_summary": None if calibration is None else _jsonify(calibration.to_dict()),
            "source_path": None if source_path is None else str(source_path),
            "source_notebook": None if source_notebook is None else str(source_notebook),
            "frequency_source": frequency_source,
            "extra": {} if extra_metadata is None else _jsonify(dict(extra_metadata)),
        },
    }
    return artifact


def write_sqr_transfer_artifact(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(dict(artifact)), indent=2), encoding="utf-8")
    return path


def load_sqr_transfer_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_sqr_pulse_from_artifact(
    artifact: Mapping[str, Any],
) -> tuple[list[Pulse], dict[str, str], dict[str, Any]]:
    if str(artifact.get("schema_version", "")) != SQR_TRANSFER_SCHEMA_VERSION:
        raise ValueError(f"Unsupported SQR transfer schema version: {artifact.get('schema_version')}")

    gate_name = str(dict(artifact.get("gate", {})).get("name", "sqr_artifact"))
    sampled = dict(artifact.get("sampled_waveform", {}))
    I = np.asarray(sampled["I"], dtype=float)
    Q = np.asarray(sampled["Q"], dtype=float)
    dt_s = float(sampled["dt_s"])
    duration_s = float(dict(artifact.get("pulse_family", {})).get("duration_s", len(I) * dt_s))
    coeff = I + 1j * Q
    pulse = Pulse(
        "qubit",
        0.0,
        duration_s,
        coeff,
        sample_rate=float(1.0 / dt_s),
        amp=1.0,
        phase=0.0,
        label=gate_name,
    )
    return [pulse], {"qubit": "qubit"}, {
        "mapping": "Transferred sampled SQR artifact waveform.",
        "artifact_kind": str(artifact.get("artifact_kind", "sqr_transfer")),
        "artifact_gate_name": gate_name,
        "artifact_duration_s": float(duration_s),
        "artifact_schema_version": str(artifact.get("schema_version")),
    }


def run_sequence_case_with_artifacts(
    gates: Sequence[Gate],
    config: Mapping[str, Any],
    artifact_map: Mapping[str, Mapping[str, Any]],
    *,
    include_dissipation: bool,
    case_label: str,
) -> dict[str, Any]:
    model = build_model(config)
    noise = build_noise_spec(config, enabled=include_dissipation)
    state = build_initial_state(config, n_cav_dim=model.n_cav)
    snapshots = [snapshot_from_state(state, 0, None, config, case_label=case_label)]
    mapping_rows: list[dict[str, Any]] = []
    for step_index, gate in enumerate(gates, start=1):
        artifact = artifact_map.get(getattr(gate, "name", ""))
        if isinstance(gate, SQRGate) and artifact is not None:
            pulses, drive_ops, meta = build_sqr_pulse_from_artifact(artifact)
        else:
            pulses, drive_ops, meta = build_gate_segment(gate, model, config)
        state = evolve_segment(model, state, pulses, drive_ops, config, noise)
        snapshots.append(snapshot_from_state(state, step_index, gate, config, case_label=case_label, extra=meta))
        mapping_rows.append({"index": step_index, "type": gate.type, "name": gate.name, **meta})
    solver_name = "mesolve" if include_dissipation else "sesolve"
    return finalize_track(
        case_label,
        snapshots,
        metadata={"solver": solver_name, "mapping_rows": mapping_rows},
    )


def run_pulse_case_from_decomposition(
    path_like: str | Path,
    config: Mapping[str, Any],
    *,
    reverse_gate_order: bool = False,
    include_dissipation: bool = False,
    case_label: str = "Case B decomposition",
) -> dict[str, Any]:
    from .sequential.pulse_unitary import run_pulse_case

    _path, gates, calibration_map, _raw = load_gate_sequence_with_sqr_corrections(path_like)
    ordered_gates = list(reversed(gates)) if bool(reverse_gate_order) else list(gates)
    return run_pulse_case(
        ordered_gates,
        config,
        include_dissipation=bool(include_dissipation),
        case_label=str(case_label),
        sqr_calibration_map=calibration_map,
    )


def prefix_bloch_rows_from_track(track: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for snapshot in list(track.get("snapshots", [])):
        rows.append(
            {
                "gate_index": int(snapshot["index"]),
                "gate_type": str(snapshot["gate_type"]),
                "gate_name": str(snapshot["gate_name"]),
                "X": float(snapshot["x"]),
                "Y": float(snapshot["y"]),
                "Z": float(snapshot["z"]),
            }
        )
    return rows


__all__ = [
    "SQR_TRANSFER_SCHEMA_VERSION",
    "build_sqr_pulse_from_artifact",
    "build_sqr_transfer_artifact",
    "gate_calibration_from_experiment_entry",
    "load_gate_sequence_with_sqr_corrections",
    "load_sqr_transfer_artifact",
    "prefix_bloch_rows_from_track",
    "run_pulse_case_from_decomposition",
    "run_sequence_case_with_artifacts",
    "write_sqr_transfer_artifact",
]
