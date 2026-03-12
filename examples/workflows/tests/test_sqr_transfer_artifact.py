from __future__ import annotations

import json

import numpy as np

from cqed_sim.calibration.sqr import SQRCalibrationResult
from cqed_sim.io.gates import SQRGate
from examples.workflows.sqr_transfer import (
    SQR_TRANSFER_SCHEMA_VERSION,
    build_sqr_transfer_artifact,
    load_gate_sequence_with_sqr_corrections,
    run_sequence_case_with_artifacts,
)


def _artifact_config() -> dict[str, float | int | bool]:
    return {
        "duration_sqr_s": 6.0e-7,
        "sqr_sigma_fraction": 0.16,
        "dt_s": 2.0e-9,
        "max_step_s": 2.0e-9,
        "omega_q_hz": 6.1503587644830475e9,
        "omega_c_hz": 0.0,
        "qubit_alpha_hz": 0.0,
        "st_chi_hz": -2.84e6,
        "st_chi2_hz": -2.1e4,
        "st_chi3_hz": 0.0,
        "st_K_hz": 0.0,
        "st_K2_hz": 0.0,
        "fock_fqs_hz": [6.1473587644830475e9, 6.1443587644830475e9, 6.1413587644830475e9, 6.1383587644830475e9, 6.1353587644830475e9],
        "use_rotating_frame": True,
        "n_cav_dim": 5,
        "sqr_theta_cutoff": 1.0e-10,
        "device_parameter_snapshot": {
            "qb_fq": 6.1503587644830475e9,
            "fock_fqs": [6.1473587644830475e9, 6.1443587644830475e9, 6.1413587644830475e9, 6.1383587644830475e9, 6.1353587644830475e9],
        },
    }


def _track_config() -> dict[str, float | int | bool | str | dict[str, float] | None]:
    cfg = dict(_artifact_config())
    cfg.update(
        {
            "cavity_fock_cutoff": 4,
            "initial_qubit": "g",
            "initial_cavity_kind": "fock",
            "initial_cavity_fock": 0,
            "initial_cavity_alpha": {"re": 0.0, "im": 0.0},
            "initial_cavity_amplitudes": None,
            "wigner_every_gate": False,
            "wigner_points": 21,
            "wigner_extent": 4.0,
            "phase_track_max_n": 2,
            "phase_reference_threshold": 1.0e-6,
            "phase_unwrap": True,
            "trajectory_gate_index": None,
            "trajectory_conditioned_max_n": 2,
            "gate_diag_probability_threshold": 1.0e-6,
            "duration_displacement_s": 48.0e-9,
            "duration_rotation_s": 16.0e-9,
            "rotation_sigma_fraction": 1.0 / 6.0,
            "qb_T1_relax_ns": None,
            "qb_T2_ramsey_ns": None,
            "qb_T2_echo_ns": None,
            "t2_source": "ramsey",
            "cavity_kappa_1_per_s": 0.0,
            "save_output_figures": False,
            "output_figure_dir": "outputs/figures",
            "output_figure_dpi": 150,
        }
    )
    return cfg


def _reference_spec() -> dict[str, float | int | str]:
    return {
        "mode": "synthesized_drag_gaussian",
        "pulse_name": "ref_transfer_gate_0_pulse",
        "duration_ns": 600,
        "sigma_fraction": 0.16,
        "amplitude": 0.0037764,
        "drag_alpha": 0.0,
        "detuning_hz": 0.0,
    }


def _calibration(name: str) -> SQRCalibrationResult:
    return SQRCalibrationResult(
        sqr_name=name,
        max_n=4,
        d_lambda=[0.15, 0.0, 0.0, 0.0, 0.0],
        d_alpha=[0.20, 0.0, 0.0, 0.0, 0.0],
        d_omega_rad_s=[2.0e5, 0.0, 0.0, 0.0, 0.0],
        theta_target=[np.pi / 2.0, 0.0, 0.0, 0.0, 0.0],
        phi_target=[0.0, 0.0, 0.0, 0.0, 0.0],
        initial_loss=[0.0] * 5,
        optimized_loss=[0.0] * 5,
        levels=[],
        metadata={},
    )


def test_build_sqr_transfer_artifact_records_waveform_and_corrections():
    cfg = _artifact_config()
    gate = SQRGate(index=0, name="sqr_gate", theta=(np.pi / 2.0, 0.0, 0.0, 0.0, 0.0), phi=(0.0, 0.0, 0.0, 0.0, 0.0))
    artifact = build_sqr_transfer_artifact(
        gate,
        cfg,
        calibration=_calibration(gate.name),
        reference_pulse=_reference_spec(),
        source_notebook="examples/studies/sqr_three_gate_duration_optimization.ipynb",
    )
    assert artifact["schema_version"] == SQR_TRANSFER_SCHEMA_VERSION
    assert artifact["gate"]["name"] == gate.name
    assert artifact["reference_pulse"]["pulse_name"] == "ref_transfer_gate_0_pulse"
    assert artifact["tones"][0]["d_lambda_norm"] == 0.15
    assert artifact["tones"][0]["d_alpha_rad"] == 0.20
    assert artifact["simulator_metadata"]["frequency_source"] == "direct_fock_fqs_hz"
    assert artifact["simulator_metadata"]["device_parameter_snapshot"]["qb_fq"] == cfg["device_parameter_snapshot"]["qb_fq"]
    expected_waveform_hz = -(cfg["fock_fqs_hz"][0] - cfg["omega_q_hz"]) + _calibration(gate.name).d_omega_hz[0]
    assert np.isclose(artifact["tones"][0]["omega_waveform_hz"], expected_waveform_hz)
    assert artifact["sampled_waveform"]["n_samples"] > 0
    assert len(artifact["sampled_waveform"]["I"]) == artifact["sampled_waveform"]["n_samples"]


def test_load_gate_sequence_with_sqr_corrections_translates_experiment_arrays(tmp_path):
    payload = [
        {
            "type": "SQR",
            "name": "sqr_gate",
            "target": "qubit",
            "params": {
                "theta": [np.pi / 2.0, 0.0, 0.0],
                "phi": [0.0, 0.0, 0.0],
                "d_lambda": [0.1, 0.0, 0.0],
                "d_alpha": [0.2, 0.0, 0.0],
                "d_omega": [1.0e5, 0.0, 0.0],
                "d_omega_is_hz": True,
                "ref_sel_pulse": "transfer_ref_pulse",
            },
        }
    ]
    path = tmp_path / "sequence.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    _chosen, gates, calibration_map, _raw = load_gate_sequence_with_sqr_corrections(path)
    assert len(gates) == 1
    calibration = calibration_map["sqr_gate"]
    assert np.isclose(calibration.d_lambda[0], 0.1)
    assert np.isclose(calibration.d_alpha[0], 0.2)
    assert np.isclose(calibration.d_omega_rad_s[0], 2.0 * np.pi * 1.0e5)
    assert calibration.metadata["ref_sel_pulse"] == "transfer_ref_pulse"


def test_run_sequence_case_with_artifacts_accepts_per_gate_durations():
    gate_a = SQRGate(index=0, name="sqr_a", theta=(np.pi / 2.0, 0.0, 0.0, 0.0, 0.0), phi=(0.0, 0.0, 0.0, 0.0, 0.0))
    gate_b = SQRGate(index=1, name="sqr_b", theta=(0.0, np.pi / 2.0, 0.0, 0.0, 0.0), phi=(0.0, np.pi / 2.0, 0.0, 0.0, 0.0))

    cfg_a = _artifact_config()
    cfg_b = dict(_artifact_config())
    cfg_b["duration_sqr_s"] = 9.0e-7
    ref_a = dict(_reference_spec())
    ref_b = dict(_reference_spec())
    ref_b["duration_ns"] = 900
    ref_b["pulse_name"] = "ref_transfer_gate_1_pulse"

    artifact_a = build_sqr_transfer_artifact(gate_a, cfg_a, calibration=None, reference_pulse=ref_a)
    artifact_b = build_sqr_transfer_artifact(gate_b, cfg_b, calibration=None, reference_pulse=ref_b)

    track = run_sequence_case_with_artifacts(
        [gate_a, gate_b],
        _track_config(),
        {gate_a.name: artifact_a, gate_b.name: artifact_b},
        include_dissipation=False,
        case_label="artifact-sequence",
    )
    rows = track["metadata"]["mapping_rows"]
    assert np.isclose(rows[0]["artifact_duration_s"], 6.0e-7)
    assert np.isclose(rows[1]["artifact_duration_s"], 9.0e-7)
    assert len(track["snapshots"]) == 3
