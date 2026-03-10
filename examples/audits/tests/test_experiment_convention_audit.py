from __future__ import annotations

import numpy as np

from examples.audits.experiment_convention_audit import (
    qubit_rotation_benchmark_rows,
    relative_phase_rows,
    sqr_addressed_axis_rows,
    tensor_order_rows,
    waveform_sign_scan,
)


def test_rotation_benchmarks_match_standard_su2_convention() -> None:
    rows = qubit_rotation_benchmark_rows()
    assert rows
    for row in rows:
        assert row["gate_process_fidelity"] > 0.999998
        assert row["global_phase_agreement"]
        assert row["state_distance_up_to_global"] < 1.0e-3


def test_rotation_benchmarks_capture_standard_and_historical_y_signs() -> None:
    rows = qubit_rotation_benchmark_rows()
    x90_g = next(
        row
        for row in rows
        if np.isclose(row["theta_rad"], np.pi / 2.0)
        and np.isclose(row["phi_rad"], 0.0)
        and row["input_state"] == "|g>"
    )
    assert x90_g["sim_bloch_y"] < -0.95
    assert x90_g["sim_legacy_y_flipped"] > 0.95


def test_rotation_benchmarks_include_standard_plus_y_state() -> None:
    rows = qubit_rotation_benchmark_rows()
    plus_i = next(
        row
        for row in rows
        if np.isclose(row["theta_rad"], np.pi / 2.0)
        and np.isclose(row["phi_rad"], np.pi / 2.0)
        and row["input_state"] == "|+i>"
    )
    assert np.isclose(plus_i["sim_bloch_x"], 0.0, atol=1.0e-12)
    assert np.isclose(plus_i["sim_bloch_y"], 1.0, atol=1.0e-6)
    assert np.isclose(plus_i["sim_bloch_z"], 0.0, atol=1.0e-12)
    assert np.isclose(plus_i["sim_legacy_y_flipped"], -1.0, atol=1.0e-6)


def test_experiment_waveform_phase_mapping_matches_simulator_sign_scan() -> None:
    sign_scan = waveform_sign_scan()
    assert sign_scan["best_rotation_match"]["phase_sign"] == 1
    assert sign_scan["best_rotation_match"]["omega_sign"] == 1
    assert sign_scan["best_rotation_match"]["process_fidelity_vs_experiment"] > 0.999999
    assert sign_scan["best_sqr_match"]["phase_sign"] == 1
    assert sign_scan["best_sqr_match"]["omega_sign"] == 1
    assert sign_scan["best_sqr_match"]["process_fidelity_vs_experiment"] > 0.999999


def test_tensor_order_is_qubit_then_cavity() -> None:
    rows = {row["check"]: row for row in tensor_order_rows()}
    assert rows["basis |g,0>"]["value"] == 0
    assert rows["basis |e,0>"]["value"] == 4
    assert rows["basis |g,1>"]["value"] == 1
    assert np.isclose(rows["qubit sigma_z on |e,2>"]["value"], -1.0, atol=1.0e-12)
    assert np.isclose(rows["cavity number on |e,2>"]["value"], 2.0, atol=1.0e-12)


def test_sqr_lowest_manifold_respects_axis_phase() -> None:
    rows = [row for row in sqr_addressed_axis_rows() if row["target_n"] == 0]
    assert len(rows) == 2
    row_x = min(rows, key=lambda row: abs(row["input_phi_rad"] - 0.0))
    row_y = min(rows, key=lambda row: abs(row["input_phi_rad"] - np.pi / 2.0))
    assert row_x["process_fidelity"] > 0.999999
    assert row_y["process_fidelity"] > 0.999999
    assert np.isclose(row_x["axis_x"], 1.0, atol=1.0e-6)
    assert np.isclose(row_x["axis_y"], 0.0, atol=1.0e-6)
    assert np.isclose(row_y["axis_x"], 0.0, atol=1.0e-6)
    assert np.isclose(row_y["axis_y"], 1.0, atol=1.0e-6)


def test_sqr_higher_manifold_keeps_quadrature_order_even_when_not_perfectly_selective() -> None:
    rows = [row for row in sqr_addressed_axis_rows() if row["target_n"] == 1]
    assert len(rows) == 2
    row_x = min(rows, key=lambda row: abs(row["input_phi_rad"] - 0.0))
    row_y = min(rows, key=lambda row: abs(row["input_phi_rad"] - np.pi / 2.0))
    assert abs(row_x["axis_x"]) > abs(row_x["axis_y"])
    assert abs(row_y["axis_y"]) > abs(row_y["axis_x"])
    assert np.isclose(row_x["process_fidelity"], row_y["process_fidelity"], atol=1.0e-12)


def test_relative_phase_audit_shows_nontrivial_fock_sector_phases() -> None:
    block_rows, state_rows = relative_phase_rows()
    assert len(block_rows) >= 3
    phase_step_1 = block_rows[1]["block_det_phase_rad"] - block_rows[0]["block_det_phase_rad"]
    phase_step_2 = block_rows[2]["block_det_phase_rad"] - block_rows[1]["block_det_phase_rad"]
    assert phase_step_1 < -0.3
    assert phase_step_2 < -0.3
    assert np.isclose(phase_step_1, phase_step_2, atol=2.0e-2)
    assert any(len(row["support_indices"]) >= 4 for row in state_rows)
