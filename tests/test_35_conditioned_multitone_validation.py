from __future__ import annotations

import numpy as np

from cqed_sim.calibration import (
    ConditionedMultitoneCorrections,
    ConditionedMultitoneRunConfig,
    ConditionedOptimizationConfig,
    ConditionedQubitTargets,
    build_conditioned_multitone_tones,
    optimize_conditioned_multitone,
    run_conditioned_multitone_validation,
)
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import manifold_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel


def _test_model(n_cav: int = 4) -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2.0 * np.pi * (-2.84e6),
        kerr=0.0,
        n_cav=int(n_cav),
        n_tr=2,
    )


def _run_config(**overrides: float | tuple[float, ...] | bool | FrameSpec | None) -> ConditionedMultitoneRunConfig:
    data = {
        "frame": FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0),
        "duration_s": 1.0e-6,
        "dt_s": 5.0e-9,
        "sigma_fraction": 1.0 / 6.0,
        "tone_cutoff": 1.0e-12,
        "include_all_levels": False,
        "max_step_s": 5.0e-9,
        "fock_fqs_hz": None,
    }
    data.update(overrides)
    return ConditionedMultitoneRunConfig(**data)


def test_conditioned_targets_from_mapping_normalizes_weights() -> None:
    targets = ConditionedQubitTargets.from_spec(
        {
            0: (0.10 * np.pi, 0.0),
            2: (0.45 * np.pi, 0.30 * np.pi),
        },
        n_levels=4,
        weights={0: 2.0, 2: 1.0},
    )
    assert targets.n_levels == 4
    assert np.isclose(sum(targets.weights), 1.0, atol=1.0e-12)
    assert np.isclose(targets.weights[0], 2.0 / 3.0, atol=1.0e-12)
    assert np.isclose(targets.weights[2], 1.0 / 3.0, atol=1.0e-12)
    assert np.isclose(targets.theta[1], 0.0, atol=1.0e-12)
    assert np.isclose(targets.phi[3], 0.0, atol=1.0e-12)


def test_build_conditioned_tones_keeps_zero_theta_correction_tone() -> None:
    model = _test_model(n_cav=4)
    targets = ConditionedQubitTargets(theta=(0.0, 0.0, 0.0, 0.0), phi=(0.0, 0.0, 0.0, 0.0))
    corrections = ConditionedMultitoneCorrections(d_lambda=(0.0, 0.2, 0.0, 0.0))
    tones = build_conditioned_multitone_tones(model, targets, _run_config(), corrections=corrections)
    assert len(tones) == 1
    assert int(tones[0].manifold) == 1
    assert abs(float(tones[0].amp_rad_s)) > 0.0


def test_reduced_and_full_sector_simulation_agree() -> None:
    model = _test_model(n_cav=3)
    targets = ConditionedQubitTargets(
        theta=(0.28 * np.pi, 0.52 * np.pi, 0.18 * np.pi),
        phi=(0.0, 0.35 * np.pi, 1.15 * np.pi),
    )
    run_config = _run_config(duration_s=0.85e-6, dt_s=4.0e-9, max_step_s=4.0e-9)

    reduced = run_conditioned_multitone_validation(model, targets, run_config, simulation_mode="reduced")
    full = run_conditioned_multitone_validation(model, targets, run_config, simulation_mode="full")

    assert np.isclose(reduced.aggregate_cost, full.aggregate_cost, atol=2.0e-4)
    for reduced_row, full_row in zip(reduced.sector_metrics, full.sector_metrics, strict=True):
        assert np.isclose(reduced_row.fidelity, full_row.fidelity, atol=2.0e-4)
        assert np.isclose(reduced_row.simulated_bloch_x, full_row.simulated_bloch_x, atol=2.0e-4)
        assert np.isclose(reduced_row.simulated_bloch_y, full_row.simulated_bloch_y, atol=2.0e-4)
        assert np.isclose(reduced_row.simulated_bloch_z, full_row.simulated_bloch_z, atol=2.0e-4)
        assert np.isclose(full_row.sector_population, 1.0, atol=1.0e-8)


def test_global_optimization_recovers_frequency_offsets() -> None:
    model = _test_model(n_cav=3)
    targets = ConditionedQubitTargets(
        theta=(0.24 * np.pi, 0.52 * np.pi, 0.42 * np.pi),
        phi=(0.0, 0.40 * np.pi, 1.05 * np.pi),
    )
    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)
    wrong_freqs_hz = tuple(
        float(manifold_transition_frequency(model, n, frame=frame) / (2.0 * np.pi) + 0.28e6)
        for n in range(3)
    )
    run_config = _run_config(
        frame=frame,
        duration_s=0.55e-6,
        dt_s=4.0e-9,
        max_step_s=4.0e-9,
        fock_fqs_hz=wrong_freqs_hz,
    )
    optimization = ConditionedOptimizationConfig(
        parameters=("d_omega",),
        maxiter_stage1=10,
        maxiter_stage2=14,
        d_omega_hz_bounds=(-1.0e6, 1.0e6),
    )

    result = optimize_conditioned_multitone(
        model,
        targets,
        run_config,
        optimization_config=optimization,
        simulation_mode="reduced",
    )

    assert result.optimized_result.aggregate_cost <= result.initial_result.aggregate_cost
    assert result.optimized_result.weighted_mean_fidelity >= result.initial_result.weighted_mean_fidelity
    optimized_detunings_hz = np.asarray(result.optimized_corrections.d_omega_rad_s, dtype=float) / (2.0 * np.pi)
    assert np.any(np.abs(optimized_detunings_hz[:3]) > 1.0e4)