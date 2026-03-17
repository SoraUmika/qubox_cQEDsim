from __future__ import annotations

import numpy as np

from cqed_sim.calibration import (
    ConditionedMultitoneRunConfig,
    ConditionedOptimizationConfig,
    ConditionedQubitTargets,
    LogicalBlockPhaseCorrection,
    TargetedSubspaceObjectiveWeights,
    TargetedSubspaceOptimizationConfig,
    analyze_targeted_subspace_operator,
    build_block_rotation_target_operator,
    build_spanning_state_transfer_set,
    optimize_targeted_subspace_multitone,
)
from cqed_sim.core.conventions import qubit_cavity_block_indices, qubit_cavity_index
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import manifold_transition_frequency
from cqed_sim.core.ideal_gates import logical_block_phase_op
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
        "duration_s": 4.5e-7,
        "dt_s": 5.0e-9,
        "sigma_fraction": 1.0 / 6.0,
        "tone_cutoff": 1.0e-12,
        "include_all_levels": False,
        "max_step_s": 5.0e-9,
        "fock_fqs_hz": None,
    }
    data.update(overrides)
    return ConditionedMultitoneRunConfig(**data)


def _embed_logical_operator(
    model: DispersiveTransmonCavityModel,
    logical_levels: tuple[int, ...],
    logical_operator: np.ndarray,
) -> np.ndarray:
    full_dim = int(model.n_tr) * int(model.n_cav)
    full = np.eye(full_dim, dtype=np.complex128)
    logical_indices: list[int] = []
    for level in logical_levels:
        logical_indices.extend(int(index) for index in qubit_cavity_block_indices(int(model.n_cav), int(level)))
    full[np.ix_(logical_indices, logical_indices)] = np.asarray(logical_operator, dtype=np.complex128)
    return full


def _permutation_operator(dim: int, mapping: dict[int, int]) -> np.ndarray:
    perm = np.arange(dim, dtype=int)
    for source, target in mapping.items():
        perm[int(source)] = int(target)
    matrix = np.zeros((dim, dim), dtype=np.complex128)
    for source, target in enumerate(perm):
        matrix[int(target), int(source)] = 1.0
    return matrix


def test_exact_logical_operator_scores_perfectly() -> None:
    model = _test_model(n_cav=4)
    targets = ConditionedQubitTargets(
        theta=(0.22 * np.pi, 0.47 * np.pi, 0.61 * np.pi),
        phi=(0.05 * np.pi, 0.35 * np.pi, 1.10 * np.pi),
    )
    logical_levels = (0, 1, 2)
    target_operator = build_block_rotation_target_operator(targets, logical_levels=logical_levels)
    transfer_set = build_spanning_state_transfer_set(target_operator)
    weights = TargetedSubspaceObjectiveWeights(
        qubit_weight=0.0,
        subspace_weight=1.0,
        preservation_weight=1.0,
        leakage_weight=1.0,
    )
    assert len(transfer_set.input_states) == 36

    full_operator = _embed_logical_operator(model, logical_levels, target_operator)
    result = analyze_targeted_subspace_operator(
        full_operator,
        model,
        targets,
        logical_levels=logical_levels,
        target_operator=target_operator,
        transfer_set=transfer_set,
        objective_weights=weights,
    )

    assert np.isclose(result.restricted_process_fidelity, 1.0, atol=1.0e-12)
    assert np.isclose(result.state_transfer_fidelity_mean, 1.0, atol=1.0e-12)
    assert np.isclose(result.state_transfer_fidelity_min, 1.0, atol=1.0e-12)
    assert np.isclose(result.same_block_population_mean, 1.0, atol=1.0e-12)
    assert np.isclose(result.same_block_population_min, 1.0, atol=1.0e-12)
    assert np.isclose(result.other_target_population_mean, 0.0, atol=1.0e-12)
    assert np.isclose(result.leakage_outside_target_mean, 0.0, atol=1.0e-12)
    assert result.weighted_loss < 1.0e-12
    for row in result.conditioned_sector_metrics:
        assert np.isclose(row.sector_population, 1.0, atol=1.0e-12)


def test_cross_talk_and_leakage_metrics_detect_bad_operator() -> None:
    model = _test_model(n_cav=4)
    targets = ConditionedQubitTargets(theta=(0.0, 0.0, 0.0), phi=(0.0, 0.0, 0.0))
    logical_levels = (0, 1, 2)
    target_operator = build_block_rotation_target_operator(targets, logical_levels=logical_levels)
    transfer_set = build_spanning_state_transfer_set(target_operator)

    dim = int(model.n_tr) * int(model.n_cav)
    idx_g0 = qubit_cavity_index(int(model.n_cav), 0, 0)
    idx_g1 = qubit_cavity_index(int(model.n_cav), 0, 1)
    idx_e0 = qubit_cavity_index(int(model.n_cav), 1, 0)
    idx_g3 = qubit_cavity_index(int(model.n_cav), 0, 3)
    bad_operator = _permutation_operator(
        dim,
        {
            idx_g0: idx_g1,
            idx_g1: idx_g0,
            idx_e0: idx_g3,
            idx_g3: idx_e0,
        },
    )
    weights = TargetedSubspaceObjectiveWeights(
        qubit_weight=0.0,
        subspace_weight=1.0,
        preservation_weight=1.0,
        leakage_weight=1.0,
    )
    result = analyze_targeted_subspace_operator(
        bad_operator,
        model,
        targets,
        logical_levels=logical_levels,
        target_operator=target_operator,
        transfer_set=transfer_set,
        objective_weights=weights,
    )

    assert result.restricted_process_fidelity < 0.8
    assert result.same_block_population_min < 0.6
    assert result.other_target_population_max > 0.4
    assert result.leakage_outside_target_max > 0.4
    assert result.state_transfer_fidelity_mean < 0.85
    assert result.weighted_loss > 0.4


def test_logical_block_phase_layer_recovers_phase_shifted_operator() -> None:
    model = _test_model(n_cav=4)
    targets = ConditionedQubitTargets(
        theta=(0.22 * np.pi, 0.47 * np.pi, 0.61 * np.pi),
        phi=(0.05 * np.pi, 0.35 * np.pi, 1.10 * np.pi),
    )
    logical_levels = (0, 1, 2)
    target_operator = build_block_rotation_target_operator(targets, logical_levels=logical_levels)
    transfer_set = build_spanning_state_transfer_set(target_operator)
    ideal_full = _embed_logical_operator(model, logical_levels, target_operator)
    block_phases = (0.0, 0.36, -0.41)
    phase_layer = np.asarray(
        logical_block_phase_op(
            block_phases,
            fock_levels=logical_levels,
            cavity_dim=int(model.n_cav),
            qubit_dim=int(model.n_tr),
        ).full(),
        dtype=np.complex128,
    )
    shifted_full = phase_layer @ ideal_full

    baseline = analyze_targeted_subspace_operator(
        shifted_full,
        model,
        targets,
        logical_levels=logical_levels,
        target_operator=target_operator,
        transfer_set=transfer_set,
    )
    corrected = analyze_targeted_subspace_operator(
        shifted_full,
        model,
        targets,
        logical_levels=logical_levels,
        logical_block_phase=LogicalBlockPhaseCorrection(logical_levels=logical_levels, phases_rad=(0.0, -0.36, 0.41)),
        target_operator=target_operator,
        transfer_set=transfer_set,
    )

    assert baseline.restricted_process_fidelity < 0.98
    assert baseline.block_phase_diagnostics is not None
    assert baseline.block_phase_diagnostics.rms_block_phase_error_rad > 0.1
    assert np.isclose(corrected.uncorrected_restricted_process_fidelity, baseline.restricted_process_fidelity, atol=1.0e-12)
    assert corrected.block_phase_diagnostics is not None
    assert corrected.block_phase_diagnostics.rms_block_phase_error_rad < 1.0e-12
    assert corrected.restricted_process_fidelity > 1.0 - 1.0e-12
    assert corrected.best_fit_restricted_process_fidelity > 1.0 - 1.0e-12


def test_full_objective_optimization_improves_restricted_fidelity() -> None:
    model = _test_model(n_cav=2)
    targets = ConditionedQubitTargets(
        theta=(0.28 * np.pi, 0.53 * np.pi),
        phi=(0.10 * np.pi, 0.85 * np.pi),
    )
    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)
    wrong_freqs_hz = tuple(
        float(manifold_transition_frequency(model, n, frame=frame) / (2.0 * np.pi) + 0.22e6)
        for n in range(2)
    )
    run_config = _run_config(
        frame=frame,
        duration_s=4.5e-7,
        dt_s=5.0e-9,
        max_step_s=5.0e-9,
        fock_fqs_hz=wrong_freqs_hz,
    )
    weights = TargetedSubspaceObjectiveWeights(
        qubit_weight=0.5,
        subspace_weight=1.0,
        preservation_weight=0.0,
        leakage_weight=0.0,
    )
    optimization = ConditionedOptimizationConfig(
        parameters=("d_omega",),
        maxiter_stage1=8,
        maxiter_stage2=10,
        d_omega_hz_bounds=(-0.6e6, 0.6e6),
    )

    result = optimize_targeted_subspace_multitone(
        model,
        targets,
        run_config,
        logical_levels=(0, 1),
        objective_weights=weights,
        optimization_config=optimization,
    )

    assert result.optimized_result.weighted_loss <= result.initial_result.weighted_loss + 1.0e-6
    assert result.optimized_result.restricted_process_fidelity >= result.initial_result.restricted_process_fidelity - 1.0e-6
    assert result.optimized_result.qubit_loss <= result.initial_result.qubit_loss + 1.0e-6
    optimized_detunings_hz = np.asarray(result.optimized_corrections.d_omega_rad_s, dtype=float) / (2.0 * np.pi)
    assert np.any(np.abs(optimized_detunings_hz[:2]) > 1.0e4)


def test_block_phase_optimization_reduces_targeted_subspace_phase_residuals() -> None:
    model = _test_model(n_cav=2)
    targets = ConditionedQubitTargets(
        theta=(0.28 * np.pi, 0.53 * np.pi),
        phi=(0.10 * np.pi, 0.85 * np.pi),
    )
    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)
    wrong_freqs_hz = tuple(
        float(manifold_transition_frequency(model, n, frame=frame) / (2.0 * np.pi) + 0.22e6)
        for n in range(2)
    )
    run_config = _run_config(
        frame=frame,
        duration_s=4.5e-7,
        dt_s=5.0e-9,
        max_step_s=5.0e-9,
        fock_fqs_hz=wrong_freqs_hz,
    )
    optimization = TargetedSubspaceOptimizationConfig(
        conditioned=ConditionedOptimizationConfig(
            parameters=("d_omega",),
            maxiter_stage1=4,
            maxiter_stage2=4,
            d_omega_hz_bounds=(0.0, 0.0),
        ),
        include_block_phase=True,
    )
    weights = TargetedSubspaceObjectiveWeights(
        qubit_weight=0.0,
        subspace_weight=1.0,
        preservation_weight=0.0,
        leakage_weight=0.0,
    )

    result = optimize_targeted_subspace_multitone(
        model,
        targets,
        run_config,
        logical_levels=(0, 1),
        objective_weights=weights,
        optimization_config=optimization,
    )

    assert result.block_phase_levels == (1,)
    assert result.optimized_result.block_phase_diagnostics is not None
    assert result.initial_result.block_phase_diagnostics is not None
    assert result.optimized_result.restricted_process_fidelity >= result.initial_result.restricted_process_fidelity - 1.0e-6
    assert (
        result.optimized_result.block_phase_diagnostics.rms_block_phase_error_rad
        <= result.initial_result.block_phase_diagnostics.rms_block_phase_error_rad + 1.0e-6
    )
    assert abs(result.optimized_logical_block_phase.phases_rad[1]) > 1.0e-6