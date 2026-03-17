from __future__ import annotations

import json

import numpy as np

from cqed_sim.calibration import (
    ConditionedMultitoneRunConfig,
    ConditionedQubitTargets,
    TargetedSubspaceObjectiveWeights,
    build_block_rotation_target_operator,
    build_spanning_state_transfer_set,
    run_targeted_subspace_multitone_validation,
)
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import manifold_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel


def build_demo_model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2.0 * np.pi * (-2.84e6),
        kerr=2.0 * np.pi * (-30.0e3),
        n_cav=5,
        n_tr=2,
    )


def build_run_config(
    model: DispersiveTransmonCavityModel,
    *,
    logical_levels: tuple[int, ...],
    frequency_bias_hz: float = 0.22e6,
) -> ConditionedMultitoneRunConfig:
    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)
    max_level = int(max(logical_levels))
    biased_frequencies_hz = tuple(
        float(manifold_transition_frequency(model, n, frame=frame) / (2.0 * np.pi) + frequency_bias_hz)
        for n in range(max_level + 1)
    )
    return ConditionedMultitoneRunConfig(
        frame=frame,
        duration_s=4.5e-7,
        dt_s=5.0e-9,
        sigma_fraction=1.0 / 6.0,
        tone_cutoff=1.0e-12,
        include_all_levels=False,
        max_step_s=5.0e-9,
        fock_fqs_hz=biased_frequencies_hz,
    )


def summarize_result(result) -> dict[str, object]:
    diagnostics = result.block_phase_diagnostics
    return {
        "restricted_process_fidelity": float(result.restricted_process_fidelity),
        "uncorrected_restricted_process_fidelity": float(result.uncorrected_restricted_process_fidelity),
        "best_fit_restricted_process_fidelity": float(result.best_fit_restricted_process_fidelity),
        "state_transfer_fidelity_mean": float(result.state_transfer_fidelity_mean),
        "state_transfer_fidelity_min": float(result.state_transfer_fidelity_min),
        "weighted_loss": float(result.weighted_loss),
        "logical_block_phase": result.logical_block_phase.as_dict(),
        "best_fit_logical_block_phase": result.best_fit_logical_block_phase.as_dict(),
        "block_phase_diagnostics": None if diagnostics is None else diagnostics.as_dict(),
    }


def run_case(
    model: DispersiveTransmonCavityModel,
    run_config: ConditionedMultitoneRunConfig,
    *,
    name: str,
    targets: ConditionedQubitTargets,
    logical_levels: tuple[int, ...],
) -> dict[str, object]:
    target_operator = build_block_rotation_target_operator(targets, logical_levels=logical_levels)
    transfer_set = build_spanning_state_transfer_set(target_operator)
    weights = TargetedSubspaceObjectiveWeights(
        qubit_weight=0.0,
        subspace_weight=1.0,
        preservation_weight=0.25,
        leakage_weight=0.25,
    )
    baseline = run_targeted_subspace_multitone_validation(
        model,
        targets,
        run_config,
        logical_levels=logical_levels,
        target_operator=target_operator,
        transfer_set=transfer_set,
        objective_weights=weights,
    )
    corrected = run_targeted_subspace_multitone_validation(
        model,
        targets,
        run_config,
        logical_levels=logical_levels,
        logical_block_phase=baseline.best_fit_logical_block_phase,
        target_operator=target_operator,
        transfer_set=transfer_set,
        objective_weights=weights,
    )
    return {
        "name": name,
        "logical_levels": [int(level) for level in logical_levels],
        "baseline": summarize_result(baseline),
        "corrected": summarize_result(corrected),
    }


def main() -> None:
    model = build_demo_model()
    logical_levels = (0, 1, 2)
    run_config = build_run_config(model, logical_levels=logical_levels)
    cases = [
        run_case(
            model,
            run_config,
            name="selective_n0_flip",
            logical_levels=logical_levels,
            targets=ConditionedQubitTargets(
                theta=(np.pi, 0.0, 0.0),
                phi=(0.0, 0.0, 0.0),
            ),
        ),
        run_case(
            model,
            run_config,
            name="mixed_xy_levels_0_1_2",
            logical_levels=logical_levels,
            targets=ConditionedQubitTargets(
                theta=(0.35 * np.pi, 0.58 * np.pi, 0.21 * np.pi),
                phi=(0.05 * np.pi, 0.62 * np.pi, -0.28 * np.pi),
            ),
        ),
    ]
    payload = {
        "model": {
            "chi_hz": -2.84e6,
            "kerr_hz": -30.0e3,
            "n_cav": int(model.n_cav),
            "n_tr": int(model.n_tr),
        },
        "run_config": {
            "duration_s": float(run_config.duration_s),
            "dt_s": float(run_config.dt_s),
            "sigma_fraction": float(run_config.sigma_fraction),
            "frequency_bias_hz": 0.22e6,
        },
        "note": "The block-phase layer appended here is an ideal cavity-only logical correction extracted from the raw restricted operator.",
        "cases": cases,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
