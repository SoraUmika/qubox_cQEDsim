from __future__ import annotations

import json

import numpy as np

from cqed_sim.calibration import (
    ConditionedMultitoneRunConfig,
    ConditionedOptimizationConfig,
    ConditionedQubitTargets,
    optimize_conditioned_multitone,
    run_conditioned_multitone_validation,
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
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )


def build_targets() -> ConditionedQubitTargets:
    return ConditionedQubitTargets(
        theta=(0.24 * np.pi, 0.52 * np.pi, 0.42 * np.pi),
        phi=(0.0, 0.40 * np.pi, 1.05 * np.pi),
    )


def build_run_config(model: DispersiveTransmonCavityModel) -> ConditionedMultitoneRunConfig:
    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)
    biased_frequencies_hz = tuple(
        float(manifold_transition_frequency(model, n, frame=frame) / (2.0 * np.pi) + 0.28e6)
        for n in range(3)
    )
    return ConditionedMultitoneRunConfig(
        frame=frame,
        duration_s=0.55e-6,
        dt_s=4.0e-9,
        sigma_fraction=1.0 / 6.0,
        tone_cutoff=1.0e-12,
        include_all_levels=False,
        max_step_s=4.0e-9,
        fock_fqs_hz=biased_frequencies_hz,
    )


def summarize_validation(result) -> dict[str, object]:
    return {
        "simulation_mode": str(result.simulation_mode),
        "aggregate_cost": float(result.aggregate_cost),
        "weighted_mean_fidelity": float(result.weighted_mean_fidelity),
        "sector_metrics": [
            {
                "n": int(row.n),
                "fidelity": float(row.fidelity),
                "bloch_radius": float(row.bloch_radius),
                "theta_error_rad": float(row.theta_error_rad),
                "phi_error_rad": float(row.phi_error_rad),
                "sector_population": float(row.sector_population),
                "dominant_error": str(row.dominant_error),
            }
            for row in result.sector_metrics
        ],
    }


def summarize_optimization(result) -> dict[str, object]:
    corrections = result.optimized_corrections
    return {
        "summary": result.improvement_summary(),
        "optimized_corrections": {
            "d_lambda": [float(value) for value in corrections.d_lambda],
            "d_alpha": [float(value) for value in corrections.d_alpha],
            "d_omega_hz": [float(value / (2.0 * np.pi)) for value in corrections.d_omega_rad_s],
        },
        "optimized_validation": summarize_validation(result.optimized_result),
    }


def main() -> None:
    model = build_demo_model()
    targets = build_targets()
    run_config = build_run_config(model)
    optimization = ConditionedOptimizationConfig(
        parameters=("d_omega",),
        maxiter_stage1=10,
        maxiter_stage2=12,
        d_omega_hz_bounds=(-1.0e6, 1.0e6),
    )

    baseline_reduced = run_conditioned_multitone_validation(
        model,
        targets,
        run_config,
        simulation_mode="reduced",
    )
    baseline_full = run_conditioned_multitone_validation(
        model,
        targets,
        run_config,
        simulation_mode="full",
    )
    optimized = optimize_conditioned_multitone(
        model,
        targets,
        run_config,
        optimization_config=optimization,
        simulation_mode="reduced",
    )

    payload = {
        "model": {
            "chi_hz": -2.84e6,
            "n_cav": int(model.n_cav),
            "n_tr": int(model.n_tr),
        },
        "targets": targets.as_rows(),
        "run_config": {
            "duration_s": float(run_config.duration_s),
            "dt_s": float(run_config.dt_s),
            "sigma_fraction": float(run_config.sigma_fraction),
            "biased_fock_fqs_hz": [float(value) for value in run_config.fock_fqs_hz or ()],
        },
        "baseline_reduced": summarize_validation(baseline_reduced),
        "baseline_full": summarize_validation(baseline_full),
        "optimization": summarize_optimization(optimized),
        "note": "Use the reduced conditioned multitone layer to test whether one common waveform can reach the requested conditioned Bloch targets before moving to full targeted-subspace validation.",
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
