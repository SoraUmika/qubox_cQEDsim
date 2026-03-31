from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    GaussianDragPulseFamily,
    GrapeConfig,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    StructuredControlChannel,
    StructuredControlConfig,
    build_structured_control_problem_from_model,
    solve_structured_then_grape,
    state_preparation_objective,
)


def build_problem():
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=-2.0 * np.pi * 200.0e6,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=3,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return build_structured_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=24, dt_s=4.0e-9),
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("I", "Q"),
                amplitude_bounds=(-8.0e7, 8.0e7),
                export_channel="qubit",
            ),
        ),
        structured_channels=(
            StructuredControlChannel(
                name="gaussian_drag",
                pulse_family=GaussianDragPulseFamily(
                    amplitude_bounds=(0.0, 7.0e7),
                    sigma_fraction_bounds=(0.1, 0.24),
                    center_fraction_bounds=(0.42, 0.58),
                    phase_bounds=(-np.pi, np.pi),
                    drag_bounds=(-0.3, 0.3),
                    default_phase=-0.5 * np.pi,
                ),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
        metadata={"example": "structured_to_grape_refinement_demo"},
    )


def main() -> None:
    output_dir = Path("outputs") / "structured_to_grape_refinement_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_problem()
    result = solve_structured_then_grape(
        problem,
        structured_config=StructuredControlConfig(maxiter=35, seed=4, initial_guess="random", random_scale=0.2),
        grape_config=GrapeConfig(maxiter=35, seed=4, random_scale=0.2),
    )
    result.structured_result.save(output_dir / "structured_result.json")
    result.grape_result.save(output_dir / "grape_result.json")

    summary = {
        "structured_objective": float(result.structured_result.objective_value),
        "grape_objective": float(result.grape_result.objective_value),
        "objective_improvement": float(result.metrics["objective_improvement"]),
        "structured_nominal_fidelity": float(result.metrics["structured_nominal_fidelity"]),
        "grape_nominal_fidelity": float(result.metrics["grape_nominal_fidelity"]),
        "nominal_fidelity_gain": float(result.metrics["nominal_fidelity_gain"]),
        "artifacts": {
            "structured_result_json": str(output_dir / "structured_result.json"),
            "grape_result_json": str(output_dir / "grape_result.json"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Structured-to-GRAPE refinement demo")
    print(f"  structured objective: {result.structured_result.objective_value:.6e}")
    print(f"  grape objective: {result.grape_result.objective_value:.6e}")
    print(f"  objective improvement: {result.metrics['objective_improvement']:.6e}")
    print(f"  structured nominal fidelity: {result.metrics['structured_nominal_fidelity']:.6f}")
    print(f"  grape nominal fidelity: {result.metrics['grape_nominal_fidelity']:.6f}")
    print(f"  artifacts: {output_dir}")


if __name__ == "__main__":
    main()