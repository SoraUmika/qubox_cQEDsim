from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np

from cqed_sim import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
    GateTimeOptimizationConfig,
    GrapeConfig,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    StateTransferObjective,
    optimize_gate_time_with_grape,
)


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def build_problem() -> ControlProblem:
    sigma_x = _sigma_x()
    parameterization = PiecewiseConstantParameterization(
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0),
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-1.0, 1.0),
                quadrature="SCALAR",
            ),
        ),
    )
    objective = StateTransferObjective.single(
        np.array([1.0, 0.0], dtype=np.complex128),
        np.array([0.0, 1.0], dtype=np.complex128),
        name="flip",
    )
    return ControlProblem(
        parameterization=parameterization,
        systems=(
            ControlSystem(
                drift_hamiltonian=np.zeros((2, 2), dtype=np.complex128),
                control_operators=(0.5 * sigma_x,),
                label="gate_time_demo",
            ),
        ),
        objectives=(objective,),
        metadata={"example": "optimal_control_gate_time_sweep_demo"},
    )


def main() -> None:
    output_dir = Path("outputs") / "optimal_control_gate_time_sweep_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_problem()
    workers = max(1, os.cpu_count() or 1)
    result = optimize_gate_time_with_grape(
        problem,
        durations_s=(2.0, 2.6, 3.2, 4.0),
        config=GrapeConfig(maxiter=35, seed=3, random_scale=0.2),
        gate_time_config=GateTimeOptimizationConfig(max_workers=workers, warm_start_strategy="none"),
        initial_schedule=np.array([[0.4]], dtype=float),
    )
    result.best_result.save(output_dir / "best_result.json")

    with (output_dir / "duration_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["duration_s", "objective_value", "nominal_fidelity"])
        for candidate in result.candidates:
            writer.writerow(
                [
                    float(candidate.duration_s),
                    float(candidate.result.objective_value),
                    float(candidate.result.metrics.get("nominal_fidelity", float("nan"))),
                ]
            )

    summary = {
        "best_duration_s": float(result.best_duration_s),
        "best_objective_value": float(result.best_result.objective_value),
        "best_nominal_fidelity": float(result.best_result.metrics.get("nominal_fidelity", float("nan"))),
        "searched_durations_s": [float(candidate.duration_s) for candidate in result.candidates],
        "max_workers": int(workers),
        "artifacts": {
            "best_result_json": str(output_dir / "best_result.json"),
            "duration_sweep_csv": str(output_dir / "duration_sweep.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Optimal-control gate-time sweep demo")
    print(f"  best duration: {result.best_duration_s:.6f}")
    print(f"  best objective: {result.best_result.objective_value:.6e}")
    print(f"  best nominal fidelity: {result.best_result.metrics.get('nominal_fidelity', float('nan')):.6f}")
    print(f"  workers used: {workers}")
    print(f"  artifacts: {output_dir}")


if __name__ == "__main__":
    main()