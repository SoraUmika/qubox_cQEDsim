from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from cqed_sim import (
    CallableParameterization,
    ControlParameterSpec,
    ControlProblem,
    ControlSystem,
    ControlTerm,
    GrapeConfig,
    PiecewiseConstantTimeGrid,
    StateTransferObjective,
    solve_grape,
)


def _sigma_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _sigma_y() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def build_problem() -> ControlProblem:
    sigma_x = _sigma_x()
    sigma_y = _sigma_y()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=1.0)

    def evaluator(values, _time_grid, _control_terms) -> np.ndarray:
        amplitude = float(values[0])
        phase = float(values[1])
        return np.array(
            [[amplitude * np.cos(phase)], [amplitude * np.sin(phase)]],
            dtype=float,
        )

    def pullback(gradient_command, values, _time_grid, _control_terms, _waveform) -> np.ndarray:
        amplitude = float(values[0])
        phase = float(values[1])
        gradient = np.asarray(gradient_command, dtype=float)
        return np.array(
            [
                gradient[0, 0] * np.cos(phase) + gradient[1, 0] * np.sin(phase),
                gradient[0, 0] * (-amplitude * np.sin(phase)) + gradient[1, 0] * (amplitude * np.cos(phase)),
            ],
            dtype=float,
        )

    parameterization = CallableParameterization(
        time_grid=time_grid,
        control_terms=(
            ControlTerm(
                name="x_drive",
                operator=0.5 * sigma_x,
                amplitude_bounds=(-2.0 * np.pi, 2.0 * np.pi),
                quadrature="SCALAR",
            ),
            ControlTerm(
                name="y_drive",
                operator=0.5 * sigma_y,
                amplitude_bounds=(-2.0 * np.pi, 2.0 * np.pi),
                quadrature="SCALAR",
            ),
        ),
        parameter_specs=(
            ControlParameterSpec("amplitude", 0.0, 2.0 * np.pi, default=1.0),
            ControlParameterSpec("phase", -np.pi, np.pi, default=0.0, units="rad"),
        ),
        evaluator=evaluator,
        pullback_evaluator=pullback,
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
                control_operators=(0.5 * sigma_x, 0.5 * sigma_y),
                label="callable_demo",
            ),
        ),
        objectives=(objective,),
        metadata={"example": "callable_optimal_control_demo"},
    )


def main() -> None:
    output_dir = Path("outputs") / "callable_optimal_control_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = build_problem()
    result = solve_grape(
        problem,
        config=GrapeConfig(maxiter=45, seed=5, random_scale=0.2),
        initial_schedule=np.array([1.1, 0.7], dtype=float),
    )
    result.save(output_dir / "result.json")

    with (output_dir / "waveforms.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["control", "slice", "value"])
        for control_index, control_name in enumerate(("x_drive", "y_drive")):
            for slice_index, value in enumerate(np.asarray(result.command_values, dtype=float)[control_index]):
                writer.writerow([control_name, int(slice_index), float(value)])

    summary = {
        "success": bool(result.success),
        "objective_value": float(result.objective_value),
        "nominal_fidelity": float(result.metrics.get("nominal_fidelity", float("nan"))),
        "parameter_values": [float(value) for value in np.asarray(result.schedule.values, dtype=float)],
        "parameterization": dict(result.parameterization_metrics),
        "artifacts": {
            "result_json": str(output_dir / "result.json"),
            "waveforms_csv": str(output_dir / "waveforms.csv"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Callable optimal-control demo")
    print(f"  success: {result.success}")
    print(f"  objective: {result.objective_value:.6e}")
    print(f"  nominal fidelity: {result.metrics.get('nominal_fidelity', float('nan')):.6f}")
    print(f"  parameter values: {np.asarray(result.schedule.values, dtype=float).tolist()}")
    print(f"  artifacts: {output_dir}")


if __name__ == "__main__":
    main()