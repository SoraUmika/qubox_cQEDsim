from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cqed_sim import (
    ControlEvaluationCase,
    ControlResult,
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    NoiseSpec,
    PiecewiseConstantTimeGrid,
    UnitaryObjective,
    build_control_problem_from_model,
    state_preparation_objective,
)
from cqed_sim.unitary_synthesis import Subspace


def _qubit_only_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return model, frame


def _state_preparation_problem() -> tuple[DispersiveTransmonCavityModel, FrameSpec, object]:
    model, frame = _qubit_only_model()
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9),
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
    )
    return model, frame, problem


def _storage_subspace_problem() -> tuple[DispersiveTransmonCavityModel, FrameSpec, object]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    storage_logical = Subspace.custom(full_dim=6, indices=(0, 1), labels=("|g,0>", "|g,1>"))
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9),
        channel_specs=(
            ModelControlChannelSpec(
                name="storage",
                target="storage",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
                export_channel="storage",
            ),
        ),
        objectives=(
            UnitaryObjective(
                target_operator=np.array(
                    [
                        [np.cos(np.pi / 4.0), -np.sin(np.pi / 4.0)],
                        [np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)],
                    ],
                    dtype=np.complex128,
                ),
                subspace=storage_logical,
                ignore_global_phase=True,
                name="storage_y90",
            ),
        ),
    )
    return model, frame, problem


def _notebook_text(path: str) -> str:
    notebook = json.loads(Path(path).read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_grape_result_is_control_result_and_supports_nominal_and_noisy_replay() -> None:
    model, frame, problem = _state_preparation_problem()
    result = GrapeSolver(GrapeConfig(maxiter=80, seed=17, random_scale=0.25)).solve(
        problem,
        initial_schedule=np.array([[3.5e7]], dtype=float),
    )

    assert isinstance(result, ControlResult)
    assert result.success

    nominal_replay = result.evaluate_with_simulator(problem, model=model, frame=frame, compiler_dt_s=1.0e-9)
    noisy_replay = result.evaluate_with_simulator(
        problem,
        cases=(
            ControlEvaluationCase(
                model=model,
                label="strong_relaxation",
                frame=frame,
                noise=NoiseSpec(t1=8.0e-9, tphi=8.0e-9),
                compiler_dt_s=1.0e-9,
            ),
        ),
    )

    assert nominal_replay.metrics["aggregate_fidelity"] > 0.99
    assert noisy_replay.metrics["aggregate_fidelity"] < nominal_replay.metrics["aggregate_fidelity"]
    assert noisy_replay.member_reports[0].label == "strong_relaxation"
    assert noisy_replay.member_reports[0].objective_reports[0].fidelity_weighted < 0.95


def test_unitary_objective_replay_reports_subspace_leakage_metric() -> None:
    model, frame, problem = _storage_subspace_problem()
    result = GrapeSolver(GrapeConfig(maxiter=80, seed=11, random_scale=0.25)).solve(
        problem,
        initial_schedule=np.array([[8.0e6]], dtype=float),
    )

    replay = result.evaluate_with_simulator(
        problem,
        cases=(
            ControlEvaluationCase(
                model=model,
                frame=frame,
                noise=NoiseSpec(kappa=2.0e5),
                compiler_dt_s=1.0e-9,
            ),
        ),
    )

    objective_report = replay.member_reports[0].objective_reports[0]
    assert objective_report.kind == "unitary"
    assert objective_report.leakage_weighted is not None
    assert 0.0 <= objective_report.leakage_weighted <= 1.0


def test_optimal_control_benchmark_smoke(tmp_path: Path) -> None:
    output_path = tmp_path / "optimal_control_benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/run_optimal_control_benchmarks.py",
            "--suite",
            "smoke",
            "--maxiter",
            "20",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["suite"] == "smoke"
    assert len(payload["results"]) == 1
    result = payload["results"][0]
    assert "solve" in result
    assert "nominal_replay" in result
    assert "configuration" in result
    assert result["solve"]["runtime_s"] >= 0.0
    assert "aggregate_fidelity" in result["nominal_replay"]["metrics"]


def test_optimal_control_tutorial_mentions_noisy_replay_and_benchmark_harness() -> None:
    content = _notebook_text("tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb")

    assert "evaluate_with_simulator(" in content
    assert "ControlEvaluationCase(" in content
    assert "NoiseSpec(" in content
    assert "benchmarks/run_optimal_control_benchmarks.py" in content