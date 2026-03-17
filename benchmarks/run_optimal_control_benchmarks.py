from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cqed_sim import (
    AmplitudePenalty,
    ControlEvaluationCase,
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    LeakagePenalty,
    ModelControlChannelSpec,
    ModelEnsembleMember,
    NoiseSpec,
    PiecewiseConstantTimeGrid,
    SlewRatePenalty,
    UnitaryObjective,
    build_control_problem_from_model,
    state_preparation_objective,
)
from cqed_sim.unitary_synthesis import Subspace


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _time_grid(*, slices: int, duration_s: float) -> PiecewiseConstantTimeGrid:
    return PiecewiseConstantTimeGrid.uniform(steps=int(slices), dt_s=float(duration_s) / float(slices))


def _constant_initial_schedule(problem, amplitude: float) -> np.ndarray:
    guess = np.zeros((problem.n_controls, problem.n_slices), dtype=float)
    guess[0, :] = float(amplitude)
    return guess


def _noise_spec(args: argparse.Namespace) -> NoiseSpec | None:
    if args.noise_t1_s is None and args.noise_tphi_s is None and args.noise_kappa is None:
        return None
    return NoiseSpec(t1=args.noise_t1_s, tphi=args.noise_tphi_s, kappa=args.noise_kappa)


def _penalties_for_case(args: argparse.Namespace, *, leakage_subspace: Subspace | None = None) -> tuple[Any, ...]:
    penalties: list[Any] = []
    if float(args.amplitude_penalty) > 0.0:
        penalties.append(AmplitudePenalty(weight=float(args.amplitude_penalty)))
    if float(args.slew_penalty) > 0.0:
        penalties.append(SlewRatePenalty(weight=float(args.slew_penalty)))
    if leakage_subspace is not None and float(args.leakage_weight) > 0.0:
        penalties.append(LeakagePenalty(subspace=leakage_subspace, weight=float(args.leakage_weight), metric="average"))
    return tuple(penalties)


def _two_mode_state_problem(args: argparse.Namespace):
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
    ensemble_members: tuple[ModelEnsembleMember, ...] = ()
    if args.robust_frequency_shift_rad_s is not None and float(args.robust_frequency_shift_rad_s) > 0.0:
        shift = float(args.robust_frequency_shift_rad_s)
        ensemble_members = (
            ModelEnsembleMember(model=replace(model, omega_q=model.omega_q + shift), label="q_plus_shift", weight=1.0),
            ModelEnsembleMember(model=replace(model, omega_q=model.omega_q - shift), label="q_minus_shift", weight=1.0),
        )
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=_time_grid(slices=args.slices, duration_s=args.duration_s),
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
        penalties=_penalties_for_case(args),
        ensemble_members=ensemble_members,
        ensemble_aggregate=args.aggregate_mode,
        metadata={"benchmark_case": args.case, "model_regime": "two_mode", "target_type": "state_transfer"},
    )
    return model, frame, problem, _constant_initial_schedule(problem, np.pi / (2.0 * float(args.duration_s)))


def _three_mode_state_problem(args: argparse.Namespace):
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=2.0 * np.pi * 5.0e9,
        omega_r=2.0 * np.pi * 7.5e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi_s=0.0,
        chi_r=0.0,
        chi_sr=0.0,
        kerr_s=0.0,
        kerr_r=0.0,
        n_storage=1,
        n_readout=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_s, omega_q_frame=model.omega_q, omega_r_frame=model.omega_r)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=_time_grid(slices=args.slices, duration_s=args.duration_s),
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0, 0), model.basis_state(1, 0, 0)),),
        penalties=_penalties_for_case(args),
        ensemble_aggregate=args.aggregate_mode,
        metadata={"benchmark_case": args.case, "model_regime": "three_mode", "target_type": "state_transfer"},
    )
    return model, frame, problem, _constant_initial_schedule(problem, np.pi / (2.0 * float(args.duration_s)))


def _storage_subspace_problem(args: argparse.Namespace):
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
        time_grid=_time_grid(slices=args.slices, duration_s=args.duration_s),
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
                target_operator=rotation_y(np.pi / 2.0),
                subspace=storage_logical,
                ignore_global_phase=True,
                name="storage_y90",
            ),
        ),
        penalties=_penalties_for_case(args, leakage_subspace=storage_logical),
        ensemble_aggregate=args.aggregate_mode,
        metadata={"benchmark_case": args.case, "model_regime": "two_mode", "target_type": "subspace_unitary"},
    )
    return model, frame, problem, _constant_initial_schedule(problem, 0.1 * np.pi / float(args.duration_s))


CASE_BUILDERS = {
    "small_state": _two_mode_state_problem,
    "larger_state": _two_mode_state_problem,
    "three_mode_state": _three_mode_state_problem,
    "storage_subspace": _storage_subspace_problem,
}


DEFAULT_SUITES = {
    "default": [
        {"case": "small_state", "slices": 4, "duration_s": 40.0e-9},
        {"case": "larger_state", "slices": 24, "duration_s": 160.0e-9},
        {"case": "storage_subspace", "slices": 16, "duration_s": 80.0e-9, "leakage_weight": 0.2},
        {"case": "small_state", "slices": 12, "duration_s": 80.0e-9, "case_label": "reduced_compare"},
        {"case": "three_mode_state", "slices": 12, "duration_s": 80.0e-9, "case_label": "three_mode_compare"},
    ],
    "smoke": [{"case": "small_state", "slices": 2, "duration_s": 40.0e-9}],
}


def _solver_result(args: argparse.Namespace, problem, initial_schedule):
    backend = str(args.backend).lower()
    if backend != "grape":
        raise ValueError(f"Unsupported optimizer backend '{args.backend}'. Only 'grape' is currently available.")
    solver = GrapeSolver(
        GrapeConfig(
            maxiter=int(args.maxiter),
            seed=int(args.seed),
            random_scale=float(args.random_scale),
            history_every=1,
        )
    )
    return solver.solve(problem, initial_schedule=initial_schedule)


def run_benchmark_case(args: argparse.Namespace) -> dict[str, Any]:
    builder = CASE_BUILDERS[str(args.case)]
    model, frame, problem, initial_schedule = builder(args)
    solve_t0 = time.perf_counter()
    result = _solver_result(args, problem, initial_schedule)
    solve_runtime_s = float(time.perf_counter() - solve_t0)

    nominal_eval = result.evaluate_with_simulator(
        problem,
        model=model,
        frame=frame,
        compiler_dt_s=args.compiler_dt_s,
        max_step_s=args.max_step_s,
    )
    noise = _noise_spec(args)
    noisy_eval = None
    if noise is not None:
        noisy_eval = result.evaluate_with_simulator(
            problem,
            cases=(
                ControlEvaluationCase(
                    model=model,
                    label="noisy",
                    frame=frame,
                    noise=noise,
                    compiler_dt_s=args.compiler_dt_s,
                    max_step_s=args.max_step_s,
                ),
            ),
        )

    return {
        "case": str(args.case),
        "backend": str(args.backend),
        "seed": int(args.seed),
        "configuration": {
            "slices": int(args.slices),
            "duration_s": float(args.duration_s),
            "target_type": str(problem.metadata.get("target_type", "unknown")),
            "model_regime": str(problem.metadata.get("model_regime", "unknown")),
            "aggregate_mode": str(args.aggregate_mode),
            "amplitude_penalty": float(args.amplitude_penalty),
            "slew_penalty": float(args.slew_penalty),
            "leakage_weight": float(args.leakage_weight),
            "robust_frequency_shift_rad_s": None
            if args.robust_frequency_shift_rad_s is None
            else float(args.robust_frequency_shift_rad_s),
            "compiler_dt_s": None if args.compiler_dt_s is None else float(args.compiler_dt_s),
            "max_step_s": None if args.max_step_s is None else float(args.max_step_s),
        },
        "model": {
            "type": type(model).__name__,
            "subsystem_dims": [int(value) for value in getattr(model, "subsystem_dims", ())],
            "full_dim": int(problem.full_dim),
        },
        "solve": {
            "success": bool(result.success),
            "message": str(result.message),
            "runtime_s": float(solve_runtime_s),
            "iteration_count": int(result.optimizer_summary.get("nit", 0)),
            "function_evaluations": int(result.optimizer_summary.get("nfev", 0)),
            "objective_value": float(result.objective_value),
            "nominal_fidelity": float(result.metrics.get("nominal_fidelity", np.nan)),
            "control_penalty_total": float(result.metrics.get("control_penalty_total", 0.0)),
            "amplitude_penalty": float(result.metrics.get("amplitude_penalty", 0.0)),
            "slew_penalty": float(result.metrics.get("slew_penalty", 0.0)),
            "max_abs_amplitude": float(result.schedule.max_abs_amplitude()),
        },
        "nominal_replay": nominal_eval.to_payload(),
        "noisy_replay": None if noisy_eval is None else noisy_eval.to_payload(),
        "history": [
            {
                "evaluation": int(record.evaluation),
                "objective": float(record.objective),
                "gradient_norm": float(record.gradient_norm),
                "elapsed_s": float(record.elapsed_s),
            }
            for record in result.history
        ],
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    suite_name = str(args.suite)
    cases = DEFAULT_SUITES[suite_name]
    results: list[dict[str, Any]] = []
    for case in cases:
        case_args = argparse.Namespace(**vars(args))
        for key, value in case.items():
            if key == "case_label":
                continue
            setattr(case_args, key, value)
        result = run_benchmark_case(case_args)
        if "case_label" in case:
            result["case_label"] = str(case["case_label"])
        results.append(result)
    return {"suite": suite_name, "results": results}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRAPE optimal-control benchmarks and noisy replay validation.")
    parser.add_argument("--suite", choices=sorted(DEFAULT_SUITES.keys()), default=None, help="Run a predefined benchmark suite.")
    parser.add_argument("--case", choices=sorted(CASE_BUILDERS.keys()), default="small_state", help="Single benchmark case to run.")
    parser.add_argument("--slices", type=int, default=8, help="Number of time slices for the control parameterization.")
    parser.add_argument("--duration-s", type=float, default=80.0e-9, help="Total pulse duration in seconds.")
    parser.add_argument("--backend", type=str, default="grape", help="Optimizer backend selector. Currently only 'grape' is supported.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the optimizer configuration.")
    parser.add_argument("--maxiter", type=int, default=80, help="Maximum number of optimizer iterations.")
    parser.add_argument("--random-scale", type=float, default=0.2, help="Random initialization scale for GRAPE when needed.")
    parser.add_argument("--aggregate-mode", choices=("mean", "worst"), default="mean", help="Aggregate mode for robust optimization or evaluation cases.")
    parser.add_argument("--amplitude-penalty", type=float, default=0.0, help="Amplitude penalty weight.")
    parser.add_argument("--slew-penalty", type=float, default=0.0, help="Slew-rate penalty weight.")
    parser.add_argument("--leakage-weight", type=float, default=0.0, help="Leakage penalty weight when the case supports it.")
    parser.add_argument("--robust-frequency-shift-rad-s", type=float, default=None, help="Optional qubit-frequency shift used to add plus/minus robust ensemble members.")
    parser.add_argument("--noise-t1-s", type=float, default=2.0e-6, help="Optional T1 in seconds for noisy replay. Set together with other noise parameters as desired.")
    parser.add_argument("--noise-tphi-s", type=float, default=1.0e-6, help="Optional Tphi in seconds for noisy replay.")
    parser.add_argument("--noise-kappa", type=float, default=None, help="Optional bosonic loss rate in 1/s for noisy replay.")
    parser.add_argument("--compiler-dt-s", type=float, default=None, help="Optional replay compilation dt in seconds.")
    parser.add_argument("--max-step-s", type=float, default=None, help="Optional solver max_step in seconds for replay simulation.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    payload = run_suite(args) if args.suite is not None else run_benchmark_case(args)
    text = json.dumps(payload, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()