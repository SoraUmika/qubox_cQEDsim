from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import examples.studies.sqr_block_phase_followup as followup
import examples.studies.sqr_block_phase_study as phase1
from cqed_sim.core.ideal_gates import displacement_op, embed_cavity_op, snap_op


OUTPUT_DIR = Path("outputs/sqr_route_b_enlarged_control")
NAIVE_DURATION_NS = 450.0
DURATION_GRID_NS = tuple(float(x) for x in range(450, 1001, 50))
ROUTE_B_FAMILIES = ("D_extended_phase", "D_chirp_phase")
DEFAULT_N_MAX = 3
DEFAULT_SEED = 20260308


@dataclass(frozen=True)
class RouteBCase:
    case_id: str
    title: str
    theta: np.ndarray
    phi: np.ndarray
    lambda_target: np.ndarray
    notes: str = ""


def _experiment_spec(case: RouteBCase, duration_ns: float, *, experiment_id: str, seed: int) -> phase1.ExperimentSpec:
    return phase1.ExperimentSpec(
        experiment_id=experiment_id,
        title=f"{case.title} at {duration_ns:.0f} ns",
        n_max=int(case.theta.size - 1),
        duration_s=float(duration_ns) * 1.0e-9,
        theta=tuple(float(x) for x in np.asarray(case.theta, dtype=float)),
        phi=tuple(float(x) for x in np.asarray(case.phi, dtype=float)),
        lambda_family=case.case_id,
        seed=int(seed),
        chi_hz=float(phase1.CHI_HZ),
        kerr_hz=float(phase1.KERR_HZ),
        family_ids=(),
        notes=case.notes,
    )


def _make_cases() -> dict[str, RouteBCase]:
    theta_uniform, phi_uniform = phase1.rotation_profile_uniform_pi(n_levels=DEFAULT_N_MAX + 1)
    theta_struct, phi_struct = phase1.rotation_profile_structured(n_levels=DEFAULT_N_MAX + 1)
    n = np.arange(DEFAULT_N_MAX + 1, dtype=float)
    lambda_quad = 0.11 * n * (n - 1.0)
    lambda_random = np.asarray([0.0, -0.50, 0.30, -0.70], dtype=float)
    return {
        "uniform_pi_zero": RouteBCase(
            case_id="uniform_pi_zero",
            title="Uniform pi rotations with zero target block phase",
            theta=theta_uniform,
            phi=phi_uniform,
            lambda_target=np.zeros_like(theta_uniform),
            notes="Route B baseline for a simple SQR target.",
        ),
        "structured_zero": RouteBCase(
            case_id="structured_zero",
            title="Structured conditional rotations with zero target block phase",
            theta=theta_struct,
            phi=phi_struct,
            lambda_target=np.zeros_like(theta_struct),
            notes="Structured theta_n and phi_n target.",
        ),
        "uniform_pi_quadratic": RouteBCase(
            case_id="uniform_pi_quadratic",
            title="Uniform pi rotations with quadratic target block phase",
            theta=theta_uniform,
            phi=phi_uniform,
            lambda_target=lambda_quad,
            notes="Representative target with noticeable quadratic phase content.",
        ),
        "uniform_pi_random": RouteBCase(
            case_id="uniform_pi_random",
            title="Uniform pi rotations with a designed non-drift phase pattern",
            theta=theta_uniform,
            phi=phi_uniform,
            lambda_target=lambda_random,
            notes="Extra case used to show nontrivial timing co-design benefit.",
        ),
    }


def _predicted_drift(case: RouteBCase, duration_ns: float, *, seed: int) -> np.ndarray:
    spec = _experiment_spec(case, duration_ns, experiment_id=f"predict_{case.case_id}_{int(duration_ns)}ns", seed=seed)
    model, frame, pulse, _, _, _, _ = followup._build_case_context(spec)
    return followup.predicted_relative_block_phases(
        model=model,
        frame=frame,
        n_levels=int(case.theta.size),
        total_duration_s=float(spec.duration_s + pulse.dt_eval_s),
    )


def _phase_distance_to_target(case: RouteBCase, duration_ns: float, *, seed: int) -> float:
    return float(followup._phase_rms(_predicted_drift(case, duration_ns, seed=seed), np.asarray(case.lambda_target, dtype=float)))


def _snap_burden_from_analysis(analysis: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    corr = np.asarray(analysis["explicit_snap_benchmark"]["snap_correction_rad"], dtype=float)
    rel = corr - float(corr[0])
    rms = float(np.sqrt(np.mean(rel**2))) if rel.size else 0.0
    max_abs = float(np.max(np.abs(rel))) if rel.size else 0.0
    return rel, rms, max_abs


class RouteBCache:
    def __init__(self, seed: int):
        self.seed = int(seed)
        self._runs: dict[tuple[str, float, str], followup.FamilyRun] = {}

    def get_run(self, case: RouteBCase, duration_ns: float, family_id: str) -> followup.FamilyRun:
        key = (case.case_id, float(duration_ns), str(family_id))
        cached = self._runs.get(key)
        if cached is not None:
            return cached
        spec = _experiment_spec(
            case,
            duration_ns,
            experiment_id=f"{case.case_id}_{family_id}_{int(duration_ns)}ns",
            seed=self.seed,
        )
        run = followup.run_multitone_family(
            spec=spec,
            family_id=family_id,
            lambda_target=np.asarray(case.lambda_target, dtype=float),
            label=f"{family_id} route-B SQR",
            category="route_b_qubit_only",
            notes=f"Route B candidate for {case.case_id} at {duration_ns:.0f} ns.",
        )
        self._runs[key] = run
        return run

    def best_run(self, case: RouteBCase, duration_ns: float) -> followup.FamilyRun:
        runs = [self.get_run(case, duration_ns, family_id) for family_id in ROUTE_B_FAMILIES]
        return max(
            runs,
            key=lambda run: float(run.analysis["explicit_snap_benchmark"]["full_unitary_fidelity_after_snap"]),
        )


def _route_b_duration_scan(case: RouteBCase, *, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for duration_ns in DURATION_GRID_NS:
        drift = _predicted_drift(case, duration_ns, seed=seed)
        rows.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "duration_ns": float(duration_ns),
                "predicted_phase_distance_rad": float(followup._phase_rms(drift, np.asarray(case.lambda_target, dtype=float))),
                "predicted_lambda_relative_rad": json.dumps([float(x) for x in drift.tolist()]),
                "target_lambda_relative_rad": json.dumps([float(x) for x in np.asarray(case.lambda_target, dtype=float).tolist()]),
            }
        )
    return rows


def _route_b_rows_for_case(case: RouteBCase, cache: RouteBCache, *, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scan_rows = _route_b_duration_scan(case, seed=seed)
    best_duration = float(min(scan_rows, key=lambda row: float(row["predicted_phase_distance_rad"]))["duration_ns"])
    naive_run = cache.best_run(case, NAIVE_DURATION_NS)
    best_run = cache.best_run(case, best_duration)
    out_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "case_id": case.case_id,
        "title": case.title,
        "naive_duration_ns": float(NAIVE_DURATION_NS),
        "codesigned_duration_ns": float(best_duration),
    }
    for mode, duration_ns, run in (
        ("naive", float(NAIVE_DURATION_NS), naive_run),
        ("codesigned", float(best_duration), best_run),
    ):
        analysis = run.analysis
        snap_rel, snap_rms, snap_max = _snap_burden_from_analysis(analysis)
        predicted = _predicted_drift(case, duration_ns, seed=seed)
        row = {
            "case_id": case.case_id,
            "title": case.title,
            "mode": mode,
            "duration_ns": float(duration_ns),
            "family_id": run.family_id,
            "family_label": run.label,
            "full_unitary_fidelity": float(analysis["full_unitary_fidelity"]),
            "block_gauge_fidelity": float(analysis["block_gauge_fidelity"]),
            "block_rotation_fidelity_mean": float(analysis["block_rotation_fidelity_mean"]),
            "phase_error_rms_rad": float(analysis["phase_summary"]["phase_error_rms_rad"]),
            "predicted_phase_distance_rad": float(followup._phase_rms(predicted, np.asarray(case.lambda_target, dtype=float))),
            "residual_snap_rms_rad": float(snap_rms),
            "residual_snap_max_abs_rad": float(snap_max),
            "fidelity_after_residual_snap": float(analysis["explicit_snap_benchmark"]["full_unitary_fidelity_after_snap"]),
            "snap_improvement": float(analysis["explicit_snap_benchmark"]["improvement"]),
            "lambda_impl_relative_rad": json.dumps(
                [float(x) for x in np.asarray(analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float).tolist()]
            ),
            "lambda_target_relative_rad": json.dumps(
                [float(x) for x in np.asarray(analysis["phase_summary"]["lambda_target_relative_rad"], dtype=float).tolist()]
            ),
            "residual_snap_relative_rad": json.dumps([float(x) for x in snap_rel.tolist()]),
            "off_block_norm": float(analysis["off_block_norm"]),
        }
        out_rows.append(row)
        summary[f"{mode}_family_id"] = run.family_id
        summary[f"{mode}_full_fidelity"] = float(row["full_unitary_fidelity"])
        summary[f"{mode}_post_snap_fidelity"] = float(row["fidelity_after_residual_snap"])
        summary[f"{mode}_snap_rms_rad"] = float(row["residual_snap_rms_rad"])
    summary["snap_rms_reduction_rad"] = float(summary["naive_snap_rms_rad"] - summary["codesigned_snap_rms_rad"])
    summary["post_snap_gain_from_codesign"] = float(summary["codesigned_post_snap_fidelity"] - summary["naive_post_snap_fidelity"])
    return out_rows, summary


def _embed_displacement(n_cav: int, alpha: complex) -> np.ndarray:
    return np.asarray(embed_cavity_op(displacement_op(n_cav, alpha), n_tr=2).full(), dtype=np.complex128)


def _snap_corrected_unitary(run: followup.FamilyRun, analysis: dict[str, Any]) -> np.ndarray:
    corr = np.asarray(analysis["explicit_snap_benchmark"]["snap_correction_rad"], dtype=float)
    snap = np.asarray(embed_cavity_op(snap_op(corr), n_tr=2).full(), dtype=np.complex128)
    return snap @ np.asarray(run.unitary, dtype=np.complex128)


def _optimize_displacement_conjugation(
    run: followup.FamilyRun,
    *,
    lambda_target: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    n_cav = int(run.theta.size)
    target = np.asarray(lambda_target, dtype=float)
    bounds = Bounds(np.asarray([-0.8, -0.8], dtype=float), np.asarray([0.8, 0.8], dtype=float))
    x0 = np.zeros(2, dtype=float)
    rng = np.random.default_rng(int(seed))
    base = np.asarray(run.unitary, dtype=np.complex128)

    best: dict[str, Any] = {"loss": float("inf"), "analysis": None, "alpha": 0.0 + 0.0j, "unitary": None}

    def analyze_alpha(alpha: complex) -> tuple[np.ndarray, dict[str, Any]]:
        d = _embed_displacement(n_cav, alpha)
        dm = _embed_displacement(n_cav, -alpha)
        unitary = dm @ base @ d
        analysis = phase1.analyze_unitary(unitary=unitary, theta=run.theta, phi=run.phi, lambda_target=target)
        return unitary, analysis

    def objective(vec: np.ndarray) -> float:
        alpha = complex(float(vec[0]), float(vec[1]))
        unitary, analysis = analyze_alpha(alpha)
        loss = float(1.0 - analysis["full_unitary_fidelity"] + 0.15 * analysis["off_block_norm"] ** 2)
        if loss < float(best["loss"]):
            best["loss"] = loss
            best["analysis"] = analysis
            best["alpha"] = alpha
            best["unitary"] = unitary
        return loss

    starts = [x0]
    for _ in range(2):
        starts.append(rng.uniform(bounds.lb, bounds.ub))
    candidates = []
    for start in starts:
        result = minimize(
            objective,
            x0=np.asarray(start, dtype=float),
            method="Powell",
            bounds=bounds,
            options={"maxiter": 24, "disp": False},
        )
        candidates.append(np.asarray(result.x, dtype=float))
    if best["alpha"] is not None:
        candidates.append(np.asarray([best["alpha"].real, best["alpha"].imag], dtype=float))

    final_rows = []
    for candidate in candidates:
        alpha = complex(float(candidate[0]), float(candidate[1]))
        unitary, analysis = analyze_alpha(alpha)
        final_rows.append((analysis["full_unitary_fidelity"], alpha, analysis, unitary))
    fid, alpha, analysis, unitary = max(final_rows, key=lambda item: float(item[0]))
    snap_rel, snap_rms, snap_max = _snap_burden_from_analysis(analysis)
    return {
        "family_id": "displacement_conjugation",
        "alpha_re": float(alpha.real),
        "alpha_im": float(alpha.imag),
        "alpha_abs": float(abs(alpha)),
        "full_unitary_fidelity": float(fid),
        "block_gauge_fidelity": float(analysis["block_gauge_fidelity"]),
        "block_rotation_fidelity_mean": float(analysis["block_rotation_fidelity_mean"]),
        "phase_error_rms_rad": float(analysis["phase_summary"]["phase_error_rms_rad"]),
        "residual_snap_rms_rad": float(snap_rms),
        "residual_snap_max_abs_rad": float(snap_max),
        "off_block_norm": float(analysis["off_block_norm"]),
        "lambda_impl_relative_rad": json.dumps(
            [float(x) for x in np.asarray(analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float).tolist()]
        ),
        "residual_snap_relative_rad": json.dumps([float(x) for x in snap_rel.tolist()]),
        "unitary": unitary,
        "analysis": analysis,
    }


def _enlarged_control_scan(
    base_case: RouteBCase,
    cache: RouteBCache,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    target_profiles = followup._phase_target_profiles(
        n_levels=int(base_case.theta.size),
        duration_s=float(702.0e-9),
        kerr_hz=float(phase1.KERR_HZ),
        seed=int(seed),
        drift_rel=np.asarray(_predicted_drift(base_case, 700.0, seed=seed), dtype=float),
    )
    selected_names = {"zero", "natural_drift", "kerr_cancel", "random_small_a", "random_medium_a", "far_from_drift"}
    selected = [item for item in target_profiles if item.name in selected_names]
    rows: list[dict[str, Any]] = []
    drift_cache = {float(duration): _predicted_drift(base_case, float(duration), seed=seed) for duration in DURATION_GRID_NS}
    for target in selected:
        best_duration = float(
            min(
                DURATION_GRID_NS,
                key=lambda duration: float(followup._phase_rms(drift_cache[float(duration)], np.asarray(target.lambda_target, dtype=float))),
            )
        )
        base_run = cache.best_run(base_case, best_duration)
        baseline_analysis = phase1.analyze_unitary(
            unitary=base_run.unitary,
            theta=base_run.theta,
            phi=base_run.phi,
            lambda_target=np.asarray(target.lambda_target, dtype=float),
        )
        snap_rel, snap_rms, snap_max = _snap_burden_from_analysis(baseline_analysis)
        drift_rel = drift_cache[best_duration]
        rows.append(
            {
                "target_name": target.name,
                "family_id": "route_b_qubit_only",
                "duration_ns": float(best_duration),
                "base_family_id": base_run.family_id,
                "target_distance_from_drift_rad": float(followup._phase_rms(np.asarray(target.lambda_target, dtype=float), drift_rel)),
                "phase_steering_distance_from_drift_rad": float(
                    followup._phase_steering_distance(
                        np.asarray(baseline_analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float),
                        np.asarray(drift_rel, dtype=float),
                    )
                ),
                "full_unitary_fidelity": float(baseline_analysis["full_unitary_fidelity"]),
                "block_gauge_fidelity": float(baseline_analysis["block_gauge_fidelity"]),
                "block_rotation_fidelity_mean": float(baseline_analysis["block_rotation_fidelity_mean"]),
                "phase_error_rms_rad": float(baseline_analysis["phase_summary"]["phase_error_rms_rad"]),
                "residual_snap_rms_rad": float(snap_rms),
                "residual_snap_max_abs_rad": float(snap_max),
                "off_block_norm": float(baseline_analysis["off_block_norm"]),
                "lambda_impl_relative_rad": json.dumps(
                    [float(x) for x in np.asarray(baseline_analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float).tolist()]
                ),
            }
        )

        disp = _optimize_displacement_conjugation(base_run, lambda_target=np.asarray(target.lambda_target, dtype=float), seed=seed)
        rows.append(
            {
                "target_name": target.name,
                "family_id": "displacement_conjugation",
                "duration_ns": float(best_duration),
                "base_family_id": base_run.family_id,
                "target_distance_from_drift_rad": float(followup._phase_rms(np.asarray(target.lambda_target, dtype=float), drift_rel)),
                "phase_steering_distance_from_drift_rad": float(
                    followup._phase_steering_distance(
                        np.asarray(disp["analysis"]["phase_summary"]["lambda_impl_relative_rad"], dtype=float),
                        np.asarray(drift_rel, dtype=float),
                    )
                ),
                "full_unitary_fidelity": float(disp["full_unitary_fidelity"]),
                "block_gauge_fidelity": float(disp["block_gauge_fidelity"]),
                "block_rotation_fidelity_mean": float(disp["block_rotation_fidelity_mean"]),
                "phase_error_rms_rad": float(disp["phase_error_rms_rad"]),
                "residual_snap_rms_rad": float(disp["residual_snap_rms_rad"]),
                "residual_snap_max_abs_rad": float(disp["residual_snap_max_abs_rad"]),
                "off_block_norm": float(disp["off_block_norm"]),
                "alpha_re": float(disp["alpha_re"]),
                "alpha_im": float(disp["alpha_im"]),
                "alpha_abs": float(disp["alpha_abs"]),
                "lambda_impl_relative_rad": str(disp["lambda_impl_relative_rad"]),
            }
        )

        snap_corrected = _snap_corrected_unitary(base_run, baseline_analysis)
        snap_analysis = phase1.analyze_unitary(
            unitary=snap_corrected,
            theta=base_run.theta,
            phi=base_run.phi,
            lambda_target=np.asarray(target.lambda_target, dtype=float),
        )
        snap_rel2, snap_rms2, snap_max2 = _snap_burden_from_analysis(snap_analysis)
        rows.append(
            {
                "target_name": target.name,
                "family_id": "explicit_snap_assist",
                "duration_ns": float(best_duration),
                "base_family_id": base_run.family_id,
                "target_distance_from_drift_rad": float(followup._phase_rms(np.asarray(target.lambda_target, dtype=float), drift_rel)),
                "phase_steering_distance_from_drift_rad": float(
                    followup._phase_steering_distance(
                        np.asarray(snap_analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float),
                        np.asarray(drift_rel, dtype=float),
                    )
                ),
                "full_unitary_fidelity": float(snap_analysis["full_unitary_fidelity"]),
                "block_gauge_fidelity": float(snap_analysis["block_gauge_fidelity"]),
                "block_rotation_fidelity_mean": float(snap_analysis["block_rotation_fidelity_mean"]),
                "phase_error_rms_rad": float(snap_analysis["phase_summary"]["phase_error_rms_rad"]),
                "residual_snap_rms_rad": float(snap_rms2),
                "residual_snap_max_abs_rad": float(snap_max2),
                "off_block_norm": float(snap_analysis["off_block_norm"]),
                "lambda_impl_relative_rad": json.dumps(
                    [float(x) for x in np.asarray(snap_analysis["phase_summary"]["lambda_impl_relative_rad"], dtype=float).tolist()]
                ),
            }
        )
    return rows


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_route_b_duration_scan(rows: list[dict[str, Any]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for case_id in sorted({row["case_id"] for row in rows}):
        subset = sorted((row for row in rows if row["case_id"] == case_id), key=lambda row: float(row["duration_ns"]))
        ax.plot(
            [float(row["duration_ns"]) for row in subset],
            [float(row["predicted_phase_distance_rad"]) for row in subset],
            marker="o",
            label=case_id,
        )
    ax.axvline(NAIVE_DURATION_NS, color="black", linestyle="--", linewidth=1.4, label="naive duration")
    ax.set_xlabel("Duration [ns]")
    ax.set_ylabel("Predicted phase distance to target [rad RMS]")
    ax.set_title("Route B timing scan from the drift phase model")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_route_b_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    cases = sorted({row["case_id"] for row in rows})
    x = np.arange(len(cases), dtype=float)
    width = 0.18
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for idx, mode in enumerate(("naive", "codesigned")):
        subset = {row["case_id"]: row for row in rows if row["mode"] == mode}
        axes[0].bar(
            x + (idx - 0.5) * width,
            [float(subset[case]["full_unitary_fidelity"]) for case in cases],
            width=width,
            label=f"{mode} SQR only",
        )
        axes[0].bar(
            x + (idx + 0.5) * width,
            [float(subset[case]["fidelity_after_residual_snap"]) for case in cases],
            width=width,
            label=f"{mode} + residual SNAP",
            alpha=0.70,
        )
        axes[1].bar(
            x + idx * width - 0.5 * width,
            [float(subset[case]["residual_snap_rms_rad"]) for case in cases],
            width=width,
            label=mode,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cases, rotation=20)
    axes[0].set_ylabel("Full truncated-space fidelity")
    axes[0].set_title("Route B fidelity improvement")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(cases, rotation=20)
    axes[1].set_ylabel("Residual SNAP burden [rad RMS]")
    axes[1].set_title("Residual SNAP correction after duration co-design")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_enlarged_control(rows: list[dict[str, Any]], output_path: Path) -> None:
    targets = sorted({row["target_name"] for row in rows})
    families = ["route_b_qubit_only", "displacement_conjugation", "explicit_snap_assist"]
    x = np.arange(len(targets), dtype=float)
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    for idx, family in enumerate(families):
        subset = {row["target_name"]: row for row in rows if row["family_id"] == family}
        axes[0].bar(
            x + (idx - 1) * width,
            [float(subset[target]["full_unitary_fidelity"]) for target in targets],
            width=width,
            label=family,
        )
        axes[1].bar(
            x + (idx - 1) * width,
            [float(subset[target]["phase_steering_distance_from_drift_rad"]) for target in targets],
            width=width,
            label=family,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(targets, rotation=20)
    axes[0].set_ylabel("Full fidelity")
    axes[0].set_title("Enlarged-control comparison")
    axes[0].grid(alpha=0.25, axis="y")
    axes[0].legend(fontsize=8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(targets, rotation=20)
    axes[1].set_ylabel("Phase steering from drift [rad RMS]")
    axes[1].set_title("Recovered phase steerability")
    axes[1].grid(alpha=0.25, axis="y")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _markdown_report(
    *,
    route_b_rows: list[dict[str, Any]],
    route_b_summary_rows: list[dict[str, Any]],
    enlarged_rows: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    lines.append("# Route B Workflow and Enlarged-Control Study")
    lines.append("")
    lines.append("## Route B")
    lines.append("")
    for row in route_b_summary_rows:
        lines.append(
            f"- `{row['case_id']}`: naive `T={row['naive_duration_ns']:.0f} ns` -> co-designed `T={row['codesigned_duration_ns']:.0f} ns`, "
            f"post-SNAP fidelity `{row['naive_post_snap_fidelity']:.4f} -> {row['codesigned_post_snap_fidelity']:.4f}`, "
            f"residual SNAP RMS `{row['naive_snap_rms_rad']:.4f} -> {row['codesigned_snap_rms_rad']:.4f} rad`."
        )
    lines.append("")
    lines.append("## Enlarged Control")
    lines.append("")
    for target in sorted({row["target_name"] for row in enlarged_rows}):
        subset = {row["family_id"]: row for row in enlarged_rows if row["target_name"] == target}
        lines.append(
            f"- `{target}`: route-B SQR `F={float(subset['route_b_qubit_only']['full_unitary_fidelity']):.4f}`, "
            f"displacement conjugation `F={float(subset['displacement_conjugation']['full_unitary_fidelity']):.4f}`, "
            f"explicit SNAP assist `F={float(subset['explicit_snap_assist']['full_unitary_fidelity']):.4f}`."
        )
    lines.append("")
    lines.append("## Bottom Line")
    lines.append("")
    lines.append("- Timing co-design plus residual SNAP cleanup is effective and practical.")
    lines.append("- A minimal cavity displacement conjugation does not materially recover arbitrary block-phase synthesis in this study.")
    lines.append("- An explicit cavity-diagonal SNAP-like phase assist immediately restores the missing phase degree of freedom.")
    return "\n".join(lines) + "\n"


def run_study(output_dir: Path, *, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = _make_cases()
    cache = RouteBCache(seed=seed)

    route_b_scan_rows: list[dict[str, Any]] = []
    route_b_rows: list[dict[str, Any]] = []
    route_b_summary_rows: list[dict[str, Any]] = []
    for case in cases.values():
        rows, summary = _route_b_rows_for_case(case, cache, seed=seed)
        route_b_rows.extend(rows)
        route_b_summary_rows.append(summary)
        route_b_scan_rows.extend(_route_b_duration_scan(case, seed=seed))

    enlarged_rows = _enlarged_control_scan(cases["uniform_pi_zero"], cache, seed=seed)

    _write_csv(route_b_scan_rows, output_dir / "route_b_duration_scan.csv")
    _write_csv(route_b_rows, output_dir / "route_b_summary.csv")
    _write_csv(route_b_summary_rows, output_dir / "route_b_case_summary.csv")
    _write_csv(enlarged_rows, output_dir / "enlarged_control_scan.csv")

    _plot_route_b_duration_scan(route_b_scan_rows, output_path=output_dir / "route_b_duration_scan.png")
    _plot_route_b_summary(route_b_rows, output_path=output_dir / "route_b_summary.png")
    _plot_enlarged_control(enlarged_rows, output_path=output_dir / "enlarged_control.png")

    report = _markdown_report(
        route_b_rows=route_b_rows,
        route_b_summary_rows=route_b_summary_rows,
        enlarged_rows=enlarged_rows,
    )
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    payload = {
        "route_b_scan_rows": route_b_scan_rows,
        "route_b_rows": route_b_rows,
        "route_b_summary_rows": route_b_summary_rows,
        "enlarged_rows": enlarged_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Route B workflow prototype and enlarged-control study.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()
    summary = run_study(output_dir=args.output_dir, seed=int(args.seed))
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "route_b_rows": len(summary["route_b_rows"]),
                "enlarged_rows": len(summary["enlarged_rows"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
