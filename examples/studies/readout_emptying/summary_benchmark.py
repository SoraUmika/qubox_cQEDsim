from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.optimal_control import refine_readout_emptying_pulse, synthesize_readout_emptying_pulse

from common import (
    comparison_payload,
    doc_asset_dir,
    hardware_models,
    nonlinear_spec,
    refinement_config,
    study_output_dir,
    verification_config,
)


DOC_ASSET_SOURCES: tuple[tuple[Path, str], ...] = (
    (Path("outputs") / "readout_emptying_qualification" / "00_linear_seed_validation" / "segment_waveform.png", "segment_waveform.png"),
    (Path("outputs") / "readout_emptying_qualification" / "00_linear_seed_validation" / "phase_space.png", "phase_space.png"),
    (Path("outputs") / "readout_emptying_qualification" / "01_kerr_replay_and_chirp" / "residual_vs_kerr.png", "residual_vs_kerr.png"),
    (Path("outputs") / "readout_emptying_qualification" / "01_kerr_replay_and_chirp" / "shared_vs_branch_specific.png", "shared_vs_branch_specific.png"),
    (Path("outputs") / "readout_emptying_qualification" / "02_dispersive_lindblad_validation" / "output_iq_trajectories.png", "output_iq_trajectories.png"),
    (Path("outputs") / "readout_emptying_qualification" / "02_dispersive_lindblad_validation" / "iq_clouds.png", "iq_clouds.png"),
    (Path("outputs") / "readout_emptying_qualification" / "02_dispersive_lindblad_validation" / "residual_vs_fidelity.png", "residual_vs_fidelity.png"),
    (Path("outputs") / "readout_emptying_qualification" / "04_hardware_sensitivity" / "mismatch_heatmap.png", "mismatch_heatmap.png"),
    (Path("outputs") / "readout_emptying_qualification" / "04_hardware_sensitivity" / "prefilter_vs_postfilter.png", "prefilter_vs_postfilter.png"),
)


def _benchmark_bars(path, report) -> None:
    labels = ("square", "analytic_seed", "kerr_corrected", "refined")
    residuals = [report.comparison_table[label]["max_final_residual_photons"] for label in labels]
    accuracies = [report.comparison_table[label]["measurement_chain_accuracy"] for label in labels]
    leakages = [report.comparison_table[label]["non_qnd_total"] for label in labels]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    axes[0].bar(x, residuals, color=["tab:red", "tab:blue", "tab:green", "tab:orange"])
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Residual photons")
    axes[1].bar(x, accuracies, color=["tab:red", "tab:blue", "tab:green", "tab:orange"])
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Assignment accuracy")
    axes[2].bar(x, leakages, color=["tab:red", "tab:blue", "tab:green", "tab:orange"])
    axes[2].set_xticks(x, labels, rotation=15)
    axes[2].set_ylabel("non-QND total")
    for axis in axes:
        axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _waveform_family(path, report) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    labels = ("analytic_seed", "kerr_corrected", "refined")
    colors = ("tab:blue", "tab:green", "tab:orange")
    for label, color in zip(labels, colors, strict=True):
        result = report.baseline_results[label]
        axes[0].step(
            result.segment_edges_s[:-1] * 1.0e9,
            result.segment_amplitudes.real / (2.0 * np.pi * 1.0e6),
            where="post",
            color=color,
            label=label,
        )
        axes[1].step(
            result.segment_edges_s[:-1] * 1.0e9,
            result.segment_amplitudes.imag / (2.0 * np.pi * 1.0e6),
            where="post",
            color=color,
            label=label,
        )
    axes[0].set_ylabel("Re[eps]/2pi (MHz)")
    axes[1].set_ylabel("Im[eps]/2pi (MHz)")
    axes[1].set_xlabel("Time (ns)")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _refresh_doc_assets(output_dir: Path, asset_dir: Path) -> dict[str, str]:
    copied: dict[str, str] = {}
    for source_name, target_name in (
        ("benchmark_comparison.png", "benchmark_comparison.png"),
        ("waveform_family.png", "waveform_family.png"),
    ):
        shutil.copyfile(output_dir / source_name, asset_dir / target_name)
        copied[target_name] = str(asset_dir / target_name)
    for source_path, target_name in DOC_ASSET_SOURCES:
        if source_path.exists():
            shutil.copyfile(source_path, asset_dir / target_name)
            copied[target_name] = str(asset_dir / target_name)
    return copied


def main() -> None:
    output_dir = study_output_dir("05_summary_benchmark")
    asset_dir = doc_asset_dir()
    spec, constraints = nonlinear_spec(include_kerr_phase_correction=True)
    hardware, variants = hardware_models()
    seed_result = synthesize_readout_emptying_pulse(spec, constraints)
    refined = refine_readout_emptying_pulse(
        seed_result,
        refinement_config(spec, hardware=hardware, hardware_variants=variants, shots_per_branch=24, maxiter=8),
    )
    report = refined.verification_report
    assert report is not None

    benchmark_plot = output_dir / "benchmark_comparison.png"
    waveform_plot = output_dir / "waveform_family.png"
    _benchmark_bars(benchmark_plot, report)
    _waveform_family(waveform_plot, report)
    copied_assets = _refresh_doc_assets(output_dir, asset_dir)

    payload = comparison_payload(report, refined=refined)
    payload["artifacts"] = {
        "benchmark_comparison": str(benchmark_plot),
        "waveform_family": str(waveform_plot),
        "doc_assets": dict(copied_assets),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved summary benchmark artifacts to {output_dir}")


if __name__ == "__main__":
    main()
