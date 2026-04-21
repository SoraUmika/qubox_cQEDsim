from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.optimal_control import synthesize_readout_emptying_pulse, verify_readout_emptying_pulse

from common import (
    comparison_payload,
    hardware_models,
    nonlinear_spec,
    study_output_dir,
    verification_config,
)


def _output_iq_trajectories(path, report) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    for axis, label in zip(axes, ("square", "kerr_corrected"), strict=True):
        lindblad = report.diagnostics["lindblad"][label]
        for state, color in (("g", "tab:blue"), ("e", "tab:orange")):
            output = np.asarray(lindblad["output_field"][state], dtype=np.complex128)
            axis.plot(output.real, output.imag, color=color, label=state)
            axis.scatter([output[-1].real], [output[-1].imag], color=color, s=25)
        axis.set_title(label)
        axis.set_xlabel("I")
        axis.set_ylabel("Q")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _integrated_pointer_separation(path, report) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 4.0))
    for label, style in (("square", "--"), ("analytic_seed", ":"), ("kerr_corrected", "-")):
        lindblad = report.diagnostics["lindblad"][label]
        time_grid = np.asarray(lindblad["time_grid_s"], dtype=float)
        diff = np.asarray(lindblad["output_field"]["e"]) - np.asarray(lindblad["output_field"]["g"])
        cumulative = np.cumsum(np.abs(diff[:-1]) ** 2 * np.diff(time_grid))
        axis.plot(time_grid[1:] * 1.0e9, cumulative, linestyle=style, label=label)
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("Cumulative |a_out,e-a_out,g|^2 dt")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _iq_clouds(path, report) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.5))
    for axis, label in zip(axes, ("square", "kerr_corrected"), strict=True):
        measurement = report.diagnostics["measurement"][label]
        for state, color in (("g", "tab:blue"), ("e", "tab:orange")):
            samples = np.asarray(measurement["sampled_iq"][state], dtype=float)
            center = np.asarray(measurement["iq_centers"][state], dtype=float)
            axis.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.45, color=color)
            axis.scatter([center[0]], [center[1]], s=45, color=color, edgecolors="black", label=state)
        axis.set_title(label)
        axis.set_xlabel("I")
        axis.set_ylabel("Q")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _measurement_overlap_error(path, report) -> None:
    labels = ("square", "analytic_seed", "kerr_corrected")
    values = [report.measurement_metrics[label]["measurement_chain_gaussian_overlap_error"] for label in labels]
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(labels, values, color=["tab:red", "tab:blue", "tab:green"])
    axis.set_ylabel("Gaussian overlap error")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _residual_vs_discrimination(path, report) -> None:
    labels = ("square", "analytic_seed", "kerr_corrected")
    fig, axis = plt.subplots(figsize=(6.0, 5.0))
    for label, color in zip(labels, ("tab:red", "tab:blue", "tab:green"), strict=True):
        axis.scatter(
            report.comparison_table[label]["max_final_residual_photons"],
            report.comparison_table[label]["measurement_chain_gaussian_overlap_error"],
            s=60,
            color=color,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Max final residual photons")
    axis.set_ylabel("Gaussian overlap error")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ringdown_tail_comparison(path, report) -> None:
    labels = ("square", "analytic_seed", "kerr_corrected")
    values = [report.ringdown_metrics[label]["ringdown_time_to_threshold"] * 1.0e9 for label in labels]
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.bar(labels, values, color=["tab:red", "tab:blue", "tab:green"])
    axis.set_ylabel("Ringdown time to 1e-2 photons (ns)")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = study_output_dir("02_dispersive_lindblad_validation")
    spec, constraints = nonlinear_spec(include_kerr_phase_correction=True)
    result = synthesize_readout_emptying_pulse(spec, constraints)
    hardware, variants = hardware_models()
    report = verify_readout_emptying_pulse(
        result,
        verification_config(spec, hardware=hardware, hardware_variants=variants, shots_per_branch=48),
    )

    _output_iq_trajectories(output_dir / "output_iq_trajectories.png", report)
    _integrated_pointer_separation(output_dir / "integrated_pointer_separation.png", report)
    _iq_clouds(output_dir / "iq_clouds.png", report)
    _measurement_overlap_error(output_dir / "measurement_overlap_error.png", report)
    _residual_vs_discrimination(output_dir / "residual_vs_discrimination.png", report)
    _ringdown_tail_comparison(output_dir / "ringdown_tail_comparison.png", report)

    payload = comparison_payload(report)
    payload["artifacts"] = {
        "output_iq_trajectories": str(output_dir / "output_iq_trajectories.png"),
        "integrated_pointer_separation": str(output_dir / "integrated_pointer_separation.png"),
        "iq_clouds": str(output_dir / "iq_clouds.png"),
        "measurement_overlap_error": str(output_dir / "measurement_overlap_error.png"),
        "residual_vs_discrimination": str(output_dir / "residual_vs_discrimination.png"),
        "ringdown_tail_comparison": str(output_dir / "ringdown_tail_comparison.png"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved dispersive Lindblad validation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
