from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.optimal_control import replay_kerr_readout_branches, synthesize_readout_emptying_pulse, verify_readout_emptying_pulse

from common import (
    hardware_models,
    nonlinear_spec,
    study_output_dir,
    verification_config,
)


def _prefilter_vs_postfilter(path, report) -> None:
    hardware = report.diagnostics["hardware"]["kerr_corrected"]
    command = np.asarray(hardware["command_values"], dtype=float)
    physical = np.asarray(hardware["physical_values"], dtype=float)
    time_ns = report.baseline_results["kerr_corrected"].segment_edges_s[:-1] * 1.0e9
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    axes[0].step(time_ns, command[0] / (2.0 * np.pi * 1.0e6), where="post", label="Command")
    axes[0].step(time_ns, physical[0] / (2.0 * np.pi * 1.0e6), where="post", label="Physical")
    axes[1].step(time_ns, command[1] / (2.0 * np.pi * 1.0e6), where="post", label="Command")
    axes[1].step(time_ns, physical[1] / (2.0 * np.pi * 1.0e6), where="post", label="Physical")
    axes[0].set_ylabel("I / 2pi (MHz)")
    axes[1].set_ylabel("Q / 2pi (MHz)")
    axes[1].set_xlabel("Time (ns)")
    axes[0].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _mismatch_heatmap(path, physical_result) -> None:
    chi_scales = np.linspace(0.94, 1.06, 7)
    kappa_scales = np.linspace(0.94, 1.06, 7)
    heatmap = np.empty((chi_scales.size, kappa_scales.size), dtype=float)
    for row, chi_scale in enumerate(chi_scales):
        for col, kappa_scale in enumerate(kappa_scales):
            variant_spec = physical_result.spec.__class__(
                kappa=float(physical_result.spec.kappa) * float(kappa_scale),
                chi=float(physical_result.spec.chi) * float(chi_scale),
                tau=float(physical_result.spec.tau),
                n_segments=int(physical_result.spec.n_segments),
                detuning_center=float(physical_result.spec.detuning_center),
                segment_times=tuple(float(value) for value in physical_result.segment_edges_s),
                allow_complex_segments=bool(physical_result.spec.allow_complex_segments),
                target_states=tuple(physical_result.spec.target_states),
                kerr=float(physical_result.spec.kerr),
                include_kerr_phase_correction=bool(physical_result.spec.include_kerr_phase_correction),
                kerr_correction_strategy=str(physical_result.spec.kerr_correction_strategy),
                replay_dt=physical_result.spec.replay_dt,
            )
            replay = replay_kerr_readout_branches(variant_spec, physical_result.segment_amplitudes)
            heatmap[row, col] = float(max(replay.final_n.values()))
    fig, axis = plt.subplots(figsize=(5.8, 4.8))
    image = axis.imshow(
        heatmap,
        origin="lower",
        aspect="auto",
        extent=(kappa_scales[0], kappa_scales[-1], chi_scales[0], chi_scales[-1]),
    )
    axis.set_xlabel("kappa scale")
    axis.set_ylabel("chi scale")
    axis.set_title("Max final residual photons")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _performance_vs_max_photons(path, amplitudes_mhz, peak_photons, accuracies) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(peak_photons, accuracies, marker="o")
    for index, label in enumerate(amplitudes_mhz):
        axis.annotate(f"{label:.1f}", (peak_photons[index], accuracies[index]), textcoords="offset points", xytext=(4, 4))
    axis.set_xlabel("Peak photons (e branch)")
    axis.set_ylabel("Synthetic assignment accuracy")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = study_output_dir("04_hardware_sensitivity")
    spec, constraints = nonlinear_spec(include_kerr_phase_correction=True)
    hardware, variants = hardware_models()
    result = synthesize_readout_emptying_pulse(spec, constraints)
    report = verify_readout_emptying_pulse(
        result,
        verification_config(spec, hardware=hardware, hardware_variants=variants, shots_per_branch=32),
    )

    _prefilter_vs_postfilter(output_dir / "prefilter_vs_postfilter.png", report)
    _mismatch_heatmap(output_dir / "mismatch_heatmap.png", report.baseline_results["kerr_corrected"])

    amplitude_caps = np.linspace(5.0, 9.0, 5)
    peak_photons: list[float] = []
    accuracies: list[float] = []
    for amplitude_cap in amplitude_caps:
        amplitude_spec, amplitude_constraints = nonlinear_spec(
            include_kerr_phase_correction=True,
            amplitude_max=2.0 * np.pi * float(amplitude_cap) * 1.0e6,
        )
        amplitude_result = synthesize_readout_emptying_pulse(amplitude_spec, amplitude_constraints)
        amplitude_report = verify_readout_emptying_pulse(
            amplitude_result,
            verification_config(amplitude_spec, hardware=hardware, hardware_variants=variants, shots_per_branch=16),
        )
        peak_photons.append(float(amplitude_report.comparison_table["kerr_corrected"]["peak_photons_e"]))
        accuracies.append(float(amplitude_report.comparison_table["kerr_corrected"]["measurement_chain_accuracy"]))
    _performance_vs_max_photons(output_dir / "performance_vs_max_photons.png", amplitude_caps, peak_photons, accuracies)

    payload = {
        "hardware_metrics": report.hardware_metrics,
        "comparison_table": report.comparison_table,
        "amplitude_sweep": {
            "amplitude_cap_mhz": [float(value) for value in amplitude_caps],
            "peak_photons_e": peak_photons,
            "measurement_chain_accuracy": accuracies,
        },
        "artifacts": {
            "prefilter_vs_postfilter": str(output_dir / "prefilter_vs_postfilter.png"),
            "mismatch_heatmap": str(output_dir / "mismatch_heatmap.png"),
            "performance_vs_max_photons": str(output_dir / "performance_vs_max_photons.png"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved hardware-sensitivity artifacts to {output_dir}")


if __name__ == "__main__":
    main()
