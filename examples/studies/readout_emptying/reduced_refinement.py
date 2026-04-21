from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.optimal_control import (
    refine_readout_emptying_pulse,
    synthesize_readout_emptying_pulse,
    verify_readout_emptying_pulse,
)

from common import (
    comparison_payload,
    hardware_models,
    nonlinear_spec,
    refinement_config,
    study_output_dir,
    verification_config,
)


def _leakage_vs_strength(path, strength_mhz, leakages, peak_photons) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(strength_mhz, leakages, marker="o", color="tab:purple", label="non-QND total")
    axis.set_xlabel("Amplitude cap / 2pi (MHz)")
    axis.set_ylabel("non-QND total")
    axis.grid(alpha=0.25)
    twin = axis.twinx()
    twin.plot(strength_mhz, peak_photons, marker="s", color="tab:gray", label="peak photons")
    twin.set_ylabel("Peak photons")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _refined_comparison(path, report) -> None:
    labels = ("square", "analytic_seed", "kerr_corrected", "refined")
    residuals = [report.comparison_table[label]["max_final_residual_photons"] for label in labels]
    separations = [report.comparison_table[label]["measurement_chain_separation"] for label in labels]
    x = np.arange(len(labels), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    axes[0].bar(x, residuals, color=["tab:red", "tab:blue", "tab:green", "tab:orange"])
    axes[0].set_xticks(x, labels, rotation=15)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Max residual photons")
    axes[0].grid(alpha=0.25, axis="y")
    axes[1].bar(x, separations, color=["tab:red", "tab:blue", "tab:green", "tab:orange"])
    axes[1].set_xticks(x, labels, rotation=15)
    axes[1].set_ylabel("Measurement-chain separation")
    axes[1].grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = study_output_dir("03_reduced_refinement")
    spec, constraints = nonlinear_spec(include_kerr_phase_correction=True)
    hardware, variants = hardware_models()
    seed_result = synthesize_readout_emptying_pulse(spec, constraints)
    refined = refine_readout_emptying_pulse(
        seed_result,
        refinement_config(spec, hardware=hardware, hardware_variants=variants, shots_per_branch=24, maxiter=8),
    )
    report = refined.verification_report
    assert report is not None

    amplitude_caps = np.linspace(5.0, 9.0, 5)
    leakages: list[float] = []
    peak_photons: list[float] = []
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
        leakages.append(float(amplitude_report.comparison_table["kerr_corrected"]["non_qnd_total"]))
        peak_photons.append(float(amplitude_report.comparison_table["kerr_corrected"]["peak_photons_e"]))

    _leakage_vs_strength(output_dir / "leakage_vs_strength.png", amplitude_caps, leakages, peak_photons)
    _refined_comparison(output_dir / "refined_comparison.png", report)

    payload = comparison_payload(report, refined=refined)
    payload["amplitude_sweep"] = {
        "amplitude_cap_mhz": [float(value) for value in amplitude_caps],
        "non_qnd_total": leakages,
        "peak_photons_e": peak_photons,
    }
    payload["artifacts"] = {
        "leakage_vs_strength": str(output_dir / "leakage_vs_strength.png"),
        "refined_comparison": str(output_dir / "refined_comparison.png"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved reduced-refinement artifacts to {output_dir}")


if __name__ == "__main__":
    main()
