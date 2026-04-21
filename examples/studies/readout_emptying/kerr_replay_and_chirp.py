from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.optimal_control import ReadoutEmptyingSpec, synthesize_readout_emptying_pulse

from common import nonlinear_spec, study_output_dir


def _residual_vs_kerr(path, kerr_values, uncorrected, corrected) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    axis.plot(kerr_values, uncorrected, marker="o", label="Analytic seed")
    axis.plot(kerr_values, corrected, marker="o", label="Shared chirp")
    axis.set_xlabel("K / 2pi (MHz)")
    axis.set_ylabel("Max final residual photons")
    axis.set_yscale("log")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _shared_vs_branch_specific(path, residuals) -> None:
    labels = list(residuals)
    values = [residuals[label] for label in labels]
    fig, axis = plt.subplots(figsize=(6.5, 4.0))
    axis.bar(labels, values, color=["tab:orange", "tab:blue", "tab:green", "tab:red"])
    axis.set_ylabel("Max final residual photons")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _chirp_plot(path, diagnostics) -> None:
    phase = np.asarray(diagnostics["phase_rad"], dtype=float)
    chirp = np.asarray(diagnostics["instantaneous_chirp_rad_s"], dtype=float)
    time_grid_s = np.asarray(diagnostics["time_grid_s"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    axes[0].plot(time_grid_s * 1.0e9, phase)
    axes[1].plot(time_grid_s * 1.0e9, chirp / (2.0 * np.pi * 1.0e6))
    axes[0].set_ylabel("phi(t) (rad)")
    axes[1].set_ylabel("dphi/dt / 2pi (MHz)")
    axes[1].set_xlabel("Time (ns)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = study_output_dir("01_kerr_replay_and_chirp")
    spec, constraints = nonlinear_spec(include_kerr_phase_correction=False)
    corrected_spec, _ = nonlinear_spec(include_kerr_phase_correction=True)

    kerr_values = np.linspace(0.0, 0.10, 6)
    uncorrected_residuals: list[float] = []
    corrected_residuals: list[float] = []
    for kerr_mhz in kerr_values:
        sweep_spec = ReadoutEmptyingSpec(
            kappa=spec.kappa,
            chi=spec.chi,
            tau=spec.tau,
            n_segments=spec.n_segments,
            kerr=2.0 * np.pi * float(kerr_mhz) * 1.0e6,
            include_kerr_phase_correction=False,
        )
        sweep_corrected = ReadoutEmptyingSpec(
            kappa=spec.kappa,
            chi=spec.chi,
            tau=spec.tau,
            n_segments=spec.n_segments,
            kerr=2.0 * np.pi * float(kerr_mhz) * 1.0e6,
            include_kerr_phase_correction=True,
        )
        uncorrected_residuals.append(
            float(synthesize_readout_emptying_pulse(sweep_spec, constraints).metrics["max_final_residual_photons"])
        )
        corrected_residuals.append(
            float(synthesize_readout_emptying_pulse(sweep_corrected, constraints).metrics["max_final_residual_photons"])
        )

    shared = synthesize_readout_emptying_pulse(corrected_spec, constraints)
    g_branch = synthesize_readout_emptying_pulse(
        ReadoutEmptyingSpec(
            kappa=corrected_spec.kappa,
            chi=corrected_spec.chi,
            tau=corrected_spec.tau,
            n_segments=corrected_spec.n_segments,
            kerr=corrected_spec.kerr,
            include_kerr_phase_correction=True,
            kerr_correction_strategy="g_branch",
        ),
        constraints,
    )
    e_branch = synthesize_readout_emptying_pulse(
        ReadoutEmptyingSpec(
            kappa=corrected_spec.kappa,
            chi=corrected_spec.chi,
            tau=corrected_spec.tau,
            n_segments=corrected_spec.n_segments,
            kerr=corrected_spec.kerr,
            include_kerr_phase_correction=True,
            kerr_correction_strategy="e_branch",
        ),
        constraints,
    )
    analytic_seed = synthesize_readout_emptying_pulse(spec, constraints)

    chirp = shared.diagnostics["kerr_correction"]
    chirp["time_grid_s"] = shared.time_grid_s

    _residual_vs_kerr(
        output_dir / "residual_vs_kerr.png",
        kerr_values,
        uncorrected_residuals,
        corrected_residuals,
    )
    _shared_vs_branch_specific(
        output_dir / "shared_vs_branch_specific.png",
        {
            "analytic_seed": analytic_seed.metrics["max_final_residual_photons"],
            "average_branch": shared.metrics["max_final_residual_photons"],
            "g_branch": g_branch.metrics["max_final_residual_photons"],
            "e_branch": e_branch.metrics["max_final_residual_photons"],
        },
    )
    _chirp_plot(output_dir / "chirp_profile.png", chirp)

    payload = {
        "kerr_mhz": [float(value) for value in kerr_values],
        "uncorrected_residuals": uncorrected_residuals,
        "corrected_residuals": corrected_residuals,
        "strategy_residuals": {
            "analytic_seed": analytic_seed.metrics["max_final_residual_photons"],
            "average_branch": shared.metrics["max_final_residual_photons"],
            "g_branch": g_branch.metrics["max_final_residual_photons"],
            "e_branch": e_branch.metrics["max_final_residual_photons"],
        },
        "artifacts": {
            "residual_vs_kerr": str(output_dir / "residual_vs_kerr.png"),
            "shared_vs_branch_specific": str(output_dir / "shared_vs_branch_specific.png"),
            "chirp_profile": str(output_dir / "chirp_profile.png"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Kerr replay artifacts to {output_dir}")


if __name__ == "__main__":
    main()
