from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.measurement import ReadoutChain, ReadoutResonator
from cqed_sim.optimal_control import (
    ReadoutEmptyingConstraints,
    ReadoutEmptyingSpec,
    evaluate_readout_emptying_with_chain,
    replay_kerr_readout_branches,
    replay_linear_readout_branches,
    synthesize_readout_emptying_pulse,
)


# Implements the segmented resonator-reset construction following the cavity-reset
# idea of McClure et al. (2016), with Kerr-aware correction guided by the analytic
# replay strategy used in Jerger et al. (2024).
# DOI: 10.1103/PhysRevApplied.5.011001
# DOI: 10.48550/arXiv.2406.04891


def _build_chain(spec: ReadoutEmptyingSpec) -> ReadoutChain:
    return ReadoutChain(
        ReadoutResonator(
            omega_r=2.0 * np.pi * 7.0e9,
            kappa=spec.kappa,
            g=2.0 * np.pi * 80.0e6,
            epsilon=1.0,
            chi=spec.chi,
        ),
        integration_time=spec.tau,
        dt=2.0e-9,
    )


def _square_segments(amplitude: float, *, n_segments: int) -> np.ndarray:
    return np.full(int(n_segments), float(amplitude), dtype=np.complex128)


def _time_separation(replay) -> np.ndarray:
    return np.abs(np.asarray(replay.trajectories["e"]) - np.asarray(replay.trajectories["g"]))


def _plot_waveform(path: Path, linear_result, corrected_result) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    for axis, key, label in zip(axes, (np.real, np.imag), ("Re", "Im")):
        axis.step(
            linear_result.segment_edges_s[:-1] * 1.0e9,
            key(linear_result.segment_amplitudes) / (2.0 * np.pi * 1.0e6),
            where="post",
            label="Linear",
        )
        axis.step(
            corrected_result.segment_edges_s[:-1] * 1.0e9,
            key(corrected_result.segment_amplitudes) / (2.0 * np.pi * 1.0e6),
            where="post",
            label="Kerr corrected",
        )
        axis.set_ylabel(f"{label}[eps] / 2pi (MHz)")
        axis.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (ns)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_phase_space(path: Path, corrected_replay, square_replay) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False, sharey=False)
    for axis, replay, title in zip(axes, (corrected_replay, square_replay), ("Readout emptying", "Square pulse")):
        for label, color in (("g", "tab:blue"), ("e", "tab:orange")):
            trajectory = np.asarray(replay.trajectories[label], dtype=np.complex128)
            axis.plot(trajectory.real, trajectory.imag, color=color, label=label)
            axis.scatter([trajectory[-1].real], [trajectory[-1].imag], color=color, s=18)
        axis.scatter([0.0], [0.0], color="black", s=20, marker="x")
        axis.set_title(title)
        axis.set_xlabel("Re(alpha)")
        axis.set_ylabel("Im(alpha)")
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_photons(path: Path, corrected_replay, square_replay) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, replay, title in zip(axes, (corrected_replay, square_replay), ("Readout emptying", "Square pulse")):
        for label, color in (("g", "tab:blue"), ("e", "tab:orange")):
            axis.plot(
                replay.time_grid_s * 1.0e9,
                replay.photon_numbers[label],
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Time (ns)")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("|alpha|^2")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_residuals(path: Path, linear_result, corrected_result, square_kerr_replay, square_linear_replay) -> None:
    labels = ["Linear replay", "Linear on Kerr", "Corrected on Kerr", "Square on Kerr"]
    residuals = [
        float(linear_result.diagnostics["linear_metrics"]["max_final_residual_photons"]),
        float(linear_result.metrics["max_final_residual_photons"]),
        float(corrected_result.metrics["max_final_residual_photons"]),
        float(max(square_kerr_replay.final_n.values())),
    ]
    baseline = float(max(square_linear_replay.final_n.values()))

    fig, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, residuals, color=["tab:green", "tab:blue", "tab:orange", "tab:red"])
    axis.axhline(baseline, color="black", linestyle="--", linewidth=1.0, label="Square on linear model")
    axis.set_ylabel("Final max residual photons")
    axis.set_yscale("log")
    axis.tick_params(axis="x", rotation=18)
    axis.legend()
    axis.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_separation(path: Path, corrected_replay, square_replay) -> None:
    fig, axis = plt.subplots(figsize=(8, 4))
    axis.plot(corrected_replay.time_grid_s * 1.0e9, _time_separation(corrected_replay), label="Readout emptying")
    axis.plot(square_replay.time_grid_s * 1.0e9, _time_separation(square_replay), label="Square pulse")
    axis.set_xlabel("Time (ns)")
    axis.set_ylabel("|alpha_e - alpha_g|")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_iq_clusters(path: Path, corrected_eval) -> None:
    fig, axis = plt.subplots(figsize=(5, 5))
    for label, color in (("g", "tab:blue"), ("e", "tab:orange")):
        samples = np.asarray(corrected_eval["sampled_iq"][label], dtype=float)
        center = np.asarray(corrected_eval["iq_centers"][label], dtype=float)
        axis.scatter(samples[:, 0], samples[:, 1], s=12, alpha=0.5, color=color, label=f"{label} samples")
        axis.scatter([center[0]], [center[1]], s=45, color=color, edgecolors="black")
    axis.set_xlabel("I")
    axis.set_ylabel("Q")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = Path("outputs") / "readout_emptying_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    linear_spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=320e-9,
        n_segments=4,
        kerr=2.0 * np.pi * 0.08e6,
        include_kerr_phase_correction=False,
    )
    corrected_spec = ReadoutEmptyingSpec(
        kappa=linear_spec.kappa,
        chi=linear_spec.chi,
        tau=linear_spec.tau,
        n_segments=linear_spec.n_segments,
        kerr=linear_spec.kerr,
        include_kerr_phase_correction=True,
    )

    linear_result = synthesize_readout_emptying_pulse(linear_spec, constraints)
    corrected_result = synthesize_readout_emptying_pulse(corrected_spec, constraints)

    square_amplitude = float(corrected_result.metrics["waveform_l2_norm"] / np.sqrt(corrected_spec.tau))
    square_segments = _square_segments(square_amplitude, n_segments=corrected_spec.n_segments)
    square_linear_replay = replay_linear_readout_branches(corrected_spec, square_segments)
    square_kerr_replay = replay_kerr_readout_branches(corrected_spec, square_segments)
    corrected_kerr_replay = replay_kerr_readout_branches(corrected_spec, corrected_result.segment_amplitudes)

    chain = _build_chain(corrected_spec)
    linear_eval = evaluate_readout_emptying_with_chain(linear_result, chain, shots_per_branch=128, seed=5)
    corrected_eval = evaluate_readout_emptying_with_chain(corrected_result, chain, shots_per_branch=128, seed=7)

    _plot_waveform(output_dir / "waveform.png", linear_result, corrected_result)
    _plot_phase_space(output_dir / "phase_space.png", corrected_kerr_replay, square_kerr_replay)
    _plot_photons(output_dir / "photons.png", corrected_kerr_replay, square_kerr_replay)
    _plot_residuals(output_dir / "residuals.png", linear_result, corrected_result, square_kerr_replay, square_linear_replay)
    _plot_separation(output_dir / "separation.png", corrected_kerr_replay, square_kerr_replay)
    _plot_iq_clusters(output_dir / "iq_clusters.png", corrected_eval)

    payload = {
        "spec": {
            "kappa": corrected_spec.kappa,
            "chi": corrected_spec.chi,
            "kerr": corrected_spec.kerr,
            "tau": corrected_spec.tau,
            "n_segments": corrected_spec.n_segments,
        },
        "constraints": {
            "amplitude_max": constraints.amplitude_max,
        },
        "linear_metrics": linear_result.metrics,
        "corrected_metrics": corrected_result.metrics,
        "square_metrics": {
            "linear_residual": float(max(square_linear_replay.final_n.values())),
            "kerr_residual": float(max(square_kerr_replay.final_n.values())),
            "integrated_branch_separation": float(np.trapezoid(_time_separation(square_kerr_replay) ** 2, x=square_kerr_replay.time_grid_s)),
        },
        "measurement": {
            "linear": linear_eval["metrics"],
            "corrected": corrected_eval["metrics"],
        },
        "artifacts": {
            "waveform": str(output_dir / "waveform.png"),
            "phase_space": str(output_dir / "phase_space.png"),
            "photons": str(output_dir / "photons.png"),
            "residuals": str(output_dir / "residuals.png"),
            "separation": str(output_dir / "separation.png"),
            "iq_clusters": str(output_dir / "iq_clusters.png"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Readout emptying demo")
    print(f"  output: {output_dir}")
    print(f"  corrected residual photons: {corrected_result.metrics['max_final_residual_photons']:.6e}")
    print(f"  square residual photons: {max(square_kerr_replay.final_n.values()):.6e}")
    print(f"  corrected IQ accuracy: {corrected_eval['metrics']['measurement_chain_accuracy']:.3f}")


if __name__ == "__main__":
    main()
