from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.optimal_control import (
    build_emptying_constraint_matrix,
    replay_linear_readout_branches,
    synthesize_readout_emptying_pulse,
)

from common import base_constraints, linear_spec, study_output_dir


def _plot_waveform(path, result) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].step(
        result.segment_edges_s[:-1] * 1.0e9,
        result.segment_amplitudes.real / (2.0 * np.pi * 1.0e6),
        where="post",
        color="tab:blue",
    )
    axes[1].step(
        result.segment_edges_s[:-1] * 1.0e9,
        result.segment_amplitudes.imag / (2.0 * np.pi * 1.0e6),
        where="post",
        color="tab:orange",
    )
    axes[0].set_ylabel("Re[eps]/2pi (MHz)")
    axes[1].set_ylabel("Im[eps]/2pi (MHz)")
    axes[1].set_xlabel("Time (ns)")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_phase_space(path, replay) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    for label, color in (("g", "tab:blue"), ("e", "tab:orange")):
        trajectory = np.asarray(replay.trajectories[label], dtype=np.complex128)
        axis.plot(trajectory.real, trajectory.imag, color=color, label=label)
        axis.scatter([trajectory[-1].real], [trajectory[-1].imag], color=color, s=26)
    axis.scatter([0.0], [0.0], marker="x", color="black", s=30)
    axis.set_xlabel("Re(alpha)")
    axis.set_ylabel("Im(alpha)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_terminal_zoom(path, replay) -> None:
    fig, axis = plt.subplots(figsize=(5.5, 5.0))
    for label, color in (("g", "tab:blue"), ("e", "tab:orange")):
        trajectory = np.asarray(replay.trajectories[label], dtype=np.complex128)
        axis.plot(trajectory.real, trajectory.imag, color=color, alpha=0.5)
        axis.scatter([trajectory[-1].real], [trajectory[-1].imag], color=color, s=35, label=f"{label} final")
    axis.set_xlim(-2.5e-6, 2.5e-6)
    axis.set_ylim(-2.5e-6, 2.5e-6)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.axvline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Re(alpha)")
    axis.set_ylabel("Im(alpha)")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_matrix_vs_ode(path, spec, result) -> None:
    amplitudes = np.asarray(result.diagnostics["linear_segment_amplitudes"], dtype=np.complex128)
    matrix_prediction = build_emptying_constraint_matrix(spec) @ amplitudes
    replay = replay_linear_readout_branches(spec, amplitudes)
    labels = list(replay.final_alpha)
    predicted = np.array([abs(matrix_prediction[index]) for index in range(len(labels))], dtype=float)
    integrated = np.array([abs(replay.final_alpha[label]) for label in labels], dtype=float)
    x = np.arange(len(labels), dtype=float)

    fig, axis = plt.subplots(figsize=(6.0, 4.0))
    axis.bar(x - 0.18, predicted + 1.0e-18, width=0.35, label="Matrix")
    axis.bar(x + 0.18, integrated + 1.0e-18, width=0.35, label="ODE")
    axis.set_xticks(x, labels)
    axis.set_ylabel("|alpha(tau)|")
    axis.set_yscale("log")
    axis.grid(alpha=0.25, axis="y")
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    output_dir = study_output_dir("00_linear_seed_validation")
    spec = linear_spec()
    result = synthesize_readout_emptying_pulse(spec, base_constraints())
    replay = replay_linear_readout_branches(spec, np.asarray(result.diagnostics["linear_segment_amplitudes"], dtype=np.complex128))

    _plot_waveform(output_dir / "segment_waveform.png", result)
    _plot_phase_space(output_dir / "phase_space.png", replay)
    _plot_terminal_zoom(output_dir / "terminal_zoom.png", replay)
    _plot_matrix_vs_ode(output_dir / "matrix_vs_ode.png", spec, result)

    payload = {
        "metrics": result.metrics,
        "linear_metrics": result.diagnostics["linear_metrics"],
        "artifacts": {
            "segment_waveform": str(output_dir / "segment_waveform.png"),
            "phase_space": str(output_dir / "phase_space.png"),
            "terminal_zoom": str(output_dir / "terminal_zoom.png"),
            "matrix_vs_ode": str(output_dir / "matrix_vs_ode.png"),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved linear seed validation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
