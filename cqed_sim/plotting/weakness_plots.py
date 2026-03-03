from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from cqed_sim.plotting.bloch_plots import add_gate_type_axis


def print_mapping_rows(track: dict[str, Any]) -> None:
    for row in track["metadata"].get("mapping_rows", []):
        print(f"k={row['index']:>2} {row['type']:<12} {row['mapping']}")


def plot_component_comparison(
    case_a: dict[str, Any],
    case_b: dict[str, Any],
    case_c: dict[str, Any],
    case_d: dict[str, Any] | None = None,
    label_stride: int = 1,
):
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 8.4), sharex=True)
    for axis, key, label in zip(axes, ("x", "y", "z"), ("X", "Y", "Z")):
        axis.plot(case_a["indices"], case_a[key], "o-", label="Case A")
        axis.plot(case_b["indices"], case_b[key], "s--", label="Case B")
        axis.plot(case_c["indices"], case_c[key], "d-.", label="Case C")
        if case_d is not None:
            axis.plot(case_d["indices"], case_d[key], "^-", label="Case D")
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Iteration index")
    add_gate_type_axis(axes[0], case_a, label_stride=label_stride, xlabel="Gate type per iteration")
    fig.suptitle("Bloch component comparison across A/B/C" + ("/D" if case_d is not None else ""))
    fig.tight_layout()
    return fig


def plot_cavity_population_comparison(
    case_a: dict[str, Any],
    case_b: dict[str, Any],
    case_c: dict[str, Any],
    case_d: dict[str, Any] | None = None,
    label_stride: int = 1,
):
    fig, ax = plt.subplots(figsize=(10.8, 4.0))
    ax.plot(case_a["indices"], case_a["n"], "o-", label="Case A")
    ax.plot(case_b["indices"], case_b["n"], "s--", label="Case B")
    ax.plot(case_c["indices"], case_c["n"], "d-.", label="Case C")
    if case_d is not None:
        ax.plot(case_d["indices"], case_d["n"], "^-", label="Case D")
    ax.set_xlabel("Iteration index")
    ax.set_ylabel(r"$\langle n \rangle$")
    ax.set_title("Cavity population comparison")
    ax.set_xticks(case_a["indices"])
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    add_gate_type_axis(ax, case_a, label_stride=label_stride, xlabel="Gate type per iteration")
    fig.tight_layout()
    return fig


def plot_weakness(
    case_b: dict[str, Any],
    case_c: dict[str, Any],
    reference_track: dict[str, Any],
    case_d: dict[str, Any] | None = None,
    label_stride: int = 1,
):
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.8), sharex=True)
    axes[0].plot(case_b["indices"], case_b["wigner_negativity"], "s--", label="Case B")
    axes[0].plot(case_c["indices"], case_c["wigner_negativity"], "d-.", label="Case C")
    if case_d is not None:
        axes[0].plot(case_d["indices"], case_d["wigner_negativity"], "^-", label="Case D")
    axes[0].set_ylabel("Wigner negativity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(case_b["indices"], case_b["fidelity_weakness_vs_a"], "s--", label="Case B")
    axes[1].plot(case_c["indices"], case_c["fidelity_weakness_vs_a"], "d-.", label="Case C")
    if case_d is not None:
        axes[1].plot(case_d["indices"], case_d["fidelity_weakness_vs_a"], "^-", label="Case D")
    axes[1].set_ylabel("1 - fidelity vs A")
    axes[1].set_xlabel("Iteration index")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")
    add_gate_type_axis(axes[0], reference_track, label_stride=label_stride, xlabel="Gate type per iteration")
    fig.suptitle("Weakness metrics")
    fig.tight_layout()
    return fig
