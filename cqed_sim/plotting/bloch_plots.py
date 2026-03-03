from __future__ import annotations

from typing import Any

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt


GATE_COLORS = {
    "INIT": "black",
    "Displacement": "tab:blue",
    "Rotation": "tab:orange",
    "SQR": "tab:green",
}


def add_gate_type_axis(ax, track: dict[str, Any], label_stride: int, xlabel: str = "Gate type per iteration") -> None:
    top = ax.twiny()
    top.set_xlim(ax.get_xlim())
    ticks = []
    labels = []
    for snapshot in track["snapshots"]:
        if snapshot["index"] == 0 or snapshot["index"] % max(1, label_stride) == 0:
            ticks.append(snapshot["index"])
            labels.append(snapshot["top_label"])
    top.set_xticks(ticks)
    top.set_xticklabels(labels, rotation=45, ha="left", fontsize=8)
    top.set_xlabel(xlabel)


def plot_bloch_track(track: dict[str, Any], title: str, label_stride: int):
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.plot(track["indices"], track["x"], "o-", label=r"$\langle \sigma_x \rangle$")
    ax.plot(track["indices"], track["y"], "o-", label=r"$\langle \sigma_y \rangle$")
    ax.plot(track["indices"], track["z"], "o-", label=r"$\langle \sigma_z \rangle$")
    for snapshot in track["snapshots"][1:]:
        ax.axvline(
            snapshot["index"],
            color=GATE_COLORS.get(snapshot["gate_type"], "black"),
            alpha=0.08,
            linewidth=1.0,
        )
    gate_handles = [
        Line2D([0], [0], color=GATE_COLORS[key], linewidth=3.0, label=key)
        for key in ("Displacement", "Rotation", "SQR")
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + gate_handles, labels + [handle.get_label() for handle in gate_handles], loc="lower right", ncol=2)
    ax.set_title(title)
    ax.set_xlabel("Iteration index")
    ax.set_ylabel("Bloch component")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks(track["indices"])
    ax.grid(alpha=0.25)
    add_gate_type_axis(ax, track, label_stride=label_stride)
    fig.tight_layout()
    return fig
