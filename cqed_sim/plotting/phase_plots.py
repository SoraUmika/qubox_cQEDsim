from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt

from cqed_sim.observables.phases import relative_phase_diagnostics
from cqed_sim.plotting.bloch_plots import add_gate_type_axis


def plot_relative_phase_track(
    track: dict[str, Any],
    max_n: int,
    threshold: float,
    unwrap: bool = False,
    label_stride: int = 1,
):
    diagnostics = relative_phase_diagnostics(track, max_n=max_n, threshold=threshold, unwrap=unwrap)
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    for label in diagnostics["labels"]:
        linestyle = "-" if label == "|g0|" else "--" if label.startswith("|g") else "-."
        linewidth = 2.2 if label == "|g0|" else 1.8
        ax.plot(track["indices"], diagnostics["traces"][label], linestyle=linestyle, linewidth=linewidth, label=label)
    ax.set_xlabel("Gate index")
    ax.set_ylabel("Relative phase [rad]")
    ax.set_title(
        f"{track['case']}: relative phase vs gate index for |g0|, |g1|, |g2|, |e0|, |e1|, |e2| "
        f"({diagnostics['phase_mode']})"
    )
    ax.set_xticks(track["indices"])
    ax.grid(alpha=0.25)
    ax.legend(loc="best", ncol=3)
    add_gate_type_axis(ax, track, label_stride=label_stride, xlabel="Gate type per iteration")
    fig.tight_layout()
    return fig
