from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cqed_sim.observables.fock import relative_phase_family_diagnostics
from cqed_sim.plotting.bloch_plots import add_gate_type_axis
from cqed_sim.plotting.gate_diagnostics import FOCK_LEVEL_COLORS


PHASE_FAMILY_STYLES = {
    "ground": {"label": r"$|g,0\rangle \rightarrow |g,n\rangle$", "marker": "o", "markersize": 4.0, "linestyle": "-"},
    "excited": {"label": r"$|g,0\rangle \rightarrow |e,n\rangle$", "marker": "s", "markersize": 4.0, "linestyle": "--"},
}


def _fock_color(level: int) -> str:
    return FOCK_LEVEL_COLORS[int(level) % len(FOCK_LEVEL_COLORS)]


def plot_relative_phase_track(
    track: dict[str, Any],
    max_n: int,
    threshold: float,
    unwrap: bool = False,
    label_stride: int = 1,
):
    diagnostics = relative_phase_family_diagnostics(
        track,
        max_n=max_n,
        probability_threshold=threshold,
        unwrap=unwrap,
        coherence_threshold=threshold,
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    indices = track["indices"]
    n_values = diagnostics["n_values"]
    for family in ("ground", "excited"):
        family_diag = diagnostics["families"][family]
        style = PHASE_FAMILY_STYLES[family]
        for row_idx, n in enumerate(n_values):
            color = _fock_color(int(n))
            ax.plot(
                indices,
                family_diag["phase"][row_idx],
                color=color,
                linewidth=2.1,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=style["markersize"],
                markerfacecolor=color,
                markeredgecolor=color,
                label=f"{family}: n={int(n)}",
            )
    ax.set_xlabel("Gate index")
    ax.set_ylabel(r"Relative phase $\phi_{g0 \rightarrow q,n}$ [rad]")
    ax.set_title(
        f"{track['case']}: relative phase between |g,0> and |g,n> / |e,n> vs gate index "
        f"({diagnostics['phase_mode']}, n_max={int(diagnostics['n_values'][-1])})"
    )
    ax.set_xticks(track["indices"])
    ax.grid(alpha=0.25)
    family_handles = [
        Line2D(
            [0],
            [0],
            color="0.25",
            linestyle=PHASE_FAMILY_STYLES[family]["linestyle"],
            linewidth=2.0,
            marker=PHASE_FAMILY_STYLES[family]["marker"],
            markersize=PHASE_FAMILY_STYLES[family]["markersize"],
            label=PHASE_FAMILY_STYLES[family]["label"],
        )
        for family in ("ground", "excited")
    ]
    n_handles = [
        Line2D([0], [0], color=_fock_color(int(n)), linewidth=3.0, label=f"n={int(n)}")
        for n in n_values
    ]
    ax.legend(handles=family_handles + n_handles, loc="best", ncol=min(3, max(1, len(n_values))))
    add_gate_type_axis(ax, track, label_stride=label_stride, xlabel="Gate type per iteration")
    fig.tight_layout()
    return fig
