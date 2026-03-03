from __future__ import annotations

from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from cqed_sim.observables.wigner import selected_wigner_snapshots


def _gate_panel_title(snapshot: dict[str, Any]) -> str:
    if snapshot["index"] == 0:
        return "k=0 (INIT)"
    return f"k={snapshot['index']} ({snapshot['gate_type']})"


def plot_wigner_grid(track: dict[str, Any], title: str, stride: int, max_cols: int | None = None):
    panels = selected_wigner_snapshots(track, stride=stride)
    if not panels:
        print(f"No Wigner panels stored for {track['case']}.")
        return None
    requested_cols = 5 if max_cols is None else int(max_cols)
    n_cols = min(max(1, requested_cols), len(panels))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.1 * n_cols, 3.5 * n_rows), squeeze=False)
    all_w = np.concatenate([panel["wigner"]["w"].ravel() for panel in panels])
    vmax = float(np.max(np.abs(all_w)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax) if vmax > 0 else None
    image = None
    for flat_idx, (axis, panel) in enumerate(zip(axes.ravel(), panels)):
        xvec = panel["wigner"]["xvec"]
        yvec = panel["wigner"]["yvec"]
        image = axis.imshow(
            panel["wigner"]["w"],
            origin="lower",
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap="RdBu_r",
            norm=norm,
            aspect="equal",
        )
        axis.set_title(_gate_panel_title(panel), fontsize=9)
        row = flat_idx // n_cols
        col = flat_idx % n_cols
        show_bottom = row == (n_rows - 1)
        show_left = col == 0
        axis.tick_params(axis="x", which="both", labelbottom=show_bottom, bottom=True)
        axis.tick_params(axis="y", which="both", labelleft=show_left, left=True)
        if not show_bottom:
            axis.set_xticklabels([])
        if not show_left:
            axis.set_yticklabels([])
    for axis in axes.ravel()[len(panels):]:
        axis.axis("off")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84, label="W(x, p)")
    fig.suptitle(f"{title} (axes: x, p)", y=1.02)
    fig.tight_layout()
    return fig
