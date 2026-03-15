from __future__ import annotations

import matplotlib.pyplot as plt


def configure_notebook_style() -> None:
    """Apply a lightweight, consistent plotting style for workflow tutorials."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["figure.figsize"] = (7.2, 4.4)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["legend.frameon"] = False
    plt.rcParams["savefig.dpi"] = 140


__all__ = ["configure_notebook_style"]
