from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from cqed_sim.core import EnergySpectrum

_TRANSMON_LEVEL_LABELS = ("g", "e", "f", "h", "i", "j", "k", "l", "m", "n")
_LEVEL_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")


def _transmon_label(level: int) -> str:
    if 0 <= int(level) < len(_TRANSMON_LEVEL_LABELS):
        return _TRANSMON_LEVEL_LABELS[int(level)]
    return f"t{int(level)}"


def _level_color(dominant_basis_levels: tuple[int, ...], *, has_transmon: bool) -> str:
    if not has_transmon or not dominant_basis_levels:
        return "tab:blue"
    return _LEVEL_COLORS[int(dominant_basis_levels[0]) % len(_LEVEL_COLORS)]


def plot_energy_levels(
    spectrum: EnergySpectrum,
    *,
    max_levels: int | None = None,
    energy_scale: float = 1.0,
    energy_unit_label: str = "rad/s",
    annotate: bool = True,
    title: str | None = None,
    ax=None,
):
    if max_levels is None:
        selected = spectrum.levels
    else:
        selected = spectrum.levels[: max(0, int(max_levels))]
    if not selected:
        raise ValueError("plot_energy_levels requires at least one level.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
    else:
        fig = ax.figure

    has_transmon = bool(spectrum.subsystem_labels) and spectrum.subsystem_labels[0] in {"qubit", "transmon"}
    x_positions = list(range(len(selected)))
    used_transmon_levels: list[int] = []

    for x_pos, level in zip(x_positions, selected):
        y_value = float(level.energy) * float(energy_scale)
        color = _level_color(level.dominant_basis_levels, has_transmon=has_transmon)
        ax.hlines(y_value, x_pos - 0.35, x_pos + 0.35, color=color, linewidth=2.0)
        ax.plot([x_pos], [y_value], marker="o", color=color, markersize=4.0)
        if has_transmon and level.dominant_basis_levels:
            transmon_level = int(level.dominant_basis_levels[0])
            if transmon_level not in used_transmon_levels:
                used_transmon_levels.append(transmon_level)
        if annotate:
            ax.text(x_pos + 0.42, y_value, level.dominant_basis_label, va="center", ha="left", fontsize=9)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(level.index) for level in selected])
    ax.set_xlim(-0.6, len(selected) - 0.4 + (1.6 if annotate else 0.2))
    ax.set_xlabel("Dressed level index")
    ax.set_ylabel(f"Energy relative to vacuum [{energy_unit_label}]")
    ax.set_title("Energy levels" if title is None else str(title))
    ax.grid(True, axis="y", alpha=0.25)

    if has_transmon and used_transmon_levels:
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=_LEVEL_COLORS[level % len(_LEVEL_COLORS)],
                lw=2.0,
                label=fr"dominant $|{_transmon_label(level)}\rangle$ manifold",
            )
            for level in used_transmon_levels
        ]
        ax.legend(handles=legend_handles, loc="best")

    fig.tight_layout()
    return fig


__all__ = ["plot_energy_levels"]
