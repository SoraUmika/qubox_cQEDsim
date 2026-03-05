from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cqed_sim.observables.fock import wrapped_phase_error
from cqed_sim.plotting.bloch_plots import add_gate_type_axis


COMPONENT_LABELS = {
    "x": r"$\langle \sigma_x \rangle_n$",
    "y": r"$\langle \sigma_y \rangle_n$",
    "z": r"$\langle \sigma_z \rangle_n$",
}
COMPONENT_COLORS = {"x": "tab:blue", "y": "tab:orange", "z": "tab:green"}
FOCK_LEVEL_COLORS = ("tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown")
IDEAL_CONTOUR_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PHASE_CONTOUR_LEVELS = (-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi)
PHASE_FAMILY_STYLES = {
    "ground": {"label": r"$|g,0\rangle \rightarrow |g,n\rangle$", "marker": "o", "linestyle": "-"},
    "excited": {"label": r"$|g,0\rangle \rightarrow |e,n\rangle$", "marker": "s", "linestyle": "--"},
}


def _masked(matrix: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_invalid(np.asarray(matrix, dtype=float))


def _bottom_ticks(indices: np.ndarray, label_stride: int) -> list[int]:
    stride = max(1, int(label_stride))
    ticks = list(np.asarray(indices, dtype=int)[::stride])
    if ticks[-1] != int(indices[-1]):
        ticks.append(int(indices[-1]))
    return ticks


def _matrix_extent(indices: np.ndarray, n_values: np.ndarray) -> list[float]:
    return [
        float(indices[0]) - 0.5,
        float(indices[-1]) + 0.5,
        float(n_values[0]) - 0.5,
        float(n_values[-1]) + 0.5,
    ]


def _style_gate_axis(ax, track: dict[str, Any], indices: np.ndarray, label_stride: int, show_top_axis: bool) -> None:
    ax.set_xlim(float(indices[0]) - 0.5, float(indices[-1]) + 0.5)
    ax.set_xticks(_bottom_ticks(indices, label_stride))
    ax.set_xlabel("Gate index")
    if show_top_axis:
        add_gate_type_axis(ax, track, label_stride=label_stride, xlabel="Gate type per iteration")


def _mesh(indices: np.ndarray, n_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.asarray(indices, dtype=float), np.asarray(n_values, dtype=float))


def _validate_alignment(simulated: dict[str, Any], ideal: dict[str, Any]) -> None:
    if not np.array_equal(simulated["indices"], ideal["indices"]):
        raise ValueError("Simulated and ideal diagnostics must share the same gate indices.")
    if not np.array_equal(simulated["n_values"], ideal["n_values"]):
        raise ValueError("Simulated and ideal diagnostics must share the same n_values.")


def _as_phase_bundle(diagnostics: dict[str, Any]) -> dict[str, Any]:
    if "families" in diagnostics:
        return diagnostics
    return {
        "case": diagnostics["case"],
        "indices": np.asarray(diagnostics["indices"], dtype=int),
        "n_values": np.asarray(diagnostics["n_values"], dtype=int),
        "top_labels": list(diagnostics.get("top_labels", [])),
        "gate_types": list(diagnostics.get("gate_types", [])),
        "phase_mode": diagnostics["phase_mode"],
        "probability_threshold": diagnostics["probability_threshold"],
        "coherence_threshold": diagnostics["coherence_threshold"],
        "phase_reference_label": diagnostics["phase_reference_label"],
        "families": {"excited": diagnostics},
        "relative_phase_definitions": {"excited": diagnostics["relative_phase_definition"]},
    }


def _validate_phase_alignment(simulated: dict[str, Any], ideal: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], tuple[str, ...]]:
    sim_bundle = _as_phase_bundle(simulated)
    ideal_bundle = _as_phase_bundle(ideal)
    _validate_alignment(sim_bundle, ideal_bundle)
    families = tuple(family for family in ("ground", "excited") if family in sim_bundle["families"] and family in ideal_bundle["families"])
    if not families:
        raise ValueError("No common phase families found between simulated and ideal diagnostics.")
    return sim_bundle, ideal_bundle, families


def _sim_vs_ideal_handles(color: str | None = None) -> list:
    contour_color = "black" if color is None else color
    return [
        Patch(facecolor="#cccccc", edgecolor="none", alpha=0.8, label="Simulated (filled heatmap)"),
        Line2D([0], [0], color=contour_color, linestyle="--", linewidth=1.5, label="Ideal (dashed contour)"),
    ]


def _fock_color(level: int) -> str:
    return FOCK_LEVEL_COLORS[int(level) % len(FOCK_LEVEL_COLORS)]


def _style_legend_handles(n_values: np.ndarray) -> list:
    handles = [
        Patch(facecolor="0.65", edgecolor="0.45", alpha=0.75, label="Simulated (filled)"),
        Patch(facecolor="none", edgecolor="0.15", hatch="//", linestyle="--", label="Ideal (hollow)"),
    ]
    handles.extend(Patch(facecolor=_fock_color(int(n)), edgecolor=_fock_color(int(n)), alpha=0.75, label=f"n={int(n)}") for n in n_values)
    return handles


def _phase_line_handles(n_values: np.ndarray, families: tuple[str, ...] = ("ground", "excited")) -> list:
    handles = [
        Line2D([0], [0], color="0.2", linestyle="-", linewidth=2.0, marker="o", markerfacecolor="0.2", markeredgecolor="0.2", label="Simulated"),
        Line2D([0], [0], color="0.2", linestyle="-", linewidth=2.0, marker="o", markerfacecolor="white", markeredgecolor="0.2", label="Ideal"),
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color="0.2",
            linestyle=PHASE_FAMILY_STYLES[family]["linestyle"],
            linewidth=2.0,
            marker=PHASE_FAMILY_STYLES[family]["marker"],
            markersize=5.0,
            label=PHASE_FAMILY_STYLES[family]["label"],
        )
        for family in families
    )
    handles.extend(Line2D([0], [0], color=_fock_color(int(n)), linewidth=3.0, label=f"n={int(n)}") for n in n_values)
    return handles


def _grouped_bar_geometry(n_levels: int) -> tuple[np.ndarray, float, float]:
    if n_levels <= 0:
        return np.asarray([], dtype=float), 0.2, 0.05
    group_span = min(0.8, 0.18 * n_levels + 0.18)
    offsets = np.linspace(-group_span / 2.0, group_span / 2.0, n_levels) if n_levels > 1 else np.asarray([0.0], dtype=float)
    pair_offset = min(0.06, group_span / max(6.0, 3.0 * n_levels))
    width = min(0.11, max(0.05, pair_offset * 1.7))
    return offsets, width, pair_offset


def _plottable_mask(valid: np.ndarray, heights: np.ndarray) -> np.ndarray:
    return np.asarray(valid, dtype=bool) & np.isfinite(np.asarray(heights, dtype=float))


def plot_fock_resolved_bloch_overlay(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    component: str,
    label_stride: int = 1,
    contour_levels: tuple[float, ...] = IDEAL_CONTOUR_LEVELS,
):
    return plot_fock_resolved_bloch_grouped_bars(
        simulated=simulated,
        ideal=ideal,
        track=track,
        component=component,
        label_stride=label_stride,
    )


def plot_fock_resolved_bloch_grouped_bars(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    component: str,
    label_stride: int = 1,
):
    _validate_alignment(simulated, ideal)
    indices = np.asarray(simulated["indices"], dtype=int)
    n_values = np.asarray(simulated["n_values"], dtype=int)
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    offsets, width, pair_offset = _grouped_bar_geometry(n_values.size)
    sim_valid = np.asarray(simulated.get("valid", np.isfinite(simulated[component])), dtype=bool)
    ideal_valid = np.asarray(ideal.get("valid", np.isfinite(ideal[component])), dtype=bool)
    sim_values = np.asarray(simulated[component], dtype=float)
    ideal_values = np.asarray(ideal[component], dtype=float)
    for row_idx, n in enumerate(n_values):
        color = _fock_color(int(n))
        center = offsets[row_idx]
        sim_x = indices + center - pair_offset
        ideal_x = indices + center + pair_offset
        sim_mask = _plottable_mask(sim_valid[row_idx], sim_values[row_idx])
        ideal_mask = _plottable_mask(ideal_valid[row_idx], ideal_values[row_idx])
        if np.any(sim_mask):
            ax.bar(
                sim_x[sim_mask],
                sim_values[row_idx, sim_mask],
                width=width,
                color=color,
                edgecolor=color,
                alpha=0.75,
                linewidth=1.0,
            )
        if np.any(ideal_mask):
            ax.bar(
                ideal_x[ideal_mask],
                ideal_values[row_idx, ideal_mask],
                width=width,
                facecolor="none",
                edgecolor=color,
                hatch="//",
                linewidth=1.4,
                linestyle="--",
            )
    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.6)
    ax.set_ylabel(COMPONENT_LABELS[component])
    _style_gate_axis(ax, track, indices, label_stride=label_stride, show_top_axis=True)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        f"{track['case']} vs ideal: {COMPONENT_LABELS[component]} by gate index "
        f"(masked when P(n) < {simulated['probability_threshold']:.1e}, n_max={int(n_values[-1])})"
    )
    ax.legend(handles=_style_legend_handles(n_values), loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_phase_heatmap_overlay(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    label_stride: int = 1,
    contour_levels: tuple[float, ...] = PHASE_CONTOUR_LEVELS,
):
    return plot_phase_overlay_lines(simulated=simulated, ideal=ideal, track=track, label_stride=label_stride)


def plot_phase_overlay_lines(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    label_stride: int = 1,
):
    sim_bundle, ideal_bundle, families = _validate_phase_alignment(simulated, ideal)
    indices = np.asarray(sim_bundle["indices"], dtype=int)
    n_values = np.asarray(sim_bundle["n_values"], dtype=int)
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    for family in families:
        style = PHASE_FAMILY_STYLES[family]
        sim_phase = np.asarray(sim_bundle["families"][family]["phase"], dtype=float)
        ideal_phase = np.asarray(ideal_bundle["families"][family]["phase"], dtype=float)
        for row_idx, n in enumerate(n_values):
            color = _fock_color(int(n))
            ax.plot(
                indices,
                sim_phase[row_idx],
                color=color,
                linewidth=2.2,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.2,
                markerfacecolor=color,
                markeredgecolor=color,
            )
            ax.plot(
                indices,
                ideal_phase[row_idx],
                color=color,
                linewidth=1.9,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=color,
            )
    ax.set_ylabel(r"Relative phase $\phi_{g0 \rightarrow q,n}$ [rad]")
    _style_gate_axis(ax, track, indices, label_stride=label_stride, show_top_axis=True)
    ax.grid(alpha=0.25)
    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.45)
    ax.set_title(
        f"{track['case']} vs ideal: relative phase between |g,0> and |g,n> / |e,n> vs gate index "
        f"({sim_bundle['phase_mode']}, n_max={int(n_values[-1])})"
    )
    ax.legend(handles=_phase_line_handles(n_values, families=families), loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_phase_error_heatmap(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    label_stride: int = 1,
):
    return plot_phase_error_track(simulated=simulated, ideal=ideal, track=track, label_stride=label_stride)


def plot_phase_error_track(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    label_stride: int = 1,
):
    sim_bundle, ideal_bundle, families = _validate_phase_alignment(simulated, ideal)
    indices = np.asarray(sim_bundle["indices"], dtype=int)
    n_values = np.asarray(sim_bundle["n_values"], dtype=int)
    error = wrapped_phase_error(simulated, ideal)
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    for family in families:
        style = PHASE_FAMILY_STYLES[family]
        family_error = error[family] if isinstance(error, dict) else error
        for row_idx, n in enumerate(n_values):
            ax.plot(
                indices,
                family_error[row_idx],
                color=_fock_color(int(n)),
                linewidth=2.0,
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=4.0,
                markerfacecolor=_fock_color(int(n)),
                markeredgecolor=_fock_color(int(n)),
            )
    ax.axhline(0.0, color="0.35", linewidth=1.0, alpha=0.5)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_ylabel(r"$\Delta \phi$ [rad]")
    _style_gate_axis(ax, track, indices, label_stride=label_stride, show_top_axis=True)
    ax.grid(alpha=0.25)
    ax.set_title(f"{track['case']}: phase error for |g,0> to |g,n> / |e,n> vs ideal (ratio-based, wrapped)")
    ax.legend(handles=_phase_line_handles(n_values, families=families)[2:], loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_gate_bloch_trajectory_overlay(simulated: dict[str, Any], ideal: dict[str, Any]):
    times_ns = np.asarray(simulated["times_ns"], dtype=float)
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.6), sharex=True)
    for axis, component in zip(axes, ("x", "y", "z")):
        axis.plot(
            times_ns,
            np.asarray(simulated[component], dtype=float),
            color=COMPONENT_COLORS[component],
            linewidth=2.2,
            linestyle="-",
            label=f"{COMPONENT_LABELS[component]} simulated",
        )
        axis.plot(
            times_ns,
            np.asarray(ideal[component], dtype=float),
            color=COMPONENT_COLORS[component],
            linewidth=2.0,
            linestyle="--",
            label=f"{COMPONENT_LABELS[component]} ideal",
        )
        axis.plot(times_ns[0], float(simulated[component][0]), marker="o", color=COMPONENT_COLORS[component], markersize=5)
        axis.plot(
            times_ns[-1],
            float(simulated[component][-1]),
            marker="o",
            color=COMPONENT_COLORS[component],
            markersize=5,
        )
        axis.plot(
            times_ns[0],
            float(ideal[component][0]),
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COMPONENT_COLORS[component],
            markersize=5,
        )
        axis.plot(
            times_ns[-1],
            float(ideal[component][-1]),
            marker="o",
            markerfacecolor="white",
            markeredgecolor=COMPONENT_COLORS[component],
            markersize=5,
        )
        axis.set_ylabel(COMPONENT_LABELS[component])
        axis.set_ylim(-1.05, 1.05)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[-1].set_xlabel("Time within selected gate [ns]")
    fig.suptitle(
        f"{simulated['case']} vs ideal: gate {simulated['gate_index']} ({simulated['gate_type']}, {simulated['gate_name']})",
        y=0.995,
    )
    fig.tight_layout()
    return fig


def plot_gate_bloch_trajectory_error(simulated: dict[str, Any], ideal: dict[str, Any]):
    times_ns = np.asarray(simulated["times_ns"], dtype=float)
    fig, ax = plt.subplots(figsize=(11.0, 4.4))
    for component in ("x", "y", "z"):
        error = np.asarray(simulated[component], dtype=float) - np.asarray(ideal[component], dtype=float)
        ax.plot(
            times_ns,
            error,
            color=COMPONENT_COLORS[component],
            linewidth=1.9,
            linestyle=":",
            label=rf"$\Delta {component.upper()}(t)$",
        )
    ax.set_xlabel("Time within selected gate [ns]")
    ax.set_ylabel("Simulated - ideal")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=3)
    ax.set_title(
        f"{simulated['case']}: trajectory error for gate {simulated['gate_index']} "
        f"({simulated['gate_type']}, {simulated['gate_name']})"
    )
    fig.tight_layout()
    return fig


def plot_combined_gate_diagnostics(
    simulated_bloch: dict[str, Any],
    ideal_bloch: dict[str, Any],
    simulated_phase: dict[str, Any],
    ideal_phase: dict[str, Any],
    track: dict[str, Any],
    trajectory_simulated: dict[str, Any] | None = None,
    trajectory_ideal: dict[str, Any] | None = None,
    label_stride: int = 1,
):
    _validate_alignment(simulated_bloch, ideal_bloch)
    _validate_alignment(simulated_phase, ideal_phase)
    indices = np.asarray(simulated_bloch["indices"], dtype=int)
    n_values = np.asarray(simulated_bloch["n_values"], dtype=int)
    extent = _matrix_extent(indices, n_values)
    fig = plt.figure(figsize=(14.0, 10.0))
    grid = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 1.0, 1.2], hspace=0.45, wspace=0.3)
    heat_axes = [fig.add_subplot(grid[row, 0:2]) for row in range(3)]
    phase_axis = fig.add_subplot(grid[0:3, 2])
    traj_axis = fig.add_subplot(grid[3, :])

    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")
    grid_x, grid_y = _mesh(indices, n_values)
    for axis, component in zip(heat_axes, ("x", "y", "z")):
        image = axis.imshow(
            _masked(simulated_bloch[component]),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        if np.count_nonzero(np.isfinite(np.asarray(ideal_bloch[component], dtype=float))) >= 4:
            axis.contour(
                grid_x,
                grid_y,
                np.ma.masked_invalid(np.asarray(ideal_bloch[component], dtype=float)),
                levels=IDEAL_CONTOUR_LEVELS,
                colors=[COMPONENT_COLORS[component]],
                linewidths=1.2,
                linestyles="--",
            )
        axis.set_ylabel("Fock n")
        axis.set_title(f"{COMPONENT_LABELS[component]}: simulated fill, ideal contour")
    _style_gate_axis(heat_axes[-1], track, indices, label_stride=label_stride, show_top_axis=False)
    add_gate_type_axis(heat_axes[0], track, label_stride=label_stride, xlabel="Gate type per iteration")
    fig.colorbar(image, ax=heat_axes, shrink=0.9, label="Simulated Bloch value")

    phase_image = phase_axis.imshow(
        _masked(simulated_phase["phase"]),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="twilight",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-np.pi, vmax=np.pi),
        interpolation="nearest",
    )
    if np.count_nonzero(np.isfinite(np.asarray(ideal_phase["phase"], dtype=float))) >= 4:
        phase_axis.contour(
            grid_x,
            grid_y,
            np.ma.masked_invalid(np.asarray(ideal_phase["phase"], dtype=float)),
            levels=PHASE_CONTOUR_LEVELS,
            colors=["black"],
            linewidths=1.0,
            linestyles="--",
        )
    phase_axis.set_ylabel("Fock n")
    _style_gate_axis(phase_axis, track, indices, label_stride=label_stride, show_top_axis=False)
    phase_axis.set_title("Phase: simulated fill, ideal contour")
    fig.colorbar(phase_image, ax=phase_axis, shrink=0.9, label="Simulated phase [rad]")

    if trajectory_simulated is not None and trajectory_ideal is not None:
        times_ns = np.asarray(trajectory_simulated["times_ns"], dtype=float)
        for component in ("x", "y", "z"):
            traj_axis.plot(times_ns, simulated_bloch := np.asarray(trajectory_simulated[component], dtype=float), color=COMPONENT_COLORS[component], linewidth=2.0, linestyle="-", label=f"{component.upper()} simulated")
            traj_axis.plot(times_ns, np.asarray(trajectory_ideal[component], dtype=float), color=COMPONENT_COLORS[component], linewidth=1.8, linestyle="--", label=f"{component.upper()} ideal")
        traj_axis.set_xlabel("Time within selected gate [ns]")
        traj_axis.set_ylabel("Bloch value")
        traj_axis.set_ylim(-1.05, 1.05)
        traj_axis.grid(alpha=0.25)
        traj_axis.legend(loc="upper right", ncol=3, fontsize=8)
        traj_axis.set_title(
            f"Selected gate {trajectory_simulated['gate_index']} ({trajectory_simulated['gate_type']}, {trajectory_simulated['gate_name']})"
        )
    else:
        traj_axis.axis("off")

    fig.suptitle(f"{track['case']} vs ideal: gate-indexed diagnostics", y=0.995)
    fig.tight_layout()
    return fig


def save_figure(fig, output_dir: str | Path, filename: str, dpi: int = 160) -> Path | None:
    if fig is None:
        return None
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / filename
    fig.savefig(target, dpi=int(dpi), bbox_inches="tight")
    return target
