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
IDEAL_CONTOUR_LEVELS = (-1.0, -0.5, 0.0, 0.5, 1.0)
PHASE_CONTOUR_LEVELS = (-np.pi, -np.pi / 2.0, 0.0, np.pi / 2.0, np.pi)


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


def _sim_vs_ideal_handles(color: str | None = None) -> list:
    contour_color = "black" if color is None else color
    return [
        Patch(facecolor="#cccccc", edgecolor="none", alpha=0.8, label="Simulated (filled heatmap)"),
        Line2D([0], [0], color=contour_color, linestyle="--", linewidth=1.5, label="Ideal (dashed contour)"),
    ]


def plot_fock_resolved_bloch_overlay(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    component: str,
    label_stride: int = 1,
    contour_levels: tuple[float, ...] = IDEAL_CONTOUR_LEVELS,
):
    _validate_alignment(simulated, ideal)
    indices = np.asarray(simulated["indices"], dtype=int)
    n_values = np.asarray(simulated["n_values"], dtype=int)
    extent = _matrix_extent(indices, n_values)
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="#f0f0f0")
    image = ax.imshow(
        _masked(simulated[component]),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
        interpolation="nearest",
    )
    grid_x, grid_y = _mesh(indices, n_values)
    ideal_data = np.asarray(ideal[component], dtype=float)
    finite = np.isfinite(ideal_data)
    if np.count_nonzero(finite) >= 4:
        contour = ax.contour(
            grid_x,
            grid_y,
            np.ma.masked_invalid(ideal_data),
            levels=contour_levels,
            colors=[COMPONENT_COLORS[component]],
            linewidths=1.4,
            linestyles="--",
        )
        if contour.allsegs:
            ax.clabel(contour, fmt="%0.1f", fontsize=7, inline=True)
    ax.set_ylabel(f"Fock level n (n_max={int(n_values[-1])})")
    _style_gate_axis(ax, track, indices, label_stride=label_stride, show_top_axis=True)
    ax.set_title(f"{track['case']} vs ideal: {COMPONENT_LABELS[component]} by gate index")
    ax.legend(handles=_sim_vs_ideal_handles(COMPONENT_COLORS[component]), loc="upper right")
    fig.colorbar(image, ax=ax, shrink=0.96, label=f"Simulated {COMPONENT_LABELS[component]}")
    fig.tight_layout()
    return fig


def plot_phase_heatmap_overlay(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    label_stride: int = 1,
    contour_levels: tuple[float, ...] = PHASE_CONTOUR_LEVELS,
):
    _validate_alignment(simulated, ideal)
    indices = np.asarray(simulated["indices"], dtype=int)
    n_values = np.asarray(simulated["n_values"], dtype=int)
    extent = _matrix_extent(indices, n_values)
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    image = ax.imshow(
        _masked(simulated["phase"]),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="twilight",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-np.pi, vmax=np.pi),
        interpolation="nearest",
    )
    grid_x, grid_y = _mesh(indices, n_values)
    ideal_phase = np.asarray(ideal["phase"], dtype=float)
    finite = np.isfinite(ideal_phase)
    if np.count_nonzero(finite) >= 4:
        contour = ax.contour(
            grid_x,
            grid_y,
            np.ma.masked_invalid(ideal_phase),
            levels=contour_levels,
            colors=["black"],
            linewidths=1.1,
            linestyles="--",
        )
        if contour.allsegs:
            ax.clabel(contour, fmt="%0.2f", fontsize=7, inline=True)
    ax.set_ylabel(f"Fock level n (n_max={int(n_values[-1])})")
    _style_gate_axis(ax, track, indices, label_stride=label_stride, show_top_axis=True)
    ax.set_title(f"{track['case']} vs ideal: conditional phase by gate index")
    ax.legend(handles=_sim_vs_ideal_handles(), loc="upper right")
    fig.colorbar(image, ax=ax, shrink=0.96, label="Simulated phase [rad]")
    fig.tight_layout()
    return fig


def plot_phase_error_heatmap(
    simulated: dict[str, Any],
    ideal: dict[str, Any],
    track: dict[str, Any],
    label_stride: int = 1,
):
    _validate_alignment(simulated, ideal)
    indices = np.asarray(simulated["indices"], dtype=int)
    n_values = np.asarray(simulated["n_values"], dtype=int)
    extent = _matrix_extent(indices, n_values)
    error = wrapped_phase_error(simulated["phase"], ideal["phase"])
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    image = ax.imshow(
        _masked(error),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="coolwarm",
        norm=TwoSlopeNorm(vcenter=0.0, vmin=-np.pi, vmax=np.pi),
        interpolation="nearest",
    )
    ax.set_ylabel(f"Fock level n (n_max={int(n_values[-1])})")
    _style_gate_axis(ax, track, indices, label_stride=label_stride, show_top_axis=True)
    ax.set_title(f"{track['case']}: wrapped phase error vs ideal")
    fig.colorbar(image, ax=ax, shrink=0.96, label=r"$\Delta \phi$ [rad]")
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
