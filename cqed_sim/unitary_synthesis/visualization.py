from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from .metrics import projected_density_matrix, state_population_distribution
from .subspace import Subspace


def _logical_indices(logical_subspace: Subspace | Sequence[int], full_dim: int) -> list[int]:
    if isinstance(logical_subspace, Subspace):
        if int(logical_subspace.full_dim) != int(full_dim):
            raise ValueError("logical_subspace.full_dim does not match operator dimension.")
        return [int(index) for index in logical_subspace.indices]
    indices = [int(index) for index in logical_subspace]
    if not indices:
        raise ValueError("logical_subspace must contain at least one basis index.")
    if min(indices) < 0 or max(indices) >= int(full_dim):
        raise ValueError("logical_subspace indices must lie within the operator dimension.")
    if len(set(indices)) != len(indices):
        raise ValueError("logical_subspace indices must be unique.")
    return indices


def reorder_operator_by_subspace(
    operator: np.ndarray,
    logical_subspace: Subspace | Sequence[int],
) -> tuple[np.ndarray, list[int], list[int]]:
    op = np.asarray(operator, dtype=np.complex128)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError("operator must be a square matrix.")
    logical = _logical_indices(logical_subspace, int(op.shape[0]))
    logical_set = set(logical)
    leakage = [index for index in range(int(op.shape[0])) if index not in logical_set]
    order = logical + leakage
    return op[np.ix_(order, order)], logical, leakage


def plot_operator_magnitude_heatmap(
    operator: np.ndarray,
    logical_subspace: Subspace | Sequence[int],
    *,
    ax: Any | None = None,
    title: str = "|U| with logical and leakage blocks",
    cmap: str = "magma",
) -> Any:
    reordered, logical, _ = reorder_operator_by_subspace(operator, logical_subspace)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(np.abs(reordered), cmap=cmap, origin="upper")
    boundary = len(logical) - 0.5
    ax.axhline(boundary, color="white", linestyle="--", linewidth=1.2)
    ax.axvline(boundary, color="white", linestyle="--", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("input basis reordered as logical then leakage")
    ax.set_ylabel("output basis reordered as logical then leakage")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_leakage_block_heatmap(
    operator: np.ndarray,
    logical_subspace: Subspace | Sequence[int],
    *,
    ax: Any | None = None,
    title: str = "|P_leak U P_logical|",
    cmap: str = "magma",
) -> Any:
    reordered, logical, leakage = reorder_operator_by_subspace(operator, logical_subspace)
    logical_dim = len(logical)
    leakage_dim = len(leakage)
    if ax is None:
        _, ax = plt.subplots(figsize=(5.0, 3.8))
    block = np.abs(reordered[logical_dim:, :logical_dim]) if leakage_dim > 0 else np.zeros((1, logical_dim), dtype=float)
    image = ax.imshow(block, cmap=cmap, aspect="auto", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("logical input index")
    ax.set_ylabel("leakage output index")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_output_population_bars(
    states: Sequence[qt.Qobj | np.ndarray],
    *,
    basis_labels: Sequence[str] | None = None,
    state_labels: Sequence[str] | None = None,
    axes: Sequence[Any] | Any | None = None,
    max_states: int | None = None,
) -> tuple[Any, list[Any]]:
    if not states:
        raise ValueError("states must contain at least one state.")
    selected = list(states[: len(states) if max_states is None else int(max_states)])
    count = len(selected)
    if axes is None:
        fig, axs = plt.subplots(count, 1, figsize=(7.0, 2.6 * count), squeeze=False)
        ax_list = [axs[row, 0] for row in range(count)]
    else:
        if isinstance(axes, Sequence) and not isinstance(axes, np.ndarray):
            ax_list = list(axes)
        elif isinstance(axes, np.ndarray):
            ax_list = list(axes.reshape(-1))
        else:
            ax_list = [axes]
        if len(ax_list) < count:
            raise ValueError("Not enough axes were provided for the requested states.")
        fig = ax_list[0].figure
    for index, (state, axis) in enumerate(zip(selected, ax_list)):
        population = state_population_distribution(state)
        x = np.arange(population.size)
        axis.bar(x, population, color="#3d6a9d")
        axis.set_ylim(0.0, max(1.0, float(np.max(population)) * 1.1 if population.size else 1.0))
        axis.set_ylabel("population")
        if basis_labels is not None and len(basis_labels) == population.size:
            axis.set_xticks(x)
            axis.set_xticklabels([str(label) for label in basis_labels], rotation=45, ha="right")
        else:
            axis.set_xticks(x)
        label = None if state_labels is None or index >= len(state_labels) else str(state_labels[index])
        axis.set_title(label or f"output state {index}")
    ax_list[-1].set_xlabel("basis index")
    fig.tight_layout()
    return fig, ax_list[:count]


def plot_density_matrix_heatmap(
    state: qt.Qobj | np.ndarray,
    *,
    ax: Any | None = None,
    title: str = "output density matrix |rho|",
    cmap: str = "magma",
) -> Any:
    obj = state if isinstance(state, qt.Qobj) else qt.Qobj(np.asarray(state, dtype=np.complex128))
    if obj.isoper:
        density = np.asarray(obj.full(), dtype=np.complex128)
    else:
        vec = np.asarray(obj.full(), dtype=np.complex128).reshape(-1)
        density = np.outer(vec, np.conjugate(vec))
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(np.abs(density), cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_projected_logical_density(
    state: qt.Qobj | np.ndarray,
    logical_projector: Any,
    *,
    ax: Any | None = None,
    title: str = "projected logical density |P rho P|",
    normalize: bool = False,
    cmap: str = "magma",
) -> tuple[Any, float, np.ndarray]:
    projected, trace_value = projected_density_matrix(state, logical_projector, normalize=normalize)
    if ax is None:
        _, ax = plt.subplots(figsize=(4.6, 4.0))
    image = ax.imshow(np.abs(projected), cmap=cmap, origin="upper")
    ax.set_title(f"{title} (trace={trace_value:.4f})")
    ax.set_xlabel("logical column")
    ax.set_ylabel("logical row")
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return ax, float(trace_value), projected


def plot_leakage_profile(
    profile: Sequence[dict[str, float]],
    *,
    ax: Any | None = None,
    x_axis: str = "step",
    title: str | None = None,
) -> Any:
    if x_axis not in {"step", "time"}:
        raise ValueError("x_axis must be 'step' or 'time'.")
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 3.8))
    if not profile:
        ax.text(0.5, 0.5, "No leakage profile available", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return ax
    x = [float(row[x_axis]) for row in profile]
    average = [float(row["average"]) for row in profile]
    worst = [float(row["worst"]) for row in profile]
    ax.plot(x, average, marker="o", linewidth=1.8, label="average leakage")
    ax.plot(x, worst, marker="s", linewidth=1.8, label="worst leakage")
    ax.set_xlabel("step" if x_axis == "step" else "time")
    ax.set_ylabel("leakage")
    ax.set_title(title or ("Leakage vs step" if x_axis == "step" else "Leakage vs time"))
    ax.legend()
    ax.grid(alpha=0.25)
    return ax


def plot_edge_population_summary(
    truncation_metrics: dict[str, Any],
    *,
    edge_metrics: dict[str, float] | None = None,
    ax: Any | None = None,
    title: str = "edge and truncation populations",
) -> Any:
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 3.8))
    labels = [
        "retained edge avg",
        "retained edge worst",
        "outside tail avg",
        "outside tail worst",
    ]
    values = [
        float(truncation_metrics.get("retained_edge_population_average", 0.0)),
        float(truncation_metrics.get("retained_edge_population_worst", 0.0)),
        float(truncation_metrics.get("outside_tail_population_average", 0.0)),
        float(truncation_metrics.get("outside_tail_population_worst", 0.0)),
    ]
    if edge_metrics is not None:
        labels.extend(["edge projector avg", "edge projector worst"])
        values.extend(
            [
                float(edge_metrics.get("edge_population_average", 0.0)),
                float(edge_metrics.get("edge_population_worst", 0.0)),
            ]
        )
    x = np.arange(len(labels))
    colors = ["#4c956c", "#2f6f4f", "#bc4b51", "#8f2d56", "#f4a259", "#e76f51"][: len(labels)]
    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("population")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    return ax


__all__ = [
    "plot_density_matrix_heatmap",
    "plot_edge_population_summary",
    "plot_leakage_block_heatmap",
    "plot_leakage_profile",
    "plot_operator_magnitude_heatmap",
    "plot_output_population_bars",
    "plot_projected_logical_density",
    "reorder_operator_by_subspace",
]