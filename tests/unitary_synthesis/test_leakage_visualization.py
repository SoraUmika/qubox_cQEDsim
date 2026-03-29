from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from cqed_sim.unitary_synthesis import (
    Subspace,
    plot_density_matrix_heatmap,
    plot_edge_population_summary,
    plot_leakage_block_heatmap,
    plot_leakage_profile,
    plot_operator_magnitude_heatmap,
    plot_output_population_bars,
    plot_projected_logical_density,
)


def test_leakage_visualization_helpers_smoke() -> None:
    logical = Subspace.custom(3, [0, 1], labels=("|0>", "|1>"))
    operator = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0 / np.sqrt(2.0), -1.0 / np.sqrt(2.0)],
            [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        ],
        dtype=np.complex128,
    )
    state = qt.Qobj(np.array([0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)], dtype=np.complex128))

    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.0))
    plot_operator_magnitude_heatmap(operator, logical, ax=axes[0, 0])
    plot_leakage_block_heatmap(operator, logical, ax=axes[0, 1])
    plot_output_population_bars(
        [state],
        basis_labels=("|0>", "|1>", "|2>"),
        state_labels=("leaky output",),
        axes=axes[0, 2],
    )
    plot_density_matrix_heatmap(state, ax=axes[1, 0])
    _, trace_value, projected = plot_projected_logical_density(state, logical.projector(), ax=axes[1, 1])
    plot_leakage_profile(
        [
            {"step": 0.0, "time": 0.0, "average": 0.0, "worst": 0.0},
            {"step": 1.0, "time": 1.0, "average": 0.2, "worst": 0.5},
        ],
        ax=axes[1, 2],
    )
    assert trace_value < 1.0
    assert projected.shape == (3, 3)
    plt.close(fig)

    fig2, axis2 = plt.subplots(figsize=(6.0, 3.5))
    plot_edge_population_summary(
        {
            "retained_edge_population_average": 0.1,
            "retained_edge_population_worst": 0.2,
            "outside_tail_population_average": 0.05,
            "outside_tail_population_worst": 0.08,
        },
        edge_metrics={"edge_population_average": 0.3, "edge_population_worst": 0.4},
        ax=axis2,
    )
    plt.close(fig2)