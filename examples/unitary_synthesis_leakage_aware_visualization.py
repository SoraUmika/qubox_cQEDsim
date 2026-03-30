from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from typing import Any

from cqed_sim.map_synthesis import (
    ExecutionOptions,
    LeakagePenalty,
    PrimitiveGate,
    QuantumMapSynthesizer,
    Subspace,
    TargetReducedStateMapping,
    plot_density_matrix_heatmap,
    plot_edge_population_summary,
    plot_leakage_block_heatmap,
    plot_leakage_profile,
    plot_operator_magnitude_heatmap,
    plot_output_population_bars,
    plot_projected_logical_density,
)


LOGICAL_SUBSPACE = Subspace.custom(6, [0, 1], labels=("|L0>", "|L1>"))
TARGET_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
BASIS_LABELS = ("|A0,q0>", "|A0,q1>", "|A1,q0>", "|A1,q1>", "|A2,q0>", "|A2,q1>")
EDGE_PROJECTOR = [4, 5]


def _rotation01(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def _rotation12(phi: float) -> np.ndarray:
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.complex128,
    )


def _effective_angles(theta_raw: float, phi_raw: float) -> tuple[float, float]:
    theta = 0.25 + 0.9 / (1.0 + np.exp(-float(theta_raw)))
    phi = 0.5 * np.pi / (1.0 + np.exp(-float(phi_raw)))
    return theta, phi


def ancilla_leakage_family(theta_raw: float, phi_raw: float) -> np.ndarray:
    theta, phi = _effective_angles(theta_raw, phi_raw)
    ancilla_mixer = _rotation12(phi) @ _rotation01(theta)
    return np.kron(ancilla_mixer, TARGET_X)


def _result_operator(result: Any) -> np.ndarray:
    parameters = result.sequence.gates[0].parameters
    return np.asarray(
        ancilla_leakage_family(float(parameters["theta_raw"]), float(parameters["phi_raw"])),
        dtype=np.complex128,
    )


def _make_synth(*, leakage_penalty: LeakagePenalty | None, seed: int) -> QuantumMapSynthesizer:
    primitive = PrimitiveGate(
        name="relevant_x_with_ancilla_leakage",
        duration=20.0e-9,
        matrix=lambda params, model: ancilla_leakage_family(float(params["theta_raw"]), float(params["phi_raw"])),
        parameters={"theta_raw": 1.0, "phi_raw": 1.0, "duration": 20.0e-9},
        parameter_bounds={"theta_raw": (-6.0, 6.0), "phi_raw": (-6.0, 6.0)},
        hilbert_dim=6,
    )
    return QuantumMapSynthesizer(
        subspace=LOGICAL_SUBSPACE,
        primitives=[primitive],
        target=TargetReducedStateMapping(
            initial_states=[
                np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            ],
            target_states=[
                np.array([0.0, 1.0], dtype=np.complex128),
                np.array([1.0, 0.0], dtype=np.complex128),
            ],
            retained_subsystems=(1,),
            subsystem_dims=(3, 2),
        ),
        leakage_penalty=leakage_penalty,
        optimizer="powell",
        execution=ExecutionOptions(engine="numpy"),
        optimize_times=False,
        seed=seed,
    )


def run_comparison(*, maxiter_unpenalized: int = 1, maxiter_penalized: int = 8) -> dict[str, Any]:
    # The reduced-state target intentionally ignores the ancilla level, so the
    # task metric is flat across a family of relevant-map matches.
    cases = {
        "no_penalty": LeakagePenalty(weight=0.0, checkpoint_weight=0.0, edge_weight=0.0, checkpoints=(1,), edge_projector=EDGE_PROJECTOR),
        "leakage_penalty": LeakagePenalty(weight=0.6, checkpoint_weight=0.0, edge_weight=0.0, checkpoints=(1,), edge_projector=EDGE_PROJECTOR),
        "leakage_and_edge_penalty": LeakagePenalty(weight=0.6, checkpoint_weight=0.0, edge_weight=0.4, checkpoints=(1,), edge_projector=EDGE_PROJECTOR),
    }
    results = {}
    for index, (name, penalty) in enumerate(cases.items()):
        maxiter = maxiter_unpenalized if name == "no_penalty" else maxiter_penalized
        results[name] = _make_synth(leakage_penalty=penalty, seed=13 + index).fit(maxiter=maxiter)
    return results


def comparison_summary(results: dict[str, Any]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, result in results.items():
        parameters = result.sequence.gates[0].parameters
        theta, phi = _effective_angles(float(parameters["theta_raw"]), float(parameters["phi_raw"]))
        summary[name] = {
            "objective": float(result.objective),
            "reduced_state_fidelity": float(result.report["metrics"].get("reduced_state_fidelity_mean", np.nan)),
            "logical_leakage": float(result.report["metrics"].get("logical_leakage_worst", np.nan)),
            "edge_population": float(result.report["metrics"].get("edge_population_worst", np.nan)),
            "effective_theta": float(theta),
            "effective_phi": float(phi),
        }
    return summary


def plot_case_overview(results: dict[str, Any]) -> list[plt.Figure]:
    figures: list[plt.Figure] = []

    fig1, axes1 = plt.subplots(2, 3, figsize=(14.0, 7.5))
    for column, (name, result) in enumerate(results.items()):
        operator = _result_operator(result)
        plot_operator_magnitude_heatmap(operator, LOGICAL_SUBSPACE, ax=axes1[0, column], title=f"{name}: |U|")
        plot_leakage_block_heatmap(operator, LOGICAL_SUBSPACE, ax=axes1[1, column], title=f"{name}: leakage block")
    fig1.tight_layout()
    figures.append(fig1)

    fig2, axes2 = plt.subplots(2, 2, figsize=(12.0, 8.0))
    representative_state = qt.Qobj(
        _result_operator(results["no_penalty"])
        @ LOGICAL_SUBSPACE.embed(np.array([1.0, 0.0], dtype=np.complex128))
    )
    plot_output_population_bars(
        [representative_state],
        basis_labels=BASIS_LABELS,
        state_labels=("no-penalty output for |L0>",),
        axes=axes2[0, 0],
    )
    plot_density_matrix_heatmap(representative_state, ax=axes2[0, 1], title="no-penalty output density |rho|")
    plot_projected_logical_density(
        representative_state,
        LOGICAL_SUBSPACE.projector(),
        ax=axes2[1, 0],
        title="logical projection of no-penalty output",
    )
    plot_edge_population_summary(
        results["no_penalty"].report["truncation"],
        edge_metrics=results["no_penalty"].report["metrics"],
        ax=axes2[1, 1],
        title="no-penalty edge and truncation summary",
    )
    fig2.tight_layout()
    figures.append(fig2)

    fig3, axis3 = plt.subplots(figsize=(8.0, 4.0))
    for name, result in results.items():
        profile = result.report["leakage_diagnostics"].get("path_profile", [])
        if not profile:
            continue
        x = [row["step"] for row in profile]
        y = [row["worst"] for row in profile]
        axis3.plot(x, y, marker="o", linewidth=1.8, label=name)
    axis3.set_xlabel("step")
    axis3.set_ylabel("worst leakage")
    axis3.set_title("Leakage-vs-step comparison")
    axis3.legend()
    axis3.grid(alpha=0.25)
    figures.append(fig3)

    fig4, axis4 = plt.subplots(figsize=(8.0, 4.0))
    plot_leakage_profile(
        results["leakage_and_edge_penalty"].report["leakage_diagnostics"].get("path_profile", []),
        ax=axis4,
        title="Leakage + edge penalty path profile",
    )
    figures.append(fig4)
    return figures


def main() -> None:
    results = run_comparison()
    for name, metrics in comparison_summary(results).items():
        print(name)
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
    plot_case_overview(results)
    plt.show()


if __name__ == "__main__":
    main()