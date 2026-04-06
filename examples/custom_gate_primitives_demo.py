"""Deterministic custom primitive example for unitary synthesis.

This example defines a user-supplied qubit X-rotation primitive with
``make_gate_from_callable(...)``, registers a factory in ``gate_registry``,
fits an ``Rx(pi/2)`` target in a two-level subspace, and validates the custom
primitive by plotting the excited-state population from ``|g>`` as a function
of the rotation angle.

The generated plot is saved to the unitary-synthesis tutorial asset path by
default so the documentation and the checked-in MkDocs site can reuse the
exact validation artifact produced by the example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.map_synthesis import (
    QuantumMapSynthesizer,
    Subspace,
    TargetUnitary,
    gate_registry,
    make_gate_from_callable,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOT_PATH = (
    REPO_ROOT
    / "documentations"
    / "assets"
    / "images"
    / "tutorials"
    / "custom_primitive_rotation_population.png"
)
DEFAULT_DURATION_S = 40.0e-9
REGISTERED_GATE_NAME = "CustomXRotation"


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def custom_x_rotation_unitary(params: dict[str, float], model: object | None) -> np.ndarray:
    del model
    theta = float(params["theta"])
    half = theta / 2.0
    c = np.cos(half)
    s = np.sin(half)
    return np.array(
        [
            [c, -1j * s],
            [-1j * s, c],
        ],
        dtype=np.complex128,
    )


def register_custom_x_rotation() -> None:
    def factory(name: str, duration: float, **kwargs: object):
        kwargs.pop("optimize_time", None)
        return make_gate_from_callable(
            name,
            custom_x_rotation_unitary,
            parameters={"theta": 0.2},
            parameter_bounds={"theta": (-np.pi, np.pi)},
            duration=duration,
            optimize_time=False,
            metadata={"example": "custom primitive smoke test"},
            **kwargs,
        )

    gate_registry.register(REGISTERED_GATE_NAME, factory)


def target_rx(theta: float) -> np.ndarray:
    return custom_x_rotation_unitary({"theta": float(theta)}, None)


def excited_state_population(theta: float) -> float:
    primitive = gate_registry.build(REGISTERED_GATE_NAME, duration=DEFAULT_DURATION_S)
    primitive.parameters["theta"] = float(theta)
    unitary = np.asarray(primitive.ideal_unitary(-1).full(), dtype=np.complex128)
    ground_state = np.array([1.0, 0.0], dtype=np.complex128)
    final_state = unitary @ ground_state
    return float(np.abs(final_state[1]) ** 2)


def run_custom_primitive_fit() -> dict[str, float | QuantumMapSynthesizer | object]:
    register_custom_x_rotation()
    primitive = gate_registry.build(REGISTERED_GATE_NAME, duration=DEFAULT_DURATION_S)
    target_theta = np.pi / 2.0
    synthesizer = QuantumMapSynthesizer(
        primitives=[primitive],
        subspace=Subspace.custom(2, [0, 1]),
        optimizer="L-BFGS-B",
        optimize_times=False,
        seed=7,
    )
    result = synthesizer.fit(target=TargetUnitary(target_rx(target_theta)), maxiter=80)
    fitted_theta = float(result.sequence.gates[0].parameters["theta"])
    fidelity = float(result.report["metrics"]["fidelity"])
    return {
        "target_theta": target_theta,
        "fitted_theta": fitted_theta,
        "fidelity": fidelity,
        "result": result,
    }


def build_population_validation(num_points: int = 121) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_values = np.linspace(0.0, np.pi, int(num_points))
    simulated = np.array([excited_state_population(theta) for theta in theta_values], dtype=float)
    expected = np.sin(theta_values / 2.0) ** 2
    return theta_values, simulated, expected


def save_population_plot(
    theta_values: np.ndarray,
    simulated: np.ndarray,
    expected: np.ndarray,
    *,
    fitted_theta: float,
    target_theta: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(theta_values / np.pi, expected, color="#1f77b4", linewidth=2.2, label=r"analytic $\sin^2(\theta/2)$")
    ax.plot(
        theta_values / np.pi,
        simulated,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        label="registered primitive",
    )
    ax.scatter(
        [target_theta / np.pi, fitted_theta / np.pi],
        [np.sin(target_theta / 2.0) ** 2, np.sin(fitted_theta / 2.0) ** 2],
        color=["#2ca02c", "#9467bd"],
        s=60,
        zorder=3,
        label="target / fitted angle",
    )
    ax.set_xlabel(r"Rotation angle $\theta / \pi$")
    ax.set_ylabel(r"Excited-state population $P_e$")
    ax.set_title("Custom callable primitive validation")
    ax.set_xlim(theta_values[0] / np.pi, theta_values[-1] / np.pi)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path for the excited-state population validation plot.",
    )
    args = parser.parse_args()

    fit_summary = run_custom_primitive_fit()
    theta_values, simulated, expected = build_population_validation()
    max_population_error = float(np.max(np.abs(simulated - expected)))
    save_population_plot(
        theta_values,
        simulated,
        expected,
        fitted_theta=float(fit_summary["fitted_theta"]),
        target_theta=float(fit_summary["target_theta"]),
        output_path=args.output,
    )

    print_section("1. Registered custom callable primitive")
    print(f"target Rx(pi/2) fidelity: {fit_summary['fidelity']:.12f}")
    print(f"  fitted theta={fit_summary['fitted_theta']:.9f} rad")
    print(f"  target theta={fit_summary['target_theta']:.9f} rad")

    print_section("2. Observable validation")
    print(f"max |P_e(custom) - P_e(analytic)| over [0, pi]: {max_population_error:.3e}")
    print(f"plot saved to: {args.output}")


if __name__ == "__main__":
    main()