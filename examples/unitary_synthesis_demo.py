from __future__ import annotations

import numpy as np

from cqed_sim.map_synthesis import (
    JaynesCummingsExchange,
    QubitRotation,
    QuantumMapSynthesizer,
    Subspace,
    TargetStateMapping,
    TargetUnitary,
)


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def xy_rotation_matrix(theta: float, phi: float) -> np.ndarray:
    half = theta / 2.0
    off_diag = -1j * np.sin(half)
    return np.array(
        [
            [np.cos(half), off_diag * np.exp(-1j * phi)],
            [off_diag * np.exp(1j * phi), np.cos(half)],
        ],
        dtype=np.complex128,
    )


def ket(dim: int, index: int) -> np.ndarray:
    state = np.zeros(dim, dtype=np.complex128)
    state[index] = 1.0
    return state


def format_populations(populations: np.ndarray, labels: list[str]) -> str:
    return ", ".join(f"{label}={populations[idx]:.6f}" for idx, label in enumerate(labels))


def run_single_qubit_rotation_demo(label: str, theta: float, phi: float) -> None:
    target = TargetUnitary(xy_rotation_matrix(theta, phi))
    synthesizer = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, [0, 1]),
        gateset=["QubitRotation"],
        backend="ideal",
        optimizer="L-BFGS-B",
        optimize_times=False,
        seed=0,
    )
    result = synthesizer.fit(target=target, maxiter=80)
    gate = result.sequence.gates[0]

    print(f"{label} fidelity: {result.report['metrics']['fidelity']:.12f}")
    print(f"  fitted theta={gate.theta:.9f} rad, phi={gate.phi:.9f} rad")
    print(f"  objective={result.objective:.12e}")


def run_cavity_fock_state_demo() -> None:
    subspace = Subspace.qubit_cavity_block(n_match=1, n_cav=3)
    initial_state = ket(subspace.full_dim, 0)
    target_state = ket(subspace.full_dim, 1)

    synthesizer = QuantumMapSynthesizer(
        subspace=subspace,
        primitives=[
            QubitRotation(name="rx", theta=np.pi, phi=0.0, duration=40.0e-9, optimize_time=False),
            JaynesCummingsExchange(
                name="jc",
                coupling=2.0 * np.pi * 5.0e6,
                phase=0.0,
                duration=50.0e-9,
                optimize_time=False,
            ),
        ],
        backend="ideal",
        optimizer="L-BFGS-B",
        optimize_times=False,
        seed=3,
    )
    result = synthesizer.fit(
        target=TargetStateMapping(initial_state=initial_state, target_state=target_state),
        init_guess="heuristic",
        multistart=1,
        maxiter=40,
    )

    final_state = np.asarray(result.simulation.state_outputs[0].full(), dtype=np.complex128).reshape(-1)
    populations = np.abs(final_state) ** 2
    rotation_gate, exchange_gate = result.sequence.gates

    print(f"state fidelity mean: {result.report['metrics']['state_fidelity_mean']:.12f}")
    print(f"  infidelity={1.0 - result.report['metrics']['state_fidelity_mean']:.12e}")
    print(
        "  sequence: "
        f"QubitRotation(theta={rotation_gate.theta:.9f}, phi={rotation_gate.phi:.9f}), "
        f"JaynesCummingsExchange(coupling={exchange_gate.coupling:.9f}, phase={exchange_gate.phase:.9f})"
    )
    print(
        "  populations: "
        + format_populations(populations[[0, 1, 3, 4]], ["|g,0>", "|g,1>", "|e,0>", "|e,1>"])
    )


def main() -> None:
    print_section("1. Single-qubit rotation synthesis using only QubitRotation")
    run_single_qubit_rotation_demo(label="Rx(pi/2)", theta=np.pi / 2.0, phi=0.0)
    run_single_qubit_rotation_demo(label="Ry(pi/2)", theta=np.pi / 2.0, phi=np.pi / 2.0)

    print_section("2. State synthesis for cavity |1> from |g,0>")
    run_cavity_fock_state_demo()


if __name__ == "__main__":
    main()
