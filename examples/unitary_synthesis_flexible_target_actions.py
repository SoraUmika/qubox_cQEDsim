from __future__ import annotations

import numpy as np

from cqed_sim.map_synthesis import (
    ExecutionOptions,
    PrimitiveGate,
    QuantumMapSynthesizer,
    Subspace,
    TargetChannel,
    TargetIsometry,
    TargetReducedStateMapping,
)


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def hadamard() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def cnot() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )


def run_channel_demo() -> None:
    primitive = PrimitiveGate(
        name="ry",
        duration=20.0e-9,
        matrix=lambda params, model: rotation_y(float(params["theta"])),
        parameters={"theta": 0.1, "duration": 20.0e-9},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
        hilbert_dim=2,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=TargetChannel(unitary=rotation_y(np.pi / 2.0), enforce_cptp=True),
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=7,
    ).fit(maxiter=40)
    print("channel overlap:", result.report["metrics"]["channel_overlap"])
    print("channel choi error:", result.report["metrics"]["channel_choi_error"])
    print("execution backend:", result.report["execution"]["selected_engine"])


def run_reduced_state_demo() -> None:
    primitive = PrimitiveGate(
        name="ix",
        duration=20.0e-9,
        matrix=np.kron(np.eye(2, dtype=np.complex128), np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)),
        hilbert_dim=4,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=TargetReducedStateMapping(
            initial_states=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
            ],
            target_states=[
                np.array([1.0, 0.0], dtype=np.complex128),
                np.array([0.0, 1.0], dtype=np.complex128),
            ],
            retained_subsystems=(0,),
            subsystem_dims=(2, 2),
        ),
        optimize_times=False,
        seed=11,
    ).fit(maxiter=1)
    print("reduced-state fidelity:", result.report["metrics"]["reduced_state_fidelity_mean"])
    print("retained subsystems:", result.report["target"]["retained_subsystems"])


def run_isometry_demo() -> None:
    encoder = cnot() @ np.kron(hadamard(), np.eye(2, dtype=np.complex128))
    primitive = PrimitiveGate(
        name="encoder",
        duration=30.0e-9,
        matrix=encoder,
        hilbert_dim=4,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=TargetIsometry(encoder[:, :2]),
        optimize_times=False,
        seed=13,
    ).fit(maxiter=1)
    print("isometry coherent fidelity:", result.report["metrics"]["isometry_coherent_fidelity"])
    print("isometry basis fidelity:", result.report["metrics"]["isometry_basis_fidelity"])


def run_metric_selection_demo() -> None:
    phase_skew = np.diag([1.0, -1.0, 1.0, 1.0]).astype(np.complex128)
    primitive = PrimitiveGate(
        name="phase_skew",
        duration=20.0e-9,
        matrix=phase_skew,
        hilbert_dim=4,
    )
    target = TargetIsometry(np.eye(4, dtype=np.complex128)[:, :2])
    coherent = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=target,
        optimize_times=False,
        seed=19,
    ).fit(maxiter=1)
    basis = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=target,
        metric="isometry_basis_fidelity",
        optimize_times=False,
        seed=19,
    ).fit(maxiter=1)
    print("default selected metric:", coherent.report["objective"]["selected_metrics"]["selected_metric_name"])
    print("default objective:", coherent.objective)
    print("basis selected metric:", basis.report["objective"]["selected_metrics"]["selected_metric_name"])
    print("basis objective:", basis.objective)


def run_explicit_input_subspace_demo() -> None:
    primitive = PrimitiveGate(
        name="identity4",
        duration=20.0e-9,
        matrix=np.eye(4, dtype=np.complex128),
        hilbert_dim=4,
    )
    target = TargetIsometry.from_basis_map(
        target_states=[
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
        ],
        input_subspace=Subspace.custom(4, [0, 2]),
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=target,
        optimize_times=False,
        seed=23,
    ).fit(maxiter=1)
    print("explicit-input isometry coherent fidelity:", result.report["metrics"]["isometry_coherent_fidelity"])


def main() -> None:
    run_channel_demo()
    run_reduced_state_demo()
    run_isometry_demo()
    run_metric_selection_demo()
    run_explicit_input_subspace_demo()


if __name__ == "__main__":
    main()