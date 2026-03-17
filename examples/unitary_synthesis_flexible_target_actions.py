from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis import (
    ExecutionOptions,
    PrimitiveGate,
    Subspace,
    TargetChannel,
    TargetIsometry,
    TargetReducedStateMapping,
    UnitarySynthesizer,
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
    result = UnitarySynthesizer(
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
    result = UnitarySynthesizer(
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
    result = UnitarySynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=TargetIsometry(encoder[:, :2]),
        optimize_times=False,
        seed=13,
    ).fit(maxiter=1)
    print("isometry fidelity:", result.report["metrics"]["state_fidelity_mean"])


def main() -> None:
    run_channel_demo()
    run_reduced_state_demo()
    run_isometry_demo()


if __name__ == "__main__":
    main()