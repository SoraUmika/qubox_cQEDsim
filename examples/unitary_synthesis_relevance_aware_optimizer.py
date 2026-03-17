from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from cqed_sim.unitary_synthesis import (
    ExecutionOptions,
    LeakagePenalty,
    MultiObjective,
    ObservableTarget,
    PrimitiveGate,
    Subspace,
    TargetStateMapping,
    TrajectoryCheckpoint,
    TrajectoryTarget,
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


@dataclass
class DemoSummary:
    label: str
    objective: float
    fidelity: float | None
    duration_s: float
    effective_gate_count: float
    execution_engine: str


def summarize(label: str, result) -> DemoSummary:
    metrics = result.report["metrics"]
    return DemoSummary(
        label=label,
        objective=float(result.objective),
        fidelity=None if np.isnan(metrics.get("fidelity", np.nan)) else float(metrics["fidelity"]),
        duration_s=float(result.sequence.total_duration()),
        effective_gate_count=float(metrics.get("gate_count_metric", np.nan)),
        execution_engine=str(result.report["execution"].get("selected_engine", "legacy")),
    )


def relevant_state_ensemble_demo() -> DemoSummary:
    primitive = PrimitiveGate(
        name="ry",
        duration=20.0e-9,
        matrix=lambda params, model: rotation_y(float(params["theta"])),
        parameters={"theta": 0.2, "duration": 20.0e-9},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
        hilbert_dim=2,
    )
    psi_g = np.array([1.0, 0.0], dtype=np.complex128)
    psi_x = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    phi_e = np.array([0.0, 1.0], dtype=np.complex128)
    phi_y = np.array([1.0j, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=TargetStateMapping(
            initial_states=[psi_g, psi_x],
            target_states=[phi_e, phi_y],
            weights=[0.7, 0.3],
        ),
        objectives=MultiObjective(task_weight=1.0, duration_weight=0.1, gate_count_weight=0.1),
        execution=ExecutionOptions(engine="numpy"),
        optimizer="powell",
        optimize_times=False,
        seed=5,
    )
    return summarize("state_ensemble", synth.fit(maxiter=50))


def observable_task_demo() -> DemoSummary:
    primitive = PrimitiveGate(
        name="ry",
        duration=20.0e-9,
        matrix=lambda params, model: rotation_y(float(params["theta"])),
        parameters={"theta": 0.1, "duration": 20.0e-9},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
        hilbert_dim=2,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=ObservableTarget(
            initial_state=np.array([1.0, 0.0], dtype=np.complex128),
            observable=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
            target_expectation=-1.0,
        ),
        objectives=MultiObjective(task_weight=1.0, duration_weight=0.05),
        execution=ExecutionOptions(engine="numpy"),
        optimizer="powell",
        optimize_times=False,
        seed=8,
    )
    return summarize("observable", synth.fit(maxiter=40))


def trajectory_and_leakage_demo() -> DemoSummary:
    leak = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    unleak = leak.conj().T
    primitives = [
        PrimitiveGate(name="leak", duration=10.0e-9, matrix=leak, hilbert_dim=3),
        PrimitiveGate(name="unleak", duration=10.0e-9, matrix=unleak, hilbert_dim=3),
    ]
    trajectory = TrajectoryTarget(
        initial_states=[np.array([0.0, 1.0, 0.0], dtype=np.complex128)],
        checkpoints=[
            TrajectoryCheckpoint(
                step=2,
                target_states=(np.array([0.0, 1.0, 0.0], dtype=np.complex128),),
                weight=1.0,
                label="return_to_logical",
            )
        ],
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(3, [0, 1]),
        primitives=primitives,
        target=trajectory,
        leakage_penalty=LeakagePenalty(weight=0.0, checkpoint_weight=1.0, checkpoints=(1,)),
        objectives=MultiObjective(task_weight=1.0, gate_count_weight=0.05),
        execution=ExecutionOptions(engine="numpy"),
        optimize_times=False,
        seed=13,
    )
    return summarize("trajectory_checkpoint_leakage", synth.fit(maxiter=1))


def main() -> None:
    rows = [
        relevant_state_ensemble_demo(),
        observable_task_demo(),
        trajectory_and_leakage_demo(),
    ]
    print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()