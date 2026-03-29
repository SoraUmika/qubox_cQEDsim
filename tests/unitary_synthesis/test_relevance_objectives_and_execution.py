from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis import (
    ExecutionOptions,
    LeakagePenalty,
    ObservableTarget,
    PrimitiveGate,
    Subspace,
    TargetStateMapping,
    TrajectoryCheckpoint,
    TrajectoryTarget,
    UnitarySynthesizer,
)


def _rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def test_observable_target_optimizes_expected_observable_value() -> None:
    primitive = PrimitiveGate(
        name="ry",
        duration=20.0e-9,
        matrix=lambda params, model: _rotation_y(float(params["theta"])),
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
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=4,
    )
    result = synth.fit(maxiter=40)
    assert result.report["metrics"]["weighted_observable_error"] < 1.0e-8


def test_trajectory_target_matches_intermediate_and_final_states() -> None:
    primitives = [
        PrimitiveGate(
            name="ry_first",
            duration=20.0e-9,
            matrix=lambda params, model: _rotation_y(float(params["theta"])),
            parameters={"theta": 0.2, "duration": 20.0e-9},
            parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
            hilbert_dim=2,
        ),
        PrimitiveGate(
            name="ry_second",
            duration=20.0e-9,
            matrix=lambda params, model: _rotation_y(float(params["theta"])),
            parameters={"theta": -0.2, "duration": 20.0e-9},
            parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
            hilbert_dim=2,
        ),
    ]
    psi_g = np.array([1.0, 0.0], dtype=np.complex128)
    psi_e = np.array([0.0, 1.0], dtype=np.complex128)
    trajectory = TrajectoryTarget(
        initial_states=[psi_g],
        checkpoints=[
            TrajectoryCheckpoint(step=1, target_states=(psi_e,), weight=1.0, label="after_first"),
            TrajectoryCheckpoint(step=2, target_states=(psi_g,), weight=1.0, label="after_second"),
        ],
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=primitives,
        target=trajectory,
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=12,
    )
    result = synth.fit(maxiter=60)
    assert result.report["metrics"]["trajectory_task_loss"] < 1.0e-6


def test_checkpoint_leakage_penalty_detects_intermediate_leakage() -> None:
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
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(3, [0, 1]),
        primitives=primitives,
        target=TargetStateMapping(
            initial_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=0.0, checkpoint_weight=1.0, checkpoints=(1,)),
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=7,
    )
    result = synth.fit(maxiter=1)
    assert result.report["metrics"]["leakage_worst"] < 1.0e-12
    assert result.report["metrics"]["checkpoint_leakage_worst"] > 0.99
    assert result.report["metrics"]["path_leakage_worst"] > 0.99
    assert result.report["objective"]["checkpoint_leakage_term"] > 0.0
    assert len(result.report["leakage_diagnostics"]["path_profile"]) == 3


def test_path_leakage_metric_alias_can_be_selected_without_checkpoint_penalty() -> None:
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
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(3, [0, 1]),
        primitives=primitives,
        target=TargetStateMapping(
            initial_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=0.0, checkpoint_weight=0.0, checkpoints=(1,)),
        metric="path_leakage_worst",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=8,
    )
    result = synth.fit(maxiter=1)
    assert result.report["objective"]["selected_metrics"]["selected_metric_name"] == "path_leakage_worst"
    assert result.report["metrics"]["path_leakage_worst"] > 0.99
    assert result.report["objective"]["checkpoint_leakage_term"] < 1.0e-12
    assert result.objective > 0.99


def test_edge_projector_penalty_is_separate_from_logical_leakage() -> None:
    swap_02 = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[PrimitiveGate(name="swap_02", duration=10.0e-9, matrix=swap_02, hilbert_dim=4)],
        target=TargetStateMapping(
            initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=0.0, checkpoint_weight=0.0, edge_weight=0.5, edge_projector=[2]),
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=9,
    )
    result = synth.fit(maxiter=1)
    assert result.report["metrics"]["state_average_fidelity"] > 0.999999
    assert result.report["metrics"]["logical_leakage_worst"] < 1.0e-12
    assert result.report["metrics"]["edge_population_worst"] > 0.999999
    assert result.report["objective"]["edge_population_term"] > 0.49
    assert result.objective > 0.49


def test_fast_path_matches_legacy_for_supported_ideal_problem() -> None:
    primitive = PrimitiveGate(
        name="ry",
        duration=20.0e-9,
        matrix=lambda params, model: _rotation_y(float(params["theta"])),
        parameters={"theta": 0.3, "duration": 20.0e-9},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
        hilbert_dim=2,
    )
    target = _rotation_y(np.pi / 2.0)
    kwargs = {
        "subspace": Subspace.custom(2, range(2)),
        "primitives": [primitive],
        "target": target,
        "optimizer": "powell",
        "optimize_times": False,
        "seed": 21,
    }
    legacy = UnitarySynthesizer(**kwargs, execution=ExecutionOptions(engine="legacy")).fit(maxiter=30)
    fast = UnitarySynthesizer(**kwargs, execution=ExecutionOptions(engine="numpy")).fit(maxiter=30)
    assert np.isclose(legacy.objective, fast.objective, atol=1.0e-9)
    assert np.isclose(legacy.report["metrics"]["fidelity"], fast.report["metrics"]["fidelity"], atol=1.0e-9)
    assert fast.report["execution"]["selected_engine"] == "numpy"


def test_jax_request_falls_back_when_jax_is_unavailable() -> None:
    primitive = PrimitiveGate(
        name="ry",
        duration=20.0e-9,
        matrix=lambda params, model: _rotation_y(float(params["theta"])),
        parameters={"theta": 0.1, "duration": 20.0e-9},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
        hilbert_dim=2,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=_rotation_y(np.pi / 4.0),
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="jax", fallback_engine="numpy"),
        seed=31,
    )
    result = synth.fit(maxiter=10)
    assert result.report["execution"]["selected_engine"] in {"jax", "numpy"}
    if result.report["execution"]["selected_engine"] == "numpy":
        assert "not installed" in result.report["execution"]["reason"].lower()