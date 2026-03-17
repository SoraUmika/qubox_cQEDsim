from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis import (
    ExecutionOptions,
    PrimitiveGate,
    Subspace,
    TargetChannel,
    TargetIsometry,
    TargetReducedStateMapping,
    TargetUnitary,
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


def _hadamard() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def _cnot() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )


def test_channel_target_matches_qubit_action_and_falls_back_cleanly() -> None:
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
        target=TargetChannel(unitary=_rotation_y(np.pi / 2.0), enforce_cptp=True),
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=41,
    )
    result = synth.fit(maxiter=40)
    assert result.report["metrics"]["channel_choi_error"] < 1.0e-8
    assert result.report["metrics"]["trace_preservation_error_max"] < 1.0e-12
    assert result.report["execution"]["selected_engine"] == "legacy"


def test_reduced_state_target_ignores_unobserved_subsystem_motion() -> None:
    primitive = PrimitiveGate(
        name="ix",
        duration=20.0e-9,
        matrix=np.kron(np.eye(2, dtype=np.complex128), np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)),
        hilbert_dim=4,
    )
    synth = UnitarySynthesizer(
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
        seed=13,
    )
    result = synth.fit(maxiter=1)
    assert result.report["metrics"]["reduced_state_fidelity_mean"] > 0.999999
    assert result.report["target"]["retained_subsystems"] == [0]


def test_isometry_target_matches_selected_logical_columns() -> None:
    bell_creator = _cnot() @ np.kron(_hadamard(), np.eye(2, dtype=np.complex128))
    primitive = PrimitiveGate(
        name="bell_creator",
        duration=30.0e-9,
        matrix=bell_creator,
        hilbert_dim=4,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=TargetIsometry(bell_creator[:, :2]),
        optimize_times=False,
        seed=17,
    )
    result = synth.fit(maxiter=1)
    assert result.report["target"]["type"] == "isometry"
    assert result.report["metrics"]["state_fidelity_mean"] > 0.999999


def test_truncation_report_detects_outside_tail_loading() -> None:
    subspace = Subspace.qubit_cavity_block(n_match=1, n_cav=3)
    operator = np.eye(subspace.full_dim, dtype=np.complex128)
    phi = float(np.arcsin(np.sqrt(0.36)))
    c = float(np.cos(phi))
    s = float(np.sin(phi))
    operator[4, 4] = c
    operator[4, 5] = -s
    operator[5, 4] = s
    operator[5, 5] = c
    primitive = PrimitiveGate(
        name="edge_leak",
        duration=20.0e-9,
        matrix=operator,
        hilbert_dim=subspace.full_dim,
    )
    synth = UnitarySynthesizer(
        subspace=subspace,
        primitives=[primitive],
        target=TargetUnitary(np.eye(subspace.full_dim, dtype=np.complex128), ignore_global_phase=True),
        optimize_times=False,
        seed=23,
    )
    result = synth.fit(maxiter=1)
    assert result.report["truncation"]["outside_tail_population_worst"] > 0.35
    assert result.report["truncation"]["retained_edge_population_worst"] > 0.6
    assert any("truncation" in warning.lower() or "outside the retained subspace" in warning.lower() for warning in result.report["warnings"])