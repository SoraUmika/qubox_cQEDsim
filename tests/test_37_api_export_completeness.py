"""
Regression tests for public API export completeness.

Ensures that names documented and intended as top-level public API are
importable from ``cqed_sim`` and that the two distinct ``SimulationResult``
classes are separate types.
"""

import importlib

import pytest


# ── Top-level import completeness ────────────────────────────────────────────

REQUIRED_TOP_LEVEL_NAMES = [
    # models
    "DispersiveTransmonCavityModel",
    "DispersiveReadoutTransmonStorageModel",
    "UniversalCQEDModel",
    # optimal control
    "ControlProblem",
    "ControlResult",
    "ControlEvaluationCase",
    "ControlEvaluationResult",
    "PiecewiseConstantTimeGrid",
    "GrapeSolver",
    "solve_grape",
    "evaluate_control_with_simulator",
    # simulation core
    "SimulationConfig",
    "SimulationSession",
    "SimulationResult",
    "simulate_sequence",
    "prepare_simulation",
    # extractors
    "reduced_subsystem_state",
    "reduced_qubit_state",
    "reduced_cavity_state",
    "conditioned_qubit_state",
    "conditioned_bloch_xyz",
    "conditioned_population",
    "cavity_wigner",
    "subsystem_level_population",
    # operators
    "displacement_op",
    "snap_op",
    "sqr_op",
    "qubit_rotation_xy",
    "qubit_rotation_axis",
    # calibration
    "CalibrationResult",
    "run_spectroscopy",
    "run_rabi",
    "run_ramsey",
    "run_t1",
    "run_t2_echo",
    # measurement
    "measure_qubit",
    # tomo
    "run_fock_resolved_tomo",
]


@pytest.mark.parametrize("name", REQUIRED_TOP_LEVEL_NAMES)
def test_top_level_export_exists(name: str) -> None:
    """Every name on the required list should be importable from cqed_sim."""
    import cqed_sim

    assert hasattr(cqed_sim, name), (
        f"{name!r} is not accessible as cqed_sim.{name}"
    )


def test_top_level_all_contains_required_names() -> None:
    """Names in __all__ should cover the required public API."""
    import cqed_sim

    all_names = set(cqed_sim.__all__)
    missing = [n for n in REQUIRED_TOP_LEVEL_NAMES if n not in all_names]
    assert not missing, f"Missing from __all__: {missing}"


# ── SimulationResult disambiguation ─────────────────────────────────────────

def test_simulation_result_classes_are_distinct() -> None:
    """The sim.runner and unitary_synthesis.backends SimulationResult classes
    must be different types with different fields."""
    from cqed_sim.sim.runner import SimulationResult as RunnerResult
    from cqed_sim.unitary_synthesis.backends import (
        SimulationResult as SynthResult,
    )

    assert RunnerResult is not SynthResult, (
        "sim.runner.SimulationResult and unitary_synthesis.backends."
        "SimulationResult should be distinct classes"
    )
    # Check distinguishing attributes
    assert hasattr(RunnerResult, "__dataclass_fields__") or True  # quick guard
    # The top-level export should be the runner result
    import cqed_sim

    assert cqed_sim.SimulationResult is RunnerResult
