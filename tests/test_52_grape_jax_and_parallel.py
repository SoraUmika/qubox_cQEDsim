"""Tests for the JAX engine and thread-based parallel multi-start in GRAPE."""

from __future__ import annotations

import numpy as np
import pytest

from cqed_sim import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeMultistartConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    UnitaryObjective,
    build_control_problem_from_model,
    solve_grape,
    solve_grape_multistart,
)

try:
    import jax  # noqa: F401

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False


def _rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _qubit_only_cavity_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return model, frame


def _rotation_problem() -> tuple[DispersiveTransmonCavityModel, FrameSpec, ControlProblem]:
    model, frame = _qubit_only_cavity_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
                export_channel="qubit",
            ),
        ),
        objectives=(
            UnitaryObjective(
                target_operator=_rotation_y(np.pi / 2.0),
                ignore_global_phase=True,
                name="ry_pi_over_two",
            ),
        ),
    )
    return model, frame, problem


# ---------------------------------------------------------------------------
# JAX engine tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_engine_reaches_same_fidelity_as_numpy() -> None:
    """JAX engine achieves comparable fidelity to the NumPy engine."""
    _model, _frame, problem = _rotation_problem()

    result_np = solve_grape(
        problem,
        config=GrapeConfig(maxiter=80, seed=11, random_scale=0.25, engine="numpy"),
        initial_schedule=np.array([[8.0e6]], dtype=float),
    )
    result_jax = solve_grape(
        problem,
        config=GrapeConfig(maxiter=80, seed=11, random_scale=0.25, engine="jax"),
        initial_schedule=np.array([[8.0e6]], dtype=float),
    )

    assert result_np.success
    assert result_jax.success
    assert result_jax.metrics["nominal_fidelity"] > 0.999
    # Both should reach high fidelity; the exact values may differ slightly
    # due to different gradient computation paths (expm_frechet vs JAX autodiff).
    np_fid = result_np.metrics["nominal_fidelity"]
    jax_fid = result_jax.metrics["nominal_fidelity"]
    assert abs(np_fid - jax_fid) < 0.01, (
        f"NumPy fidelity {np_fid:.6f} vs JAX fidelity {jax_fid:.6f} differ too much"
    )


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_engine_unitary_fidelity_metric_reported() -> None:
    """JAX engine reports exact_unitary_fidelity in system metrics."""
    _model, _frame, problem = _rotation_problem()

    result = solve_grape(
        problem,
        config=GrapeConfig(maxiter=40, seed=7, engine="jax"),
        initial_schedule=np.array([[8.0e6]], dtype=float),
    )

    objectives = result.system_metrics[0]["objectives"]
    assert len(objectives) == 1
    assert "exact_unitary_fidelity" in objectives[0]
    assert objectives[0]["exact_unitary_fidelity"] > 0.9


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_propagators_module_is_available() -> None:
    """The propagators_jax module reports availability correctly."""
    from cqed_sim.optimal_control.propagators_jax import is_available

    assert is_available() is True


# ---------------------------------------------------------------------------
# Thread-based multi-start tests
# ---------------------------------------------------------------------------


def test_thread_multistart_returns_sorted_results() -> None:
    """Thread-based multi-start returns results sorted by objective value."""
    _model, _frame, problem = _rotation_problem()

    results = solve_grape_multistart(
        problem,
        config=GrapeConfig(maxiter=30, seed=0, random_scale=0.3),
        multistart_config=GrapeMultistartConfig(
            n_restarts=3, max_workers=2, mp_context="thread",
        ),
    )

    assert len(results) == 3
    # Sorted best-first
    for i in range(len(results) - 1):
        assert results[i].objective_value <= results[i + 1].objective_value


def test_thread_multistart_single_worker_is_serial() -> None:
    """max_workers=1 executes serially regardless of mp_context."""
    _model, _frame, problem = _rotation_problem()

    results = solve_grape_multistart(
        problem,
        config=GrapeConfig(maxiter=20, seed=42),
        multistart_config=GrapeMultistartConfig(
            n_restarts=2, max_workers=1, mp_context="thread",
        ),
    )

    assert len(results) == 2
    for r in results:
        assert r.metrics["nominal_fidelity"] > 0.0


@pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")
def test_jax_engine_thread_multistart() -> None:
    """JAX engine + thread parallelism works together."""
    _model, _frame, problem = _rotation_problem()

    results = solve_grape_multistart(
        problem,
        config=GrapeConfig(maxiter=30, seed=0, random_scale=0.2, engine="jax"),
        multistart_config=GrapeMultistartConfig(
            n_restarts=2, max_workers=2, mp_context="thread",
        ),
    )

    assert len(results) == 2
    assert results[0].metrics["nominal_fidelity"] > 0.9


# ---------------------------------------------------------------------------
# GrapeConfig validation
# ---------------------------------------------------------------------------


def test_grape_config_rejects_invalid_engine() -> None:
    """GrapeConfig raises on unknown engine names."""
    with pytest.raises(ValueError, match="engine"):
        GrapeConfig(engine="cuda")


def test_grape_multistart_config_rejects_invalid_context() -> None:
    """GrapeMultistartConfig raises on unknown mp_context names."""
    with pytest.raises(ValueError, match="mp_context"):
        GrapeMultistartConfig(mp_context="invalid_context")


def test_grape_multistart_config_default_is_thread() -> None:
    """Default mp_context is now 'thread'."""
    cfg = GrapeMultistartConfig()
    assert cfg.mp_context == "thread"
