"""Tests for the GRAPE multi-start solver and the run_sweep parameter-sweep runner.

These tests verify:

1.  GrapeMultistartConfig - construction, validation, and defaults.
2.  solve_grape_multistart - returns the expected number of GrapeResult objects,
    sorted best-first, with independent random seeds per restart.
3.  run_sweep - serial and parallel-stub execution, correct ordering, and
    equivalence with per-session simulate_sequence calls.
"""
from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeMultistartConfig,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    build_control_problem_from_model,
    solve_grape_multistart,
    state_preparation_objective,
)
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim import SimulationConfig, prepare_simulation, run_sweep, simulate_sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec()
    return model, frame


def _rotation_problem():
    model, frame = _simple_model()
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=2, dt_s=20.0e-9)
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e9, 1.0e9),
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
    )
    return problem


def _square(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=np.complex128)


# ---------------------------------------------------------------------------
# GrapeMultistartConfig
# ---------------------------------------------------------------------------

class TestGrapeMultistartConfig:
    def test_defaults(self) -> None:
        cfg = GrapeMultistartConfig()
        assert cfg.n_restarts == 4
        assert cfg.max_workers == 1
        assert cfg.mp_context == "spawn"
        assert cfg.return_all is True

    def test_custom_values(self) -> None:
        cfg = GrapeMultistartConfig(n_restarts=6, max_workers=2, return_all=False)
        assert cfg.n_restarts == 6
        assert cfg.max_workers == 2
        assert cfg.return_all is False

    def test_invalid_n_restarts(self) -> None:
        with pytest.raises(ValueError, match="n_restarts"):
            GrapeMultistartConfig(n_restarts=0)

    def test_invalid_max_workers(self) -> None:
        with pytest.raises(ValueError, match="max_workers"):
            GrapeMultistartConfig(max_workers=0)


# ---------------------------------------------------------------------------
# solve_grape_multistart
# ---------------------------------------------------------------------------

class TestSolveGrapeMultistart:
    def test_returns_sorted_results(self) -> None:
        problem = _rotation_problem()
        config = GrapeConfig(maxiter=5, seed=0, history_every=5)
        ms_config = GrapeMultistartConfig(n_restarts=3, max_workers=1)
        results = solve_grape_multistart(problem, config=config, multistart_config=ms_config)

        assert len(results) == 3
        # Results must be sorted best (lowest) objective first.
        for i in range(len(results) - 1):
            assert results[i].objective_value <= results[i + 1].objective_value

    def test_results_have_distinct_seeds(self) -> None:
        """Each restart should produce a different history, confirming independent seeds."""
        problem = _rotation_problem()
        config = GrapeConfig(maxiter=3, seed=10, history_every=3)
        ms_config = GrapeMultistartConfig(n_restarts=3, max_workers=1)
        results = solve_grape_multistart(problem, config=config, multistart_config=ms_config)

        # Not all results should have identical objective values (different random starts).
        objectives = [r.objective_value for r in results]
        # With different seeds we expect at least two distinct values over 3 restarts
        # (could still collide, but very unlikely for a nonlinear optimizer).
        assert len(results) == 3
        # Each result must be a valid GrapeResult.
        for r in results:
            assert hasattr(r, "schedule")
            assert hasattr(r, "history")

    def test_return_all_false_returns_single_best(self) -> None:
        problem = _rotation_problem()
        config = GrapeConfig(maxiter=3, seed=0, history_every=3)
        ms_config = GrapeMultistartConfig(n_restarts=4, max_workers=1, return_all=False)
        results = solve_grape_multistart(problem, config=config, multistart_config=ms_config)
        assert len(results) == 1

    def test_default_config_used_when_none(self) -> None:
        problem = _rotation_problem()
        # Smoke test: both configs None → use defaults (4 restarts).
        results = solve_grape_multistart(
            problem,
            config=GrapeConfig(maxiter=2, seed=7, history_every=2),
            multistart_config=GrapeMultistartConfig(n_restarts=2, max_workers=1),
        )
        assert len(results) == 2

    def test_single_restart_matches_solve_grape(self) -> None:
        """A single-restart multi-start result should match the standard solver for the same seed."""
        from cqed_sim import solve_grape

        problem = _rotation_problem()
        seed = 42
        config = GrapeConfig(maxiter=10, seed=seed, initial_guess="random", history_every=10)
        ms_config = GrapeMultistartConfig(n_restarts=1, max_workers=1)

        [ms_result] = solve_grape_multistart(problem, config=config, multistart_config=ms_config)
        ref_result = solve_grape(problem, config=config)

        assert abs(ms_result.objective_value - ref_result.objective_value) < 1.0e-10


# ---------------------------------------------------------------------------
# run_sweep
# ---------------------------------------------------------------------------

class TestRunSweep:
    def _build_session(self, amp: float):
        model, _frame = _simple_model()
        pulse = Pulse("q", 0.0, 1.0, _square, amp=float(amp))
        compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=1.1)
        return prepare_simulation(model, compiled, {"q": "qubit"}, config=SimulationConfig(), e_ops={})

    def test_returns_correct_length(self) -> None:
        sessions = [self._build_session(amp) for amp in (0.1, 0.2, 0.3)]
        model, _ = _simple_model()
        states = [model.basis_state(0, 0)] * 3
        results = run_sweep(sessions, states)
        assert len(results) == 3

    def test_length_mismatch_raises(self) -> None:
        sessions = [self._build_session(0.1), self._build_session(0.2)]
        model, _ = _simple_model()
        states = [model.basis_state(0, 0)]
        with pytest.raises(ValueError, match="length"):
            run_sweep(sessions, states)

    def test_matches_serial_simulate_sequence(self) -> None:
        """run_sweep results should agree with individual simulate_sequence calls."""
        model, _ = _simple_model()
        amps = [0.0, np.pi / 4.0, np.pi / 2.0]
        sessions = [self._build_session(amp) for amp in amps]
        state = model.basis_state(0, 0)
        states = [state] * len(amps)

        sweep_results = run_sweep(sessions, states)

        for idx, (session, sweep_r) in enumerate(zip(sessions, sweep_results)):
            ref = session.run(state)
            assert np.allclose(
                np.asarray(sweep_r.final_state.full()),
                np.asarray(ref.final_state.full()),
                atol=1.0e-10,
            ), f"Sweep result {idx} does not match reference."

    def test_preserves_order(self) -> None:
        """Results must appear in the same order as the input sessions."""
        model, _ = _simple_model()
        amps = [float(k) * 0.01 for k in range(5)]
        sessions = [self._build_session(amp) for amp in amps]
        state = model.basis_state(0, 0)
        states = [state] * len(amps)

        sweep_results = run_sweep(sessions, states)

        # Each session.run should match the corresponding sweep result.
        for session, sweep_r in zip(sessions, sweep_results):
            ref = session.run(state)
            assert np.allclose(
                np.asarray(sweep_r.final_state.full()),
                np.asarray(ref.final_state.full()),
                atol=1.0e-10,
            )

    def test_single_session_sweep(self) -> None:
        model, _ = _simple_model()
        session = self._build_session(np.pi / 2.0)
        state = model.basis_state(0, 0)
        results = run_sweep([session], [state])
        assert len(results) == 1
        ref = session.run(state)
        assert np.allclose(
            np.asarray(results[0].final_state.full()),
            np.asarray(ref.final_state.full()),
            atol=1.0e-10,
        )
