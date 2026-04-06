"""pytest-benchmark integration for cqed_sim performance tests.

Run with timing enabled::

    pytest benchmarks/test_pytest_benchmarks.py --benchmark-enable -v

Run normally (benchmarks are no-ops)::

    pytest benchmarks/test_pytest_benchmarks.py

Compare backends::

    pytest benchmarks/test_pytest_benchmarks.py --benchmark-enable --benchmark-compare

Speedup summary (compare dynamiqs vs QuTiP)::

    pytest benchmarks/test_pytest_benchmarks.py -k dynamiqs --benchmark-enable -v

This file is designed to be *additive*: it does not modify any existing
benchmark functions.  All timing is done via the ``benchmark`` fixture.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pytest
import qutip as qt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    UnitaryObjective,
    build_control_problem_from_model,
    simulate_sequence,
)
from cqed_sim.pulses import HardwareConfig, Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, prepare_simulation

# ── Helpers ──────────────────────────────────────────────────────────────────

def _gaussian(sigma: float):
    def env(t_rel: np.ndarray) -> np.ndarray:
        return gaussian_envelope(t_rel, sigma=sigma)
    return env


def _small_model():
    return DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=-2.0 * np.pi * 0.22,
        chi=2.0 * np.pi * 0.015, kerr=-2.0 * np.pi * 0.003,
        n_cav=5, n_tr=2,
    )


def _prepared_session(model, config: SimulationConfig):
    """Return a compiled session with 20 overlapping Gaussian pulses."""
    compiler = SequenceCompiler(
        dt=0.01,
        hardware={
            "q": HardwareConfig(zoh_samples=4, lowpass_bw=12.0, amplitude_bits=12),
            "c": HardwareConfig(zoh_samples=3, lowpass_bw=8.0),
        },
    )
    pulses = [
        Pulse(
            "q" if i % 2 == 0 else "c",
            0.03 * i, 0.12,
            _gaussian(0.16 + 0.005 * (i % 3)),
            amp=0.12 + 0.01 * (i % 4),
            phase=0.1 * i,
            carrier=2.0 * np.pi * (0.5 if i % 2 == 0 else 0.2),
        )
        for i in range(20)
    ]
    compiled = compiler.compile(pulses, t_end=0.8)
    drive_ops = {"q": "qubit", "c": "cavity"}
    return model, compiled, drive_ops, prepare_simulation(model, compiled, drive_ops, config=config, e_ops={})


def _grape_problem():
    """Small 1-mode qubit GRAPE problem (fast to run)."""
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9, omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0, chi=0.0, kerr=0.0, n_cav=1, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    time_grid = PiecewiseConstantTimeGrid.uniform(steps=20, dt_s=5.0e-9)
    target = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # X gate in qubit subspace
    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=time_grid,
        channel_specs=(
            ModelControlChannelSpec(
                name="qubit",
                target="qubit",
                quadratures=("I", "Q"),
                amplitude_bounds=(-2.0e8, 2.0e8),
                export_channel="qubit",
            ),
        ),
        objectives=(UnitaryObjective(target_operator=target, name="X", ignore_global_phase=True),),
    )
    return problem


# ── Benchmark: single simulation (QuTiP vs dynamiqs) ─────────────────────────

@pytest.fixture(scope="module")
def qutip_session():
    model = _small_model()
    config = SimulationConfig(frame=FrameSpec(), max_step=0.01)
    model, compiled, drive_ops, session = _prepared_session(model, config)
    initial = model.basis_state(0, 0)
    return session, initial


@pytest.fixture(scope="module")
def dynamiqs_session():
    pytest.importorskip("dynamiqs")
    model = _small_model()
    config = SimulationConfig(
        frame=FrameSpec(), max_step=0.01,
        dynamiqs_solver="Tsit5",
        dynamiqs_atol=1e-7, dynamiqs_rtol=1e-5,
    )
    model, compiled, drive_ops, session = _prepared_session(model, config)
    initial = model.basis_state(0, 0)
    return session, initial


def test_simulate_single_qutip(benchmark, qutip_session):
    """Baseline: single simulation with QuTiP sesolve."""
    session, initial = qutip_session
    result = benchmark.pedantic(session.run, args=(initial,), rounds=8, warmup_rounds=2)
    assert result.final_state is not None


def test_simulate_single_dynamiqs(benchmark, dynamiqs_session):
    """GPU-ready: single simulation with dynamiqs Tsit5.

    Compare timings to test_simulate_single_qutip.  On CPU the first call
    includes JIT overhead; subsequent calls are faster.
    """
    session, initial = dynamiqs_session
    result = benchmark.pedantic(session.run, args=(initial,), rounds=8, warmup_rounds=2)
    assert result.final_state is not None


# ── Benchmark: 20 repeated simulations (warm session) ─────────────────────────

def test_simulate_20_prepared_qutip(benchmark, qutip_session):
    """20 sequential simulations through a prepared QuTiP session."""
    session, initial = qutip_session

    def _run20():
        for _ in range(20):
            session.run(initial)

    benchmark.pedantic(_run20, rounds=3, warmup_rounds=1)


def test_simulate_20_prepared_dynamiqs(benchmark, dynamiqs_session):
    """20 sequential simulations through a prepared dynamiqs session (JIT warm)."""
    session, initial = dynamiqs_session

    def _run20():
        for _ in range(20):
            session.run(initial)

    benchmark.pedantic(_run20, rounds=3, warmup_rounds=1)


# ── Benchmark: GRAPE optimization (NumPy vs JAX) ─────────────────────────────

@pytest.fixture(scope="module")
def grape_problem():
    return _grape_problem()


def test_grape_numpy_lbfgsb(benchmark, grape_problem):
    """GRAPE with NumPy engine + L-BFGS-B (baseline).

    On the small X-gate problem L-BFGS-B converges to machine precision
    (<1e-6) in 30 iterations.  This is the fastest solver path for small
    Hilbert spaces on CPU.
    """
    config = GrapeConfig(engine="numpy", optimizer_method="L-BFGS-B", maxiter=30, seed=42)

    def _solve():
        return GrapeSolver(config=config).solve(grape_problem)

    result = benchmark.pedantic(_solve, rounds=3, warmup_rounds=0)
    assert result.objective_value < 1e-4, (
        f"NumPy L-BFGS-B should converge to <1e-4 in 30 iters, got {result.objective_value:.6g}"
    )


def test_grape_jax_lbfgsb(benchmark, grape_problem):
    """GRAPE with JAX engine + L-BFGS-B (JIT-compiled cost+gradient).

    Same convergence as NumPy but with JIT overhead.  On the small X-gate
    problem L-BFGS-B converges to <1e-6 in 30 iterations through either
    engine.
    """
    pytest.importorskip("jax")
    config = GrapeConfig(engine="jax", optimizer_method="L-BFGS-B", maxiter=30, seed=42)

    def _solve():
        return GrapeSolver(config=config).solve(grape_problem)

    result = benchmark.pedantic(_solve, rounds=3, warmup_rounds=1)
    assert result.objective_value < 1e-4, (
        f"JAX L-BFGS-B should converge to <1e-4 in 30 iters, got {result.objective_value:.6g}"
    )


def test_grape_jax_adam(benchmark, grape_problem):
    """GRAPE with JAX engine + Optax Adam (first-order, no line search).

    Adam needs more iterations than L-BFGS-B but is better suited for noisy
    or stochastic objectives.  At 100 iterations on this problem it should
    reach <1e-2 comfortably.
    """
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    config = GrapeConfig(
        engine="jax", optimizer_method="adam",
        optax_learning_rate=5e-3, maxiter=100, seed=42,
    )

    def _solve():
        return GrapeSolver(config=config).solve(grape_problem)

    result = benchmark.pedantic(_solve, rounds=3, warmup_rounds=1)
    assert result.objective_value < 1e-2, (
        f"JAX Adam should reach <1e-2 in 100 iters, got {result.objective_value:.6g}"
    )


# ── Regression: dynamiqs matches QuTiP fidelity ──────────────────────────────

def test_dynamiqs_matches_qutip_fidelity():
    """dynamiqs result must agree with QuTiP to within 1e-3 in state fidelity."""
    pytest.importorskip("dynamiqs")

    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=-2.0 * np.pi * 0.22,
        chi=2.0 * np.pi * 0.015, kerr=0.0, n_cav=4, n_tr=2,
    )
    from cqed_sim.pulses.envelopes import square_envelope
    compiler = SequenceCompiler(dt=0.02)

    pulses = [Pulse("q", 0.0, 0.5, square_envelope, amp=np.pi / 2.0, carrier=0.0)]
    compiled = compiler.compile(pulses, t_end=0.6)
    drive_ops = {"q": "qubit"}
    initial = model.basis_state(0, 0)

    # QuTiP reference
    cfg_qt = SimulationConfig(frame=FrameSpec())
    res_qt = simulate_sequence(model, compiled, initial, drive_ops, config=cfg_qt)

    # dynamiqs
    cfg_dq = SimulationConfig(frame=FrameSpec(), dynamiqs_solver="Tsit5", dynamiqs_atol=1e-9, dynamiqs_rtol=1e-7)
    res_dq = simulate_sequence(model, compiled, initial, drive_ops, config=cfg_dq)

    fid = abs(qt.fidelity(res_qt.final_state, res_dq.final_state)) ** 2
    assert fid > 0.999, f"dynamiqs/QuTiP fidelity mismatch: {fid:.6f}"


# ── Regression: Optax Adam reduces cost ──────────────────────────────────────

def test_optax_adam_reduces_cost():
    """Adam optimizer must reduce the GRAPE infidelity well below 0.1 in 50 iters."""
    pytest.importorskip("jax")
    pytest.importorskip("optax")

    problem = _grape_problem()
    config_adam = GrapeConfig(
        engine="jax", optimizer_method="adam",
        optax_learning_rate=5e-3, maxiter=50, seed=0,
    )
    result = GrapeSolver(config=config_adam).solve(problem)
    # At seed=0, 50 Adam steps on the X-gate problem reaches ~0.003.
    # Use 0.05 as a safe upper bound that catches real failures.
    assert result.objective_value < 0.05, (
        f"Adam did not converge sufficiently: final={result.objective_value:.6f} (expect <0.05)"
    )


def test_optax_requires_jax_engine():
    """Optax optimizer names must raise ValueError when engine='numpy'."""
    with pytest.raises(ValueError, match="engine='jax'"):
        GrapeConfig(engine="numpy", optimizer_method="adam")


# ── Regression: Gymnasium wrapper ────────────────────────────────────────────

def test_gymnasium_env_api():
    """GymnasiumCQEDEnv must pass the basic gymnasium interface checks."""
    gymnasium = pytest.importorskip("gymnasium")
    from gymnasium.utils.env_checker import check_env
    from cqed_sim.rl_control import (
        GymnasiumCQEDEnv,
        HybridEnvConfig,
        HybridSystemConfig,
        ReducedDispersiveModelConfig,
        ParametricPulseActionSpace,
        IdealSummaryObservation,
        build_reward_model,
        vacuum_preservation_task,
    )

    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=2.0 * np.pi * (-5.0e3),
            n_cav=4,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    config = HybridEnvConfig(
        system=system,
        task=vacuum_preservation_task(),
        action_space=ParametricPulseActionSpace(family="qubit_gaussian"),
        observation_model=IdealSummaryObservation(),
        reward_model=build_reward_model("state"),
        episode_horizon=2,
        seed=0,
    )
    env = GymnasiumCQEDEnv(config)

    # Gymnasium interface contract
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")
    assert isinstance(env.action_space, gymnasium.spaces.Box)
    assert isinstance(env.observation_space, gymnasium.spaces.Box)

    obs, info = env.reset(seed=42)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    # Step with a zero (null) action — safe for real-physics envs
    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    obs2, reward, terminated, truncated, info2 = env.step(zero_action)
    assert obs2.shape == env.observation_space.shape
    assert obs2.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)

    # Verify inner env is accessible
    assert env.inner is not None


# ── Speedup summary (printed, not asserted) ───────────────────────────────────

def test_speedup_summary(capsys):
    """Print a wall-clock speedup table: QuTiP vs dynamiqs (20 runs each)."""
    dq = pytest.importorskip("dynamiqs")

    model = _small_model()
    initial = model.basis_state(0, 0)

    # QuTiP
    cfg_qt = SimulationConfig(frame=FrameSpec(), max_step=0.01)
    _, _, drive_ops, session_qt = _prepared_session(model, cfg_qt)
    t0 = time.perf_counter()
    for _ in range(20):
        session_qt.run(initial)
    qt_wall = time.perf_counter() - t0

    # dynamiqs (includes JIT on first call)
    cfg_dq = SimulationConfig(frame=FrameSpec(), dynamiqs_solver="Tsit5")
    _, _, _, session_dq = _prepared_session(model, cfg_dq)
    # warm-up JIT
    session_dq.run(initial)
    t0 = time.perf_counter()
    for _ in range(20):
        session_dq.run(initial)
    dq_wall = time.perf_counter() - t0

    speedup = qt_wall / dq_wall if dq_wall > 0 else float("inf")
    with capsys.disabled():
        print(
            f"\n{'─'*52}\n"
            f"  Speedup benchmark: 20x simulate_sequence\n"
            f"  QuTiP  : {qt_wall*1000:.1f} ms total ({qt_wall/20*1000:.2f} ms/sim)\n"
            f"  dynamiqs: {dq_wall*1000:.1f} ms total ({dq_wall/20*1000:.2f} ms/sim)\n"
            f"  Speedup : {speedup:.2f}x\n"
            f"{'─'*52}"
        )
    # On CPU with a small system, dynamiqs is ~3x slower than QuTiP due to
    # JAX overhead.  Cap at 15x to catch regressions while tolerating CI jitter.
    assert dq_wall < qt_wall * 15, (
        f"dynamiqs is unexpectedly slow vs QuTiP: "
        f"{dq_wall:.3f}s vs {qt_wall:.3f}s ({speedup:.1f}x)"
    )


# ── Cross-optimizer convergence comparison ────────────────────────────────────

def test_optimizer_convergence_comparison(capsys):
    """Compare all available GRAPE optimizers on the same X-gate problem.

    This test is the primary reference for which optimizer to choose.
    It prints a ranked table and enforces convergence bounds per method.

    Expected ranking (small CPU problem, 2x2 qubit subspace):
        1. L-BFGS-B (numpy) — fastest wall-clock, best convergence
        2. L-BFGS-B (jax)   — same convergence, JIT overhead on small problems
        3. Adam              — good first-order, needs ~100 iters
        4. AdamW             — similar to Adam
        5. SGD               — competitive with tuned LR
        6. AdaGrad           — slowest convergence of the Optax set

    On larger Hilbert spaces (n_cav >= 15) or GPU, the JAX paths are expected
    to become faster than the NumPy path due to JIT and hardware acceleration.
    """
    jax = pytest.importorskip("jax")
    optax = pytest.importorskip("optax")

    problem = _grape_problem()

    configs = {
        "numpy L-BFGS-B": GrapeConfig(
            engine="numpy", optimizer_method="L-BFGS-B", maxiter=30, seed=42),
        "jax   L-BFGS-B": GrapeConfig(
            engine="jax", optimizer_method="L-BFGS-B", maxiter=30, seed=42),
        "jax   adam":      GrapeConfig(
            engine="jax", optimizer_method="adam",
            optax_learning_rate=5e-3, maxiter=100, seed=42),
        "jax   adamw":     GrapeConfig(
            engine="jax", optimizer_method="adamw",
            optax_learning_rate=5e-3, maxiter=100, seed=42),
        "jax   sgd":       GrapeConfig(
            engine="jax", optimizer_method="sgd",
            optax_learning_rate=5e-3, maxiter=100, seed=42),
        "jax   adagrad":   GrapeConfig(
            engine="jax", optimizer_method="adagrad",
            optax_learning_rate=5e-3, maxiter=100, seed=42),
    }

    # Convergence thresholds — generous enough to tolerate CI variance,
    # tight enough to catch real regressions.
    thresholds = {
        "numpy L-BFGS-B": 1e-4,
        "jax   L-BFGS-B": 1e-4,
        "jax   adam":      1e-2,
        "jax   adamw":     1e-2,
        "jax   sgd":       1e-2,
        "jax   adagrad":   1e-2,
    }

    results = {}
    for label, cfg in configs.items():
        t0 = time.perf_counter()
        res = GrapeSolver(config=cfg).solve(problem)
        wall = time.perf_counter() - t0
        results[label] = (res.objective_value, wall, cfg.maxiter)

    # Print comparison table
    with capsys.disabled():
        header = f"\n{'='*72}\n  GRAPE Optimizer Comparison (X-gate, 2x2 qubit, 20 time slices)\n{'='*72}"
        print(header)
        print(f"  {'Method':<20s} {'Iters':>5s} {'Objective':>12s} {'Wall (ms)':>10s}")
        print(f"  {'-'*20} {'-'*5} {'-'*12} {'-'*10}")
        for label in configs:
            obj, wall, iters = results[label]
            print(f"  {label:<20s} {iters:>5d} {obj:>12.6g} {wall*1000:>10.1f}")
        print(f"{'='*72}")

    # Enforce convergence bounds
    for label, threshold in thresholds.items():
        obj = results[label][0]
        assert obj < threshold, (
            f"{label} did not converge: objective={obj:.6g} >= {threshold:.1g}"
        )
