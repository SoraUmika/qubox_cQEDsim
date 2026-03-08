from __future__ import annotations

import os
import time

import numpy as np
import pytest

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer
from cqed_sim.unitary_synthesis.constraints import enforce_slew_limit, evaluate_tone_spacing, piecewise_constant_samples
from cqed_sim.unitary_synthesis.sequence import GateSequence, QubitRotation


def _sleep_worker(delay_s: float) -> float:
    time.sleep(float(delay_s))
    return float(delay_s)


def test_c1_grid_time_enforcement_and_report_fields() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target = GateSequence(gates=[QubitRotation(name="t", theta=np.pi / 3, phi=0.2, duration=95e-9)], n_cav=3).unitary("pulse")

    synth = UnitarySynthesizer(
        subspace=sub,
        backend="pulse",
        gateset=["QubitRotation"],
        optimize_times=True,
        time_policy={"default": {"optimize": True, "bounds": (21.3e-9, 222.7e-9), "init": 37.4e-9}},
        time_grid={"dt": 1e-9, "mode": "soft", "lambda_grid": 1e6},
        seed=100,
    )
    result = synth.fit(target=target, init_guess="random", multistart=3, maxiter=120)

    dt = 1e-9
    durations = np.asarray(result.report["parameters"]["durations"], dtype=float)
    ticks = np.rint(durations / dt)
    assert np.allclose(durations, ticks * dt, atol=1e-15)

    tg = result.report["time_grid"]
    assert tg["mode"] in {"hard", "soft"}
    assert np.isclose(float(tg["dt"]), dt)
    assert len(tg["raw"]) == len(tg["snapped"]) == len(tg["ticks"])
    assert len(result.report["parameters"]["time_grid_per_param"]) == len(tg["ticks"])


def test_c2_slew_constraint_detects_and_projects() -> None:
    dt = 1e-9
    amplitudes = [0.0, 1.0, 0.0]
    durations = [2e-9, 2e-9, 2e-9]
    samples = piecewise_constant_samples(amplitudes, durations, dt=dt)

    s_max = 2.0e8
    penalty = enforce_slew_limit(samples, dt=dt, s_max=s_max, mode="penalty")
    assert penalty.violation_count > 0
    assert penalty.penalty > 0.0

    projected = enforce_slew_limit(samples, dt=dt, s_max=s_max, mode="project")
    assert projected.max_violation <= 1e-12
    assert projected.penalty <= penalty.penalty


def test_c3_tone_spacing_penalty_and_projection() -> None:
    domega_min = 1.0
    far = evaluate_tone_spacing([0.0, 2.0], domega_min=domega_min)
    close = evaluate_tone_spacing([0.0, 0.2], domega_min=domega_min)
    assert close.total_penalty > far.total_penalty

    proj = evaluate_tone_spacing([0.0, 0.2, 0.35], domega_min=domega_min, project=True)
    if proj.freqs_projected.size >= 2:
        assert np.min(np.diff(np.sort(proj.freqs_projected))) >= domega_min - 1e-12

    sub = Subspace.qubit_cavity_block(n_match=2)
    synth = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["SQR"],
        optimize_times=False,
        constraints={"tone_spacing": {"enabled": True, "domega_min": 1.0, "projection": True, "lambda": 1.0}},
        seed=7,
    )
    synth.sequence.gates[0].tone_freqs = [0.0, 0.1, 0.15]
    target = np.eye(sub.full_dim, dtype=np.complex128)
    result = synth.fit(target=target, multistart=1, maxiter=5)
    freqs = np.asarray(result.sequence.gates[0].tone_freqs, dtype=float)
    if freqs.size >= 2:
        assert np.min(np.diff(np.sort(freqs))) >= 1.0 - 1e-9


def test_c4_parallel_multistart_equivalence() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target = GateSequence(gates=[QubitRotation(name="t", theta=1.1, phi=-0.4, duration=120e-9)], n_cav=3).unitary("ideal")

    serial = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["QubitRotation"],
        optimize_times=True,
        time_policy={"default": {"optimize": True, "bounds": (20e-9, 200e-9), "init": 80e-9}},
        parallel={"enabled": False, "n_jobs": 1, "backend": "multiprocessing"},
        seed=314,
    ).fit(target=target, init_guess="random", multistart=4, maxiter=100)

    parallel = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["QubitRotation"],
        optimize_times=True,
        time_policy={"default": {"optimize": True, "bounds": (20e-9, 200e-9), "init": 80e-9}},
        parallel={"enabled": True, "n_jobs": 2, "backend": "multiprocessing"},
        seed=314,
    ).fit(target=target, init_guess="random", multistart=4, maxiter=100)

    assert np.isclose(serial.objective, parallel.objective, atol=1e-9)
    assert serial.report["optimizer"]["selected_start_index"] == parallel.report["optimizer"]["selected_start_index"]


@pytest.mark.slow
@pytest.mark.skipif(os.getenv("RUN_SLOW_PARALLEL_TESTS") != "1", reason="Enable with RUN_SLOW_PARALLEL_TESTS=1")
def test_c5_parallel_performance_sanity_synthetic() -> None:
    delays = [0.03] * 12

    t0 = time.perf_counter()
    _ = [_sleep_worker(d) for d in delays]
    serial_s = time.perf_counter() - t0

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t1 = time.perf_counter()
    with ctx.Pool(processes=4) as pool:
        _ = pool.map(_sleep_worker, delays)
    parallel_s = time.perf_counter() - t1

    assert parallel_s < serial_s
