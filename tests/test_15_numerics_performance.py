from __future__ import annotations

import time

import numpy as np

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_solver_step_control_respects_max_step():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    p = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 2.0, carrier=2 * np.pi * 6.0)
    c = SequenceCompiler(dt=0.002).compile([p], t_end=1.05)
    ref = simulate_sequence(m, c, m.basis_state( 0,0), {"q": "qubit"}, SimulationConfig(max_step=0.001))
    coarse = simulate_sequence(m, c, m.basis_state( 0,0), {"q": "qubit"}, SimulationConfig(max_step=0.08))
    tight = simulate_sequence(m, c, m.basis_state( 0,0), {"q": "qubit"}, SimulationConfig(max_step=0.004))
    e_coarse = abs(coarse.expectations["P_e"][-1] - ref.expectations["P_e"][-1])
    e_tight = abs(tight.expectations["P_e"][-1] - ref.expectations["P_e"][-1])
    assert e_tight < e_coarse + 1e-5
    assert (time.perf_counter() - start) < 2.0


def test_compile_cache_reuse_speeds_up_repeated_runs():
    start = time.perf_counter()
    pulses = [Pulse("q", 0.0, 1.0, _square, amp=0.3), Pulse("q", 0.8, 1.0, _square, amp=0.2, phase=0.3)]
    sc = SequenceCompiler(dt=0.001, enable_cache=True)
    t0 = time.perf_counter()
    sc.compile(pulses, t_end=2.0)
    first = time.perf_counter() - t0
    t1 = time.perf_counter()
    sc.compile(pulses, t_end=2.0)
    second = time.perf_counter() - t1
    assert second < 0.4 * first
    assert (time.perf_counter() - start) < 1.2


def test_runtime_scales_reasonably_with_dimensionality_smoke():
    start = time.perf_counter()
    runtimes = []
    for n_cav in [8, 12, 16]:
        m = DispersiveTransmonCavityModel(
            omega_c=0.0, omega_q=0.0, alpha=-2 * np.pi * 0.2, chi=2 * np.pi * 0.02, kerr=0.0, n_cav=n_cav, n_tr=2
        )
        p = Pulse("q", 0.0, 0.8, _square, amp=0.6)
        c = SequenceCompiler(dt=0.02).compile([p], t_end=1.0)
        t0 = time.perf_counter()
        simulate_sequence(m, c, m.basis_state( 0,0), {"q": "qubit"}, SimulationConfig())
        runtimes.append(time.perf_counter() - t0)
    assert runtimes[2] < 10.0 * runtimes[0]
    assert (time.perf_counter() - start) < 4.0

