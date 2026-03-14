from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import pure_dephasing_time_from_t1_t2
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def test_t1_relaxation_matches_exponential():
    start = time.perf_counter()
    gamma1 = 0.18
    t1 = 1.0 / gamma1
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    b = m.operators()["b"]
    c = SequenceCompiler(dt=0.05).compile([], t_end=8.0)
    r = simulate_sequence(m, c, m.basis_state( 1,0), {}, SimulationConfig(), c_ops=[np.sqrt(gamma1) * b])
    pe = np.asarray(r.expectations["P_e"])
    t = c.tlist
    fit = np.polyfit(t[pe > 1e-4], np.log(pe[pe > 1e-4]), 1)[0]
    t1_fit = -1.0 / fit
    assert np.isclose(t1_fit, t1, rtol=0.12, atol=0.2)
    assert (time.perf_counter() - start) < 1.3


def test_tphi_dephasing_matches_ramsey_envelope():
    start = time.perf_counter()
    gphi = 0.12
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    n_q = m.operators()["n_q"]
    c = SequenceCompiler(dt=0.05).compile([], t_end=8.0)
    psi = (m.basis_state( 0,0) + m.basis_state( 1,0)).unit()
    r = simulate_sequence(m, c, psi, {}, SimulationConfig(store_states=True), c_ops=[np.sqrt(gphi) * n_q])
    coh = np.array([abs(qt.ptrace(s, 0)[0, 1]) for s in r.states], dtype=float)
    t = c.tlist
    fit = np.polyfit(t[coh > 1e-5], np.log(coh[coh > 1e-5] / coh[0]), 1)[0]
    assert np.isclose(-fit, gphi / 2.0, rtol=0.2, atol=0.03)
    assert (time.perf_counter() - start) < 1.5


def test_chi_with_t1_does_not_break_trace_or_positivity():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=-2 * np.pi * 0.2, chi=2 * np.pi * 0.03, kerr=0.0, n_cav=6, n_tr=3
    )
    b = m.operators()["b"]
    psi = (m.basis_state( 0,1) + m.basis_state( 1,1)).unit()
    c = SequenceCompiler(dt=0.05).compile([], t_end=5.0)
    r = simulate_sequence(m, c, psi, {}, SimulationConfig(store_states=True), c_ops=[np.sqrt(0.12) * b])
    for s in r.states[::6]:
        assert abs(s.tr() - 1.0) < 1e-8
        ev = np.linalg.eigvalsh(s.full())
        assert np.min(ev) > -1e-8
    assert (time.perf_counter() - start) < 1.6


def test_pure_dephasing_time_helper_returns_none_when_extra_rate_vanishes():
    assert pure_dephasing_time_from_t1_t2(t1_s=12.0e-6, t2_s=24.0e-6) is None

    inferred = pure_dephasing_time_from_t1_t2(t1_s=20.0e-6, t2_s=8.0e-6)
    expected = 1.0 / (1.0 / (8.0e-6) - 1.0 / (2.0 * 20.0e-6))
    assert inferred is not None
    assert np.isclose(inferred, expected, rtol=1.0e-12, atol=0.0)

