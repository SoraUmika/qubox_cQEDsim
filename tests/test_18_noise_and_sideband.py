from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.ideal_gates import beamsplitter_unitary
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_qubit_t1_decay_matches_exponential():
    start = time.perf_counter()
    t1 = 6.0
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    c = SequenceCompiler(dt=0.05).compile([], t_end=8.0)
    r = simulate_sequence(m, c, m.basis_state(0, 1), {}, SimulationConfig(), noise=NoiseSpec(t1=t1))
    pe = np.asarray(r.expectations["P_e"])
    t = c.tlist
    fit = np.polyfit(t[pe > 1e-4], np.log(pe[pe > 1e-4]), 1)[0]
    t1_fit = -1.0 / fit
    assert np.isclose(t1_fit, t1, rtol=0.12, atol=0.35)
    assert (time.perf_counter() - start) < 1.5


def test_qubit_tphi_dephasing_ramsey_envelope():
    start = time.perf_counter()
    tphi = 5.0
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    psi = (m.basis_state(0, 0) + m.basis_state(0, 1)).unit()
    c = SequenceCompiler(dt=0.05).compile([], t_end=8.0)
    r = simulate_sequence(m, c, psi, {}, SimulationConfig(store_states=True), noise=NoiseSpec(tphi=tphi))
    coh = np.array([abs(qt.ptrace(s, 1)[0, 1]) for s in r.states], dtype=float)
    t = c.tlist
    fit = np.polyfit(t[coh > 1e-5], np.log(coh[coh > 1e-5] / coh[0]), 1)[0]
    assert np.isclose(-fit, 1.0 / tphi, rtol=0.15, atol=0.03)
    assert (time.perf_counter() - start) < 1.6


def test_cavity_ringdown_matches_kappa():
    start = time.perf_counter()
    kappa = 0.25
    alpha = 1.1
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=14, n_tr=2)
    psi = qt.tensor(qt.coherent(m.n_cav, alpha), qt.basis(m.n_tr, 0))
    c = SequenceCompiler(dt=0.05).compile([], t_end=8.0)
    r = simulate_sequence(m, c, psi, {}, SimulationConfig(), noise=NoiseSpec(kappa=kappa, nth=0.0))
    n = np.asarray(r.expectations["n_c"])
    t = c.tlist
    fit = np.polyfit(t[n > 1e-4], np.log(n[n > 1e-4]), 1)[0]
    assert np.isclose(-fit, kappa, rtol=0.12, atol=0.03)
    assert (time.perf_counter() - start) < 1.8


def test_thermal_steady_state_matches_nth():
    start = time.perf_counter()
    kappa = 0.3
    nth = 0.6
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=16, n_tr=2)
    c = SequenceCompiler(dt=0.1).compile([], t_end=30.0)
    r = simulate_sequence(m, c, m.basis_state(0, 0), {}, SimulationConfig(), noise=NoiseSpec(kappa=kappa, nth=nth))
    assert np.isclose(r.expectations["n_c"][-1], nth, rtol=0.15, atol=0.08)
    assert (time.perf_counter() - start) < 2.0


def test_trace_preserved_and_positivity_bounded_open_system():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=-2 * np.pi * 0.2, chi=2 * np.pi * 0.02, kerr=0.0, n_cav=8, n_tr=3
    )
    psi = (m.basis_state(1, 0) + 1j * m.basis_state(2, 1)).unit()
    c = SequenceCompiler(dt=0.05).compile([], t_end=5.0)
    r = simulate_sequence(m, c, psi, {}, SimulationConfig(store_states=True), noise=NoiseSpec(t1=8.0, tphi=7.0, kappa=0.15, nth=0.1))
    for s in r.states[::5]:
        assert abs(s.tr() - 1.0) < 1e-8
        ev = np.linalg.eigvalsh(s.full())
        assert np.min(ev) > -1e-8
    assert (time.perf_counter() - start) < 2.2


def test_sideband_swap_one_photon_pi_pulse():
    start = time.perf_counter()
    g = 0.35
    t_pi = np.pi / (2 * g)
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)
    p = Pulse("sb", 0.0, t_pi, _square, amp=g)
    c = SequenceCompiler(dt=0.01).compile([p], t_end=t_pi)
    r = simulate_sequence(m, c, m.basis_state(0, 1), {"sb": "sideband"}, SimulationConfig(frame=FrameSpec()))
    p_g1 = abs(m.basis_state(1, 0).overlap(r.final_state)) ** 2
    p_e0 = abs(m.basis_state(0, 1).overlap(r.final_state)) ** 2
    assert p_g1 > 0.98
    assert p_e0 < 0.02
    assert (time.perf_counter() - start) < 1.1


def test_sideband_oscillation_frequency_matches_g():
    start = time.perf_counter()
    g = 0.22
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)
    p = Pulse("sb", 0.0, 8.0, _square, amp=g)
    c = SequenceCompiler(dt=0.02).compile([p], t_end=8.0)
    r = simulate_sequence(m, c, m.basis_state(0, 1), {"sb": "sideband"}, SimulationConfig(store_states=True))
    pg1 = np.array([abs(m.basis_state(1, 0).overlap(s)) ** 2 for s in r.states], dtype=float)
    pred = np.sin(g * c.tlist) ** 2
    assert np.max(np.abs(pg1 - pred)) < 0.05
    assert (time.perf_counter() - start) < 1.5


def test_sideband_selectivity_breaks_with_detuning():
    start = time.perf_counter()
    g = 0.25
    t_pi = np.pi / (2 * g)
    p = Pulse("sb", 0.0, t_pi, _square, amp=g)
    m_res = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)
    m_det = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.9, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)
    c = SequenceCompiler(dt=0.01).compile([p], t_end=t_pi)
    rr = simulate_sequence(m_res, c, m_res.basis_state(0, 1), {"sb": "sideband"}, SimulationConfig())
    rd = simulate_sequence(m_det, c, m_det.basis_state(0, 1), {"sb": "sideband"}, SimulationConfig())
    p_res = abs(m_res.basis_state(1, 0).overlap(rr.final_state)) ** 2
    p_det = abs(m_det.basis_state(1, 0).overlap(rd.final_state)) ** 2
    assert p_res > p_det + 0.25
    assert (time.perf_counter() - start) < 1.2


def test_sideband_with_kerr_still_conserves_excitation_number_closed():
    start = time.perf_counter()
    g = 0.15
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=-2 * np.pi * 0.03, n_cav=5, n_tr=2)
    p = Pulse("sb", 0.0, 5.0, _square, amp=g)
    c = SequenceCompiler(dt=0.02).compile([p], t_end=5.0)
    r = simulate_sequence(m, c, m.basis_state(0, 1), {"sb": "sideband"}, SimulationConfig(store_states=True))
    n_tot = m.operators()["n_c"] + m.operators()["n_q"]
    vals = np.array([qt.expect(n_tot, s) for s in r.states], dtype=float)
    assert np.max(np.abs(vals - vals[0])) < 2e-6
    assert (time.perf_counter() - start) < 1.6


def test_beamsplitter_swap_between_modes():
    start = time.perf_counter()
    n = 4
    # theta = pi/2 swaps one-photon states up to phase.
    u = beamsplitter_unitary(n, n, theta=np.pi / 2)
    psi = qt.tensor(qt.basis(n, 1), qt.basis(n, 0))
    out = u * psi
    target = qt.tensor(qt.basis(n, 0), qt.basis(n, 1))
    assert abs(out.overlap(target)) ** 2 > 0.99
    assert (time.perf_counter() - start) < 0.8

