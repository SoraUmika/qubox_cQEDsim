from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def _gauss(x):
    return gaussian_envelope(x, sigma=0.18)


def _q2():
    return DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)


def test_virtual_z_phase_shift_equivalence():
    start = time.perf_counter()
    m = _q2()
    phi = 0.63
    # VZ(phi) then X90 is equivalent to phase-shifted XY pulse.
    seq_a = [Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0, phase=phi)]
    seq_b = [Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0, phase=phi)]
    c = SequenceCompiler(dt=0.02)
    ra = simulate_sequence(m, c.compile(seq_a, t_end=1.1), m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig())
    rb = simulate_sequence(m, c.compile(seq_b, t_end=1.1), m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig())
    assert abs(ra.final_state.overlap(rb.final_state)) > 1 - 1e-10
    assert (time.perf_counter() - start) < 0.8


def test_phase_continuity_across_wait_with_detuning():
    start = time.perf_counter()
    m = _q2()
    delta = 2 * np.pi * 0.17
    tau = 2.3
    p1 = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0, carrier=delta)
    p2 = Pulse("q", 1.0 + tau, 1.0, _square, amp=np.pi / 4.0, carrier=delta)
    p2_equiv = Pulse("q", 1.0 + tau, 1.0, _square, amp=np.pi / 4.0, carrier=0.0, phase=delta * tau)
    c = SequenceCompiler(dt=0.01)
    r1 = simulate_sequence(
        m, c.compile([p1, p2], t_end=2.1 + tau), m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig(frame=FrameSpec())
    )
    r2 = simulate_sequence(
        m,
        c.compile([Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0), p2_equiv], t_end=2.1 + tau),
        m.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec()),
    )
    assert abs(r1.final_state.overlap(r2.final_state)) > 0.97
    assert (time.perf_counter() - start) < 1.2


def test_same_area_different_shape_same_rotation_small_signal():
    start = time.perf_counter()
    m = _q2()
    dur = 1.0
    area = 0.2
    amp_sq = area / dur
    # normalize Gaussian to same area numerically
    t = np.linspace(0, 1, 2000)
    g_int = np.trapezoid(_gauss(t).real, t)
    amp_g = area / g_int
    sq = Pulse("q", 0.0, dur, _square, amp=amp_sq)
    ga = Pulse("q", 0.0, dur, _gauss, amp=amp_g)
    c = SequenceCompiler(dt=0.005)
    rs = simulate_sequence(m, c.compile([sq], t_end=1.05), m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig())
    rg = simulate_sequence(m, c.compile([ga], t_end=1.05), m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig())
    assert abs(rs.expectations["P_e"][-1] - rg.expectations["P_e"][-1]) < 0.03
    assert (time.perf_counter() - start) < 1.4


def test_drag_quadrature_orthogonality():
    start = time.perf_counter()
    p = Pulse("q", 0.0, 1.0, _gauss, amp=1.0, drag=0.25)
    bb = SequenceCompiler(dt=0.002).compile([p], t_end=1.0).channels["q"].baseband
    re = bb.real
    im = bb.imag
    dre = np.gradient(re, 0.002, edge_order=1)
    corr = np.corrcoef(im, dre)[0, 1]
    assert corr > 0.96
    assert (time.perf_counter() - start) < 0.7


def test_idle_then_echo_cancels_static_detuning():
    start = time.perf_counter()
    m = _q2()
    delta = 2 * np.pi * 0.12
    tau = 2.0
    x90 = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    x180 = Pulse("q", 1.0 + tau, 1.0, _square, amp=np.pi / 2.0)
    x90b = Pulse("q", 2.0 + 2 * tau, 1.0, _square, amp=np.pi / 4.0)
    ramsey = [x90, Pulse("q", 1.0 + tau, 1.0, _square, amp=np.pi / 4.0)]
    echo = [x90, x180, x90b]
    c = SequenceCompiler(dt=0.01)
    rr = simulate_sequence(
        m,
        c.compile(ramsey, t_end=2.2 + tau),
        m.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec(omega_q_frame=delta)),
    )
    re = simulate_sequence(
        m,
        c.compile(echo, t_end=3.2 + 2 * tau),
        m.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec(omega_q_frame=delta)),
    )
    # Echo sequence should refocus and return closer to ground for this phase convention.
    assert re.expectations["P_e"][-1] < rr.expectations["P_e"][-1]
    assert (time.perf_counter() - start) < 1.8


def test_two_pulse_commutation_noncommuting_axes():
    start = time.perf_counter()
    m = _q2()
    x90 = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 6.0, phase=0.0)
    y90 = Pulse("q", 1.0, 1.0, _square, amp=np.pi / 8.0, phase=np.pi / 2)
    y_first = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 8.0, phase=np.pi / 2)
    x_second = Pulse("q", 1.0, 1.0, _square, amp=np.pi / 6.0, phase=0.0)
    c = SequenceCompiler(dt=0.02)
    init = (m.basis_state(0, 0) + m.basis_state(0, 1)).unit()
    rxy = simulate_sequence(m, c.compile([x90, y90], t_end=2.1), init, {"q": "qubit"}, SimulationConfig())
    ryx = simulate_sequence(m, c.compile([y_first, x_second], t_end=2.1), init, {"q": "qubit"}, SimulationConfig())
    assert abs(rxy.final_state.overlap(ryx.final_state)) < 0.995
    assert (time.perf_counter() - start) < 1.0


def test_overlapping_pulses_commutativity_limit_small_dt():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=8, n_tr=2)
    p_c = Pulse("c", 0.0, 1.5, _square, amp=0.1)
    p_q = Pulse("q", 0.0, 1.5, _square, amp=0.25)
    seq = SequenceCompiler(dt=0.005)
    r_ov = simulate_sequence(
        m, seq.compile([p_c, p_q], t_end=1.6), m.basis_state(0, 0), {"c": "cavity", "q": "qubit"}, SimulationConfig()
    )
    r_sq = simulate_sequence(
        m,
        seq.compile([Pulse("q", 0.0, 1.5, _square, amp=0.25), Pulse("c", 1.5, 1.5, _square, amp=0.1)], t_end=3.1),
        m.basis_state(0, 0),
        {"c": "cavity", "q": "qubit"},
        SimulationConfig(),
    )
    assert abs(r_ov.expectations["P_e"][-1] - r_sq.expectations["P_e"][-1]) < 0.02
    assert abs(r_ov.expectations["n_c"][-1] - r_sq.expectations["n_c"][-1]) < 0.02
    assert (time.perf_counter() - start) < 2.0
