from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def _xy_model():
    return DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2
    )


def _simulate_qubit(pulses):
    model = _xy_model()
    compiled = SequenceCompiler(dt=0.02).compile(pulses, t_end=max(p.t1 for p in pulses) + 0.05)
    return simulate_sequence(
        model,
        compiled,
        model.basis_state( 0,0),
        {"q": "qubit"},
        config=SimulationConfig(frame=FrameSpec()),
    )


def test_xpi2_plus_xpi2_equals_xpi():
    start = time.perf_counter()
    amp = np.pi / 4.0
    p1 = Pulse("q", 0.0, 1.0, _square, amp=amp)
    p2 = Pulse("q", 1.0, 1.0, _square, amp=amp)
    res = _simulate_qubit([p1, p2])
    assert res.expectations["P_e"][-1] > 0.97
    assert (time.perf_counter() - start) < 1.2


def test_x_then_y_composition_phase_shift():
    start = time.perf_counter()
    amp = np.pi / 4.0
    p1 = Pulse("q", 0.0, 1.0, _square, amp=amp, phase=0.0)
    p2 = Pulse("q", 1.0, 1.0, _square, amp=amp, phase=np.pi / 2)
    res = _simulate_qubit([p1, p2])
    rho_q = qt.ptrace(res.final_state, 0)
    bloch_y = 2 * np.imag(rho_q[1, 0])
    assert bloch_y < -0.6
    assert (time.perf_counter() - start) < 1.2


def test_overlap_additivity_and_cancellation():
    start = time.perf_counter()
    full = Pulse("q", 0.0, 1.0, _square, amp=0.21, phase=0.0)
    half_a = Pulse("q", 0.0, 1.0, _square, amp=0.105, phase=0.0)
    half_b = Pulse("q", 0.0, 1.0, _square, amp=0.105, phase=0.0)
    opp = Pulse("q", 0.0, 1.0, _square, amp=0.105, phase=np.pi)
    compiler = SequenceCompiler(dt=0.01)
    c_full = compiler.compile([full], t_end=1.0).channels["q"].baseband
    c_add = compiler.compile([half_a, half_b], t_end=1.0).channels["q"].baseband
    c_cancel = compiler.compile([half_a, opp], t_end=1.0).channels["q"].baseband
    assert np.allclose(c_full, c_add, atol=1e-12)
    assert np.max(np.abs(c_cancel)) < 1e-10
    assert (time.perf_counter() - start) < 0.8
