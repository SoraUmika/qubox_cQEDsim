from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from cqed_sim.sequence.scheduler import SequenceCompiler


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def _run(model, amp, dt):
    t_p = 3.0
    pulse = Pulse(channel="cavity", t0=0.0, duration=t_p, envelope=_square, amp=amp)
    compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=t_p)
    psi0 = model.basis_state( 0,0)
    return simulate_sequence(
        model,
        compiled,
        psi0,
        {"cavity": "cavity"},
        config=SimulationConfig(frame=FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)),
    )


def test_linear_cavity_drive_coherent_signature():
    start = time.perf_counter()
    model = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 0.9,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=18,
        n_tr=2,
    )
    amp = 0.11
    t_p = 3.0
    res = _run(model, amp=amp, dt=0.03)
    n_expect = res.expectations["n_c"][-1]
    n_target = (amp * t_p) ** 2
    assert np.isclose(n_expect, n_target, rtol=0.08, atol=0.01)
    assert (time.perf_counter() - start) < 2.5


def test_kerr_signature_and_timestep_refinement():
    start = time.perf_counter()
    base = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 0.9,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=18,
        n_tr=2,
    )
    kerr = DispersiveTransmonCavityModel(
        omega_c=base.omega_c,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=-2 * np.pi * 0.015,
        n_cav=18,
        n_tr=2,
    )
    r0 = _run(base, amp=0.45, dt=0.05)
    r1 = _run(kerr, amp=0.45, dt=0.05)
    rk = _run(kerr, amp=0.45, dt=0.025)
    assert abs(r0.expectations["n_c"][-1] - r1.expectations["n_c"][-1]) > 0.008
    diff_coarse = abs(r1.expectations["n_c"][-1] - rk.expectations["n_c"][-1])
    assert diff_coarse < 0.08
    assert (time.perf_counter() - start) < 3.5
