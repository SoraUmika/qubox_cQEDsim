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


def _gauss(t):
    return gaussian_envelope(t, sigma=0.17)


def test_multilevel_leakage_and_drag_directionality():
    start = time.perf_counter()
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-2 * np.pi * 0.35,
        chi=0.0,
        kerr=0.0,
        n_cav=2,
        n_tr=3,
    )
    base = Pulse("q", 0.0, 0.8, _gauss, amp=4.2, drag=0.0)
    drag = Pulse("q", 0.0, 0.8, _gauss, amp=4.2, drag=0.23)
    compiler = SequenceCompiler(dt=0.004)
    init = model.basis_state( 0,0)
    r0 = simulate_sequence(
        model, compiler.compile([base], t_end=0.85), init, {"q": "qubit"}, SimulationConfig(frame=FrameSpec())
    )
    r1 = simulate_sequence(
        model, compiler.compile([drag], t_end=0.85), init, {"q": "qubit"}, SimulationConfig(frame=FrameSpec())
    )
    proj_f = qt.tensor(qt.basis(model.n_tr, 2) * qt.basis(model.n_tr, 2).dag(), qt.qeye(model.n_cav))
    pf0 = qt.expect(proj_f, r0.final_state)
    pf1 = qt.expect(proj_f, r1.final_state)
    assert pf0 > 2e-3
    assert pf1 < pf0
    assert (time.perf_counter() - start) < 2.5


def test_higher_order_terms_smoke():
    start = time.perf_counter()
    t_end = 4.0
    model_linear = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2 * np.pi * 0.02,
        chi_higher=(),
        kerr=0.0,
        n_cav=8,
        n_tr=2,
    )
    model_high = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2 * np.pi * 0.02,
        chi_higher=(2 * np.pi * 0.01,),
        kerr=0.0,
        n_cav=8,
        n_tr=2,
    )
    compiled = SequenceCompiler(dt=0.05).compile([], t_end=t_end)

    def rel_phase(model, n):
        psi = (model.basis_state( 0,n) + model.basis_state( 1,n)).unit()
        rho = qt.ptrace(simulate_sequence(model, compiled, psi, {}, SimulationConfig(frame=FrameSpec())).final_state, 0)
        return np.angle(rho[0, 1])

    p1_lin = rel_phase(model_linear, 1)
    p2_lin = rel_phase(model_linear, 2)
    p1_hi = rel_phase(model_high, 1)
    p2_hi = rel_phase(model_high, 2)
    lin_err = abs(np.angle(np.exp(1j * (p2_lin - 2 * p1_lin))))
    hi_err = abs(np.angle(np.exp(1j * (p2_hi - 2 * p1_hi))))
    assert hi_err > lin_err + 0.03
    assert (time.perf_counter() - start) < 2.5


def test_higher_order_kerr_matches_falling_factorial_energy_shift():
    kerr = 2 * np.pi * 0.004
    kerr2 = -2 * np.pi * 0.0012
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        chi_higher=(),
        kerr=kerr,
        kerr_higher=(kerr2,),
        n_cav=8,
        n_tr=2,
    )
    h0 = model.static_hamiltonian(FrameSpec())
    for n in range(5):
        energy = float(qt.expect(h0, model.basis_state( 0,n)))
        expected = -0.5 * kerr * n * (n - 1)
        expected += -(kerr2 / 6.0) * n * (n - 1) * (n - 2)
        assert np.isclose(energy, expected, atol=1e-9)
