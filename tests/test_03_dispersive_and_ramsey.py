from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_chi_conditional_phase_scales_with_photon_number():
    start = time.perf_counter()
    chi = 2 * np.pi * 0.03
    t_end = 5.0
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=chi, kerr=0.0, n_cav=8, n_tr=2
    )
    compiled = SequenceCompiler(dt=0.1).compile([], t_end=t_end)

    def phase_for_n(n):
        psi = (model.basis_state( 0,n) + model.basis_state( 1,n)).unit()
        res = simulate_sequence(model, compiled, psi, {}, config=SimulationConfig(frame=FrameSpec()))
        rho_q = qt.ptrace(res.final_state, 0)
        return np.angle(rho_q[0, 1])

    p1 = phase_for_n(1)
    p2 = phase_for_n(2)
    d = np.angle(np.exp(1j * (p2 - 2 * p1)))
    assert abs(d) < 6e-2
    assert (time.perf_counter() - start) < 2.0


def test_ramsey_pull_with_cavity_photons():
    start = time.perf_counter()
    chi = 2 * np.pi * 0.025
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=chi, kerr=0.0, n_cav=8, n_tr=2
    )
    waits = np.linspace(0.5, 6.0, 14)
    ph0 = []
    ph1 = []
    for tw in waits:
        compiled = SequenceCompiler(dt=0.05).compile([], t_end=tw)
        psi0 = (model.basis_state( 0,0) + model.basis_state( 1,0)).unit()
        psi1 = (model.basis_state( 0,1) + model.basis_state( 1,1)).unit()
        res0 = simulate_sequence(
            model,
            compiled,
            psi0,
            {},
            config=SimulationConfig(frame=FrameSpec()),
        )
        res1 = simulate_sequence(
            model,
            compiled,
            psi1,
            {},
            config=SimulationConfig(frame=FrameSpec()),
        )
        ph0.append(np.angle(qt.ptrace(res0.final_state, 0)[0, 1]))
        ph1.append(np.angle(qt.ptrace(res1.final_state, 0)[0, 1]))
    slope0 = np.polyfit(waits, np.unwrap(ph0), 1)[0]
    slope1 = np.polyfit(waits, np.unwrap(ph1), 1)[0]
    # Under project convention omega_ge(n)=omega_ge(0)-n*chi, rho_ge phase slope decreases with n.
    omega_shift = slope0 - slope1
    assert omega_shift > 0.0
    assert np.isclose(omega_shift, chi, rtol=0.2, atol=0.03)
    assert (time.perf_counter() - start) < 5.0
