from __future__ import annotations

import time

import numpy as np
import pytest
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, default_observables, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def _spectroscopy_peak(model: DispersiveTransmonCavityModel, n: int, w_scan: np.ndarray) -> float:
    pe = []
    for wd in w_scan:
        pulse = Pulse("q", 0.0, 14.0, _square, amp=0.06, carrier=wd)
        compiled = SequenceCompiler(dt=0.1).compile([pulse], t_end=14.2)
        res = simulate_sequence(
            model,
            compiled,
            model.basis_state( 0,n),
            {"q": "qubit"},
            SimulationConfig(frame=FrameSpec(omega_q_frame=model.omega_q)),
        )
        pe.append(res.expectations["P_e"][-1])
    return float(w_scan[int(np.argmax(pe))])


@pytest.mark.slow
def test_chi_is_qubit_fock_peak_spacing_definition():
    start = time.perf_counter()
    chi = 2 * np.pi * 0.035
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=2 * np.pi * 1.0, alpha=0.0, chi=chi, chi_higher=(), kerr=0.0, n_cav=8, n_tr=2
    )
    w_scan = np.linspace(-0.8 * chi, 2.8 * chi, 33)
    # Drive carrier sign convention in this simulator is opposite to extracted omega_ge.
    w0 = -_spectroscopy_peak(model, 0, w_scan)
    w1 = -_spectroscopy_peak(model, 1, w_scan)
    w2 = -_spectroscopy_peak(model, 2, w_scan)
    assert np.isclose(w0 - w1, chi, rtol=0.15, atol=0.02)
    assert np.isclose(w1 - w2, chi, rtol=0.15, atol=0.02)
    assert (time.perf_counter() - start) < 8.0


def test_chi_ramsey_phase_slope_equals_n_times_chi():
    start = time.perf_counter()
    chi = 2 * np.pi * 0.03
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=chi, chi_higher=(), kerr=0.0, n_cav=8, n_tr=2
    )
    times = np.linspace(0.0, 8.0, 41)
    slopes = []
    for n in [0, 1, 2]:
        phases = []
        for t in times:
            compiled = SequenceCompiler(dt=0.1).compile([], t_end=t)
            psi = (model.basis_state( 0,n) + model.basis_state( 1,n)).unit()
            res = simulate_sequence(model, compiled, psi, {}, SimulationConfig(frame=FrameSpec()))
            rho_q = qt.ptrace(res.final_state, 0)
            # For rho_ge = <g|rho|e>, dispersive pull appears with opposite sign; negate to recover +n*chi slope.
            phases.append(-np.angle(rho_q[0, 1]))
        slopes.append(float(np.polyfit(times, np.unwrap(phases), 1)[0]))
    assert np.isclose(slopes[1], chi, rtol=0.12, atol=0.02)
    assert np.isclose(slopes[2], 2 * chi, rtol=0.12, atol=0.03)
    assert np.isclose(slopes[2] - slopes[1], chi, rtol=0.12, atol=0.02)
    assert (time.perf_counter() - start) < 2.5


def test_chi_projector_vs_pauli_equivalence_with_offset():
    start = time.perf_counter()
    chi = 2 * np.pi * 0.028
    n_cav = 6
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=chi, chi_higher=(), kerr=0.0, n_cav=n_cav, n_tr=2
    )
    ops = model.operators()
    n_c = ops["n_c"]
    n_q = ops["n_q"]
    i_tot = qt.tensor( qt.qeye(2),qt.qeye(n_cav))
    sigma_z_eg = 2 * n_q - i_tot

    h_proj = -chi * n_c * n_q
    h_pauli = -(chi / 2.0) * n_c * sigma_z_eg
    h_pauli_adj = h_pauli - (chi / 2.0) * n_c

    compiled = SequenceCompiler(dt=0.1).compile([], t_end=6.0)
    psi0 = (model.basis_state( 0,1) + model.basis_state( 1,1)).unit()
    obs = default_observables(model)
    ra = qt.sesolve([h_proj], psi0, compiled.tlist, e_ops=list(obs.values()), options={"store_states": False})
    rb = qt.sesolve([h_pauli_adj], psi0, compiled.tlist, e_ops=list(obs.values()), options={"store_states": False})
    pe_a = np.asarray(ra.expect[0])
    pe_b = np.asarray(rb.expect[0])
    nc_a = np.asarray(ra.expect[1])
    nc_b = np.asarray(rb.expect[1])
    assert np.allclose(pe_a, pe_b, atol=1e-9)
    assert np.allclose(nc_a, nc_b, atol=1e-9)
    assert (time.perf_counter() - start) < 1.2


def test_chi_sign_matches_pull_direction():
    start = time.perf_counter()
    chi = 2 * np.pi * 0.03
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=chi, chi_higher=(), kerr=0.0, n_cav=6, n_tr=2
    )
    times = np.linspace(0.0, 6.0, 31)
    phases0 = []
    phases1 = []
    for t in times:
        compiled = SequenceCompiler(dt=0.1).compile([], t_end=t)
        p0 = (model.basis_state( 0,0) + model.basis_state( 1,0)).unit()
        p1 = (model.basis_state( 0,1) + model.basis_state( 1,1)).unit()
        r0 = simulate_sequence(model, compiled, p0, {}, SimulationConfig(frame=FrameSpec()))
        r1 = simulate_sequence(model, compiled, p1, {}, SimulationConfig(frame=FrameSpec()))
        phases0.append(-np.angle(qt.ptrace(r0.final_state, 0)[0, 1]))
        phases1.append(-np.angle(qt.ptrace(r1.final_state, 0)[0, 1]))
    s0 = float(np.polyfit(times, np.unwrap(phases0), 1)[0])
    s1 = float(np.polyfit(times, np.unwrap(phases1), 1)[0])
    assert s1 - s0 > 0.0
    assert np.isclose(s1 - s0, chi, rtol=0.12, atol=0.02)
    assert (time.perf_counter() - start) < 2.0


def test_higher_order_chi_matches_fock_falling_factorial_transition_shift():
    chi = 2 * np.pi * 0.031
    chi2 = 2 * np.pi * 0.006
    chi3 = -2 * np.pi * 0.0015
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=2 * np.pi * 1.2,
        alpha=0.0,
        chi=chi,
        chi_higher=(chi2, chi3),
        kerr=0.0,
        n_cav=8,
        n_tr=2,
    )
    h0 = model.static_hamiltonian(FrameSpec())
    for n in range(4):
        eg = float(qt.expect(h0, model.basis_state( 1,n)) - qt.expect(h0, model.basis_state( 0,n)))
        expected = model.omega_q - chi * n - chi2 * (n * (n - 1)) - chi3 * (n * (n - 1) * (n - 2))
        assert np.isclose(eg, expected, atol=1e-9)
