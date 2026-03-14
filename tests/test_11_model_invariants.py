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


def test_unitarity_closed_system_sesolve():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=8, n_tr=2)
    p = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 3)
    c = SequenceCompiler(dt=0.02).compile([p], t_end=1.2)
    r = simulate_sequence(m, c, m.basis_state( 0,0), {"q": "qubit"}, SimulationConfig(store_states=True))
    norms = [float(s.norm()) for s in r.states]
    assert max(abs(n - 1.0) for n in norms) < 1e-8
    assert (time.perf_counter() - start) < 1.2


def test_master_equation_trace_preserved_with_collapse_ops():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)
    ops = m.operators()
    c_ops = [np.sqrt(0.08) * ops["b"]]
    c = SequenceCompiler(dt=0.05).compile([], t_end=5.0)
    r = simulate_sequence(m, c, m.basis_state( 1,0), {}, SimulationConfig(store_states=True), c_ops=c_ops)
    traces = [abs(s.tr() - 1.0) for s in r.states]
    assert max(traces) < 1e-8
    assert (time.perf_counter() - start) < 1.2


def test_energy_conservation_when_time_independent():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 0.7, omega_q=2 * np.pi * 1.1, alpha=-2 * np.pi * 0.2, chi=2 * np.pi * 0.02, kerr=0.0, n_cav=6, n_tr=3
    )
    c = SequenceCompiler(dt=0.05).compile([], t_end=4.0)
    h = m.static_hamiltonian(FrameSpec())
    r = simulate_sequence(m, c, (m.basis_state( 0,1) + 1j * m.basis_state( 1,2)).unit(), {}, SimulationConfig(store_states=True))
    e = np.array([qt.expect(h, s) for s in r.states], dtype=float)
    assert np.max(np.abs(e - e[0])) < 4e-5
    assert (time.perf_counter() - start) < 1.3


def test_chi_zero_reduces_to_decoupled_tensor_evolution():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 0.6, omega_q=2 * np.pi * 1.0, alpha=-2 * np.pi * 0.2, chi=0.0, kerr=0.0, n_cav=6, n_tr=3
    )
    psi = qt.tensor( (qt.basis(m.n_tr, 0) + qt.basis(m.n_tr, 1)).unit(),(qt.basis(m.n_cav, 0) + qt.basis(m.n_cav, 1)).unit())
    c = SequenceCompiler(dt=0.05).compile([], t_end=3.0)
    r = simulate_sequence(m, c, psi, {}, SimulationConfig())
    rho = r.final_state.proj()
    rho_c = qt.ptrace(r.final_state, 1)
    rho_q = qt.ptrace(r.final_state, 0)
    prod = qt.tensor( rho_q,rho_c)
    assert (rho - prod).norm() < 3e-5
    assert (time.perf_counter() - start) < 1.2


def test_truncation_monotonicity_cavity_cutoff():
    start = time.perf_counter()
    cuts = [10, 14, 18]
    vals = []
    for n_cav in cuts:
        m = DispersiveTransmonCavityModel(
            omega_c=0.0, omega_q=0.0, alpha=-2 * np.pi * 0.2, chi=2 * np.pi * 0.02, kerr=-2 * np.pi * 0.004, n_cav=n_cav, n_tr=3
        )
        pulses = [Pulse("c", 0.0, 2.0, _square, amp=0.14), Pulse("q", 0.6, 1.2, _square, amp=0.7)]
        c = SequenceCompiler(dt=0.03).compile(pulses, t_end=2.4)
        r = simulate_sequence(m, c, m.basis_state( 0,0), {"c": "cavity", "q": "qubit"}, SimulationConfig())
        vals.append((float(r.expectations["n_c"][-1]), float(r.expectations["P_e"][-1])))
    d1 = abs(vals[0][0] - vals[1][0]) + abs(vals[0][1] - vals[1][1])
    d2 = abs(vals[1][0] - vals[2][0]) + abs(vals[1][1] - vals[2][1])
    assert d2 < d1 + 2e-3
    assert (time.perf_counter() - start) < 3.5


def test_singleton_cavity_cutoff_supports_default_observables():
    m = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 5.0,
        omega_q=2 * np.pi * 6.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    c = SequenceCompiler(dt=0.05).compile([], t_end=0.2)
    r = simulate_sequence(m, c, m.basis_state(0, 0), {}, SimulationConfig(frame=FrameSpec(omega_q_frame=m.omega_q)))
    assert "n_c" in r.expectations
    assert np.allclose(np.asarray(r.expectations["n_c"], dtype=float), 0.0, atol=1.0e-12)
