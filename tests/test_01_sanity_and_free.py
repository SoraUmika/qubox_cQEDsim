from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import hamiltonian_time_slices, simulate_sequence, SimulationConfig


def test_hamiltonian_hermitian_over_grid(base_model):
    start = time.perf_counter()
    compiler = SequenceCompiler(dt=0.25)
    compiled = compiler.compile([], t_end=8.0)
    h_terms = hamiltonian_time_slices(base_model, compiled, {"unused": "cavity"} if False else {}, frame=FrameSpec())
    h0 = h_terms[0]
    for _ in range(0, len(compiled.tlist), 5):
        delta = (h0 - h0.dag()).norm()
        assert delta < 1e-10
    elapsed = time.perf_counter() - start
    assert elapsed < 0.8


def test_free_evolution_phase_k0_chi0():
    start = time.perf_counter()
    model = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 0.8,
        omega_q=2 * np.pi * 1.1,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=6,
        n_tr=2,
    )
    psi0 = model.basis_state( 1,2)
    compiler = SequenceCompiler(dt=0.1)
    compiled = compiler.compile([], t_end=3.0)
    result = simulate_sequence(model, compiled, psi0, {}, config=SimulationConfig(frame=FrameSpec()))
    phase = psi0.overlap(result.final_state)
    assert np.isclose(np.abs(phase), 1.0, atol=1e-8)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0


def test_no_drive_populations_constant(base_model):
    start = time.perf_counter()
    psi0 = (base_model.basis_state( 0,1) + 1j * base_model.basis_state( 1,2)).unit()
    compiler = SequenceCompiler(dt=0.2)
    compiled = compiler.compile([], t_end=6.0)
    res = simulate_sequence(base_model, compiled, psi0, {}, config=SimulationConfig(frame=FrameSpec()))
    p_start = np.abs(psi0.full().ravel()) ** 2
    p_end = np.abs(res.final_state.full().ravel()) ** 2
    assert np.allclose(p_start, p_end, atol=2e-5)
    assert (time.perf_counter() - start) < 1.0


def test_kerr_only_phase_matches_analytic():
    start = time.perf_counter()
    k = -2 * np.pi * 0.02
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=k,
        n_cav=8,
        n_tr=2,
    )
    n = 3
    t_end = 4.0
    psi0 = model.basis_state( 0,n)
    compiled = SequenceCompiler(dt=0.1).compile([], t_end=t_end)
    res = simulate_sequence(model, compiled, psi0, {}, config=SimulationConfig(frame=FrameSpec()))
    energy = 0.5 * k * n * (n - 1)
    expected = np.exp(-1j * energy * t_end)
    overlap = psi0.overlap(res.final_state)
    assert np.allclose(overlap / np.abs(overlap), expected, atol=2e-3)
    assert (time.perf_counter() - start) < 1.0
