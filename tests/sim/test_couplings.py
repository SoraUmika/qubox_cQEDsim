from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core import CrossKerrSpec, DispersiveTransmonCavityModel, ExchangeSpec, SelfKerrSpec
from cqed_sim.sim.couplings import cross_kerr, exchange, self_kerr


def test_cross_kerr_shifts_resonator_frequency_by_qubit_excitation_number():
    model = DispersiveTransmonCavityModel(
        omega_c=5.0,
        omega_q=7.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        cross_kerr_terms=(CrossKerrSpec("a", "b", 0.3),),
        n_cav=4,
        n_tr=2,
    )
    hamiltonian = model.static_hamiltonian()
    e_g0 = float(qt.expect(hamiltonian, model.basis_state(0, 0)))
    e_g1 = float(qt.expect(hamiltonian, model.basis_state(0, 1)))
    e_e0 = float(qt.expect(hamiltonian, model.basis_state(1, 0)))
    e_e1 = float(qt.expect(hamiltonian, model.basis_state(1, 1)))
    freq_g = e_g1 - e_g0
    freq_e = e_e1 - e_e0
    assert np.isclose(freq_e - freq_g, 0.3, atol=1.0e-12)


def test_self_kerr_breaks_harmonic_level_spacing():
    a = qt.destroy(5)
    hamiltonian = 2.0 * (a.dag() * a) + self_kerr(a, -0.4)
    energies = np.real(np.diag(hamiltonian.full()))
    spacing_01 = energies[1] - energies[0]
    spacing_12 = energies[2] - energies[1]
    assert np.isclose(spacing_12 - spacing_01, -0.4, atol=1.0e-12)


def test_exchange_coupling_generates_vacuum_rabi_swap():
    a = qt.tensor(qt.destroy(2), qt.qeye(2))
    b = qt.tensor(qt.qeye(2), qt.destroy(2))
    coupling = 0.7
    hamiltonian = exchange(a, b, coupling)
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    t_swap = np.pi / (2.0 * coupling)
    result = qt.sesolve(hamiltonian, psi0, [0.0, t_swap])
    final_state = result.states[-1]
    target = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
    assert abs(final_state.overlap(target)) > 0.999


def test_exchange_term_is_injected_into_static_model_hamiltonian():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        exchange_terms=(ExchangeSpec("a", "b", 0.2),),
        n_cav=2,
        n_tr=2,
    )
    hamiltonian = model.static_hamiltonian()
    psi0 = model.basis_state(1, 0)
    target = model.basis_state(0, 1)
    assert abs((hamiltonian * psi0).overlap(target)) > 0.0
