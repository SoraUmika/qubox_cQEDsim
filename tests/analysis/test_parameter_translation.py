from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.analysis import from_measured, from_transmon_params
from cqed_sim.sim.couplings import exchange


def _exact_dressed_shifts(omega_q: float, alpha: float, g: float, omega_r: float) -> tuple[float, float]:
    a = qt.tensor(qt.qeye(6), qt.destroy(5))
    b = qt.tensor(qt.destroy(6), qt.qeye(5))
    hamiltonian = (
        omega_r * (a.dag() * a)
        + omega_q * (b.dag() * b)
        + 0.5 * alpha * (b.dag() * b.dag() * b * b)
        + exchange(a, b, g)
    )
    eigenvalues, eigenstates = hamiltonian.eigenstates()

    def matched_energy(q_level: int, n_level: int) -> float:
        bare = qt.tensor(qt.basis(6, q_level), qt.basis(5, n_level))
        overlaps = np.array([abs(complex(bare.overlap(state))) ** 2 for state in eigenstates], dtype=float)
        return float(eigenvalues[int(np.argmax(overlaps))])

    omega0 = matched_energy(1, 0) - matched_energy(0, 0)
    omega1 = matched_energy(1, 1) - matched_energy(0, 1)
    omega2 = matched_energy(1, 2) - matched_energy(0, 2)
    chi = float(omega1 - omega0)
    chi_2 = float(0.5 * (omega2 - omega0) - chi)
    return chi, chi_2


def test_from_transmon_params_matches_low_dimensional_exact_diagonalization():
    ej = 2.0 * np.pi * 15.0e9
    ec = 2.0 * np.pi * 0.22e9
    g = 2.0 * np.pi * 0.09e9
    omega_r = 2.0 * np.pi * 6.3e9

    translated = from_transmon_params(ej, ec, g, omega_r)
    chi_exact, chi2_exact = _exact_dressed_shifts(translated.omega_q, translated.alpha, g, omega_r)

    assert np.isclose(translated.omega_q, np.sqrt(8.0 * ej * ec) - ec, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(translated.alpha, -ec, rtol=1.0e-12, atol=1.0e-12)
    assert np.isclose(translated.chi, chi_exact, rtol=1.0e-2, atol=1.0e-6)
    assert np.isclose(translated.chi_2, chi2_exact, rtol=1.0e-2, atol=1.0e-6)


def test_from_measured_recovers_bare_circuit_parameters_to_within_one_percent():
    ej = 2.0 * np.pi * 16.0e9
    ec = 2.0 * np.pi * 0.24e9
    g = 2.0 * np.pi * 0.08e9
    omega_r = 2.0 * np.pi * 6.1e9

    translated = from_transmon_params(ej, ec, g, omega_r)
    recovered = from_measured(translated.omega_q, translated.alpha, translated.chi, g, omega_r=omega_r)

    assert np.isclose(recovered.ec, ec, rtol=1.0e-2, atol=1.0e-6)
    assert np.isclose(recovered.ej, ej, rtol=1.0e-2, atol=1.0e-6)
    assert np.isclose(recovered.chi, translated.chi, rtol=1.0e-2, atol=1.0e-6)
