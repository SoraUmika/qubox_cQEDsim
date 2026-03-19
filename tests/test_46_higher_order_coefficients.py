"""Analytic tests for chi_higher and kerr_higher Hamiltonian coefficients.

These tests verify the falling-factorial expansion implemented in
``cqed_sim.core.universal_model`` against closed-form energy expressions.

The Hamiltonian convention used in this codebase is:

  H_kerr = kerr * n*(n-1)/2  +  kerr_higher[0] * n*(n-1)*(n-2)/6  +  ...
  H_chi  = chi * n_c * n_q   +  chi_higher[0] * n_c*(n_c-1) * n_q  +  ...

where n = a†a is the photon-number operator for the bosonic mode and n_q = b†b
is the transmon number operator.

The falling factorial of order k evaluated at integer n is:
  n^(k)_falling = n * (n-1) * ... * (n-k+1)

so the coefficient in the Hamiltonian reads:
  coeff * n^(order)_falling / order!
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.core.universal_model import (
    BosonicModeSpec,
    DispersiveCouplingSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _falling_factorial(n: int, order: int) -> float:
    """Compute n * (n-1) * ... * (n - order + 1)."""
    result = 1.0
    for k in range(order):
        result *= float(n - k)
    return result


def _kerr_energy(n: int, kerr: float, kerr_higher: tuple[float, ...]) -> float:
    """Analytic energy contribution from self-Kerr and higher-order Kerr at Fock level n."""
    energy = kerr * _falling_factorial(n, 2) / math.factorial(2)
    for order_index, coeff in enumerate(kerr_higher, start=2):
        order = order_index + 1
        energy += coeff * _falling_factorial(n, order) / math.factorial(order)
    return energy


def _chi_energy(n: int, q_level: int, chi: float, chi_higher: tuple[float, ...]) -> float:
    """Analytic energy contribution from dispersive coupling at (q_level, n)."""
    energy = chi * float(n) * float(q_level)
    for order, coeff in enumerate(chi_higher, start=2):
        energy += coeff * _falling_factorial(n, order) * float(q_level)
    return energy


# ---------------------------------------------------------------------------
# kerr_higher tests
# ---------------------------------------------------------------------------


class TestKerrHigher:
    """Verify that kerr_higher coefficients enter the Hamiltonian with the
    correct falling-factorial prefactors."""

    def _model(self, kerr: float, kerr_higher: tuple[float, ...], n_cav: int = 10) -> DispersiveTransmonCavityModel:
        return DispersiveTransmonCavityModel(
            omega_c=0.0,
            omega_q=0.0,
            alpha=0.0,
            chi=0.0,
            kerr=kerr,
            kerr_higher=kerr_higher,
            n_cav=n_cav,
            n_tr=2,
        )

    def test_kerr_only_matches_analytic_for_fock_levels_0_to_4(self):
        kerr = 2.0 * np.pi * 0.004
        model = self._model(kerr=kerr, kerr_higher=())
        h = model.static_hamiltonian(FrameSpec())
        for n in range(5):
            energy = float(qt.expect(h, model.basis_state(0, n)))
            expected = 0.5 * kerr * n * (n - 1)  # kerr * n*(n-1) / 2!
            assert np.isclose(energy, expected, atol=1.0e-12), (
                f"n={n}: got {energy}, expected {expected}"
            )

    def test_kerr_higher_single_coefficient_matches_falling_factorial_formula(self):
        """kerr_higher[0] enters as coeff * n*(n-1)*(n-2) / 6."""
        kerr = 2.0 * np.pi * 0.004
        kerr2 = -2.0 * np.pi * 0.0012
        model = self._model(kerr=kerr, kerr_higher=(kerr2,))
        h = model.static_hamiltonian(FrameSpec())
        for n in range(7):
            energy = float(qt.expect(h, model.basis_state(0, n)))
            expected = _kerr_energy(n, kerr, (kerr2,))
            assert np.isclose(energy, expected, atol=1.0e-12), (
                f"n={n}: got {energy}, expected {expected}"
            )

    def test_kerr_higher_low_fock_zeros(self):
        """At n=0 and n=1 the Kerr contribution (including higher order) must be zero."""
        kerr = 2.0 * np.pi * 0.004
        kerr2 = -2.0 * np.pi * 0.001
        model = self._model(kerr=kerr, kerr_higher=(kerr2,))
        h = model.static_hamiltonian(FrameSpec())
        for n in (0, 1):
            energy = float(qt.expect(h, model.basis_state(0, n)))
            # No chi, no bare frequency -> only Kerr contributions, which are zero for n<=1
            assert np.isclose(energy, 0.0, atol=1.0e-12), (
                f"n={n}: expected 0 Kerr energy, got {energy}"
            )

    def test_kerr_higher_two_coefficients_match_falling_factorial(self):
        """Two higher-order Kerr coefficients both enter correctly."""
        kerr = 2.0 * np.pi * 0.003
        kerr2 = -2.0 * np.pi * 0.0008
        kerr3 = 2.0 * np.pi * 0.00005
        model = self._model(kerr=kerr, kerr_higher=(kerr2, kerr3), n_cav=12)
        h = model.static_hamiltonian(FrameSpec())
        for n in range(8):
            energy = float(qt.expect(h, model.basis_state(0, n)))
            expected = _kerr_energy(n, kerr, (kerr2, kerr3))
            assert np.isclose(energy, expected, atol=1.0e-12), (
                f"n={n}: got {energy}, expected {expected}"
            )

    def test_kerr_higher_basis_energy_matches_hamiltonian_expectation(self):
        """``basis_energy`` should agree with the Hamiltonian expectation value."""
        kerr = 2.0 * np.pi * 0.005
        kerr2 = -2.0 * np.pi * 0.0015
        model = self._model(kerr=kerr, kerr_higher=(kerr2,))
        frame = FrameSpec()
        h = model.static_hamiltonian(frame)
        for n in range(6):
            h_energy = float(qt.expect(h, model.basis_state(0, n)))
            be_energy = model.basis_energy(0, n, frame=frame)
            assert np.isclose(h_energy, be_energy, atol=1.0e-12), (
                f"n={n}: Hamiltonian expectation {h_energy} != basis_energy {be_energy}"
            )


# ---------------------------------------------------------------------------
# chi_higher tests
# ---------------------------------------------------------------------------


class TestChiHigher:
    """Verify that chi_higher coefficients enter the dispersive Hamiltonian with
    the correct falling-factorial prefactors."""

    def _model(self, chi: float, chi_higher: tuple[float, ...], n_cav: int = 10) -> DispersiveTransmonCavityModel:
        return DispersiveTransmonCavityModel(
            omega_c=0.0,
            omega_q=0.0,
            alpha=0.0,
            chi=chi,
            chi_higher=chi_higher,
            kerr=0.0,
            n_cav=n_cav,
            n_tr=2,
        )

    def test_chi_only_energy_shift_matches_analytic(self):
        """chi * n is added when qubit is in |e>."""
        chi = -2.0 * np.pi * 0.002
        model = self._model(chi=chi, chi_higher=())
        frame = FrameSpec()
        for n in range(6):
            e_g = model.basis_energy(0, n, frame=frame)
            e_e = model.basis_energy(1, n, frame=frame)
            shift = e_e - e_g
            # dispersive shift only; no bare frequencies or kerr in this model
            assert np.isclose(shift, chi * n, atol=1.0e-12), (
                f"n={n}: chi shift = {shift}, expected {chi * n}"
            )

    def test_chi_higher_single_coefficient_matches_falling_factorial(self):
        """chi_higher[0] (order=2) contributes chi2 * n*(n-1) to the e-state shift."""
        chi = -2.0 * np.pi * 0.002
        chi2 = -2.0 * np.pi * 0.0001
        model = self._model(chi=chi, chi_higher=(chi2,))
        frame = FrameSpec()
        h = model.static_hamiltonian(frame)
        for n in range(7):
            psi_g = model.basis_state(0, n)
            psi_e = model.basis_state(1, n)
            energy_g = float(qt.expect(h, psi_g))
            energy_e = float(qt.expect(h, psi_e))
            shift = energy_e - energy_g
            # expected: chi*n + chi2 * n*(n-1)
            expected_shift = _chi_energy(n, q_level=1, chi=chi, chi_higher=(chi2,))
            assert np.isclose(shift, expected_shift, atol=1.0e-12), (
                f"n={n}: chi shift = {shift}, expected {expected_shift}"
            )

    def test_chi_higher_zero_for_low_fock_orders(self):
        """At n=0 and n=1 the chi_higher term (order 2) vanishes."""
        chi = -2.0 * np.pi * 0.002
        chi2 = -2.0 * np.pi * 0.0001
        model = self._model(chi=chi, chi_higher=(chi2,))
        frame = FrameSpec()
        for n in (0, 1):
            # chi2 term = chi2 * n*(n-1); at n=0 and n=1 this is 0
            e_g = model.basis_energy(0, n, frame=frame)
            e_e = model.basis_energy(1, n, frame=frame)
            shift = e_e - e_g
            linear_chi = chi * n
            # For n<=1, chi_higher term should not contribute
            assert np.isclose(shift, linear_chi, atol=1.0e-12), (
                f"n={n}: expected only linear chi ({linear_chi}), got {shift}"
            )

    def test_chi_higher_ground_state_unaffected(self):
        """The |g, n> energies should not be affected by chi or chi_higher."""
        chi = -2.0 * np.pi * 0.002
        chi2 = -2.0 * np.pi * 0.0001
        model = self._model(chi=chi, chi_higher=(chi2,))
        frame = FrameSpec()
        # In this model omega_c = omega_q = kerr = 0; |g,n> has zero energy
        for n in range(6):
            e_g = model.basis_energy(0, n, frame=frame)
            assert np.isclose(e_g, 0.0, atol=1.0e-12), (
                f"|g, {n}> energy = {e_g}, expected 0"
            )

    def test_chi_higher_basis_energy_matches_hamiltonian_expectation(self):
        """``basis_energy`` agrees with Hamiltonian expectation value."""
        chi = -2.0 * np.pi * 0.003
        chi2 = 2.0 * np.pi * 0.00015
        model = self._model(chi=chi, chi_higher=(chi2,))
        frame = FrameSpec()
        h = model.static_hamiltonian(frame)
        for q in (0, 1):
            for n in range(6):
                h_energy = float(qt.expect(h, model.basis_state(q, n)))
                be_energy = model.basis_energy(q, n, frame=frame)
                assert np.isclose(h_energy, be_energy, atol=1.0e-12), (
                    f"|{q},{n}>: H expectation {h_energy} != basis_energy {be_energy}"
                )


# ---------------------------------------------------------------------------
# UniversalCQEDModel consistency check
# ---------------------------------------------------------------------------


class TestUniversalModelHigherOrder:
    """Smoke tests verifying the UniversalCQEDModel chi/kerr higher-order paths."""

    def test_universal_model_kerr_higher_matches_dispersive_cavity_model(self):
        kerr = 2.0 * np.pi * 0.004
        kerr2 = -2.0 * np.pi * 0.001
        simple = DispersiveTransmonCavityModel(
            omega_c=0.0, omega_q=0.0, alpha=0.0,
            chi=0.0, kerr=kerr, kerr_higher=(kerr2,),
            n_cav=8, n_tr=2,
        )
        universal = UniversalCQEDModel(
            transmon=TransmonModeSpec(omega=0.0, dim=2, alpha=0.0),
            bosonic_modes=(
                BosonicModeSpec(
                    label="storage", omega=0.0, dim=8,
                    kerr=kerr, kerr_higher=(kerr2,),
                    aliases=("storage", "cavity"),
                ),
            ),
            dispersive_couplings=(
                DispersiveCouplingSpec(mode="storage", chi=0.0, chi_higher=()),
            ),
        )
        frame = FrameSpec()
        h_simple = simple.static_hamiltonian(frame)
        h_universal = universal.static_hamiltonian(frame)
        assert (h_simple - h_universal).norm() < 1.0e-12, (
            "UniversalCQEDModel and DispersiveTransmonCavityModel Hamiltonians differ."
        )
