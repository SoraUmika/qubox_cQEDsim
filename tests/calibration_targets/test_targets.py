from __future__ import annotations

import numpy as np

from cqed_sim.calibration_targets import run_drag_tuning, run_rabi, run_ramsey, run_spectroscopy, run_t1, run_t2_echo
from cqed_sim.core import DispersiveTransmonCavityModel


def _model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 6.0e9,
        omega_q=2.0 * np.pi * 5.2e9,
        alpha=2.0 * np.pi * (-0.22e9),
        chi=2.0 * np.pi * 2.0e6,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )


def test_spectroscopy_returns_known_transition_frequencies():
    model = _model()
    freqs = np.linspace(model.omega_q + 1.2 * model.alpha, model.omega_q + 0.2 * abs(model.alpha), 400)
    result = run_spectroscopy(model, freqs, excited_state_fraction=0.35)
    assert np.isclose(result.fitted_parameters["omega_01"], model.omega_q, atol=abs(model.alpha) / 100.0)
    assert np.isclose(result.fitted_parameters["omega_12"], model.omega_q + model.alpha, atol=abs(model.alpha) / 100.0)


def test_rabi_ramsey_and_relaxation_targets_recover_input_scales():
    model = _model()
    rabi = run_rabi(model, np.linspace(0.0, 2.0, 200), duration=50.0e-9, omega_scale=8.0e7)
    ramsey = run_ramsey(model, np.linspace(0.0, 4.0e-6, 200), detuning=2.0 * np.pi * 1.5e6, t2_star=18.0e-6)
    t1 = run_t1(model, np.linspace(0.0, 60.0e-6, 200), t1=32.0e-6)
    t2_echo = run_t2_echo(model, np.linspace(0.0, 80.0e-6, 200), t2_echo=44.0e-6)

    assert np.isclose(rabi.fitted_parameters["omega_scale"], 8.0e7, rtol=1.0e-3)
    assert np.isclose(ramsey.fitted_parameters["delta_omega"], 2.0 * np.pi * 1.5e6, rtol=1.0e-3)
    assert np.isclose(ramsey.fitted_parameters["t2_star"], 18.0e-6, rtol=1.0e-3)
    assert np.isclose(t1.fitted_parameters["t1"], 32.0e-6, rtol=1.0e-3)
    assert np.isclose(t2_echo.fitted_parameters["t2_echo"], 44.0e-6, rtol=1.0e-3)


def test_drag_tuning_finds_quadratic_minimum():
    model = _model()
    drag_values = np.linspace(-3.0e-9, 9.0e-9, 121)
    result = run_drag_tuning(model, drag_values, optimal_drag=2.5e-9)
    assert np.isclose(result.fitted_parameters["drag_optimal"], 2.5e-9, atol=1.0e-10)
