from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.measurement import (
    AmplifierChain,
    PurcellFilter,
    QubitMeasurementSpec,
    ReadoutChain,
    ReadoutResonator,
    measure_qubit,
)


def _make_chain(*, chi: float, with_filter: bool) -> ReadoutChain:
    resonator = ReadoutResonator(
        omega_r=2.0 * np.pi * 7.0e9,
        kappa=2.0 * np.pi * 8.0e6,
        g=2.0 * np.pi * 90.0e6,
        epsilon=2.0 * np.pi * 0.6e6,
        chi=chi,
    )
    purcell_filter = None if not with_filter else PurcellFilter(omega_f=resonator.omega_r, bandwidth=2.0 * np.pi * 40.0e6)
    return ReadoutChain(
        resonator=resonator,
        purcell_filter=purcell_filter,
        amplifier=AmplifierChain(noise_temperature=4.0, gain=12.0),
        integration_time=300.0e-9,
        dt=5.0e-9,
    )


def test_purcell_filter_improves_purcell_limited_t1():
    omega_q = 2.0 * np.pi * 6.0e9
    bare = _make_chain(chi=2.0 * np.pi * 1.2e6, with_filter=False)
    filtered = _make_chain(chi=2.0 * np.pi * 1.2e6, with_filter=True)

    t1_bare = bare.purcell_limited_t1(omega_q, include_filter=False)
    t1_filtered = filtered.purcell_limited_t1(omega_q, include_filter=True)

    assert t1_filtered > 100.0 * t1_bare


def test_measurement_induced_dephasing_scales_like_chi_squared_nbar_over_kappa():
    chi_values = 2.0 * np.pi * np.array([0.25e6, 0.5e6, 0.75e6], dtype=float)
    scaled_rates = []
    for chi in chi_values:
        chain = _make_chain(chi=chi, with_filter=False)
        nbar_g = chain.resonator.mean_photon_numbers(drive_frequency=chain.resonator.omega_r)["g"]
        gamma = chain.gamma_meas(drive_frequency=chain.resonator.omega_r)
        scaled_rates.append(gamma / (chi * chi * nbar_g / chain.resonator.kappa))
    assert np.max(scaled_rates) - np.min(scaled_rates) < 0.08 * np.mean(scaled_rates)


def test_measurement_chain_produces_distinct_iq_clusters_and_matches_steady_state_amplitudes():
    chain = _make_chain(chi=2.0 * np.pi * 1.8e6, with_filter=True)
    trace_g = chain.simulate_trace("g", include_noise=False)
    trace_e = chain.simulate_trace("e", include_noise=False)
    analytic = chain.steady_state_amplitudes()

    assert np.isclose(trace_g.cavity_field[-1], analytic["g"], rtol=0.0, atol=5.0e-3 * abs(analytic["g"]) + 1.0e-10)
    assert np.isclose(trace_e.cavity_field[-1], analytic["e"], rtol=0.0, atol=5.0e-3 * abs(analytic["e"]) + 1.0e-10)

    state_g = qt.tensor(qt.basis(2, 0), qt.basis(2, 0))
    state_e = qt.tensor(qt.basis(2, 1), qt.basis(2, 0))
    spec = QubitMeasurementSpec(
        shots=500,
        seed=7,
        readout_chain=chain,
        readout_duration=chain.integration_time,
        readout_dt=chain.dt,
        classify_from_iq=True,
    )
    result_g = measure_qubit(state_g, spec)
    result_e = measure_qubit(state_e, spec)

    assert result_g.iq_samples is not None
    assert result_e.iq_samples is not None
    mean_g = np.mean(result_g.iq_samples, axis=0)
    mean_e = np.mean(result_e.iq_samples, axis=0)
    assert np.linalg.norm(mean_e - mean_g) > 5.0 * chain.integrated_noise_sigma()
