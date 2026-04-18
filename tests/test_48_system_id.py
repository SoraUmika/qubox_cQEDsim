"""Tests for cqed_sim.system_id — calibration bridge to domain randomization."""
from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.calibration_targets import CalibrationResult
from cqed_sim.rl_control.domain_randomization import DomainRandomizer
from cqed_sim.system_id import (
    CalibrationEvidence,
    ChoicePrior,
    FixedPrior,
    NormalPrior,
    UniformPrior,
    evidence_from_fit,
    fit_rabi_trace,
    fit_ramsey_trace,
    fit_spectroscopy_trace,
    fit_t1_trace,
    fit_t2_echo_trace,
    merge_calibration_evidence,
    prior_from_fit,
    randomizer_from_calibration,
)


class TestCalibrationEvidence:
    def test_empty_evidence_constructs(self):
        ev = CalibrationEvidence()
        assert ev.model_posteriors == {}
        assert ev.noise_posteriors == {}
        assert ev.hardware_posteriors == {}
        assert ev.measurement_posteriors == {}

    def test_evidence_with_model_posteriors(self):
        ev = CalibrationEvidence(
            model_posteriors={"chi": NormalPrior(mean=-2.84e6, sigma=50e3)}
        )
        assert "chi" in ev.model_posteriors
        assert isinstance(ev.model_posteriors["chi"], NormalPrior)

    def test_evidence_with_all_posterior_types(self):
        ev = CalibrationEvidence(
            model_posteriors={"chi": FixedPrior(value=-2.84e6)},
            noise_posteriors={"t1": UniformPrior(low=8e-6, high=12e-6)},
            hardware_posteriors={"drive_q": {"amp_scale": FixedPrior(1.0)}},
            measurement_posteriors={"p_e_given_g": FixedPrior(0.02)},
        )
        assert "chi" in ev.model_posteriors
        assert "t1" in ev.noise_posteriors
        assert "drive_q" in ev.hardware_posteriors
        assert "p_e_given_g" in ev.measurement_posteriors


class TestRandomizerFromCalibration:
    def test_empty_evidence_produces_randomizer(self):
        ev = CalibrationEvidence()
        randomizer = randomizer_from_calibration(ev)
        assert isinstance(randomizer, DomainRandomizer)

    def test_model_posteriors_propagate_to_randomizer(self):
        ev = CalibrationEvidence(
            model_posteriors={"chi": NormalPrior(mean=-2.84e6, sigma=50e3)}
        )
        randomizer = randomizer_from_calibration(ev)
        assert "chi" in randomizer.model_priors_train
        assert "chi" in randomizer.model_priors_eval

    def test_noise_posteriors_propagate(self):
        ev = CalibrationEvidence(
            noise_posteriors={"t1": FixedPrior(value=10e-6)}
        )
        randomizer = randomizer_from_calibration(ev)
        assert "t1" in randomizer.noise_priors_train
        assert "t1" in randomizer.noise_priors_eval

    def test_hardware_posteriors_propagate(self):
        ev = CalibrationEvidence(
            hardware_posteriors={"drive_q": {"amp_scale": FixedPrior(1.0)}}
        )
        randomizer = randomizer_from_calibration(ev)
        assert "drive_q" in randomizer.hardware_priors_train

    def test_train_and_eval_have_same_keys(self):
        ev = CalibrationEvidence(
            model_posteriors={"chi": NormalPrior(mean=-2.84e6, sigma=50e3)},
        )
        randomizer = randomizer_from_calibration(ev)
        # train and eval should have the same keys
        assert randomizer.model_priors_train.keys() == randomizer.model_priors_eval.keys()

    def test_train_and_eval_are_separate_dicts(self):
        ev = CalibrationEvidence(
            model_posteriors={"chi": NormalPrior(mean=-2.84e6, sigma=50e3)},
        )
        randomizer = randomizer_from_calibration(ev)
        # They should be separate dict objects (not the same reference)
        assert randomizer.model_priors_train is not randomizer.model_priors_eval

    def test_fixed_prior_sample_returns_value(self):
        prior = FixedPrior(value=42.0)
        rng = np.random.default_rng(0)
        samples = [prior.sample(rng) for _ in range(5)]
        assert all(s == 42.0 for s in samples)

    def test_normal_prior_sample_is_numeric(self):
        prior = NormalPrior(mean=0.0, sigma=1.0)
        rng = np.random.default_rng(0)
        sample = prior.sample(rng)
        assert isinstance(sample, float)

    def test_uniform_prior_sample_in_range(self):
        prior = UniformPrior(low=1.0, high=2.0)
        rng = np.random.default_rng(0)
        for _ in range(10):
            s = prior.sample(rng)
            assert 1.0 <= s <= 2.0

    def test_choice_prior_sample_from_values(self):
        prior = ChoicePrior(values=(10, 20, 30))
        rng = np.random.default_rng(0)
        for _ in range(10):
            s = prior.sample(rng)
            assert s in (10, 20, 30)


class TestCalibrationEvidenceAdditional:
    def test_measurement_posteriors_propagate(self):
        ev = CalibrationEvidence(
            measurement_posteriors={"iq_sigma": UniformPrior(low=0.03, high=0.07)}
        )
        randomizer = randomizer_from_calibration(ev)
        assert "iq_sigma" in randomizer.measurement_priors_train
        assert "iq_sigma" in randomizer.measurement_priors_eval

    def test_notes_preserved(self):
        ev = CalibrationEvidence(notes={"source": "test", "date": "2025-01-01"})
        assert ev.notes["source"] == "test"

    def test_combined_evidence_keys_propagate(self):
        ev = CalibrationEvidence(
            model_posteriors={"chi": NormalPrior(mean=-2.84e6, sigma=50e3)},
            noise_posteriors={"t1": FixedPrior(value=10e-6)},
            hardware_posteriors={"drive_q": {"amp_scale": FixedPrior(1.0)}},
            measurement_posteriors={"iq_sigma": UniformPrior(low=0.03, high=0.07)},
        )
        randomizer = randomizer_from_calibration(ev)
        assert "chi" in randomizer.model_priors_train
        assert "t1" in randomizer.noise_priors_train
        assert "drive_q" in randomizer.hardware_priors_train
        assert "iq_sigma" in randomizer.measurement_priors_train

    def test_evidence_is_frozen(self):
        ev = CalibrationEvidence()
        with pytest.raises((AttributeError, TypeError)):
            ev.notes = {"mutated": True}  # type: ignore[misc]

    def test_normal_prior_bounded_sample(self):
        prior = NormalPrior(mean=0.0, sigma=1.0, low=-0.5, high=0.5)
        rng = np.random.default_rng(42)
        for _ in range(50):
            s = prior.sample(rng)
            assert -0.5 <= s <= 0.5


class TestMeasuredTraceFitting:
    def test_fit_t1_trace_recovers_decay_with_offset(self):
        delays = np.linspace(0.0, 60.0e-6, 301)
        excited_population = 0.12 + 0.78 * np.exp(-delays / 32.0e-6)
        result = fit_t1_trace(delays, excited_population)

        assert np.isclose(result.fitted_parameters["t1"], 32.0e-6, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["amplitude"], 0.78, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["offset"], 0.12, atol=1.0e-3)

    def test_fit_t2_echo_trace_recovers_decay_with_offset(self):
        delays = np.linspace(0.0, 80.0e-6, 401)
        excited_population = 0.18 + 0.55 * np.exp(-delays / 44.0e-6)
        result = fit_t2_echo_trace(delays, excited_population)

        assert np.isclose(result.fitted_parameters["t2_echo"], 44.0e-6, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["amplitude"], 0.55, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["offset"], 0.18, atol=1.0e-3)

    def test_fit_ramsey_trace_recovers_detuning_and_t2_star(self):
        delays = np.linspace(0.0, 30.0e-6, 1001)
        detuning = 2.0 * np.pi * 0.75e6
        t2_star = 18.0e-6
        excited_population = 0.48 + 0.41 * np.exp(-delays / t2_star) * np.cos(detuning * delays + 0.23)
        result = fit_ramsey_trace(delays, excited_population)

        assert np.isclose(result.fitted_parameters["delta_omega"], detuning, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["t2_star"], t2_star, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["offset"], 0.48, atol=1.0e-3)

    def test_fit_rabi_trace_recovers_scale(self):
        amplitudes = np.linspace(0.0, 2.0, 1001)
        omega_scale = 8.0e7
        duration = 50.0e-9
        excited_population = 0.05 + 0.90 * np.sin(0.5 * omega_scale * amplitudes * duration + 0.31) ** 2
        result = fit_rabi_trace(amplitudes, excited_population, duration=duration)

        assert np.isclose(result.fitted_parameters["omega_scale"], omega_scale, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["duration"], duration, rtol=1.0e-12)
        assert np.isclose(result.fitted_parameters["offset"], 0.05, atol=1.0e-3)

    def test_fit_spectroscopy_trace_recovers_peak_and_linewidth(self):
        drive_frequencies = np.linspace(5.18e9, 5.22e9, 1001)
        omega_peak = 5.2015e9
        linewidth = 2.4e6
        response = 0.09 + 0.82 / (1.0 + ((drive_frequencies - omega_peak) / linewidth) ** 2)
        result = fit_spectroscopy_trace(drive_frequencies, response)

        assert np.isclose(result.fitted_parameters["omega_peak"], omega_peak, rtol=1.0e-5)
        assert np.isclose(result.fitted_parameters["linewidth"], linewidth, rtol=1.0e-2)
        assert np.isclose(result.fitted_parameters["offset"], 0.09, atol=1.0e-3)


class TestFitPosteriors:
    def test_prior_from_fit_uses_fixed_prior_for_zero_sigma(self):
        prior = prior_from_fit(5.0, 0.0)
        assert isinstance(prior, FixedPrior)
        assert prior.value == 5.0

    def test_prior_from_fit_applies_bounds_and_sigma_floor(self):
        prior = prior_from_fit(5.0, 1.0e-9, low=0.0, high=4.0, min_sigma=0.25)
        assert isinstance(prior, NormalPrior)
        assert prior.mean == 4.0
        assert prior.sigma == 0.25
        assert prior.low == 0.0
        assert prior.high == 4.0

    def test_evidence_from_fit_maps_model_parameters(self):
        result = CalibrationResult(
            fitted_parameters={"omega_peak": 5.2e9},
            uncertainties={"omega_peak": 2.0e6},
            raw_data={},
        )
        evidence = evidence_from_fit(
            result,
            category="model",
            parameter_map={"omega_peak": "omega_q"},
            bounds={"omega_q": (0.0, None)},
        )

        assert "omega_q" in evidence.model_posteriors
        prior = evidence.model_posteriors["omega_q"]
        assert isinstance(prior, NormalPrior)
        assert prior.mean == 5.2e9
        assert prior.low == 0.0

    def test_evidence_from_fit_requires_channel_for_hardware(self):
        result = CalibrationResult(fitted_parameters={"channel_gain": 0.9}, uncertainties={}, raw_data={})
        with pytest.raises(ValueError, match="channel"):
            evidence_from_fit(result, category="hardware")

    def test_merge_calibration_evidence_combines_categories(self):
        model_result = CalibrationResult(
            fitted_parameters={"omega_peak": 5.2e9},
            uncertainties={"omega_peak": 2.0e6},
            raw_data={},
        )
        noise_result = CalibrationResult(
            fitted_parameters={"t1": 24.0e-6},
            uncertainties={"t1": 0.8e-6},
            raw_data={},
        )
        model_evidence = evidence_from_fit(
            model_result,
            category="model",
            parameter_map={"omega_peak": "omega_q"},
        )
        noise_evidence = evidence_from_fit(noise_result, category="noise")

        merged = merge_calibration_evidence(
            model_evidence,
            noise_evidence,
            CalibrationEvidence(model_posteriors={"chi": FixedPrior(value=-2.2e6)}),
            notes={"workflow": "test"},
        )

        assert sorted(merged.model_posteriors.keys()) == ["chi", "omega_q"]
        assert sorted(merged.noise_posteriors.keys()) == ["t1"]
        assert merged.notes["workflow"] == "test"
        assert len(merged.notes["fit_sources"]) == 2

    def test_merge_calibration_evidence_rejects_duplicate_keys(self):
        first = CalibrationEvidence(model_posteriors={"omega_q": FixedPrior(value=5.2e9)})
        second = CalibrationEvidence(model_posteriors={"omega_q": FixedPrior(value=5.3e9)})
        with pytest.raises(ValueError, match="Duplicate model posterior"):
            merge_calibration_evidence(first, second)
