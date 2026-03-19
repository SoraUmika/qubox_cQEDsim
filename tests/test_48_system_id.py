"""Tests for cqed_sim.system_id — calibration bridge to domain randomization."""
from __future__ import annotations
import numpy as np
import pytest
from cqed_sim.system_id import (
    CalibrationEvidence,
    randomizer_from_calibration,
    FixedPrior,
    NormalPrior,
    UniformPrior,
    ChoicePrior,
)
from cqed_sim.rl_control.domain_randomization import DomainRandomizer


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
