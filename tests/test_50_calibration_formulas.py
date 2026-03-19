"""Analytic tests for closed-form calibration formulas in pulses/calibration.py."""
from __future__ import annotations
import numpy as np
import pytest
from cqed_sim.pulses.calibration import (
    displacement_square_amplitude,
    rotation_gaussian_amplitude,
    sqr_lambda0_rad_s,
    sqr_rotation_coefficient,
    sqr_tone_amplitude_rad_s,
)


class TestDisplacementSquareAmplitude:
    """displacement_square_amplitude(alpha, duration_s) = 1j * alpha / duration_s"""

    def test_real_alpha(self):
        alpha = 1.0 + 0.0j
        duration = 100.0
        amp = displacement_square_amplitude(alpha, duration)
        assert abs(amp - 1j * alpha / duration) < 1e-14

    def test_complex_alpha(self):
        alpha = 1.0 + 0.5j
        duration = 50.0
        amp = displacement_square_amplitude(alpha, duration)
        assert abs(amp - 1j * alpha / duration) < 1e-14

    def test_imaginary_prefactor_is_correct(self):
        """The 1j factor ensures the integral of a square envelope gives alpha."""
        alpha = 2.0 + 0.0j
        duration = 1.0
        amp = displacement_square_amplitude(alpha, duration)
        # amp = 1j * alpha / duration = 1j * 2.0 / 1.0 = 2j
        # imag part = real(alpha) / duration
        assert amp.imag == pytest.approx(float(alpha.real) / duration)
        # real part = -imag(alpha) / duration
        assert amp.real == pytest.approx(-float(alpha.imag) / duration)

    def test_scales_inversely_with_duration(self):
        alpha = 1.0 + 0.0j
        amp1 = displacement_square_amplitude(alpha, 100.0)
        amp2 = displacement_square_amplitude(alpha, 200.0)
        assert abs(amp2) == pytest.approx(abs(amp1) / 2.0)

    def test_zero_alpha_gives_zero(self):
        amp = displacement_square_amplitude(0.0 + 0.0j, 100.0)
        assert amp == 0.0

    def test_purely_imaginary_alpha(self):
        alpha = 0.0 + 1.0j
        duration = 1.0
        amp = displacement_square_amplitude(alpha, duration)
        # 1j * (0 + 1j) / 1.0 = 1j * 1j = -1
        assert abs(amp - (-1.0 + 0j)) < 1e-14


class TestRotationGaussianAmplitude:
    """rotation_gaussian_amplitude(theta, duration_s) = theta / (2 * duration_s)"""

    def test_pi_rotation_unit_duration(self):
        amp = rotation_gaussian_amplitude(np.pi, 1.0)
        assert amp == pytest.approx(np.pi / 2.0)

    def test_half_pi_rotation(self):
        amp = rotation_gaussian_amplitude(np.pi / 2, 100.0)
        assert amp == pytest.approx(np.pi / 4 / 100.0)

    def test_scales_linearly_with_theta(self):
        amp1 = rotation_gaussian_amplitude(np.pi / 4, 100.0)
        amp2 = rotation_gaussian_amplitude(np.pi / 2, 100.0)
        assert amp2 == pytest.approx(2.0 * amp1)

    def test_scales_inversely_with_duration(self):
        amp1 = rotation_gaussian_amplitude(np.pi, 100.0)
        amp2 = rotation_gaussian_amplitude(np.pi, 200.0)
        assert amp2 == pytest.approx(amp1 / 2.0)

    def test_zero_rotation_gives_zero(self):
        amp = rotation_gaussian_amplitude(0.0, 100.0)
        assert amp == 0.0

    def test_negative_rotation(self):
        amp = rotation_gaussian_amplitude(-np.pi, 100.0)
        assert amp == pytest.approx(-np.pi / 200.0)

    def test_two_pi_rotation(self):
        amp = rotation_gaussian_amplitude(2 * np.pi, 50.0)
        assert amp == pytest.approx(2 * np.pi / 100.0)


class TestSqrLambda0:
    """sqr_lambda0_rad_s(duration_s) = pi / (2 * duration_s)"""

    def test_unit_duration(self):
        lam = sqr_lambda0_rad_s(1.0)
        assert lam == pytest.approx(np.pi / 2.0)

    def test_100_duration(self):
        lam = sqr_lambda0_rad_s(100.0)
        assert lam == pytest.approx(np.pi / 200.0)

    def test_scales_inversely_with_duration(self):
        lam1 = sqr_lambda0_rad_s(100.0)
        lam2 = sqr_lambda0_rad_s(200.0)
        assert lam2 == pytest.approx(lam1 / 2.0)

    def test_zero_duration_returns_zero(self):
        lam = sqr_lambda0_rad_s(0.0)
        assert lam == 0.0

    def test_nearly_zero_duration_returns_zero(self):
        lam = sqr_lambda0_rad_s(1e-16)
        assert lam == 0.0

    def test_positive_value_for_positive_duration(self):
        lam = sqr_lambda0_rad_s(50.0)
        assert lam > 0.0

    def test_large_duration(self):
        lam = sqr_lambda0_rad_s(1e6)
        assert lam == pytest.approx(np.pi / (2e6))


class TestSqrRotationCoefficient:
    """sqr_rotation_coefficient(theta, d_lambda_norm) = theta/pi + d_lambda_norm"""

    def test_pi_rotation_no_correction(self):
        coeff = sqr_rotation_coefficient(np.pi, 0.0)
        assert coeff == pytest.approx(1.0)

    def test_half_pi_rotation_no_correction(self):
        coeff = sqr_rotation_coefficient(np.pi / 2, 0.0)
        assert coeff == pytest.approx(0.5)

    def test_correction_adds_linearly(self):
        coeff = sqr_rotation_coefficient(np.pi, 0.1)
        assert coeff == pytest.approx(1.0 + 0.1)

    def test_zero_rotation_zero_correction(self):
        coeff = sqr_rotation_coefficient(0.0, 0.0)
        assert coeff == pytest.approx(0.0)

    def test_negative_correction(self):
        coeff = sqr_rotation_coefficient(np.pi, -0.05)
        assert coeff == pytest.approx(1.0 - 0.05)

    def test_two_pi_rotation(self):
        coeff = sqr_rotation_coefficient(2 * np.pi, 0.0)
        assert coeff == pytest.approx(2.0)


class TestSqrToneAmplitude:
    """sqr_tone_amplitude_rad_s = lambda0 * rotation_coefficient"""

    def test_pi_rotation_tone_amplitude(self):
        duration = 100.0
        lam0 = np.pi / (2 * duration)
        amp = sqr_tone_amplitude_rad_s(np.pi, duration, d_lambda_norm=0.0)
        assert amp == pytest.approx(lam0 * 1.0)

    def test_no_rotation_no_amplitude(self):
        amp = sqr_tone_amplitude_rad_s(0.0, 100.0, d_lambda_norm=0.0)
        assert amp == pytest.approx(0.0)

    def test_half_pi_rotation(self):
        duration = 100.0
        lam0 = np.pi / (2 * duration)
        amp = sqr_tone_amplitude_rad_s(np.pi / 2, duration, d_lambda_norm=0.0)
        assert amp == pytest.approx(lam0 * 0.5)

    def test_default_d_lambda_norm_is_zero(self):
        # calling without d_lambda_norm should behave as d_lambda_norm=0
        duration = 50.0
        amp_explicit = sqr_tone_amplitude_rad_s(np.pi, duration, d_lambda_norm=0.0)
        amp_default = sqr_tone_amplitude_rad_s(np.pi, duration)
        assert amp_default == pytest.approx(amp_explicit)

    def test_correction_shifts_amplitude(self):
        duration = 100.0
        lam0 = np.pi / (2 * duration)
        d_lambda = 0.1
        amp = sqr_tone_amplitude_rad_s(np.pi, duration, d_lambda_norm=d_lambda)
        expected = lam0 * (1.0 + d_lambda)
        assert amp == pytest.approx(expected)
