from __future__ import annotations

import numpy as np

from cqed_sim.calibration.sqr import SQRCalibrationResult
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import SQRGate
from cqed_sim.pulses.builders import build_sqr_multitone_pulse
from cqed_sim.pulses.calibration import build_sqr_tone_specs, sqr_lambda0_rad_s, sqr_tone_amplitude_rad_s


def _test_model(n_cav: int = 6) -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2.0 * np.pi * (-2.84e6),
        kerr=0.0,
        n_cav=int(n_cav),
        n_tr=2,
    )


def _test_config() -> dict[str, float | bool]:
    return {
        "duration_sqr_s": 1.0e-6,
        "sqr_sigma_fraction": 1.0 / 6.0,
        "sqr_theta_cutoff": 1.0e-10,
        "use_rotating_frame": True,
        "omega_c_hz": 0.0,
        "omega_q_hz": 0.0,
    }


def test_sqr_tone_specs_include_zero_theta_correction_tone():
    model = _test_model(n_cav=6)
    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)
    theta = [0.0] * 6
    phi = [0.0] * 6
    d_lambda_norm = [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]
    tones = build_sqr_tone_specs(
        model=model,
        frame=frame,
        theta_values=theta,
        phi_values=phi,
        duration_s=1.0e-6,
        d_lambda_values=d_lambda_norm,
        tone_cutoff=1.0e-12,
    )
    assert len(tones) == 1
    assert int(tones[0].manifold) == 1
    expected = sqr_tone_amplitude_rad_s(theta=0.0, duration_s=1.0e-6, d_lambda_norm=0.2)
    assert np.isclose(float(tones[0].amp_rad_s), float(expected), atol=1.0e-9, rtol=0.0)


def test_build_sqr_multitone_pulse_uses_additive_amplitude_correction():
    model = _test_model(n_cav=6)
    config = _test_config()
    gate = SQRGate(index=0, name="sqr_test", theta=tuple([0.0] * 6), phi=tuple([0.0] * 6))
    calibration = SQRCalibrationResult(
        sqr_name=gate.name,
        max_n=5,
        d_lambda=[0.0, 0.15, 0.0, 0.0, 0.0, 0.0],
        d_alpha=[0.0] * 6,
        d_omega_rad_s=[0.0] * 6,
        theta_target=[0.0] * 6,
        phi_target=[0.0] * 6,
        initial_loss=[0.0] * 6,
        optimized_loss=[0.0] * 6,
        levels=[],
        metadata={},
    )
    _, _, meta = build_sqr_multitone_pulse(gate, model, config, calibration=calibration)
    tones = meta["active_tones"]
    assert len(tones) == 1
    assert int(tones[0]["n"]) == 1
    expected_amp = float(sqr_lambda0_rad_s(1.0e-6) * 0.15)
    assert np.isclose(float(tones[0]["amp_rad_s"]), expected_amp, atol=1.0e-9, rtol=0.0)
