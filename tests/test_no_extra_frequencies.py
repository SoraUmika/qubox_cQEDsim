from __future__ import annotations

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.snap_prl133.model import SnapModelConfig, manifold_transition_frequency
from cqed_sim.snap_prl133.pulses import (
    SnapToneParameters,
    has_only_allowed_tones,
    landgraf_envelope,
    slow_stage_multitone_pulse,
)


def test_no_extra_frequencies():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, 0.4, -0.2, 0.8], dtype=float)
    params = SnapToneParameters(
        amplitudes=np.array([1.0, 1.05, 0.95, 1.1]),
        detunings=np.array([0.0, 0.01, -0.008, 0.004]),
        phases=target + np.array([0.0, 0.02, -0.03, 0.01]),
    )
    frame = FrameSpec(omega_q_frame=model.omega_q)
    pulse = slow_stage_multitone_pulse(
        model=model,
        target_phases=target,
        params=params,
        duration=150.0,
        base_amp=0.010,
        frame=frame,
    )
    t_rel = np.linspace(0.0, 1.0, 800, endpoint=False)
    coeff = pulse.envelope(t_rel)
    allowed = np.array(
        [manifold_transition_frequency(model, n, frame=frame) + params.detunings[n] for n in range(target.size)],
        dtype=float,
    )
    ok = has_only_allowed_tones(
        coeff,
        dt=(150.0 / 800.0),
        allowed_omegas=allowed,
        envelope=landgraf_envelope(t_rel),
    )
    assert ok

    # FFT-based leakage ratio on demodulated tone comb (Hann window, +/-1-bin keep rule).
    env = landgraf_envelope(t_rel)
    mask = np.abs(env) > 1e-6 * np.max(np.abs(env))
    sig = np.zeros_like(coeff, dtype=np.complex128)
    sig[mask] = coeff[mask] / env[mask]
    sig_w = sig * np.hanning(sig.size)
    dt = 150.0 / 800.0
    omega = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(sig_w.size, d=dt))
    power = np.abs(np.fft.fftshift(np.fft.fft(sig_w))) ** 2
    keep = np.zeros(power.size, dtype=bool)
    for w in np.concatenate([allowed, -allowed]):
        idx = int(np.argmin(np.abs(omega - w)))
        keep[max(0, idx - 2) : min(power.size, idx + 3)] = True
    r_out = float(np.sum(power[~keep]) / max(np.sum(power), 1e-15))
    assert r_out < 1e-3
