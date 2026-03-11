from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import carrier_for_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _mhz_to_rad_per_ns(mhz: float) -> float:
    return float(2.0 * np.pi * mhz * 1e-3)


def _gauss_sigma_018(t_rel: np.ndarray) -> np.ndarray:
    return gaussian_envelope(t_rel, sigma=0.18)


def _selective_gaussian_calibrated(t_rel: np.ndarray, sigma_rel: float = 1.0 / 6.0) -> np.ndarray:
    env = np.asarray(gaussian_envelope(t_rel, sigma=sigma_rel), dtype=np.complex128)
    mean_val = float(np.mean(np.real(env)))
    if abs(mean_val) < 1e-12:
        return env
    return env / mean_val


def _local_maxima_indices(y: np.ndarray) -> np.ndarray:
    idx = [i for i in range(1, len(y) - 1) if y[i] > y[i - 1] and y[i] >= y[i + 1]]
    return np.asarray(idx, dtype=int)


@pytest.mark.slow
def test_displacement_selective_spectroscopy_peak_spacing_tracks_chi():
    chi_mhz = -2.84
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=_mhz_to_rad_per_ns(chi_mhz),
        chi_higher=(),
        kerr=0.0,
        n_cav=12,
        n_tr=2,
    )

    dt_ns = 1.0
    disp_duration_ns = 80.0
    spec_duration_ns = 2000.0
    gap_ns = 20.0
    disp_amp = 0.035
    spec_amp = 0.004
    detunings_mhz = np.arange(-12.0, 2.0 + 1e-12, 0.4)

    disp_pulse = Pulse(
        channel="c",
        t0=0.0,
        duration=disp_duration_ns,
        envelope=_gauss_sigma_018,
        carrier=0.0,
        phase=0.0,
        amp=disp_amp,
    )
    spec_env = lambda t_rel: _selective_gaussian_calibrated(t_rel, sigma_rel=1.0 / 6.0)

    rho0 = model.basis_state( 0,0)
    pe = np.zeros_like(detunings_mhz, dtype=float)
    for idx, det_mhz in enumerate(detunings_mhz):
        spec_pulse = Pulse(
            channel="q",
            t0=disp_duration_ns + gap_ns,
            duration=spec_duration_ns,
            envelope=spec_env,
            carrier=carrier_for_transition_frequency(_mhz_to_rad_per_ns(float(det_mhz))),
            phase=0.0,
            amp=spec_amp,
        )
        t_end = disp_duration_ns + gap_ns + spec_duration_ns + dt_ns
        compiled = SequenceCompiler(dt=dt_ns).compile([disp_pulse, spec_pulse], t_end=t_end)
        res = simulate_sequence(
            model,
            compiled,
            rho0,
            {"c": "cavity", "q": "qubit"},
            config=SimulationConfig(frame=FrameSpec(omega_q_frame=model.omega_q)),
        )
        pe[idx] = float(np.real(res.expectations["P_e"][-1]))

    maxima_idx = _local_maxima_indices(pe)
    assert maxima_idx.size >= 3
    maxima_idx = maxima_idx[pe[maxima_idx] >= 0.12 * np.max(pe)]
    assert maxima_idx.size >= 3

    peak_detunings = np.sort(detunings_mhz[maxima_idx])
    dominant = float(detunings_mhz[int(np.argmax(pe))])
    nearby = np.abs(peak_detunings - dominant) <= 6.5
    peak_detunings = peak_detunings[nearby]
    assert peak_detunings.size >= 3

    spacings = np.diff(peak_detunings)
    target = abs(float(chi_mhz))
    assert np.all(np.abs(spacings - target) < 0.75)
    assert dominant < 0.0
    assert np.isclose(dominant, chi_mhz, atol=0.8)
