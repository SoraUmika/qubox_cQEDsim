from __future__ import annotations

import time

import numpy as np

from cqed_sim.pulses.hardware import HardwareConfig, image_ratio_db
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def _tone_mag(sig: np.ndarray, dt: float, freq: float) -> float:
    n = sig.size
    freqs = np.fft.fftfreq(n, d=dt)
    fft = np.fft.fft(sig)
    idx = np.argmin(np.abs(freqs - freq / (2 * np.pi)))
    return float(np.abs(fft[idx]) / n)


def test_timeline_overlap_boundaries_alignment():
    start = time.perf_counter()
    p1 = Pulse("q", 0.0, 2.0, _square, amp=1.0)
    p2 = Pulse("q", 1.0, 2.0, _square, amp=0.5)
    c = SequenceCompiler(dt=0.5).compile([p1, p2], t_end=3.0).channels["q"].baseband
    expected = np.array([1.0, 1.0, 1.5, 1.5, 0.5, 0.5, 0.0], dtype=np.complex128)
    assert np.allclose(c, expected)
    assert (time.perf_counter() - start) < 0.6


def test_iq_imbalance_image_tone():
    start = time.perf_counter()
    ifreq = 2 * np.pi * 3.0
    hw = {"q": HardwareConfig(if_freq=0.0, gain_i=1.05, gain_q=0.95, lo_freq=0.0)}
    p = Pulse("q", 0.0, 40.0, _square, amp=1.0, carrier=ifreq)
    compiled = SequenceCompiler(dt=0.01, hardware=hw).compile([p], t_end=40.0)
    sig = compiled.channels["q"].distorted
    main = _tone_mag(sig, 0.01, ifreq)
    image = _tone_mag(sig, 0.01, -ifreq)
    measured_db = 20 * np.log10(max(image / main, 1e-12))
    predicted_db = image_ratio_db(1.05, 0.95)
    assert np.isclose(measured_db, predicted_db, atol=2.5)
    assert (time.perf_counter() - start) < 1.2


def test_dc_offset_generates_lo_leakage():
    start = time.perf_counter()
    lof = 2 * np.pi * 2.0
    hw = {"q": HardwareConfig(lo_freq=lof, dc_i=0.2)}
    p = Pulse("q", 0.0, 30.0, _square, amp=0.0)
    compiled = SequenceCompiler(dt=0.01, hardware=hw).compile([p], t_end=30.0)
    rf = compiled.channels["q"].rf
    lo_mag = _tone_mag(rf.astype(np.complex128), 0.01, lof)
    assert lo_mag > 0.06
    assert (time.perf_counter() - start) < 1.0

