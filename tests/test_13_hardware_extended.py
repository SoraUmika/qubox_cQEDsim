from __future__ import annotations

import time

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.hardware import HardwareConfig
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_zoh_hold_equivalence_to_piecewise_constant():
    start = time.perf_counter()
    p = Pulse("q", 0.0, 1.0, _square, amp=0.4)
    c1 = SequenceCompiler(dt=0.01, hardware={"q": HardwareConfig(zoh_samples=4)}).compile([p], t_end=1.0)
    c2 = SequenceCompiler(dt=0.01).compile([p], t_end=1.0)
    manual = c2.channels["q"].baseband.copy()
    for i in range(0, manual.size, 4):
        manual[i : i + 4] = manual[i]
    assert np.allclose(c1.channels["q"].baseband, manual)
    assert (time.perf_counter() - start) < 0.7


def test_lowpass_filter_group_delay_and_amplitude_rolloff():
    start = time.perf_counter()
    dt = 0.002
    t_end = 4.0
    c = SequenceCompiler(dt=dt, hardware={"q": HardwareConfig(lowpass_bw=8.0)})
    low = c.compile([Pulse("q", 0.0, t_end, _square, amp=1.0, carrier=2 * np.pi * 2.0)], t_end=t_end).channels["q"].baseband
    high = c.compile([Pulse("q", 0.0, t_end, _square, amp=1.0, carrier=2 * np.pi * 20.0)], t_end=t_end).channels["q"].baseband
    assert np.mean(np.abs(high)) < np.mean(np.abs(low))
    step = c.compile([Pulse("q", 0.5, t_end - 0.5, _square, amp=1.0)], t_end=t_end).channels["q"].baseband.real
    t10 = np.argmax(step > 0.1)
    assert t10 > int(0.5 / dt)
    assert (time.perf_counter() - start) < 1.2


def test_iq_sideband_selection_usb_lsb():
    start = time.perf_counter()
    delta = 2 * np.pi * 0.35
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    hw_plus = {"q": HardwareConfig(if_freq=delta)}
    hw_minus = {"q": HardwareConfig(if_freq=-delta)}
    p = Pulse("q", 0.0, 2.0, _square, amp=np.pi / 4.0, carrier=0.0)
    r_plus = simulate_sequence(
        m,
        SequenceCompiler(dt=0.01, hardware=hw_plus).compile([p], t_end=2.1),
        m.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec(omega_q_frame=delta)),
    )
    r_minus = simulate_sequence(
        m,
        SequenceCompiler(dt=0.01, hardware=hw_minus).compile([p], t_end=2.1),
        m.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec(omega_q_frame=delta)),
    )
    assert r_plus.expectations["P_e"][-1] > r_minus.expectations["P_e"][-1] + 0.15
    assert (time.perf_counter() - start) < 1.3


def test_amplitude_quantization_effect_is_bounded():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    p = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    rc = simulate_sequence(
        m, SequenceCompiler(dt=0.01).compile([p], t_end=1.1), m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig()
    )
    rq = simulate_sequence(
        m,
        SequenceCompiler(dt=0.01, hardware={"q": HardwareConfig(amplitude_bits=8)}).compile([p], t_end=1.1),
        m.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(),
    )
    err = abs(rc.expectations["P_e"][-1] - rq.expectations["P_e"][-1])
    assert err < 0.015
    assert (time.perf_counter() - start) < 1.2


def test_channel_crosstalk_matrix_smoke():
    start = time.perf_counter()
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=10, n_tr=2)
    p = Pulse("q", 0.0, 1.8, _square, amp=0.3)
    c_off = SequenceCompiler(dt=0.01, crosstalk_matrix={}).compile([p], t_end=2.0)
    c_on = SequenceCompiler(dt=0.01, crosstalk_matrix={"q": {"c": 0.15}}).compile([p], t_end=2.0)
    r_off = simulate_sequence(m, c_off, m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig())
    r_on = simulate_sequence(m, c_on, m.basis_state(0, 0), {"q": "qubit", "c": "cavity"}, SimulationConfig())
    assert r_on.expectations["n_c"][-1] > r_off.expectations["n_c"][-1] + 5e-3
    assert (time.perf_counter() - start) < 1.8


def test_timing_quantization_rounding_rules():
    start = time.perf_counter()
    hw = {"q": HardwareConfig(timing_quantum=0.1)}
    p = Pulse("q", 0.24, 0.3, _square, amp=1.0)
    c = SequenceCompiler(dt=0.1, hardware=hw).compile([p], t_end=0.8)
    sig = c.channels["q"].baseband.real
    # 0.24 rounds to 0.2 with half-up rule.
    assert np.isclose(sig[2], 1.0)
    assert np.isclose(sig[1], 0.0)
    assert (time.perf_counter() - start) < 0.7
