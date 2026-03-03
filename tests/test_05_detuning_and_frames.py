from __future__ import annotations

import time

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_detuned_drive_reduces_inversion():
    start = time.perf_counter()
    wq = 2 * np.pi * 1.2
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=wq, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    amp = np.pi / 2
    p_res = Pulse("q", 0.0, 1.0, _square, amp=amp, carrier=0.0)
    p_det = Pulse("q", 0.0, 1.0, _square, amp=amp, carrier=2 * np.pi * 0.4)
    comp = SequenceCompiler(dt=0.01)
    r_res = simulate_sequence(
        model,
        comp.compile([p_res], t_end=1.1),
        model.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec(omega_q_frame=wq)),
    )
    r_det = simulate_sequence(
        model,
        comp.compile([p_det], t_end=1.1),
        model.basis_state(0, 0),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec(omega_q_frame=wq)),
    )
    assert r_res.expectations["P_e"][-1] > r_det.expectations["P_e"][-1] + 0.15
    assert (time.perf_counter() - start) < 1.4


def test_frame_invariance_for_observables():
    start = time.perf_counter()
    wq = 2 * np.pi * 1.0
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=wq, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    compiler = SequenceCompiler(dt=0.01)
    r_lab = simulate_sequence(
        model,
        compiler.compile([], t_end=1.5),
        (model.basis_state(0, 0) + model.basis_state(0, 1)).unit(),
        {},
        SimulationConfig(frame=FrameSpec(omega_q_frame=0.0)),
    )
    r_rot = simulate_sequence(
        model,
        compiler.compile([], t_end=1.5),
        (model.basis_state(0, 0) + model.basis_state(0, 1)).unit(),
        {},
        SimulationConfig(frame=FrameSpec(omega_q_frame=wq)),
    )
    assert np.isclose(r_lab.expectations["P_e"][-1], r_rot.expectations["P_e"][-1], atol=1e-6)
    assert np.isclose(r_lab.expectations["n_c"][-1], r_rot.expectations["n_c"][-1], atol=1e-6)
    assert (time.perf_counter() - start) < 2.0
