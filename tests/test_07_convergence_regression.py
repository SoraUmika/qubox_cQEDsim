from __future__ import annotations

import time

import numpy as np
import pytest

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from tests.utils import load_golden


def _gauss(t):
    return gaussian_envelope(t, sigma=0.2)


def _hard_run(dt: float):
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-2 * np.pi * 0.25,
        chi=2 * np.pi * 0.02,
        kerr=-2 * np.pi * 0.005,
        n_cav=12,
        n_tr=3,
    )
    pulses = [
        Pulse("c", 0.0, 2.6, _gauss, amp=0.18, phase=0.0),
        Pulse("q", 1.0, 1.4, _gauss, amp=0.8, phase=0.2),
        Pulse("c", 1.8, 2.2, _gauss, amp=0.11, phase=np.pi / 3),
        Pulse("q", 2.5, 1.0, _gauss, amp=0.6, phase=-0.4),
    ]
    compiled = SequenceCompiler(dt=dt).compile(pulses, t_end=4.4)
    res = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0),
        {"q": "qubit", "c": "cavity"},
        SimulationConfig(frame=FrameSpec()),
    )
    return {
        "P_e": float(res.expectations["P_e"][-1]),
        "n_c": float(res.expectations["n_c"][-1]),
        "x_c": float(res.expectations["x_c"][-1]),
    }


@pytest.mark.slow
def test_numerical_convergence_and_regression():
    start = time.perf_counter()
    r1 = _hard_run(0.08)
    r2 = _hard_run(0.04)
    r3 = _hard_run(0.02)
    err12 = abs(r1["n_c"] - r2["n_c"])
    err23 = abs(r2["n_c"] - r3["n_c"])
    assert err23 < 8e-4
    assert abs(r1["n_c"] - r3["n_c"]) < 1.2e-3
    golden = load_golden()
    assert np.isclose(r3["P_e"], golden["P_e"], atol=2e-3)
    assert np.isclose(r3["n_c"], golden["n_c"], atol=2e-3)
    assert np.isclose(r3["x_c"], golden["x_c"], atol=3e-3)
    assert (time.perf_counter() - start) < 8.0
