from __future__ import annotations

import time

import numpy as np

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import simulate_sequence, SimulationConfig


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_fast_path_runtime_budget():
    start = time.perf_counter()
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=8, n_tr=2
    )
    pulse = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 3)
    compiled = SequenceCompiler(dt=0.02).compile([pulse], t_end=1.2)
    simulate_sequence(model, compiled, model.basis_state(0, 0), {"q": "qubit"}, SimulationConfig())
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0

