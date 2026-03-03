from __future__ import annotations

import time

import numpy as np
import pytest
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.hardware import HardwareConfig
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


@pytest.fixture(autouse=True)
def _fixed_seed():
    np.random.seed(12345)


@pytest.fixture
def base_model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-2 * np.pi * 0.22,
        chi=2 * np.pi * 0.015,
        kerr=-2 * np.pi * 0.003,
        n_cav=14,
        n_tr=3,
    )


def square(_: np.ndarray) -> np.ndarray:
    return np.ones_like(_, dtype=np.complex128)


def run_sequence(
    model: DispersiveTransmonCavityModel,
    pulses: list[Pulse],
    drive_ops: dict[str, str],
    initial: qt.Qobj,
    dt: float = 0.5,
    t_end: float | None = None,
    frame: FrameSpec | None = None,
    hardware: dict[str, HardwareConfig] | None = None,
):
    compiler = SequenceCompiler(dt=dt, hardware=hardware)
    compiled = compiler.compile(pulses, t_end=t_end)
    cfg = SimulationConfig(frame=frame or FrameSpec(), atol=1e-9, rtol=1e-8)
    result = simulate_sequence(model, compiled, initial, drive_ops, config=cfg)
    return compiled, result


def assert_runtime_under(start: float, threshold_s: float):
    elapsed = time.perf_counter() - start
    assert elapsed < threshold_s, f"runtime {elapsed:.3f}s exceeded {threshold_s:.3f}s"

