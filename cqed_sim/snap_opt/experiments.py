from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from cqed_sim.snap_opt.pulses import SnapToneParameters, slow_stage_multitone_pulse


@dataclass(frozen=True)
class SnapRunConfig:
    duration: float
    dt: float
    base_amp: float
    apply_fast_reset: bool = False


def run_snap_stage(
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
    params: SnapToneParameters,
    cfg: SnapRunConfig,
    initial_state: qt.Qobj,
    frame: FrameSpec | None = None,
) -> tuple[qt.Qobj, np.ndarray, np.ndarray]:
    frame = frame or FrameSpec(omega_q_frame=model.omega_q)
    pulse = slow_stage_multitone_pulse(
        model=model,
        target_phases=target_phases,
        params=params,
        duration=cfg.duration,
        base_amp=cfg.base_amp,
        frame=frame,
        channel="q",
    )
    compiler = SequenceCompiler(dt=cfg.dt)
    compiled = compiler.compile([pulse], t_end=cfg.duration + cfg.dt)
    res = simulate_sequence(
        model=model,
        compiled=compiled,
        initial_state=initial_state,
        drive_ops={"q": "qubit"},
        config=SimulationConfig(frame=frame),
    )
    out = res.final_state
    if cfg.apply_fast_reset:
        # Idealized global qubit X reset pulse approximation.
        x = qt.tensor(qt.sigmax(), qt.qeye(model.n_cav))
        out = x * out if not out.isoper else x * out * x.dag()
    return out, compiled.tlist, compiled.channels["q"].baseband


def target_difficulty_metric(target_phases: np.ndarray) -> float:
    # Phase roughness + span as simple target complexity scalar.
    ph = np.unwrap(np.asarray(target_phases, dtype=float))
    rough = np.linalg.norm(np.diff(ph), ord=2)
    span = float(np.max(ph) - np.min(ph))
    return float(rough + 0.5 * span)

