from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import qutip as qt

from cqed_sim.solvers.master_equation import MasterEquationConfig, solve_master_equation


@dataclass(frozen=True)
class TrajectoryConfig:
    ntraj: int = 128
    heterodyne: bool = True
    eta: float = 1.0
    additive_noise_std: float = 0.0
    seed: int | None = None
    store_states: bool = False
    master_equation_config: MasterEquationConfig | None = None


@dataclass
class MeasurementTrajectory:
    final_state: qt.Qobj
    I: np.ndarray
    Q: np.ndarray | None
    states: list[qt.Qobj] | None = None


@dataclass
class TrajectoryResult:
    times: np.ndarray
    trajectories: list[MeasurementTrajectory]
    mean_I: np.ndarray
    mean_Q: np.ndarray | None
    deterministic_state: qt.Qobj


def _expectation_trace(states: Sequence[qt.Qobj], op: qt.Qobj) -> np.ndarray:
    return np.asarray([complex((op * (state if state.isoper else state.proj())).tr()) for state in states], dtype=np.complex128)


def simulate_measurement_trajectories(
    hamiltonian,
    rho0: qt.Qobj,
    *,
    tlist: Sequence[float],
    output_operator: qt.Qobj,
    c_ops: Sequence[qt.Qobj] = (),
    config: TrajectoryConfig | None = None,
) -> TrajectoryResult:
    """Generate homodyne or heterodyne measurement records.

    This routine uses deterministic Lindblad evolution for the conditional mean
    field and samples Gaussian measurement noise around that mean.  It is a
    lightweight validation path; use the existing `measurement.simulate_continuous_readout`
    SME wrapper when full quantum backaction trajectories are required.
    """

    config = TrajectoryConfig() if config is None else config
    eta = float(config.eta)
    if eta < 0.0 or eta > 1.0:
        raise ValueError("eta must lie in [0, 1].")
    times = np.asarray(tlist, dtype=float)
    if times.size < 2:
        raise ValueError("tlist must contain at least two times.")
    master_config = config.master_equation_config or MasterEquationConfig()
    master_config = replace(master_config, store_states=True)
    master = solve_master_equation(
        hamiltonian,
        rho0,
        tlist=times,
        c_ops=c_ops,
        e_ops={
            "output_I": output_operator + output_operator.dag(),
            "output_Q": -1j * (output_operator - output_operator.dag()),
        },
        config=master_config,
    )
    states = master.states or [master.final_state]
    mean_i = np.asarray(master.expectations["output_I"], dtype=float) * np.sqrt(eta)
    mean_q = np.asarray(master.expectations["output_Q"], dtype=float) * np.sqrt(eta)
    dt = np.diff(times, prepend=times[0])
    dt[0] = dt[1] if dt.size > 1 else 1.0
    sigma = 1.0 / np.sqrt(np.maximum(dt, 1.0e-30))
    if config.additive_noise_std > 0.0:
        sigma = np.sqrt(sigma * sigma + float(config.additive_noise_std) ** 2)
    rng = np.random.default_rng(config.seed)
    trajectories: list[MeasurementTrajectory] = []
    for _ in range(int(config.ntraj)):
        I = mean_i + rng.normal(scale=sigma, size=mean_i.shape)
        Q = None
        if config.heterodyne:
            Q = mean_q + rng.normal(scale=sigma, size=mean_q.shape)
        trajectories.append(
            MeasurementTrajectory(
                final_state=master.final_state,
                I=np.asarray(I, dtype=float),
                Q=None if Q is None else np.asarray(Q, dtype=float),
                states=states if config.store_states else None,
            )
        )
    mean_I = np.mean(np.vstack([traj.I for traj in trajectories]), axis=0)
    mean_Q = None
    if config.heterodyne:
        mean_Q = np.mean(np.vstack([traj.Q for traj in trajectories if traj.Q is not None]), axis=0)
    return TrajectoryResult(
        times=times,
        trajectories=trajectories,
        mean_I=np.asarray(mean_I, dtype=float),
        mean_Q=None if mean_Q is None else np.asarray(mean_Q, dtype=float),
        deterministic_state=master.final_state,
    )


__all__ = [
    "MeasurementTrajectory",
    "TrajectoryConfig",
    "TrajectoryResult",
    "simulate_measurement_trajectories",
]
