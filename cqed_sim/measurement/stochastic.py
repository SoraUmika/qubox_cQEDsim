from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.drive_targets import SidebandDriveSpec, TransmonTransitionDriveSpec
from cqed_sim.core.frame import FrameSpec
from cqed_sim.sequence.scheduler import CompiledSequence
from cqed_sim.sim.noise import NoiseSpec, split_collapse_operators
from cqed_sim.sim.runner import default_observables, hamiltonian_time_slices
from cqed_sim.solvers.options import build_qutip_solver_options


@dataclass(frozen=True)
class ContinuousReadoutSpec:
    """Configuration for stochastic continuous readout replay via QuTiP SME solving."""

    frame: FrameSpec = FrameSpec()
    monitored_subsystem: str = "cavity"
    ntraj: int = 64
    heterodyne: bool = False
    store_states: bool = False
    keep_runs_results: bool = True
    store_measurement: str = "end"
    atol: float = 1e-8
    rtol: float = 1e-7
    max_step: float | None = None
    solver_options: Mapping[str, Any] = field(default_factory=dict)
    seeds: int | Sequence[int] | None = None
    progress_bar: str | bool = ""


@dataclass
class ContinuousReadoutTrajectory:
    measurement: np.ndarray | None
    final_state: qt.Qobj | None
    states: list[qt.Qobj] | None
    expectations: dict[str, np.ndarray]


@dataclass
class ContinuousReadoutResult:
    average_final_state: qt.Qobj
    average_states: list[qt.Qobj] | None
    average_expectations: dict[str, np.ndarray]
    trajectories: list[ContinuousReadoutTrajectory]
    times: np.ndarray
    monitored_ops: tuple[qt.Qobj, ...]
    unmonitored_ops: tuple[qt.Qobj, ...]
    solver_result: Any

    @property
    def measurement_records(self) -> list[np.ndarray]:
        return [record for record in (traj.measurement for traj in self.trajectories) if record is not None]


def _stochastic_solver_options(spec: ContinuousReadoutSpec) -> dict[str, Any]:
    extra_options: dict[str, Any] = {
        "keep_runs_results": bool(spec.keep_runs_results),
        "store_measurement": spec.store_measurement,
        "tol": max(float(spec.atol), float(spec.rtol)),
    }
    if spec.max_step is not None:
        extra_options["dt"] = float(spec.max_step)
    return build_qutip_solver_options(
        store_states=bool(spec.store_states),
        store_final_state=True,
        progress_bar=spec.progress_bar,
        extra_options=extra_options,
        solver_options=spec.solver_options,
    )


def _trajectory_expectation_slice(expectation_data: np.ndarray, traj_index: int) -> np.ndarray:
    array = np.asarray(expectation_data)
    if array.ndim == 1:
        return np.asarray(array)
    return np.asarray(array[traj_index])


def _result_sequence(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value)


def integrate_measurement_record(
    record: np.ndarray,
    *,
    dt: float = 1.0,
    kernel: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate a homodyne/heterodyne record over its final time axis."""

    samples = np.asarray(record, dtype=float)
    if samples.ndim < 1:
        raise ValueError("record must have at least one dimension.")
    if kernel is None:
        weights = np.ones(samples.shape[-1], dtype=float)
    else:
        weights = np.asarray(kernel, dtype=float)
        if weights.ndim != 1 or weights.shape[0] != samples.shape[-1]:
            raise ValueError("kernel must be one-dimensional and match the time axis of the record.")
    return np.asarray(np.tensordot(samples, weights * float(dt), axes=([-1], [0])))


def simulate_continuous_readout(
    model: Any,
    compiled: CompiledSequence,
    initial_state: qt.Qobj,
    drive_ops: dict[str, str | TransmonTransitionDriveSpec | SidebandDriveSpec],
    *,
    noise: NoiseSpec,
    spec: ContinuousReadoutSpec | None = None,
    c_ops: Sequence[qt.Qobj] | None = None,
    sc_ops: Sequence[qt.Qobj] | None = None,
    e_ops: dict[str, qt.Qobj] | None = None,
) -> ContinuousReadoutResult:
    """Run stochastic continuous-measurement replay on a compiled pulse sequence."""

    spec = ContinuousReadoutSpec() if spec is None else spec
    observables = default_observables(model) if e_ops is None else dict(e_ops)
    observable_names = tuple(observables.keys())
    observable_ops = tuple(observables.values())
    hamiltonian = hamiltonian_time_slices(model, compiled, drive_ops, frame=spec.frame)
    noise_unmonitored, noise_monitored = split_collapse_operators(
        model,
        noise,
        monitored_subsystem=spec.monitored_subsystem,
    )
    effective_c_ops = tuple([*(c_ops or ()), *noise_unmonitored])
    effective_sc_ops = tuple([*(sc_ops or ()), *noise_monitored])
    if not effective_sc_ops:
        raise ValueError(
            "No monitored stochastic operators were resolved. "
            "Provide `sc_ops` explicitly or choose a monitored_subsystem with readout loss."
        )

    rho0 = initial_state if initial_state.isoper else initial_state.proj()
    result = qt.smesolve(
        hamiltonian,
        rho0,
        compiled.tlist,
        c_ops=list(effective_c_ops),
        sc_ops=list(effective_sc_ops),
        heterodyne=bool(spec.heterodyne),
        e_ops=list(observable_ops),
        ntraj=int(spec.ntraj),
        seeds=spec.seeds,
        options=_stochastic_solver_options(spec),
    )

    average_expectations = {
        name: np.asarray(result.average_expect[idx])
        for idx, name in enumerate(observable_names)
    }
    average_states = list(result.states) if spec.store_states and getattr(result, "states", None) is not None else None
    runs_expect = _result_sequence(getattr(result, "runs_expect", None))
    runs_states = _result_sequence(getattr(result, "runs_states", None))
    runs_final_states = _result_sequence(getattr(result, "runs_final_states", None))
    measurements = _result_sequence(getattr(result, "measurement", None))

    trajectories: list[ContinuousReadoutTrajectory] = []
    n_traj = max(
        len(measurements),
        len(runs_states),
        len(runs_final_states),
        int(spec.ntraj),
    )
    for traj_index in range(n_traj):
        trajectory_expectations = {
            name: _trajectory_expectation_slice(runs_expect[idx], traj_index)
            for idx, name in enumerate(observable_names)
        } if runs_expect else {}
        trajectories.append(
            ContinuousReadoutTrajectory(
                measurement=None
                if traj_index >= len(measurements)
                else np.asarray(np.real(measurements[traj_index]), dtype=float),
                final_state=None if traj_index >= len(runs_final_states) else runs_final_states[traj_index],
                states=None if traj_index >= len(runs_states) else list(runs_states[traj_index]),
                expectations=trajectory_expectations,
            )
        )

    return ContinuousReadoutResult(
        average_final_state=result.average_final_state,
        average_states=average_states,
        average_expectations=average_expectations,
        trajectories=trajectories,
        times=np.asarray(compiled.tlist, dtype=float),
        monitored_ops=effective_sc_ops,
        unmonitored_ops=effective_c_ops,
        solver_result=result,
    )


__all__ = [
    "ContinuousReadoutSpec",
    "ContinuousReadoutTrajectory",
    "ContinuousReadoutResult",
    "integrate_measurement_record",
    "simulate_continuous_readout",
]
