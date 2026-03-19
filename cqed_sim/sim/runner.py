from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Any, Iterable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.backends.base_backend import BaseBackend
from cqed_sim.core.drive_targets import SidebandDriveSpec, TransmonTransitionDriveSpec
from cqed_sim.core.frame import FrameSpec
from cqed_sim.sequence.scheduler import CompiledSequence
from cqed_sim.sim.noise import NoiseSpec, collapse_operators
from cqed_sim.sim.solver import solve_with_backend


@dataclass(frozen=True)
class SimulationConfig:
    frame: FrameSpec = FrameSpec()
    atol: float = 1e-8
    rtol: float = 1e-7
    max_step: float | None = None
    store_states: bool = False
    backend: BaseBackend | None = None


@dataclass
class SimulationResult:
    final_state: qt.Qobj
    states: list[qt.Qobj] | None
    expectations: dict[str, np.ndarray]
    solver_result: Any


def _projector_onto_first_excited_state(subsystem_dims: tuple[int, ...]) -> qt.Qobj:
    factors = [qt.basis(subsystem_dims[0], 1) * qt.basis(subsystem_dims[0], 1).dag()]
    factors.extend(qt.qeye(dim) for dim in subsystem_dims[1:])
    return qt.tensor(*factors)


def _projector_onto_transmon_level(subsystem_dims: tuple[int, ...], level: int) -> qt.Qobj:
    factors = [qt.basis(subsystem_dims[0], int(level)) * qt.basis(subsystem_dims[0], int(level)).dag()]
    factors.extend(qt.qeye(dim) for dim in subsystem_dims[1:])
    return qt.tensor(*factors)


def _mode_quadratures(lowering: qt.Qobj, raising: qt.Qobj) -> tuple[qt.Qobj, qt.Qobj]:
    return lowering + raising, -1j * (lowering - raising)


def default_observables(model: Any) -> dict[str, qt.Qobj]:
    cached = getattr(model, "_default_observables_cache", None)
    if cached is not None:
        return cached

    ops = model.operators()
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    has_transmon = bool(getattr(model, "has_transmon", "n_q" in ops and "b" in ops))
    observables: dict[str, qt.Qobj] = {}
    if has_transmon:
        for level in range(dims[0]):
            projector = _projector_onto_transmon_level(dims, level)
            observables[f"P_q{level}"] = projector
            if level == 0:
                observables["P_g"] = projector
            elif level == 1:
                observables["P_e"] = projector
            elif level == 2:
                observables["P_f"] = projector

    if "a" in ops:
        x_c, p_c = _mode_quadratures(ops["a"], ops["adag"])
        observables.update({"n_c": ops["n_c"], "x_c": x_c, "p_c": p_c})
    if "a_s" in ops:
        x_s, p_s = _mode_quadratures(ops["a_s"], ops["adag_s"])
        observables.update({"n_s": ops["n_s"], "x_s": x_s, "p_s": p_s})
    if "a_r" in ops:
        x_r, p_r = _mode_quadratures(ops["a_r"], ops["adag_r"])
        observables.update({"n_r": ops["n_r"], "x_r": x_r, "p_r": p_r})

    setattr(model, "_default_observables_cache", observables)
    return observables


def _legacy_drive_couplings(model: Any) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
    ops = model.operators()
    couplings: dict[str, tuple[qt.Qobj, qt.Qobj]] = {}
    if "a" in ops:
        couplings["cavity"] = (ops["adag"], ops["a"])
        couplings["storage"] = (ops["adag"], ops["a"])
    if "b" in ops:
        couplings["qubit"] = (ops["bdag"], ops["b"])
    if {"a", "adag", "b", "bdag"}.issubset(ops):
        couplings["sideband"] = (ops["adag"] * ops["b"], ops["a"] * ops["bdag"])
    return couplings


def _resolve_drive_target(
    model: Any,
    target: str | TransmonTransitionDriveSpec | SidebandDriveSpec,
    couplings: dict[str, tuple[qt.Qobj, qt.Qobj]],
) -> tuple[qt.Qobj, qt.Qobj]:
    if isinstance(target, str):
        if target not in couplings:
            raise ValueError(f"Unsupported target '{target}'.")
        return couplings[target]
    if isinstance(target, TransmonTransitionDriveSpec):
        if not hasattr(model, "transmon_transition_operators"):
            raise ValueError("Model does not support structured transmon transition targets.")
        return model.transmon_transition_operators(target.lower_level, target.upper_level)
    if isinstance(target, SidebandDriveSpec):
        if not hasattr(model, "sideband_drive_operators"):
            raise ValueError("Model does not support structured sideband targets.")
        return model.sideband_drive_operators(
            mode=target.mode,
            lower_level=target.lower_level,
            upper_level=target.upper_level,
            sideband=target.sideband,
        )
    raise TypeError(f"Unsupported drive target type '{type(target).__name__}'.")


def hamiltonian_time_slices(
    model: Any,
    compiled: CompiledSequence,
    drive_ops: dict[str, str | TransmonTransitionDriveSpec | SidebandDriveSpec],
    frame: FrameSpec | None = None,
) -> list:
    frame = frame or FrameSpec()
    couplings = model.drive_coupling_operators() if hasattr(model, "drive_coupling_operators") else _legacy_drive_couplings(model)

    h = [model.static_hamiltonian(frame)]
    for channel, target in drive_ops.items():
        coeff = compiled.channels[channel].distorted
        raising, lowering = _resolve_drive_target(model, target, couplings)
        h.append([raising, coeff])
        h.append([lowering, np.conj(coeff)])
    return h


def _solver_options(cfg: SimulationConfig) -> dict[str, Any]:
    options: dict[str, Any] = {
        "atol": cfg.atol,
        "rtol": cfg.rtol,
        "store_states": bool(cfg.store_states),
        "store_final_state": True,
    }
    if cfg.max_step is not None:
        options["max_step"] = cfg.max_step
    return options


def _final_state_from_result(result: qt.solver.Result) -> qt.Qobj:
    final_state = getattr(result, "final_state", None)
    if final_state is not None:
        return final_state
    return result.states[-1]


@dataclass
class SimulationSession:
    model: Any
    compiled: CompiledSequence
    drive_ops: dict[str, str | TransmonTransitionDriveSpec | SidebandDriveSpec]
    config: SimulationConfig = field(default_factory=SimulationConfig)
    c_ops: Sequence[qt.Qobj] | None = None
    noise: NoiseSpec | None = None
    e_ops: dict[str, qt.Qobj] | None = None

    hamiltonian: list = field(init=False, repr=False)
    effective_c_ops: tuple[qt.Qobj, ...] = field(init=False, repr=False)
    observables: dict[str, qt.Qobj] = field(init=False)
    observable_names: tuple[str, ...] = field(init=False, repr=False)
    observable_ops: tuple[qt.Qobj, ...] = field(init=False, repr=False)
    solver_options: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.observables = default_observables(self.model) if self.e_ops is None else dict(self.e_ops)
        self.observable_names = tuple(self.observables.keys())
        self.observable_ops = tuple(self.observables.values())
        self.hamiltonian = hamiltonian_time_slices(self.model, self.compiled, self.drive_ops, frame=self.config.frame)
        eff_c_ops = list(self.c_ops) if self.c_ops else []
        eff_c_ops.extend(collapse_operators(self.model, self.noise))
        self.effective_c_ops = tuple(eff_c_ops)
        self.solver_options = _solver_options(self.config)

    def run(self, initial_state: qt.Qobj) -> SimulationResult:
        if self.config.backend is not None:
            result = solve_with_backend(
                self.hamiltonian,
                self.compiled.tlist,
                initial_state,
                observables=list(self.observable_ops),
                collapse_ops=list(self.effective_c_ops),
                backend=self.config.backend,
                store_states=self.config.store_states,
            )
        elif self.effective_c_ops or initial_state.isoper:
            result = qt.mesolve(
                self.hamiltonian,
                initial_state,
                self.compiled.tlist,
                c_ops=list(self.effective_c_ops),
                e_ops=list(self.observable_ops),
                options=self.solver_options,
            )
        else:
            result = qt.sesolve(
                self.hamiltonian,
                initial_state,
                self.compiled.tlist,
                e_ops=list(self.observable_ops),
                options=self.solver_options,
            )
        expectations = {name: np.asarray(result.expect[idx]) for idx, name in enumerate(self.observable_names)}
        return SimulationResult(
            final_state=_final_state_from_result(result),
            states=result.states if self.config.store_states else None,
            expectations=expectations,
            solver_result=result,
        )

    def run_many(
        self,
        initial_states: Iterable[qt.Qobj],
        *,
        max_workers: int = 1,
        mp_context: str = "spawn",
    ) -> list[SimulationResult]:
        """Run the session over multiple initial states.

        Note: On Windows, multiprocessing uses the ``spawn`` start method.
        This carries noticeable per-worker startup overhead. For small jobs
        (few initial states, short simulations), the overhead may dominate and
        a sequential approach may be faster. The serial prepared-session path
        (``SimulationSession.run_many`` with ``max_workers=1``) is generally
        preferred for most workloads on Windows.
        """
        states = list(initial_states)
        if max_workers <= 1 or len(states) <= 1:
            return [self.run(state) for state in states]

        ctx = mp.get_context(mp_context)
        with ProcessPoolExecutor(
            max_workers=int(max_workers),
            mp_context=ctx,
            initializer=_init_parallel_session,
            initargs=(self,),
        ) as executor:
            return list(executor.map(_run_parallel_state, states))


def prepare_simulation(
    model: Any,
    compiled: CompiledSequence,
    drive_ops: dict[str, str | TransmonTransitionDriveSpec | SidebandDriveSpec],
    *,
    config: SimulationConfig | None = None,
    c_ops: Sequence[qt.Qobj] | None = None,
    noise: NoiseSpec | None = None,
    e_ops: dict[str, qt.Qobj] | None = None,
) -> SimulationSession:
    return SimulationSession(
        model=model,
        compiled=compiled,
        drive_ops=drive_ops,
        config=config or SimulationConfig(),
        c_ops=c_ops,
        noise=noise,
        e_ops=e_ops,
    )


def simulate_batch(
    session: SimulationSession,
    initial_states: Iterable[qt.Qobj],
    *,
    max_workers: int = 1,
    mp_context: str = "spawn",
) -> list[SimulationResult]:
    """Run a prepared session over a batch of initial states.

    Note: On Windows, multiprocessing uses the ``spawn`` start method.
    This carries noticeable per-worker startup overhead. For small jobs
    (few initial states, short simulations), the overhead may dominate and
    a sequential approach may be faster. The serial prepared-session path
    (``SimulationSession.run_many``) is generally preferred for most workloads
    on Windows.
    """
    return session.run_many(initial_states, max_workers=max_workers, mp_context=mp_context)


# Module-level parallel worker state.  These globals are set by
# ProcessPoolExecutor initializers in forked/spawned workers, so each
# process has its own copy.  They are NOT safe for concurrent use from
# multiple threads within a single process — always use max_workers=1
# if calling from threads.
_PARALLEL_SESSION: SimulationSession | None = None


def _init_parallel_session(session: SimulationSession) -> None:
    global _PARALLEL_SESSION
    _PARALLEL_SESSION = session


def _run_parallel_state(initial_state: qt.Qobj) -> SimulationResult:
    if _PARALLEL_SESSION is None:
        raise RuntimeError("Parallel simulation session was not initialized.")
    return _PARALLEL_SESSION.run(initial_state)


# ---------------------------------------------------------------------------
# Parameter sweep runner
# ---------------------------------------------------------------------------

def run_sweep(
    sessions: Sequence["SimulationSession"],
    initial_states: Sequence[qt.Qobj],
    *,
    max_workers: int = 1,
    mp_context: str = "spawn",
) -> list[SimulationResult]:
    """Run a parameter sweep over a list of (session, initial_state) pairs.

    Each ``(session, state)`` pair is an independent simulation job.  When
    ``max_workers == 1`` (default), the sweep is executed serially.  Set
    ``max_workers > 1`` to distribute jobs across worker processes via
    :class:`~concurrent.futures.ProcessPoolExecutor`.

    This is useful for sweeps over system parameters (frequency, chi, …) where
    each point requires a *different* :class:`SimulationSession` (because the
    Hamiltonian changes between points).  For sweeps over initial states where
    the Hamiltonian is the same, prefer :func:`simulate_batch` instead.

    On Windows the ``spawn`` context adds significant process-startup overhead;
    parallel execution only pays off when each solve takes several hundred
    milliseconds or more.

    Args:
        sessions: Sequence of :class:`SimulationSession` instances, one per
            sweep point.  Must have the same length as *initial_states*.
        initial_states: Initial quantum state for each sweep point.
        max_workers: Number of parallel worker processes.  ``1`` runs serially.
        mp_context: Multiprocessing start method.  ``"spawn"`` is the only safe
            choice on Windows.

    Returns:
        List of :class:`SimulationResult`, one per sweep point, in the same
        order as *sessions*.
    """
    session_list = list(sessions)
    state_list = list(initial_states)
    if len(session_list) != len(state_list):
        raise ValueError(
            f"run_sweep: sessions length ({len(session_list)}) must equal "
            f"initial_states length ({len(state_list)})."
        )

    if max_workers <= 1 or len(session_list) <= 1:
        return [session.run(state) for session, state in zip(session_list, state_list)]

    ctx = mp.get_context(mp_context)
    with ProcessPoolExecutor(
        max_workers=int(max_workers),
        mp_context=ctx,
        initializer=_init_parallel_sweep_worker,
        initargs=(session_list,),
    ) as executor:
        indexed_states = list(enumerate(state_list))
        results_with_index = list(executor.map(_run_parallel_sweep_point_indexed, indexed_states))
    results_with_index.sort(key=lambda pair: pair[0])
    return [result for _, result in results_with_index]


# Workers for indexed parallel sweep (one pool, sessions broadcast by index).
_PARALLEL_SWEEP_SESSIONS: list[SimulationSession] = []


def _init_parallel_sweep_worker(sessions: list[SimulationSession]) -> None:
    global _PARALLEL_SWEEP_SESSIONS
    _PARALLEL_SWEEP_SESSIONS = sessions


def _run_parallel_sweep_point_indexed(args: tuple[int, qt.Qobj]) -> tuple[int, SimulationResult]:
    index, state = args
    if not _PARALLEL_SWEEP_SESSIONS:
        raise RuntimeError("Parallel sweep worker sessions were not initialized.")
    return index, _PARALLEL_SWEEP_SESSIONS[index].run(state)


def simulate_sequence(
    model: Any,
    compiled: CompiledSequence,
    initial_state: qt.Qobj,
    drive_ops: dict[str, str | TransmonTransitionDriveSpec | SidebandDriveSpec],
    config: SimulationConfig | None = None,
    c_ops: Sequence[qt.Qobj] | None = None,
    noise: NoiseSpec | None = None,
    e_ops: dict[str, qt.Qobj] | None = None,
) -> SimulationResult:
    session = prepare_simulation(
        model,
        compiled,
        drive_ops,
        config=config,
        c_ops=c_ops,
        noise=noise,
        e_ops=e_ops,
    )
    return session.run(initial_state)
