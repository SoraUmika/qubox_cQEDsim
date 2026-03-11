from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import multiprocessing as mp
from typing import Any, Iterable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.backends.base_backend import BaseBackend
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


def _mode_quadratures(lowering: qt.Qobj, raising: qt.Qobj) -> tuple[qt.Qobj, qt.Qobj]:
    return lowering + raising, -1j * (lowering - raising)


def default_observables(model: Any) -> dict[str, qt.Qobj]:
    cached = getattr(model, "_default_observables_cache", None)
    if cached is not None:
        return cached

    ops = model.operators()
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    observables: dict[str, qt.Qobj] = {"P_e": _projector_onto_first_excited_state(dims)}

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


def hamiltonian_time_slices(
    model: Any,
    compiled: CompiledSequence,
    drive_ops: dict[str, str],
    frame: FrameSpec | None = None,
) -> list:
    frame = frame or FrameSpec()
    couplings = model.drive_coupling_operators() if hasattr(model, "drive_coupling_operators") else _legacy_drive_couplings(model)

    h = [model.static_hamiltonian(frame)]
    for channel, target in drive_ops.items():
        coeff = compiled.channels[channel].distorted
        if target not in couplings:
            raise ValueError(f"Unsupported target '{target}' for channel '{channel}'.")
        raising, lowering = couplings[target]
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
    drive_ops: dict[str, str]
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
    drive_ops: dict[str, str],
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
    return session.run_many(initial_states, max_workers=max_workers, mp_context=mp_context)


_PARALLEL_SESSION: SimulationSession | None = None


def _init_parallel_session(session: SimulationSession) -> None:
    global _PARALLEL_SESSION
    _PARALLEL_SESSION = session


def _run_parallel_state(initial_state: qt.Qobj) -> SimulationResult:
    if _PARALLEL_SESSION is None:
        raise RuntimeError("Parallel simulation session was not initialized.")
    return _PARALLEL_SESSION.run(initial_state)


def simulate_sequence(
    model: Any,
    compiled: CompiledSequence,
    initial_state: qt.Qobj,
    drive_ops: dict[str, str],
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
