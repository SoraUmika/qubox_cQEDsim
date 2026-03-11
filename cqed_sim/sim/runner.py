from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.sequence.scheduler import CompiledSequence
from cqed_sim.sim.noise import NoiseSpec, collapse_operators


@dataclass(frozen=True)
class SimulationConfig:
    frame: FrameSpec = FrameSpec()
    atol: float = 1e-8
    rtol: float = 1e-7
    max_step: float | None = None
    store_states: bool = False


@dataclass
class SimulationResult:
    final_state: qt.Qobj
    states: list[qt.Qobj] | None
    expectations: dict[str, np.ndarray]
    solver_result: qt.solver.Result


def _projector_onto_first_excited_state(subsystem_dims: tuple[int, ...]) -> qt.Qobj:
    factors = [qt.basis(subsystem_dims[0], 1) * qt.basis(subsystem_dims[0], 1).dag()]
    factors.extend(qt.qeye(dim) for dim in subsystem_dims[1:])
    return qt.tensor(*factors)


def _mode_quadratures(lowering: qt.Qobj, raising: qt.Qobj) -> tuple[qt.Qobj, qt.Qobj]:
    return lowering + raising, -1j * (lowering - raising)


def default_observables(model: Any) -> dict[str, qt.Qobj]:
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
    cfg = config or SimulationConfig()
    e_ops = e_ops or default_observables(model)
    h = hamiltonian_time_slices(model, compiled, drive_ops, frame=cfg.frame)
    options = {"atol": cfg.atol, "rtol": cfg.rtol, "store_states": True}
    if cfg.max_step is not None:
        options["max_step"] = cfg.max_step
    eff_c_ops = list(c_ops) if c_ops else []
    eff_c_ops.extend(collapse_operators(model, noise))
    if eff_c_ops or initial_state.isoper:
        result = qt.mesolve(
            h,
            initial_state,
            compiled.tlist,
            c_ops=eff_c_ops,
            e_ops=list(e_ops.values()),
            options=options,
        )
    else:
        result = qt.sesolve(
            h,
            initial_state,
            compiled.tlist,
            e_ops=list(e_ops.values()),
            options=options,
        )
    expectations = {name: np.asarray(result.expect[idx]) for idx, name in enumerate(e_ops.keys())}
    final_state = result.states[-1]
    return SimulationResult(
        final_state=final_state,
        states=result.states if cfg.store_states else None,
        expectations=expectations,
        solver_result=result,
    )
