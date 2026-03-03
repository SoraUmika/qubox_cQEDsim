from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
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


def default_observables(model: DispersiveTransmonCavityModel) -> dict[str, qt.Qobj]:
    ops = model.operators()
    proj_e = qt.tensor(qt.qeye(model.n_cav), qt.basis(model.n_tr, 1) * qt.basis(model.n_tr, 1).dag())
    x_c = ops["a"] + ops["adag"]
    p_c = -1j * (ops["a"] - ops["adag"])
    return {"P_e": proj_e, "n_c": ops["n_c"], "x_c": x_c, "p_c": p_c}


def hamiltonian_time_slices(
    model: DispersiveTransmonCavityModel,
    compiled: CompiledSequence,
    drive_ops: dict[str, str],
    frame: FrameSpec | None = None,
) -> list:
    frame = frame or FrameSpec()
    ops = model.operators()
    h = [model.static_hamiltonian(frame)]
    for channel, target in drive_ops.items():
        coeff = compiled.channels[channel].distorted
        if target == "cavity":
            h.append([ops["adag"], coeff])
            h.append([ops["a"], np.conj(coeff)])
        elif target == "qubit":
            h.append([ops["bdag"], coeff])
            h.append([ops["b"], np.conj(coeff)])
        elif target == "sideband":
            # e,n <-> g,n+1 exchange: a^\dagger b + a b^\dagger
            h.append([ops["adag"] * ops["b"], coeff])
            h.append([ops["a"] * ops["bdag"], np.conj(coeff)])
        else:
            raise ValueError(f"Unsupported target '{target}' for channel '{channel}'.")
    return h


def simulate_sequence(
    model: DispersiveTransmonCavityModel,
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
