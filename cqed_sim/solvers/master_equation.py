from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import qutip as qt

from cqed_sim.models.multilevel_cqed import HamiltonianData
from cqed_sim.solvers.options import build_qutip_solver_options


@dataclass(frozen=True)
class DressedDecay:
    source: int
    target: int
    rate: float


@dataclass(frozen=True)
class MasterEquationConfig:
    atol: float = 1.0e-8
    rtol: float = 1.0e-7
    max_step: float | None = None
    nsteps: int | None = None
    store_states: bool = False
    progress_bar: str | bool = ""
    solver_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class MasterEquationResult:
    final_state: qt.Qobj
    times: np.ndarray
    states: list[qt.Qobj] | None
    expectations: dict[str, np.ndarray]
    populations: dict[str, np.ndarray]
    photon_numbers: dict[str, np.ndarray]
    output_means: np.ndarray | None
    solver_result: object


def _solver_options(config: MasterEquationConfig) -> dict[str, object]:
    return build_qutip_solver_options(
        atol=config.atol,
        rtol=config.rtol,
        max_step=config.max_step,
        nsteps=config.nsteps,
        store_states=bool(config.store_states),
        store_final_state=True,
        progress_bar=config.progress_bar,
        solver_options=config.solver_options,
    )


def dressed_collapse_operators(
    dressed_states: Sequence[qt.Qobj],
    decays: Sequence[DressedDecay],
) -> tuple[qt.Qobj, ...]:
    c_ops: list[qt.Qobj] = []
    states = list(dressed_states)
    for decay in decays:
        if float(decay.rate) <= 0.0:
            continue
        source = int(decay.source)
        target = int(decay.target)
        if source < 0 or source >= len(states) or target < 0 or target >= len(states):
            raise IndexError("DressedDecay source/target index is outside the retained dressed basis.")
        c_ops.append(np.sqrt(float(decay.rate)) * states[target] * states[source].dag())
    return tuple(c_ops)


def collapse_operators_from_model(
    model: object,
    *,
    kappa_r: float = 0.0,
    kappa_f: float = 0.0,
    transmon_t1: Sequence[float] | None = None,
    transmon_tphi: float | None = None,
    dressed_states: Sequence[qt.Qobj] | None = None,
    dressed_decays: Sequence[DressedDecay] = (),
    include_purcell_qubit_decay: bool = False,
) -> tuple[qt.Qobj, ...]:
    c_ops: list[qt.Qobj] = []
    if hasattr(model, "collapse_operators"):
        try:
            c_ops.extend(
                model.collapse_operators(
                    kappa_r=kappa_r,
                    transmon_t1=transmon_t1,
                    transmon_tphi=transmon_tphi,
                )
            )
        except TypeError:
            c_ops.extend(
                model.collapse_operators(
                    kappa_r_internal=kappa_r,
                    transmon_t1=transmon_t1,
                    transmon_tphi=transmon_tphi,
                    include_purcell_qubit_decay=include_purcell_qubit_decay,
                )
            )
        if dressed_states is not None:
            c_ops.extend(dressed_collapse_operators(dressed_states, dressed_decays))
        return tuple(c_ops)

    ops = model.operators()  # type: ignore[attr-defined]
    if float(kappa_r) > 0.0 and "a" in ops:
        c_ops.append(np.sqrt(float(kappa_r)) * ops["a"])
    if float(kappa_f) > 0.0 and "f" in ops:
        c_ops.append(np.sqrt(float(kappa_f)) * ops["f"])
    if transmon_t1 is not None and hasattr(model, "transmon_transition_operators"):
        for lower, time in enumerate(tuple(transmon_t1)):
            if float(time) > 0.0 and np.isfinite(float(time)):
                _up, down = model.transmon_transition_operators(lower, lower + 1)  # type: ignore[attr-defined]
                c_ops.append(np.sqrt(1.0 / float(time)) * down)
    if transmon_tphi is not None and float(transmon_tphi) > 0.0 and np.isfinite(float(transmon_tphi)) and "n_q" in ops:
        c_ops.append(np.sqrt(1.0 / float(transmon_tphi)) * ops["n_q"])
    if dressed_states is not None:
        c_ops.extend(dressed_collapse_operators(dressed_states, dressed_decays))
    return tuple(c_ops)


def default_readout_observables(model: object, output_operator: qt.Qobj | None = None) -> dict[str, qt.Qobj]:
    ops = model.operators()  # type: ignore[attr-defined]
    observables: dict[str, qt.Qobj] = {}
    if hasattr(model, "subsystem_dims"):
        dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
        if dims:
            for level in range(dims[0]):
                factors = [qt.qeye(dim) for dim in dims]
                factors[0] = qt.basis(dims[0], level) * qt.basis(dims[0], level).dag()
                observables[f"P_q{level}"] = qt.tensor(*factors)
    if "n_r" in ops:
        observables["n_r"] = ops["n_r"]
    if "n_f" in ops:
        observables["n_f"] = ops["n_f"]
    if output_operator is not None:
        observables["output_I"] = output_operator + output_operator.dag()
        observables["output_Q"] = -1j * (output_operator - output_operator.dag())
    return observables


def solve_master_equation(
    hamiltonian_data: HamiltonianData | Sequence | qt.Qobj | qt.QobjEvo,
    rho0: qt.Qobj,
    *,
    tlist: Sequence[float] | None = None,
    c_ops: Sequence[qt.Qobj] = (),
    e_ops: Mapping[str, qt.Qobj] | None = None,
    config: MasterEquationConfig | None = None,
    output_operator: qt.Qobj | None = None,
) -> MasterEquationResult:
    """Run a Lindblad master-equation solve.

    The equation is ``drho/dt = -i[H(t),rho] + sum_k D[L_k] rho``.  Hamiltonian
    coefficients are angular frequencies and ``tlist`` is in seconds.
    """

    config = MasterEquationConfig() if config is None else config
    if isinstance(hamiltonian_data, HamiltonianData):
        hamiltonian = hamiltonian_data.as_qobjevo()
        times = np.asarray(hamiltonian_data.tlist, dtype=float)
        output_operator = hamiltonian_data.output_operator if output_operator is None else output_operator
    else:
        hamiltonian = hamiltonian_data
        if tlist is None:
            raise ValueError("tlist must be provided when hamiltonian_data is not HamiltonianData.")
        times = np.asarray(tlist, dtype=float)
    observables = dict(e_ops or {})
    if output_operator is not None:
        observables.setdefault("output_I", output_operator + output_operator.dag())
        observables.setdefault("output_Q", -1j * (output_operator - output_operator.dag()))
    names = tuple(observables)
    rho_initial = rho0 if rho0.isoper else rho0.proj()
    result = qt.mesolve(
        hamiltonian,
        rho_initial,
        times,
        c_ops=list(c_ops),
        e_ops=[observables[name] for name in names],
        options=_solver_options(config),
    )
    expectations = {name: np.asarray(result.expect[idx]) for idx, name in enumerate(names)}
    populations = {name: values for name, values in expectations.items() if name.startswith("P_q")}
    photon_numbers = {name: values for name, values in expectations.items() if name in {"n_r", "n_f"}}
    output_means = None
    if "output_I" in expectations and "output_Q" in expectations:
        output_means = 0.5 * (np.asarray(expectations["output_I"]) + 1j * np.asarray(expectations["output_Q"]))
    final_state = getattr(result, "final_state", None)
    if final_state is None:
        final_state = result.states[-1]
    return MasterEquationResult(
        final_state=final_state,
        times=times,
        states=list(result.states) if config.store_states else None,
        expectations=expectations,
        populations=populations,
        photon_numbers=photon_numbers,
        output_means=None if output_means is None else np.asarray(output_means, dtype=np.complex128),
        solver_result=result,
    )


__all__ = [
    "DressedDecay",
    "MasterEquationConfig",
    "MasterEquationResult",
    "collapse_operators_from_model",
    "default_readout_observables",
    "dressed_collapse_operators",
    "solve_master_equation",
]
