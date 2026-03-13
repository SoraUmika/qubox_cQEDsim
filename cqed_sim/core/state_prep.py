from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import qutip as qt


@dataclass(frozen=True)
class SubsystemStateSpec:
    kind: str
    label: str | None = None
    level: int | None = None
    alpha: complex | None = None
    amplitudes: Any | None = None
    density_matrix: Any | None = None


def qubit_state(label: str) -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="label", label=str(label))


def qubit_level(level: int) -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="level", level=int(level))


def vacuum_state() -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="vacuum", level=0)


def fock_state(level: int) -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="fock", level=int(level))


def coherent_state(alpha: complex) -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="coherent", alpha=complex(alpha))


def amplitude_state(amplitudes: Any) -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="amplitudes", amplitudes=amplitudes)


def density_matrix_state(rho: Any) -> SubsystemStateSpec:
    return SubsystemStateSpec(kind="density_matrix", density_matrix=rho)


@dataclass(frozen=True)
class StatePreparationSpec:
    qubit: SubsystemStateSpec = field(default_factory=lambda: qubit_state("g"))
    storage: SubsystemStateSpec = field(default_factory=vacuum_state)
    readout: SubsystemStateSpec | None = None


def _qubit_from_label(dim: int, label: str) -> qt.Qobj:
    g = qt.basis(dim, 0)
    e = qt.basis(dim, 1)
    lookup = {
        "g": g,
        "e": e,
        "+x": (g + e).unit(),
        "-x": (g - e).unit(),
        "+y": (g + 1j * e).unit(),
        "-y": (g - 1j * e).unit(),
    }
    if label not in lookup:
        raise ValueError(f"Unsupported qubit label '{label}'.")
    return lookup[label]


def _dm_from_any(rho_like: Any, dim: int) -> qt.Qobj:
    if isinstance(rho_like, qt.Qobj):
        rho = rho_like
    else:
        arr = np.asarray(rho_like, dtype=np.complex128)
        rho = qt.Qobj(arr, dims=[[dim], [dim]])
    if not rho.isoper:
        raise ValueError("density_matrix spec must be an operator.")
    return rho


def _ket_from_amplitudes(amplitudes: Any, dim: int) -> qt.Qobj:
    arr = np.asarray(amplitudes, dtype=np.complex128).reshape(-1)
    if arr.size > dim:
        arr = arr[:dim]
    if arr.size < dim:
        arr = np.pad(arr, (0, dim - arr.size))
    return qt.Qobj(arr.reshape((-1, 1)), dims=[[dim], [1]]).unit()


def _build_subsystem_state(spec: SubsystemStateSpec, dim: int, *, qubit: bool) -> qt.Qobj:
    kind = spec.kind.lower()
    if kind == "label":
        if not qubit:
            raise ValueError("label specs are only supported for the qubit subsystem.")
        return _qubit_from_label(dim, str(spec.label).lower())
    if kind == "level":
        if spec.level is None:
            raise ValueError("level spec requires a level value.")
        return qt.basis(dim, int(spec.level))
    if kind in {"vacuum", "fock"}:
        level = 0 if spec.level is None else int(spec.level)
        return qt.basis(dim, level)
    if kind == "coherent":
        return qt.coherent(dim, complex(0.0 if spec.alpha is None else spec.alpha))
    if kind == "amplitudes":
        return _ket_from_amplitudes(spec.amplitudes, dim)
    if kind == "density_matrix":
        return _dm_from_any(spec.density_matrix, dim)
    raise ValueError(f"Unsupported subsystem state kind '{spec.kind}'.")


def prepare_state(model: Any, spec: StatePreparationSpec | None = None) -> qt.Qobj:
    spec = StatePreparationSpec() if spec is None else spec
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    if len(dims) not in {2, 3}:
        raise ValueError(f"prepare_state only supports two- and three-subsystem models, got dims={dims}.")
    if len(dims) < 3 and spec.readout is not None:
        raise ValueError("readout state specs are only valid for three-mode models.")

    subsystem_specs: list[SubsystemStateSpec] = [spec.qubit, spec.storage]
    if len(dims) >= 3:
        subsystem_specs.append(spec.readout if spec.readout is not None else vacuum_state())
    if len(subsystem_specs) != len(dims):
        raise ValueError(f"StatePreparationSpec does not match model dims {dims}.")

    states = [
        _build_subsystem_state(subsystem_spec, dim, qubit=(idx == 0))
        for idx, (subsystem_spec, dim) in enumerate(zip(subsystem_specs, dims, strict=True))
    ]
    if any(state.isoper for state in states):
        return qt.tensor(*(state if state.isoper else state.proj() for state in states))
    return qt.tensor(*states)


def prepare_ground_state(model: Any) -> qt.Qobj:
    return prepare_state(model, StatePreparationSpec())
