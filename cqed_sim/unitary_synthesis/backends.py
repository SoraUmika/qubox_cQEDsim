from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import qutip as qt

from .metrics import objective_breakdown, unitarity_error
from .sequence import GateSequence
from .subspace import Subspace


@dataclass
class SimulationResult:
    full_operator: np.ndarray | None
    subspace_operator: np.ndarray | None
    state_outputs: list[qt.Qobj] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    backend: str = "ideal"
    settings: dict[str, Any] = field(default_factory=dict)


def simulate_sequence(
    sequence: GateSequence,
    subspace: Subspace | None,
    backend: str = "ideal",
    target_subspace: np.ndarray | None = None,
    leakage_weight: float = 0.0,
    gauge: str = "global",
    block_slices: Sequence[slice | Sequence[int] | np.ndarray] | None = None,
    state_inputs: Sequence[qt.Qobj | np.ndarray] | None = None,
    need_operator: bool = True,
    system: Any | None = None,
    **backend_settings: Any,
) -> SimulationResult:
    if backend not in {"ideal", "pulse"}:
        raise ValueError("backend must be 'ideal' or 'pulse'.")

    settings = dict(backend_settings)
    system = settings.pop("system", system)

    if system is not None:
        full = system.simulate_unitary(sequence, backend=backend, **settings) if need_operator else None
    else:
        full = sequence.unitary(backend=backend, backend_settings=settings) if need_operator else None
    sub = None
    if full is not None and subspace is not None:
        sub = subspace.restrict_operator(full)

    state_outputs = None
    if state_inputs is not None:
        if system is not None:
            state_outputs = system.simulate_states(sequence, list(state_inputs), backend=backend, **settings)
        else:
            state_outputs = sequence.propagate_states(
                list(state_inputs),
                backend=backend,
                backend_settings=settings,
            )

    metrics: dict[str, float] = {}
    if full is not None:
        metrics["unitarity_error"] = unitarity_error(full)
    if target_subspace is not None:
        if full is None or subspace is None or sub is None:
            raise ValueError("Unitary-target evaluation requires a full operator and a subspace.")
        metrics.update(
            objective_breakdown(
                actual_subspace=sub,
                target_subspace=np.asarray(target_subspace, dtype=np.complex128),
                full_operator=full,
                subspace=subspace,
                leakage_weight=leakage_weight,
                gauge=gauge,
                block_slices=block_slices,
                leakage_n_jobs=int(settings.get("leakage_n_jobs", 1)),
            )
        )
    return SimulationResult(
        full_operator=full,
        subspace_operator=sub,
        state_outputs=state_outputs,
        metrics=metrics,
        backend=backend,
        settings=dict(settings),
    )
