from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .metrics import objective_breakdown, unitarity_error
from .sequence import GateSequence
from .subspace import Subspace


@dataclass
class SimulationResult:
    full_operator: np.ndarray
    subspace_operator: np.ndarray
    metrics: dict[str, float]
    backend: str
    settings: dict[str, Any]


def simulate_sequence(
    sequence: GateSequence,
    subspace: Subspace,
    backend: str = "ideal",
    target_subspace: np.ndarray | None = None,
    leakage_weight: float = 0.0,
    gauge: str = "global",
    **backend_settings: Any,
) -> SimulationResult:
    if backend not in {"ideal", "pulse"}:
        raise ValueError("backend must be 'ideal' or 'pulse'.")
    full = sequence.unitary(backend=backend, backend_settings=backend_settings)
    sub = subspace.restrict_operator(full)

    metrics = {
        "unitarity_error": unitarity_error(full),
    }
    if target_subspace is not None:
        leakage_n_jobs = int(backend_settings.get("leakage_n_jobs", 1))
        metrics.update(
            objective_breakdown(
                actual_subspace=sub,
                target_subspace=np.asarray(target_subspace, dtype=np.complex128),
                full_operator=full,
                subspace=subspace,
                leakage_weight=leakage_weight,
                gauge=gauge,
                leakage_n_jobs=leakage_n_jobs,
            )
        )
    return SimulationResult(
        full_operator=full,
        subspace_operator=sub,
        metrics=metrics,
        backend=backend,
        settings=dict(backend_settings),
    )
