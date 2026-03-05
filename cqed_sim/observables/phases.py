from __future__ import annotations

from typing import Any

import numpy as np

from cqed_sim.observables.fock import relative_phase_family_diagnostics


def relative_phase_diagnostics(
    track: dict[str, Any],
    max_n: int,
    threshold: float,
    unwrap: bool = False,
) -> dict[str, Any]:
    diagnostics = relative_phase_family_diagnostics(
        track,
        max_n=max_n,
        probability_threshold=threshold,
        unwrap=unwrap,
        coherence_threshold=threshold,
    )
    labels: list[str] = []
    traces: dict[str, np.ndarray] = {}
    amplitudes: dict[str, np.ndarray] = {}
    for family in ("ground", "excited"):
        family_diag = diagnostics["families"][family]
        family_tag = "g" if family == "ground" else "e"
        for row_idx, n in enumerate(family_diag["n_values"]):
            label = f"|{family_tag},{int(n)}>"
            labels.append(label)
            traces[label] = np.asarray(family_diag["phase"][row_idx], dtype=float)
            amplitudes[label] = np.asarray(family_diag["coherence_magnitude"][row_idx], dtype=float)
    return {
        "labels": labels,
        "traces": traces,
        "amplitudes": amplitudes,
        "phase_mode": diagnostics["phase_mode"],
        "reference_label": diagnostics["phase_reference_label"],
        "target_templates": {
            family: diagnostics["families"][family]["phase_target_template"]
            for family in ("ground", "excited")
        },
        "definitions": diagnostics["relative_phase_definitions"],
    }
