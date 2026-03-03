from __future__ import annotations

from typing import Any

import numpy as np

from cqed_sim.sim.extractors import bloch_xyz_from_joint, conditioned_bloch_xyz


def bloch_trajectory_from_states(
    states: list,
    conditioned_n_levels: list[int] | tuple[int, ...] | None = None,
    probability_threshold: float = 1.0e-8,
) -> dict[str, Any]:
    x = np.asarray([bloch_xyz_from_joint(state)[0] for state in states], dtype=float)
    y = np.asarray([bloch_xyz_from_joint(state)[1] for state in states], dtype=float)
    z = np.asarray([bloch_xyz_from_joint(state)[2] for state in states], dtype=float)

    conditioned: dict[int, dict[str, np.ndarray]] = {}
    for n in sorted(set([] if conditioned_n_levels is None else conditioned_n_levels)):
        x_n = np.full(len(states), np.nan, dtype=float)
        y_n = np.full(len(states), np.nan, dtype=float)
        z_n = np.full(len(states), np.nan, dtype=float)
        p_n = np.zeros(len(states), dtype=float)
        valid = np.zeros(len(states), dtype=bool)
        for idx, state in enumerate(states):
            bx, by, bz, prob, is_valid = conditioned_bloch_xyz(state, n=n, fallback="nan")
            p_n[idx] = float(prob)
            if is_valid and prob >= probability_threshold:
                x_n[idx] = float(bx)
                y_n[idx] = float(by)
                z_n[idx] = float(bz)
                valid[idx] = True
        conditioned[int(n)] = {
            "x": x_n,
            "y": y_n,
            "z": z_n,
            "probability": p_n,
            "valid": valid,
        }

    return {
        "x": x,
        "y": y,
        "z": z,
        "conditioned": conditioned,
        "probability_threshold": float(probability_threshold),
    }
