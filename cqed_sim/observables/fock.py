from __future__ import annotations

from typing import Any

import numpy as np

from cqed_sim.sim.extractors import conditioned_bloch_xyz, conditioned_qubit_state


def _available_max_n(track: dict[str, Any], max_n: int) -> int:
    n_cav_dim = int(track["snapshots"][0]["rho_c"].dims[0][0])
    return min(int(max_n), n_cav_dim - 1)


def fock_resolved_bloch_diagnostics(
    track: dict[str, Any],
    max_n: int,
    probability_threshold: float = 1.0e-8,
) -> dict[str, Any]:
    available_max_n = _available_max_n(track, max_n)
    n_values = np.arange(available_max_n + 1, dtype=int)
    n_steps = len(track["snapshots"])
    x = np.full((n_values.size, n_steps), np.nan, dtype=float)
    y = np.full((n_values.size, n_steps), np.nan, dtype=float)
    z = np.full((n_values.size, n_steps), np.nan, dtype=float)
    p = np.zeros((n_values.size, n_steps), dtype=float)
    valid = np.zeros((n_values.size, n_steps), dtype=bool)

    for step_idx, snapshot in enumerate(track["snapshots"]):
        state = snapshot["state"]
        for row_idx, n in enumerate(n_values):
            x_n, y_n, z_n, p_n, is_valid = conditioned_bloch_xyz(state, n=int(n), fallback="nan")
            p[row_idx, step_idx] = float(p_n)
            if is_valid and p_n >= probability_threshold:
                x[row_idx, step_idx] = float(x_n)
                y[row_idx, step_idx] = float(y_n)
                z[row_idx, step_idx] = float(z_n)
                valid[row_idx, step_idx] = True

    return {
        "case": track["case"],
        "indices": np.asarray(track["indices"], dtype=int),
        "n_values": n_values,
        "x": x,
        "y": y,
        "z": z,
        "probability": p,
        "valid": valid,
        "top_labels": [snapshot["top_label"] for snapshot in track["snapshots"]],
        "gate_types": [snapshot["gate_type"] for snapshot in track["snapshots"]],
        "probability_threshold": float(probability_threshold),
    }


def conditional_phase_diagnostics(
    track: dict[str, Any],
    max_n: int,
    probability_threshold: float = 1.0e-8,
    unwrap: bool = False,
) -> dict[str, Any]:
    available_max_n = _available_max_n(track, max_n)
    n_values = np.arange(available_max_n + 1, dtype=int)
    n_steps = len(track["snapshots"])
    phase = np.full((n_values.size, n_steps), np.nan, dtype=float)
    coherence_mag = np.zeros((n_values.size, n_steps), dtype=float)
    probability = np.zeros((n_values.size, n_steps), dtype=float)

    for step_idx, snapshot in enumerate(track["snapshots"]):
        state = snapshot["state"]
        for row_idx, n in enumerate(n_values):
            rho_q, p_n, is_valid = conditioned_qubit_state(state, n=int(n), fallback="nan")
            probability[row_idx, step_idx] = float(p_n)
            if not is_valid or p_n < probability_threshold:
                continue
            coherence = complex(rho_q.full()[0, 1])
            coherence_mag[row_idx, step_idx] = abs(coherence)
            if abs(coherence) < probability_threshold:
                continue
            phase[row_idx, step_idx] = float(np.angle(coherence))

    if unwrap:
        for row_idx in range(phase.shape[0]):
            row = phase[row_idx]
            valid = np.isfinite(row)
            if np.count_nonzero(valid) <= 1:
                continue
            valid_idx = np.where(valid)[0]
            split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
            for segment in np.split(valid_idx, split_points):
                phase[row_idx, segment] = np.unwrap(row[segment])
        phase_mode = "unwrapped"
    else:
        phase_mode = "wrapped"

    return {
        "case": track["case"],
        "indices": np.asarray(track["indices"], dtype=int),
        "n_values": n_values,
        "phase": phase,
        "coherence_magnitude": coherence_mag,
        "probability": probability,
        "top_labels": [snapshot["top_label"] for snapshot in track["snapshots"]],
        "gate_types": [snapshot["gate_type"] for snapshot in track["snapshots"]],
        "phase_mode": phase_mode,
        "probability_threshold": float(probability_threshold),
    }


def wrapped_phase_error(simulated_phase: np.ndarray, ideal_phase: np.ndarray) -> np.ndarray:
    simulated = np.asarray(simulated_phase, dtype=float)
    ideal = np.asarray(ideal_phase, dtype=float)
    if simulated.shape != ideal.shape:
        raise ValueError(f"Phase arrays must have matching shapes, got {simulated.shape} and {ideal.shape}.")
    valid = np.isfinite(simulated) & np.isfinite(ideal)
    error = np.full(simulated.shape, np.nan, dtype=float)
    if np.any(valid):
        wrapped = (simulated[valid] - ideal[valid] + np.pi) % (2.0 * np.pi) - np.pi
        error[valid] = wrapped
    return error
