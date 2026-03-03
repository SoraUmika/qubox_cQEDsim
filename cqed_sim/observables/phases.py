from __future__ import annotations

from typing import Any

import numpy as np

from cqed_sim.operators.basic import joint_basis_state


def relative_phase_diagnostics(
    track: dict[str, Any],
    max_n: int,
    threshold: float,
    unwrap: bool = False,
) -> dict[str, Any]:
    n_cav_dim = int(track["snapshots"][0]["rho_c"].dims[0][0])
    available_max_n = min(int(max_n), n_cav_dim - 1)
    labels_and_states = []
    for qubit_label in ("g", "e"):
        for n in range(available_max_n + 1):
            labels_and_states.append((f"|{qubit_label}{n}|", joint_basis_state(n_cav_dim, qubit_label, n)))

    reference_state = joint_basis_state(n_cav_dim, "g", 0)
    traces = {label: [] for label, _ in labels_and_states}
    amplitudes = {label: [] for label, _ in labels_and_states}

    for snapshot in track["snapshots"]:
        rho = snapshot["state"]
        c_ref = complex((reference_state.dag() * rho * reference_state)[0, 0])
        for label, ket in labels_and_states:
            c_j = complex((reference_state.dag() * rho * ket)[0, 0])
            amplitudes[label].append(abs(c_j))
            if label == "|g0|":
                traces[label].append(0.0)
            elif abs(c_ref) < threshold or abs(c_j) < threshold:
                traces[label].append(np.nan)
            else:
                traces[label].append(float(np.angle(c_j / c_ref)))

    if unwrap:
        for label, values in traces.items():
            arr = np.asarray(values, dtype=float)
            valid = np.isfinite(arr)
            if np.count_nonzero(valid) <= 1:
                traces[label] = arr
                continue
            unwrapped = arr.copy()
            valid_idx = np.where(valid)[0]
            split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
            for segment in np.split(valid_idx, split_points):
                unwrapped[segment] = np.unwrap(arr[segment])
            traces[label] = unwrapped
        phase_mode = "unwrapped"
    else:
        traces = {label: np.asarray(values, dtype=float) for label, values in traces.items()}
        phase_mode = "wrapped"

    return {
        "labels": [label for label, _ in labels_and_states],
        "traces": traces,
        "amplitudes": {label: np.asarray(values, dtype=float) for label, values in amplitudes.items()},
        "phase_mode": phase_mode,
    }
