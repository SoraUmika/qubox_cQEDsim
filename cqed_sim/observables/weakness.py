from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.observables.wigner import wigner_negativity


def attach_weakness_metrics(reference_track: dict[str, Any], track: dict[str, Any]) -> dict[str, Any]:
    track["wigner_negativity"] = np.asarray(
        [wigner_negativity(snapshot) for snapshot in track["snapshots"]],
        dtype=float,
    )
    track["fidelity_weakness_vs_a"] = np.asarray(
        [
            1.0 - float(qt.metrics.fidelity(ref_snapshot["state"], snapshot["state"]))
            for ref_snapshot, snapshot in zip(reference_track["snapshots"], track["snapshots"])
        ],
        dtype=float,
    )
    return track


def comparison_metrics(track_a: dict[str, Any], track_b: dict[str, Any]) -> dict[str, float]:
    n_common = min(len(track_a["snapshots"]), len(track_b["snapshots"]))
    return {
        "x_rmse": float(np.sqrt(np.mean((track_a["x"][:n_common] - track_b["x"][:n_common]) ** 2))),
        "y_rmse": float(np.sqrt(np.mean((track_a["y"][:n_common] - track_b["y"][:n_common]) ** 2))),
        "z_rmse": float(np.sqrt(np.mean((track_a["z"][:n_common] - track_b["z"][:n_common]) ** 2))),
        "n_rmse": float(np.sqrt(np.mean((track_a["n"][:n_common] - track_b["n"][:n_common]) ** 2))),
        "final_fidelity": float(
            qt.metrics.fidelity(track_a["snapshots"][n_common - 1]["state"], track_b["snapshots"][n_common - 1]["state"])
        ),
    }
