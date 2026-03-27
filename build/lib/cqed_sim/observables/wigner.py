from __future__ import annotations

from typing import Any

import numpy as np

from cqed_sim.sim.extractors import cavity_wigner


def selected_wigner_snapshots(track: dict[str, Any], stride: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    for snapshot in track["wigner_snapshots"]:
        if snapshot["index"] == 0 or snapshot["index"] % max(1, stride) == 0:
            chosen.append(snapshot)
    if track["wigner_snapshots"] and track["wigner_snapshots"][-1] not in chosen:
        chosen.append(track["wigner_snapshots"][-1])
    return chosen


def wigner_negativity(snapshot: dict[str, Any]) -> float:
    if snapshot["wigner"] is None:
        return float("nan")
    xvec = snapshot["wigner"]["xvec"]
    yvec = snapshot["wigner"]["yvec"]
    w = snapshot["wigner"]["w"]
    dx = float(xvec[1] - xvec[0]) if len(xvec) > 1 else 1.0
    dy = float(yvec[1] - yvec[0]) if len(yvec) > 1 else 1.0
    return float(max(0.5 * (np.sum(np.abs(w)) * dx * dy - 1.0), 0.0))


__all__ = ["cavity_wigner", "selected_wigner_snapshots", "wigner_negativity"]
