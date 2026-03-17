from __future__ import annotations

from typing import Any

import numpy as np


def summarize_blocks(subspace_operator: np.ndarray) -> list[dict[str, Any]]:
    u = np.asarray(subspace_operator, dtype=np.complex128)
    if u.shape[0] % 2 != 0:
        return []
    rows: list[dict[str, Any]] = []
    n_blk = u.shape[0] // 2
    for n in range(n_blk):
        block = u[2 * n : 2 * n + 2, 2 * n : 2 * n + 2]
        rows.append(
            {
                "n": n,
                "block": block.tolist(),
                "det_phase": float(np.angle(np.linalg.det(block))),
            }
        )
    return rows


def make_run_report(base_report: dict[str, Any], subspace_operator: np.ndarray) -> dict[str, Any]:
    out = dict(base_report)
    out["per_fock_blocks"] = summarize_blocks(subspace_operator)
    if "phase_decomposition" not in out:
        out["phase_decomposition"] = []
    if "warnings" not in out:
        out["warnings"] = []
    metrics = out.get("metrics", {})
    if float(metrics.get("leakage_worst", 0.0)) > 1e-2:
        out["warnings"].append("High leakage detected; consider increasing cavity truncation or leakage penalty.")
    truncation = out.get("truncation", {})
    if float(truncation.get("outside_tail_population_worst", 0.0)) > 1e-2:
        out["warnings"].append("Population reaches levels outside the retained subspace; consider enlarging the truncation or tightening leakage suppression.")
    if float(truncation.get("retained_edge_population_worst", 0.0)) > 0.2:
        out["warnings"].append("Population accumulates near the truncation edge; validate that the chosen Hilbert-space cutoff is sufficient.")
    return out
