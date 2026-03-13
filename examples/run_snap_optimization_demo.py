from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from examples.studies.snap_opt import SnapModelConfig, SnapRunConfig, SnapToneParameters, optimize_snap_parameters


def main():
    out = Path(__file__).resolve().parent / "outputs_snap_opt"
    out.mkdir(parents=True, exist_ok=True)
    model = SnapModelConfig(n_cav=7, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, 1.1, -0.7, 0.4], dtype=float)
    cfg = SnapRunConfig(duration=170.0, dt=0.2, base_amp=0.010)
    vanilla = SnapToneParameters.vanilla(target)
    res = optimize_snap_parameters(model, target, cfg, initial_params=vanilla, max_iter=40, learning_rate=0.3, threshold=6e-3)
    with (out / "demo_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "initial_error": float(res.history_error[0]),
                "final_error": float(res.history_error[-1]),
                "iterations": len(res.history_error),
                "converged": bool(res.converged),
                "A": res.params.amplitudes.tolist(),
                "delta": res.params.detunings.tolist(),
                "phi": res.params.phases.tolist(),
            },
            f,
            indent=2,
        )
    print("SNAP optimization summary saved to", out / "demo_summary.json")


if __name__ == "__main__":
    main()

