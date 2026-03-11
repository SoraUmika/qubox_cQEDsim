from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import coherent_state, plot_kerr_wigner_snapshots, run_kerr_free_evolution, times_us_to_seconds

TIMES_US = [0, 1, 2, 4, 6, 8, 10, 12, 14]


def main() -> None:
    out_dir = Path("examples") / "outputs" / "kerr_free_evolution"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_kerr_free_evolution(
        times_us_to_seconds(TIMES_US),
        cavity_state=coherent_state(1.8),
        parameter_set="phase_evolution",
        n_cav=30,
        wigner_n_points=121,
        wigner_extent=5.0,
    )

    summary = {
        "parameter_set": result.parameter_set.name,
        "times_us": TIMES_US,
        "omega_q_hz": result.parameter_set.omega_q_hz,
        "omega_c_hz": result.parameter_set.omega_c_hz,
        "omega_ro_hz": result.parameter_set.omega_ro_hz,
        "alpha_q_hz": result.parameter_set.alpha_q_hz,
        "snapshots": [
            {
                "time_us": snapshot.time_us,
                "n": snapshot.cavity_photon_number,
                "a_real": float(np.real(snapshot.cavity_mean)),
                "a_imag": float(np.imag(snapshot.cavity_mean)),
            }
            for snapshot in result.snapshots
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        fig = plot_kerr_wigner_snapshots(result, max_cols=3, show_colorbar=True)
        fig.savefig(out_dir / "wigner_snapshots.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    print(f"Saved Kerr free-evolution outputs to {out_dir}")


if __name__ == "__main__":
    main()