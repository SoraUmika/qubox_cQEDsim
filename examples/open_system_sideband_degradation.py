from __future__ import annotations

import json
from pathlib import Path
import sys

import qutip as qt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import (
    DispersiveTransmonCavityModel,
    NoiseSpec,
    Pulse,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    simulate_sequence,
)


def _square(t_rel):
    import numpy as np

    return np.ones_like(t_rel, dtype=np.complex128)


def main() -> None:
    out_dir = ROOT / "examples" / "outputs" / "open_system_sideband_degradation"
    out_dir.mkdir(parents=True, exist_ok=True)

    g_sb = 0.3
    duration = 3.141592653589793 / (2.0 * g_sb)
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    pulse = Pulse("sb", 0.0, duration, _square, amp=g_sb, label="gf_swap")
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=duration)
    drive_ops = {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)}
    target = model.basis_state(0, 1)

    closed = simulate_sequence(model, compiled, model.basis_state(2, 0), drive_ops, SimulationConfig())
    noisy = simulate_sequence(
        model,
        compiled,
        model.basis_state(2, 0),
        drive_ops,
        SimulationConfig(),
        noise=NoiseSpec(transmon_t1=(120.0, 35.0), tphi=90.0, kappa=0.02),
    )

    summary = {
        "closed_fidelity": float(abs(target.overlap(closed.final_state)) ** 2),
        "open_system_fidelity": float(qt.fidelity(noisy.final_state, target.proj())),
        "noise_model": {
            "transmon_t1_ge_s": 120.0,
            "transmon_t1_fe_s": 35.0,
            "tphi_s": 90.0,
            "kappa_s_inv": 0.02,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
