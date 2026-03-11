from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import DispersiveReadoutTransmonStorageModel, SequenceCompiler, SimulationConfig, simulate_sequence


def main() -> None:
    out_dir = Path("examples") / "outputs" / "multimode_crosskerr_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    chi_sr = 0.14
    evolution_time = 8.0
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi_s=0.0,
        chi_r=0.0,
        chi_sr=chi_sr,
        kerr_s=0.0,
        kerr_r=0.0,
        n_storage=3,
        n_readout=3,
        n_tr=3,
    )
    initial = (model.basis_state(0, 0, 1) + model.basis_state(0, 1, 1)).unit()
    compiled = SequenceCompiler(dt=evolution_time).compile([], t_end=evolution_time)
    result = simulate_sequence(model, compiled, initial, {}, SimulationConfig())

    amp_ref = model.basis_state(0, 0, 1).overlap(result.final_state)
    amp_shifted = model.basis_state(0, 1, 1).overlap(result.final_state)
    relative_phase = float(np.angle(amp_shifted / amp_ref))
    expected_phase = float(np.angle(np.exp(-1j * chi_sr * evolution_time)))

    summary = {
        "model": "storage-readout cross-Kerr free evolution",
        "chi_sr_rad_s": chi_sr,
        "evolution_time_s": evolution_time,
        "simulated_relative_phase_rad": relative_phase,
        "expected_relative_phase_rad": expected_phase,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
