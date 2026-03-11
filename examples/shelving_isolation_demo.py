from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import (
    DispersiveTransmonCavityModel,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    build_sideband_pulse,
    compute_shelving_leakage,
    simulate_sequence,
    subsystem_level_population,
)


def main() -> None:
    out_dir = Path("examples") / "outputs" / "shelving_isolation_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    duration = 6.0
    amplitude = np.pi / (2.0 * duration)
    pulses, drive_ops, _meta = build_sideband_pulse(
        SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2),
        duration_s=duration,
        amplitude_rad_s=amplitude,
        sigma_fraction=0.18,
        channel="sb",
        label="gaussian_gf_sideband",
    )
    compiled = SequenceCompiler(dt=0.01).compile(pulses, t_end=duration)

    initial = (np.sqrt(0.4) * model.basis_state(1, 0) + np.sqrt(0.6) * model.basis_state(2, 0)).unit()
    result = simulate_sequence(model, compiled, initial, drive_ops, SimulationConfig())

    summary = {
        "model": "effective gf sideband with population shelved in |e>",
        "shelved_level": "e",
        "initial_p_e": 0.4,
        "final_p_e": subsystem_level_population(result.final_state, "transmon", 1),
        "shelving_leakage": compute_shelving_leakage(initial, result.final_state, shelved_level=1),
        "final_p_g1": float(abs(model.basis_state(0, 1).overlap(result.final_state)) ** 2),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
