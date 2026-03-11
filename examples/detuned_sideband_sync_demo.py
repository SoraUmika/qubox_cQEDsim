from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import (
    DispersiveTransmonCavityModel,
    Pulse,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    carrier_for_transition_frequency,
    sideband_transition_frequency,
    simulate_sequence,
)


def _square(t_rel):
    import numpy as np

    return np.ones_like(t_rel, dtype=np.complex128)


def main() -> None:
    out_dir = Path("examples") / "outputs" / "detuned_sideband_sync_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    g_sb = 0.35
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=-0.2,
        kerr=0.0,
        n_cav=5,
        n_tr=3,
    )
    target_spec = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)
    initial = (model.basis_state(2, 0) + model.basis_state(2, 1)).unit()
    target = (model.basis_state(0, 1) + model.basis_state(2, 1)).unit()
    base_frequency = sideband_transition_frequency(model, cavity_level=0, lower_level=0, upper_level=2)

    naive_duration = 3.141592653589793 / (2.0 * g_sb)
    naive_pulse = Pulse(
        "sb",
        0.0,
        naive_duration,
        _square,
        amp=g_sb,
        carrier=carrier_for_transition_frequency(base_frequency),
        label="resonant_reference",
    )
    naive_compiled = SequenceCompiler(dt=0.02).compile([naive_pulse], t_end=naive_duration)
    naive_result = simulate_sequence(model, naive_compiled, initial, {"sb": target_spec}, SimulationConfig())
    naive_fidelity = abs(target.overlap(naive_result.final_state)) ** 2

    optimized_detuning = 0.16
    optimized_duration = 5.11
    optimized_pulse = Pulse(
        "sb",
        0.0,
        optimized_duration,
        _square,
        amp=g_sb,
        carrier=carrier_for_transition_frequency(base_frequency + optimized_detuning),
        label="detuned_sync",
    )
    optimized_compiled = SequenceCompiler(dt=0.02).compile([optimized_pulse], t_end=optimized_duration)
    optimized_result = simulate_sequence(model, optimized_compiled, initial, {"sb": target_spec}, SimulationConfig())
    optimized_fidelity = abs(target.overlap(optimized_result.final_state)) ** 2

    summary = {
        "model": "effective gf sideband with photon-number-dependent detuning",
        "assumptions": [
            "single effective sideband tone",
            "branch mismatch caused by dispersive offset chi",
            "improvement measured against a simple target superposition",
        ],
        "naive_duration": naive_duration,
        "naive_fidelity": float(naive_fidelity),
        "optimized_detuning": optimized_detuning,
        "optimized_duration": optimized_duration,
        "optimized_fidelity": float(optimized_fidelity),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
