from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec, StatePreparationSpec, fock_state, prepare_state, qubit_state
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit
from cqed_sim.pulses import Pulse
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def main() -> None:
    out_dir = ROOT / "examples" / "outputs" / "protocol_style_simulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )
    frame = FrameSpec()
    initial = prepare_state(
        model,
        StatePreparationSpec(
            qubit=qubit_state("g"),
            storage=fock_state(0),
        ),
    )
    pulse = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=1.1)
    simulation = simulate_sequence(
        model,
        compiled,
        initial,
        {"q": "qubit"},
        config=SimulationConfig(frame=frame),
    )
    measurement = measure_qubit(simulation.final_state, QubitMeasurementSpec(shots=2048, seed=7))

    summary = {
        "compiled_samples": int(compiled.tlist.size),
        "final_excited_population": float(simulation.expectations["P_e"][-1]),
        "measurement_counts": measurement.counts,
        "measurement_probabilities": measurement.probabilities,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
