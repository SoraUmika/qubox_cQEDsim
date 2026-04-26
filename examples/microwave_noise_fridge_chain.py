from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import NoiseCascade, PassiveLoss


def main() -> None:
    freq_hz = 6.0e9
    cascade = NoiseCascade(
        [
            PassiveLoss("4K", temp_K=4.0, loss_db=20.0),
            PassiveLoss("still", temp_K=0.7, loss_db=10.0),
            PassiveLoss("100mK", temp_K=0.1, loss_db=0.01),
            PassiveLoss("MXC", temp_K=0.06, loss_db=40.0),
            PassiveLoss("MXC2", temp_K=0.02, loss_db=0.01),
        ]
    )
    result = cascade.propagate(freq_hz, source_temp_K=300.0)
    summary = {
        "freq_hz": freq_hz,
        "n_out": float(result.n_out),
        "effective_temperature_K": float(result.effective_temperature),
        "weights": {name: float(value) for name, value in result.budget.weights.items()},
        "contributions": {name: float(value) for name, value in result.budget.contributions.items()},
        "weight_sum": float(result.budget.weight_sum),
        "contribution_sum": float(result.budget.contribution_sum),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
