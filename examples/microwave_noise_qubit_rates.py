from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import bose_occupation, qubit_thermal_rates


def main() -> None:
    qubit_freq_hz = 6.0e9
    bath_temp_K = 0.06
    gamma_zero_temp = 1.0 / 80.0e-6

    n_bath = bose_occupation(qubit_freq_hz, bath_temp_K)
    gamma_down, gamma_up, gamma_1 = qubit_thermal_rates(gamma_zero_temp, n_bath)

    summary = {
        "qubit_freq_hz": qubit_freq_hz,
        "bath_temp_K": bath_temp_K,
        "bath_occupation": float(n_bath),
        "gamma_zero_temp_rad_s_or_consistent_units": gamma_zero_temp,
        "Gamma_down": gamma_down,
        "Gamma_up": gamma_up,
        "Gamma_1": gamma_1,
        "up_down_ratio": gamma_up / gamma_down,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
