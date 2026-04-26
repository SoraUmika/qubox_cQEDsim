from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import thermal_photon_dephasing


def main() -> None:
    kappa = 2.0 * np.pi * 5.0e6
    chi = 2.0 * np.pi * 50.0e6
    n_cav = 8.696185e-3

    exact = thermal_photon_dephasing(kappa, chi, n_cav)
    weak = thermal_photon_dephasing(kappa, chi, n_cav, approximation="weak")
    strong = thermal_photon_dephasing(kappa, chi, n_cav, approximation="strong_low_occupation")

    summary = {
        "kappa_over_2pi_hz": kappa / (2.0 * np.pi),
        "chi_over_2pi_hz": chi / (2.0 * np.pi),
        "n_cav": n_cav,
        "Gamma_phi_exact_over_2pi_hz": exact / (2.0 * np.pi),
        "Gamma_phi_weak_over_2pi_hz": weak / (2.0 * np.pi),
        "Gamma_phi_strong_low_occupation_over_2pi_hz": strong / (2.0 * np.pi),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
