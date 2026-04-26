from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import (
    NoiseCascade,
    PassiveLoss,
    bose_occupation,
    resonator_thermal_occupation,
)


def main() -> None:
    storage_freq_hz = 5.0e9
    readout_freq_hz = 7.0e9

    drive_line = NoiseCascade(
        [
            PassiveLoss("4K", temp_K=4.0, loss_db=20.0),
            PassiveLoss("MXC", temp_K=0.02, loss_db=40.0),
        ]
    )
    storage_port_n = drive_line.propagate(storage_freq_hz, source_temp_K=300.0).n_out
    weak_internal_bath_n = bose_occupation(storage_freq_hz, 0.02)

    kappa_storage_ports = [2.0e3, 8.0e3]
    n_storage_baths = [storage_port_n, weak_internal_bath_n]
    n_storage = resonator_thermal_occupation(kappa_storage_ports, n_storage_baths)

    readout_port_n = drive_line.propagate(readout_freq_hz, source_temp_K=300.0).n_out
    n_readout = resonator_thermal_occupation([1.0e6], [readout_port_n])

    summary = {
        "storage": {
            "freq_hz": storage_freq_hz,
            "kappas_rad_s_or_consistent_units": kappa_storage_ports,
            "bath_occupations": [float(value) for value in n_storage_baths],
            "steady_state_occupation": float(n_storage),
        },
        "readout": {
            "freq_hz": readout_freq_hz,
            "bath_occupations": [float(readout_port_n)],
            "steady_state_occupation": float(n_readout),
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
