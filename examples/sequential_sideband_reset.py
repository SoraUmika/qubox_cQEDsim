from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.workflows.sequential_sideband_reset import (
    SequentialSidebandResetCalibration,
    SequentialSidebandResetDevice,
    build_sideband_reset_frame,
    build_sideband_reset_model,
    build_sideband_reset_noise,
    run_sequential_sideband_reset,
)


def _device() -> SequentialSidebandResetDevice:
    return SequentialSidebandResetDevice(
        readout_frequency_hz=8596222556.078796,
        qubit_frequency_hz=6150358764.4830475,
        storage_frequency_hz=5240932800.0,
        readout_kappa_hz=4.156e6,
        qubit_anharmonicity_hz=-255669694.5244608,
        chi_storage_hz=-2840421.0,
        chi_readout_hz=-3.0e6,
        storage_gf_sideband_frequency_hz=6803533628.0,
        storage_t1_s=250.0e-6,
        storage_t2_ramsey_s=150.0e-6,
    )


def _calibration() -> SequentialSidebandResetCalibration:
    return SequentialSidebandResetCalibration(
        storage_sideband_rate_hz=8.0e6,
        readout_sideband_rate_hz=10.0e6,
        ef_rate_hz=12.0e6,
        ringdown_multiple=3.0,
    )


def main() -> None:
    out_dir = ROOT / "examples" / "outputs" / "sequential_sideband_reset"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _device()
    calibration = _calibration()
    model = build_sideband_reset_model(device, n_storage=5, n_readout=3)
    result = run_sequential_sideband_reset(
        model,
        model.basis_state(0, 2, 0),
        calibration=calibration,
        initial_storage_level=2,
        frame=build_sideband_reset_frame(model),
        noise=build_sideband_reset_noise(device),
        pulse_dt_s=0.25e-9,
        ringdown_dt_s=4.0e-9,
    )

    summary = {
        "n_cycles": int(result.cycle_final_storage_photon_number.size),
        "final_storage_photon_number": float(result.cycle_final_storage_photon_number[-1]),
        "final_readout_photon_number": float(result.cycle_final_readout_photon_number[-1]),
        "n_stage_records": len(result.stage_records),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
