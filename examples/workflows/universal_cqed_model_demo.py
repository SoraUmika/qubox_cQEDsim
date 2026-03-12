from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import (
    BosonicModeSpec,
    DispersiveCouplingSpec,
    DispersiveTransmonCavityModel,
    FrameSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
)
from physics_and_conventions.conventions import from_internal_units, to_internal_units


def _mhz(value_mhz: float) -> float:
    return to_internal_units(float(value_mhz) * 1.0e6)


def main() -> None:
    out_dir = Path("examples") / "outputs" / "universal_cqed_model_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0, omega_r_frame=0.0)
    universal = UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=0.0,
            dim=5,
            alpha=_mhz(-200.0),
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=0.0,
                dim=12,
                kerr=_mhz(-0.002),
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
        ),
        dispersive_couplings=(
            DispersiveCouplingSpec(
                mode="storage",
                chi=_mhz(-3.0),
                transmon="qubit",
            ),
        ),
    )

    wrapper = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=_mhz(-200.0),
        chi=_mhz(-3.0),
        kerr=_mhz(-0.002),
        n_cav=12,
        n_tr=5,
    )

    summary = {
        "universal_subsystem_labels": list(universal.subsystem_labels),
        "universal_subsystem_dims": list(universal.subsystem_dims),
        "wrapper_subsystem_dims": list(wrapper.subsystem_dims),
        "hamiltonian_difference_norm": float((universal.hamiltonian(frame) - wrapper.hamiltonian(frame)).norm()),
        "ge_mhz": from_internal_units(universal.transmon_transition_frequency(mode_levels={"storage": 0})) / 1.0e6,
        "ef_mhz": from_internal_units(
            universal.transmon_transition_frequency(mode_levels={"storage": 0}, lower_level=1, upper_level=2)
        )
        / 1.0e6,
        "g1_minus_f0_sideband_mhz": from_internal_units(
            universal.sideband_transition_frequency(mode="storage", mode_levels={"storage": 0}, lower_level=0, upper_level=2)
        )
        / 1.0e6,
        "storage_0_to_1_mhz": from_internal_units(
            universal.mode_transition_frequency("storage", mode_levels={"storage": 0}, transmon_level=0)
        )
        / 1.0e6,
        "wrapper_compatibility": "DispersiveTransmonCavityModel now delegates to UniversalCQEDModel.",
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
