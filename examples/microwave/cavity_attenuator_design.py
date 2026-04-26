from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import (
    LosslessFilterStage,
    MicrowaveNoiseChain,
    PassiveLossStage,
    TwoModeCavityAttenuatorModel,
    n_bose,
    sweep_cavity_attenuator_design,
)


def main() -> None:
    output_dir = ROOT / "documentations" / "assets" / "images" / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cavity_attenuator_design.png"

    f_ro = 7.573e9
    omega_ro = 2.0 * np.pi * f_ro
    kappa_c = 2.0 * np.pi * 1.9e6
    chi = 2.0 * np.pi * 1.2e6
    external_nbar = 7.0e-4
    internal_nbar = n_bose(f_ro, 0.020)
    ratios = np.logspace(-2, 2, 160)
    design = sweep_cavity_attenuator_design(
        ratios,
        omega_ro_rad_s=omega_ro,
        kappa_external_rad_s=kappa_c,
        external_nbar=external_nbar,
        chi_rad_s=chi,
        T1_s=100e-6,
        internal_nbar=internal_nbar,
    )

    detuning = 2.0 * np.pi * np.linspace(-60e6, 60e6, 200)
    readout_participation = []
    attenuator_participation = []
    for delta in detuning:
        model = TwoModeCavityAttenuatorModel(
            omega_readout_rad_s=omega_ro,
            omega_attenuator_rad_s=omega_ro + delta,
            coupling_J_rad_s=2.0 * np.pi * 10e6,
            kappa_readout_internal_rad_s=2.0 * np.pi * 0.05e6,
            kappa_attenuator_internal_rad_s=2.0 * np.pi * 11.4e6,
            kappa_external_rad_s=kappa_c,
            chi_bare_readout_rad_s=chi,
        )
        parts = model.participation_ratios()
        readout_like = int(np.argmax(parts["readout"]))
        readout_participation.append(parts["readout"][readout_like])
        attenuator_participation.append(parts["attenuator"][readout_like])

    lossless = MicrowaveNoiseChain(
        [LosslessFilterStage("lossless-control", f_ro, 20e6, 40.0)],
        input_temperature_K=0.060,
    )
    dissipative = MicrowaveNoiseChain(
        [PassiveLossStage("cold-cavity-attenuator", 8.45, 0.020, center_frequency_hz=f_ro, bandwidth_hz=40e6)],
        input_temperature_K=0.060,
    )
    no_attenuator_nbar = n_bose(f_ro, 0.060)
    filter_nbar = lossless.propagate_nbar(f_ro)
    dissipative_nbar = dissipative.propagate_nbar(f_ro)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    axes[0, 0].loglog(ratios, design["nbar_eff"])
    axes[0, 0].axvline(6.0, color="0.5", ls="--", label="Wang-like 6:1")
    axes[0, 0].set_xlabel("$\\kappa_i/\\kappa_c$")
    axes[0, 0].set_ylabel("$\\bar n_\\mathrm{eff}$")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].semilogx(ratios, design["T2_over_2T1"])
    axes[0, 1].set_xlabel("$\\kappa_i/\\kappa_c$")
    axes[0, 1].set_ylabel("$T_2/(2T_1)$")

    axes[1, 0].plot(detuning / (2.0 * np.pi * 1e6), readout_participation, label="readout")
    axes[1, 0].plot(detuning / (2.0 * np.pi * 1e6), attenuator_participation, label="attenuator")
    axes[1, 0].set_xlabel("attenuator detuning (MHz)")
    axes[1, 0].set_ylabel("readout-like mode participation")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].bar(["none", "lossless filter", "cold attenuator"], [no_attenuator_nbar, filter_nbar, dissipative_nbar])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("in-band $\\bar n$")

    fig.savefig(output_path, dpi=180)
    print(
        json.dumps(
            {
                "plot": str(output_path),
                "nbar_at_ratio_6": float(np.interp(6.0, ratios, design["nbar_eff"])),
                "lossless_filter_in_band_nbar": float(filter_nbar),
                "cold_attenuator_in_band_nbar": float(dissipative_nbar),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
