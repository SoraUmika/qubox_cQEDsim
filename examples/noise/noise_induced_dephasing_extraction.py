from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import fit_noise_induced_dephasing, simulate_noise_induced_dephasing


def main() -> None:
    output_dir = ROOT / "documentations" / "assets" / "images" / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "noise_induced_dephasing_extraction.png"

    nadd = np.linspace(0.0, 0.08, 13)
    result = simulate_noise_induced_dephasing(
        nadd,
        nth=2.0e-4,
        kappa_rad_s=2.0 * np.pi * 1.9e6,
        chi_rad_s=2.0 * np.pi * 1.2e6,
        T1_s=100e-6,
        measurement_noise={"relative_T2_sigma": 0.005, "seed": 7},
    )
    fit = fit_noise_induced_dephasing(
        result.nadd_values,
        result.T1_values_s,
        result.T2e_values_s,
        2.0 * np.pi * 1.9e6,
        2.0 * np.pi * 1.2e6,
    )
    gamma_fit = fit.fitted_slope_rad_s * nadd + fit.metadata["model_slope_rad_s_per_photon"] * fit.fitted_nth
    gamma_data = 1.0 / fit.Tphi_values_s

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    axes[0].plot(nadd, result.T1_values_s * 1e6, "o-", label="$T_1$")
    axes[0].plot(nadd, result.T2e_values_s * 1e6, "o-", label="$T_{2e}$")
    axes[0].plot(nadd, result.Tphi_values_s * 1e6, "o-", label="$T_\\phi$")
    axes[0].set_xlabel("$\\bar n_\\mathrm{add}$")
    axes[0].set_ylabel("time ($\\mu$s)")
    axes[0].legend(fontsize=8)

    axes[1].plot(nadd, gamma_data / (2.0 * np.pi), "o", label="synthetic data")
    axes[1].plot(nadd, gamma_fit / (2.0 * np.pi), "-", label="linear fit")
    axes[1].set_xlabel("$\\bar n_\\mathrm{add}$")
    axes[1].set_ylabel("$\\Gamma_\\phi/2\\pi$ (Hz)")
    axes[1].legend(fontsize=8)

    axes[2].errorbar(
        [0.0],
        [fit.fitted_nth],
        yerr=[[fit.fitted_nth - fit.confidence_interval[0]], [fit.confidence_interval[1] - fit.fitted_nth]],
        fmt="o",
        capsize=4,
    )
    axes[2].axhline(2.0e-4, color="0.5", ls="--", label="true")
    axes[2].set_xlim(-0.5, 0.5)
    axes[2].set_xticks([])
    axes[2].set_ylabel("extracted $\\bar n_\\mathrm{th}$")
    axes[2].legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    print(
        json.dumps(
            {
                "plot": str(output_path),
                "fitted_nth": float(fit.fitted_nth),
                "confidence_interval": [float(value) for value in fit.confidence_interval],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
