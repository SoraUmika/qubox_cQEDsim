from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim.microwave_noise import T2_from_T1_Tphi, Tphi_from_gamma, gamma_phi_multimode, n_bose


def main() -> None:
    output_dir = ROOT / "documentations" / "assets" / "images" / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "multimode_thermal_photons.png"

    names = ["storage", "readout", "package"]
    frequencies = np.array([5.0e9, 7.5e9, 9.2e9])
    kappas = 2.0 * np.pi * np.array([3.0e3, 1.0e6, 60.0e3])
    chis = 2.0 * np.pi * np.array([2.8e6, 1.2e6, 0.35e6])
    temperatures = np.array([0.035, 0.060, 0.090])
    nbars = np.asarray([n_bose(freq, temp) for freq, temp in zip(frequencies, temperatures)])
    total, contributions = gamma_phi_multimode(nbars, kappas, chis, mode_names=names)

    temp_grid = np.linspace(0.015, 0.12, 100)
    tphi_by_mode = []
    for freq, kappa, chi in zip(frequencies, kappas, chis):
        gammas = [gamma_phi_multimode([n_bose(freq, temp)], [kappa], [chi])[0] for temp in temp_grid]
        tphi_by_mode.append([Tphi_from_gamma(gamma) for gamma in gammas])
    total_t2 = []
    storage_only_t2 = []
    T1 = 100e-6
    for temp in temp_grid:
        all_n = [n_bose(freq, temp) for freq in frequencies]
        gamma_all = gamma_phi_multimode(all_n, kappas, chis)[0]
        gamma_storage = gamma_phi_multimode([all_n[0]], [kappas[0]], [chis[0]])[0]
        total_t2.append(T2_from_T1_Tphi(T1, Tphi_from_gamma(gamma_all)))
        storage_only_t2.append(T2_from_T1_Tphi(T1, Tphi_from_gamma(gamma_storage)))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6), constrained_layout=True)
    axes[0].bar(names, [item.gamma_phi_rad_s / (2.0 * np.pi) for item in contributions])
    axes[0].set_ylabel("$\\Gamma_\\phi/2\\pi$ (Hz)")
    axes[0].set_title("Mode dephasing budget")

    for name, tphi_values in zip(names, tphi_by_mode):
        axes[1].semilogy(temp_grid * 1e3, np.asarray(tphi_values) * 1e6, label=name)
    axes[1].set_xlabel("effective mode temperature (mK)")
    axes[1].set_ylabel("$T_\\phi$ ($\\mu$s)")
    axes[1].legend(fontsize=8)

    axes[2].plot(temp_grid * 1e3, np.asarray(total_t2) * 1e6, label="all modes")
    axes[2].plot(temp_grid * 1e3, np.asarray(storage_only_t2) * 1e6, "--", label="storage only")
    axes[2].set_xlabel("common bath temperature (mK)")
    axes[2].set_ylabel("$T_2$ ($\\mu$s)")
    axes[2].legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    print(
        json.dumps(
            {
                "plot": str(output_path),
                "total_Gamma_phi_over_2pi_Hz": float(total / (2.0 * np.pi)),
                "contributions_over_2pi_Hz": {
                    item.mode_name: float(item.gamma_phi_rad_s / (2.0 * np.pi)) for item in contributions
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
