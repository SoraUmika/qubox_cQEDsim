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
    T2_from_T1_Tphi,
    Tphi_from_gamma,
    gamma_phi_lorentzian_interpolation,
    gamma_phi_strong_dispersive_N,
    gamma_phi_thermal,
)


def main() -> None:
    output_dir = ROOT / "documentations" / "assets" / "images" / "tutorials"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "photon_shot_noise_dephasing.png"

    T1 = 100e-6
    kappa = 2.0 * np.pi * 1.0e6
    chi = 2.0 * np.pi * 1.0e6
    nbar = np.logspace(-5, -1, 120)
    kappa_values = 2.0 * np.pi * np.array([0.1e6, 1.0e6, 10.0e6])
    chi_over_kappa = np.logspace(-3, 3, 160)
    times = np.linspace(0.0, 60e-6, 300)
    trace_nbars = [2.0e-4, 2.0e-3, 2.0e-2]

    gamma = np.asarray([gamma_phi_thermal(value, kappa, chi) for value in nbar])
    tphi = np.asarray([Tphi_from_gamma(value) for value in gamma])
    t2 = np.asarray([T2_from_T1_Tphi(T1, value) for value in tphi])

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.loglog(nbar, 1e6 * t2)
    ax.set_xlabel("thermal occupation $\\bar n$")
    ax.set_ylabel("$T_2^*$ estimate ($\\mu$s)")
    ax.set_title("Coherence versus residual photons")

    ax = axes[0, 1]
    for kappa_i in kappa_values:
        values = [gamma_phi_thermal(value, kappa_i, chi) / (2.0 * np.pi) for value in nbar]
        ax.loglog(nbar, values, label=f"$\\kappa/2\\pi={kappa_i/(2*np.pi*1e6):.1f}$ MHz")
    ax.set_xlabel("thermal occupation $\\bar n$")
    ax.set_ylabel("$\\Gamma_\\phi/2\\pi$ (Hz)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    exact = [gamma_phi_thermal(2.0e-4, kappa, ratio * kappa) / kappa for ratio in chi_over_kappa]
    lorentz = [gamma_phi_lorentzian_interpolation(2.0e-4, kappa, ratio * kappa) / kappa for ratio in chi_over_kappa]
    ax.loglog(chi_over_kappa, exact, label="canonical")
    ax.loglog(chi_over_kappa, lorentz, "--", label="Lorentzian interpolation")
    ax.axhline(2.0e-4, color="0.5", lw=1, label="strong limit")
    ax.set_xlabel("$|\\chi|/\\kappa$")
    ax.set_ylabel("$\\Gamma_\\phi/\\kappa$")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    for nth in trace_nbars:
        gamma_i = gamma_phi_thermal(nth, kappa, chi)
        ax.plot(times * 1e6, np.exp(-gamma_i * times), label=f"$\\bar n={nth:g}$")
    gamma_N0 = gamma_phi_strong_dispersive_N(1.0e-3, 2.0 * np.pi * 100e3, 0)
    gamma_N1 = gamma_phi_strong_dispersive_N(1.0e-3, 2.0 * np.pi * 100e3, 1)
    ax.set_xlabel("time ($\\mu$s)")
    ax.set_ylabel("Ramsey envelope")
    ax.set_title(f"Strong dispersive N=0: {gamma_N0/(2*np.pi):.0f} Hz, N=1: {gamma_N1/(2*np.pi):.0f} Hz")
    ax.legend(fontsize=8)

    fig.savefig(output_path, dpi=180)
    summary = {
        "plot": str(output_path),
        "T2star_us_at_nbar_2e-4": float(1e6 * t2[np.argmin(np.abs(nbar - 2.0e-4))]),
        "Gamma_phi_over_2pi_Hz_at_nbar_2e-4": float(gamma[np.argmin(np.abs(nbar - 2.0e-4))] / (2.0 * np.pi)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
