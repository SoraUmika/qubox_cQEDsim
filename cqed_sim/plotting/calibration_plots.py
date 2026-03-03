from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.calibration.sqr import SQRCalibrationResult


def plot_sqr_calibration_result(result: SQRCalibrationResult):
    n_values = np.arange(result.max_n + 1, dtype=int)
    fig, axes = plt.subplots(4, 1, figsize=(9.5, 12.0), sharex=True)
    axes[0].plot(n_values, result.d_lambda, "o-")
    axes[0].set_ylabel(r"$d_{\lambda,n}$")
    axes[0].grid(alpha=0.25)

    axes[1].plot(n_values, result.d_alpha, "o-")
    axes[1].set_ylabel(r"$d_{\alpha,n}$ [rad]")
    axes[1].grid(alpha=0.25)

    axes[2].plot(n_values, result.d_omega_hz, "o-")
    axes[2].set_ylabel(r"$d_{\omega,n}/2\pi$ [Hz]")
    axes[2].grid(alpha=0.25)

    safe_loss = np.maximum(np.asarray(result.optimized_loss, dtype=float), 1.0e-16)
    axes[3].semilogy(n_values, safe_loss, "o-")
    axes[3].set_ylabel("Final loss")
    axes[3].set_xlabel("Fock level n")
    axes[3].grid(alpha=0.25)

    fig.suptitle(f"SQR calibration corrections for {result.sqr_name}")
    fig.tight_layout()
    return fig
