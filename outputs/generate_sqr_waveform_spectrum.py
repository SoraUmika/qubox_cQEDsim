from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import SQRGate
from cqed_sim.pulses.builders import build_sqr_multitone_pulse


def main():
    chi_sqr_hz = -2.84e6
    n_cav = 20
    sqr_duration = 1.0e-6
    sqr_sigma_fraction = 1.0 / 6.0
    n_levels_sqr = (0, 1, 2)

    theta_targets = {
        0: 0.5 * np.pi,
        1: 0.33 * np.pi,
        2: 0.25 * np.pi,
    }
    phi_targets = {
        0: 0.0,
        1: 0.5 * np.pi,
        2: -0.33 * np.pi,
    }

    theta_values = [0.0] * n_cav
    phi_values = [0.0] * n_cav
    for n_level in n_levels_sqr:
        theta_values[n_level] = float(theta_targets[n_level])
        phi_values[n_level] = float(phi_targets[n_level])

    gate_sqr_waveform = SQRGate(
        index=0,
        name="sqr_n012_custom_angles",
        theta=tuple(theta_values),
        phi=tuple(phi_values),
    )

    model_sqr_waveform = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2 * np.pi * chi_sqr_hz,
        n_cav=n_cav,
        n_tr=2,
    )

    pulses_sqr_waveform, _drive_ops_sqr_waveform, meta_sqr_waveform = build_sqr_multitone_pulse(
        gate_sqr_waveform,
        model_sqr_waveform,
        {
            "duration_sqr_s": sqr_duration,
            "sqr_sigma_fraction": sqr_sigma_fraction,
            "sqr_theta_cutoff": 1.0e-10,
            "use_rotating_frame": True,
        },
    )

    if len(pulses_sqr_waveform) != 1:
        raise RuntimeError(f"Expected a single effective SQR pulse, got {len(pulses_sqr_waveform)} pulses.")

    sqr_pulse = pulses_sqr_waveform[0]
    num_samples = 8192
    t_waveform = np.linspace(0.0, sqr_pulse.duration, num_samples, endpoint=False)
    waveform_complex = sqr_pulse.sample(t_waveform)
    i_waveform = np.real(waveform_complex)
    q_waveform = np.imag(waveform_complex)

    sample_dt = float(t_waveform[1] - t_waveform[0])
    n_fft = 2 ** int(np.ceil(np.log2(num_samples * 16)))
    freq_hz = np.fft.fftfreq(n_fft, d=sample_dt)
    spectrum = np.fft.fft(waveform_complex, n=n_fft)
    positive = freq_hz >= 0.0
    freq_mhz = freq_hz[positive] / 1e6
    spectrum_mag = np.abs(spectrum[positive])
    spectrum_mag /= np.max(spectrum_mag)

    active_tones = sorted(meta_sqr_waveform["active_tones"], key=lambda tone: tone["omega_rad_s"])
    peak_freqs_hz = np.array([tone["omega_rad_s"] / (2 * np.pi) for tone in active_tones], dtype=float)
    peak_freqs_mhz = peak_freqs_hz / 1e6
    positive_freq_hz = freq_hz[positive]
    peak_indices = [int(np.argmin(np.abs(positive_freq_hz - freq_hz_val))) for freq_hz_val in peak_freqs_hz]
    peak_heights = spectrum_mag[peak_indices]
    chi_sep_mhz = abs(chi_sqr_hz) / 1e6
    theta_over_pi = [theta_targets[n_level] / np.pi for n_level in n_levels_sqr]
    phi_over_pi = [phi_targets[n_level] / np.pi for n_level in n_levels_sqr]

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 15,
            "legend.fontsize": 11,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.8), dpi=180)

    axes[0].plot(t_waveform * 1e9, i_waveform, label="I = Re[w(t)]", color="#2563eb")
    axes[0].plot(t_waveform * 1e9, q_waveform, label="Q = Im[w(t)]", color="#dc2626")
    axes[0].set_xlabel("Time (ns)")
    axes[0].set_ylabel("Drive amplitude (rad/s)")
    axes[0].set_title("SQR I/Q waveform")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="upper right", framealpha=0.92)

    axes[1].plot(freq_mhz, spectrum_mag, color="#0f172a", lw=2.2, label=r"$|\mathcal{F}[w(t)]|$")
    for tone, peak_freq_mhz, peak_height in zip(active_tones, peak_freqs_mhz, peak_heights):
        axes[1].axvline(peak_freq_mhz, color="#94a3b8", lw=1.0, linestyle="--", alpha=0.8)
        axes[1].scatter([peak_freq_mhz], [peak_height], color="#2563eb", s=42, zorder=4)
        axes[1].annotate(
            f"n={tone['n']}\n{peak_freq_mhz:.2f} MHz",
            xy=(peak_freq_mhz, peak_height),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.95},
        )

    left_idx, right_idx = 0, 1
    x0 = peak_freqs_mhz[left_idx]
    x1 = peak_freqs_mhz[right_idx]
    y_arrow = 0.88
    axes[1].annotate(
        "",
        xy=(x0, y_arrow),
        xytext=(x1, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": "#d97706", "lw": 1.8},
    )
    axes[1].text(
        0.5 * (x0 + x1),
        y_arrow + 0.04,
        rf"$\Delta f \approx {abs(x1 - x0):.2f}\,\mathrm{{MHz}} \approx |\chi|/2\pi$",
        ha="center",
        va="bottom",
        color="#92400e",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "#fff7ed", "edgecolor": "#fdba74", "alpha": 0.95},
    )

    axes[1].set_xlim(-0.2, max(peak_freqs_mhz[-1] + 2.0 * chi_sep_mhz, 8.5))
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_xlabel("Frequency (MHz)")
    axes[1].set_ylabel("Normalized spectral magnitude")
    axes[1].set_title("SQR waveform Fourier spectrum")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", framealpha=0.92)

    theta_text = ", ".join(f"{value:.2f}" for value in theta_over_pi)
    phi_text = ", ".join(f"{value:.2f}" for value in phi_over_pi)
    fig.suptitle(
        rf"SQR waveform with $\theta/\pi = [{theta_text}]$ and $\phi/\pi = [{phi_text}]$",
        fontsize=16,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = REPO_ROOT / "outputs" / "sqr_waveform_spectrum_custom_angles.png"
    fig.savefig(output_path, dpi=300, facecolor="white")

    print(f"Saved figure to: {output_path}")
    print(f"SQR pulse duration = {sqr_duration * 1e9:.1f} ns")
    print(f"Gaussian sigma fraction = {sqr_sigma_fraction:.4f}")
    print(f"Expected tone frequencies (MHz) = {[round(val, 4) for val in peak_freqs_mhz]}")
    print(f"Neighbor spacings (MHz) = {[round(peak_freqs_mhz[i + 1] - peak_freqs_mhz[i], 4) for i in range(len(peak_freqs_mhz) - 1)]}")
    print(f"|chi| / 2pi = {chi_sep_mhz:.4f} MHz")


if __name__ == "__main__":
    main()