from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim import FloquetConfig, FloquetProblem, SidebandDriveSpec
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.floquet import build_target_drive_term, run_floquet_sweep


def build_model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.05,
        omega_q=2.0 * np.pi * 6.25,
        alpha=2.0 * np.pi * (-0.25),
        chi=2.0 * np.pi * (-0.015),
        kerr=0.0,
        n_cav=3,
        n_tr=3,
    )


def build_problem(model: DispersiveTransmonCavityModel, drive_frequency: float) -> FloquetProblem:
    sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
    drive = build_target_drive_term(
        model,
        sideband,
        amplitude=2.0 * np.pi * 0.03,
        frequency=drive_frequency,
        waveform="cos",
        label="effective_red_sideband_drive",
    )
    return FloquetProblem(
        model=model,
        periodic_terms=(drive,),
        period=2.0 * np.pi / drive_frequency,
        label="Transmon-cavity sideband Floquet scan",
    )


def main() -> None:
    model = build_model()
    frame = FrameSpec()
    center_frequency = model.sideband_transition_frequency(
        cavity_level=0,
        lower_level=0,
        upper_level=1,
        sideband="red",
        frame=frame,
    )
    drive_frequencies = center_frequency + 2.0 * np.pi * np.linspace(-0.25, 0.25, 61)
    problems = [build_problem(model, frequency) for frequency in drive_frequencies]

    sweep = run_floquet_sweep(
        problems,
        parameter_values=drive_frequencies / (2.0 * np.pi),
        config=FloquetConfig(n_time_samples=128),
    )

    tracked = sweep.tracked_quasienergies / (2.0 * np.pi)
    minimum_gap = np.min(np.abs(tracked[:, 1:] - tracked[:, :-1]), axis=1)

    print(f"Reference red-sideband frequency / (2 pi): {center_frequency / (2.0 * np.pi):.4f}")
    print(f"Minimum tracked quasienergy gap / (2 pi): {np.min(minimum_gap):.6f}")

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)
    for branch in range(tracked.shape[1]):
        axes[0].plot(np.asarray(sweep.parameter_values), tracked[:, branch], linewidth=1.2)
    axes[0].axvline(center_frequency / (2.0 * np.pi), color="black", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("Tracked quasienergy / (2 pi)")
    axes[0].set_title("Transmon-cavity sideband Floquet scan")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(np.asarray(sweep.parameter_values), minimum_gap, color="tab:red", linewidth=1.5)
    axes[1].axvline(center_frequency / (2.0 * np.pi), color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Drive frequency / (2 pi)")
    axes[1].set_ylabel("Minimum adjacent branch gap / (2 pi)")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()