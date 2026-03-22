from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim import FloquetConfig, FloquetProblem, PeriodicDriveTerm
from cqed_sim.core import TransmonModeSpec, UniversalCQEDModel
from cqed_sim.floquet import identify_multiphoton_resonances, run_floquet_sweep


def build_model() -> UniversalCQEDModel:
    return UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=2.0 * np.pi * 5.2,
            dim=4,
            alpha=2.0 * np.pi * (-0.22),
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(),
    )


def build_problem(model: UniversalCQEDModel, drive_frequency: float, drive_amplitude: float) -> FloquetProblem:
    drive = PeriodicDriveTerm(
        target="qubit",
        amplitude=drive_amplitude,
        frequency=drive_frequency,
        waveform="cos",
        label="continuous_qubit_drive",
    )
    return FloquetProblem(
        model=model,
        periodic_terms=(drive,),
        period=2.0 * np.pi / drive_frequency,
        label="Driven multilevel transmon",
    )


def main() -> None:
    model = build_model()
    drive_amplitude = 2.0 * np.pi * 0.08
    drive_frequencies = 2.0 * np.pi * np.linspace(4.6, 5.8, 61)
    problems = [build_problem(model, frequency, drive_amplitude) for frequency in drive_frequencies]

    sweep = run_floquet_sweep(
        problems,
        parameter_values=drive_frequencies / (2.0 * np.pi),
        config=FloquetConfig(n_time_samples=128),
    )

    midpoint = sweep.results[len(sweep.results) // 2]
    resonances = identify_multiphoton_resonances(midpoint, max_photon_order=3, tolerance=2.0 * np.pi * 0.05)
    print("Top multiphoton candidates near the midpoint drive frequency:")
    for candidate in resonances[:5]:
        print(
            f"  states ({candidate.lower_state}, {candidate.upper_state}), "
            f"order={candidate.photon_order}, detuning={candidate.detuning / (2.0 * np.pi):+.4f}"
        )

    print("Dominant bare-state overlaps at the midpoint drive frequency:")
    for branch_index, quasienergy in enumerate(midpoint.quasienergies[: min(4, len(midpoint.quasienergies))]):
        bare_index = int(midpoint.dominant_bare_state_indices[branch_index])
        overlap = float(midpoint.bare_state_overlaps[branch_index, bare_index])
        print(
            f"  branch {branch_index}: quasi={quasienergy / (2.0 * np.pi):+.4f}, "
            f"dominant bare state={bare_index}, overlap={overlap:.4f}"
        )

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    tracked = sweep.tracked_quasienergies / (2.0 * np.pi)
    for branch in range(tracked.shape[1]):
        ax.plot(np.asarray(sweep.parameter_values), tracked[:, branch], linewidth=1.4)
    ax.set_xlabel("Drive frequency / (2 pi)")
    ax.set_ylabel("Tracked quasienergy / (2 pi)")
    ax.set_title("Driven transmon Floquet quasienergy sweep")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()