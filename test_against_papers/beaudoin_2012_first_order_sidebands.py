"""Literature validation for first-order red-sideband resonance in circuit QED.

Papers checked:
[1] F. Beaudoin, M. P. da Silva, Z. Dutton, and A. Blais,
    "First-order sidebands in circuit QED using qubit frequency modulation,"
    Physical Review A 86, 022305 (2012). DOI: 10.1103/PhysRevA.86.022305
[2] J. D. Strand, M. Ware, F. Beaudoin, T. A. Ohki, B. R. Johnson,
    A. Blais, and B. L. T. Plourde,
    "First-order sideband transitions with flux-driven asymmetric transmon qubits,"
    Physical Review B 87, 220505(R) (2013). DOI: 10.1103/PhysRevB.87.220505

Target result:
- First-order red-sideband resonance occurs when the modulation frequency matches
  the qubit-cavity sideband transition frequency.

What this script validates:
- The Floquet solver reproduces maximal hybridization of the effective red-sideband
  pair at the predicted resonance frequency in the repository's sideband-drive
  abstraction.

Assumptions and approximations:
- This is an effective-Hamiltonian validation using cqed_sim's structured
  `SidebandDriveSpec(...)` path, not a full microscopic flux-modulation model.
- Closed-system Floquet analysis only.
- The resonance marker is maximal mixing between the bare red-sideband pair.
"""

from __future__ import annotations

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
    sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
    drive_frequencies = center_frequency + 2.0 * np.pi * np.linspace(-0.12, 0.12, 13)

    problems = []
    for frequency in drive_frequencies:
        drive = build_target_drive_term(
            model,
            sideband,
            amplitude=2.0 * np.pi * 0.03,
            frequency=frequency,
            waveform="cos",
            label="effective_red_sideband_drive",
        )
        problems.append(FloquetProblem(model=model, periodic_terms=(drive,), period=2.0 * np.pi / frequency))

    sweep = run_floquet_sweep(
        problems,
        parameter_values=drive_frequencies / (2.0 * np.pi),
        config=FloquetConfig(n_time_samples=96, overlap_reference_time=0.17),
    )

    hybridization_scores = []
    for result in sweep.results:
        red_sideband_mixing = np.max(
            np.minimum(result.bare_state_overlaps[:, 1], result.bare_state_overlaps[:, 2])
        )
        hybridization_scores.append(float(red_sideband_mixing))

    best_index = int(np.argmax(hybridization_scores))
    best_frequency = float(drive_frequencies[best_index] / (2.0 * np.pi))
    target_frequency = float(center_frequency / (2.0 * np.pi))
    detuning = best_frequency - target_frequency

    print("Beaudoin 2012 / Strand 2013 red-sideband resonance validation")
    print(f"predicted red-sideband frequency / (2 pi): {target_frequency:.6f}")
    print(f"maximum hybridization frequency / (2 pi): {best_frequency:.6f}")
    print(f"detuning at maximum hybridization: {detuning:+.6f}")
    print("hybridization scores:")
    for frequency, score in zip(drive_frequencies / (2.0 * np.pi), hybridization_scores):
        print(f"  {float(frequency):.6f}: {float(score):.6f}")

    if abs(detuning) > 0.02:
        raise SystemExit("Validation failed: sideband hybridization peak moved too far from the predicted resonance.")
    if hybridization_scores[best_index] <= 0.45:
        raise SystemExit("Validation failed: sideband hybridization at resonance is weaker than expected.")


if __name__ == "__main__":
    main()