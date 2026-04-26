"""Paper-aligned validation for strong-readout depletion and MIST penalties.

Papers checked:
[1] D. T. McClure, H. Paik, L. S. Bishop, M. Steffen, J. M. Chow, and
    J. M. Gambetta, "Rapid Driven Reset of a Qubit Readout Resonator,"
    Physical Review Applied 5, 011001 (2016). DOI: 10.1103/PhysRevApplied.5.011001
[2] W. Dai, S. Hazra, D. K. Weiss, P. D. Kurilovich, T. Connolly,
    H. K. Babla, S. Singh, V. R. Joshi, A. Z. Ding, P. D. Parakh,
    J. Venkatraman, X. Xiao, L. Frunzio, and M. H. Devoret,
    "Spectroscopy of drive-induced unwanted state transitions in superconducting circuits,"
    arXiv:2506.24070 (2025). DOI: 10.48550/arXiv.2506.24070

What this script validates:
- A CLEAR-like kick/plateau/depletion pulse leaves fewer linear-regime residual
  photons than a square measurement segment followed by passive ring-down.
- The semiclassical MIST scanner assigns a larger penalty to a designed
  multiphoton resonance than to a nearby off-resonant control and scales the
  risk upward with drive amplitude.

Assumptions:
- This is a compact regression-style validation, not a reproduction of the full
  experimental calibrations in either paper.
- Frequencies and amplitudes are angular-frequency-like internal units; only
  relative residual photon number and relative MIST penalty are asserted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from cqed_sim.models import MISTScanConfig, TransmonCosineSpec, diagonalize_transmon, scan_mist
from cqed_sim.pulses.clear import clear_readout_seed
from cqed_sim.readout.input_output import linear_pointer_response


@dataclass(frozen=True)
class ValidationResult:
    name: str
    passed: bool
    details: dict[str, float]


def validate_clear_linear_depletion() -> ValidationResult:
    dt = 0.005
    kappa = 1.4
    amplitude = 0.35
    total_duration = 1.0
    measurement_fraction = 0.65
    n_total = int(round(total_duration / dt))
    n_measure = int(round(total_duration * measurement_fraction / dt))
    passive = np.concatenate(
        [
            np.full(n_measure, amplitude, dtype=np.complex128),
            np.zeros(n_total - n_measure, dtype=np.complex128),
        ]
    )
    clear = clear_readout_seed(
        amplitude=amplitude,
        duration=total_duration,
        dt=dt,
        drive_frequency=0.0,
        kick_fraction=0.12,
        depletion_fraction=1.0 - measurement_fraction,
        kick_amplitude=0.62,
        depletion_amplitude=-0.44,
    )
    _t, alpha_passive = linear_pointer_response(passive, dt=dt, kappa=kappa)
    _t, alpha_clear = linear_pointer_response(clear.samples, dt=dt, kappa=kappa)
    passive_residual = float(abs(alpha_passive[-1]) ** 2)
    clear_residual = float(abs(alpha_clear[-1]) ** 2)
    return ValidationResult(
        name="CLEAR linear depletion",
        passed=clear_residual < passive_residual,
        details={
            "passive_residual_photons": passive_residual,
            "clear_residual_photons": clear_residual,
            "residual_ratio": clear_residual / max(passive_residual, 1.0e-300),
        },
    )


def validate_mist_resonance_penalty() -> ValidationResult:
    spec = TransmonCosineSpec(EJ=35.0, EC=0.60, ng=0.0, n_cut=9, levels=5)
    spectrum = diagonalize_transmon(spec)
    candidates = []
    for initial in (0, 1):
        for target in range(2, spec.levels):
            candidates.append((abs(spectrum.n_matrix[target, initial]), initial, target))
    _matrix_element, initial_level, target_level = max(candidates, key=lambda item: item[0])
    resonant_frequency = float(abs(spectrum.shifted_energies[target_level] - spectrum.shifted_energies[initial_level]))
    off_frequency = resonant_frequency + 1.0
    scan = scan_mist(
        MISTScanConfig(
            EJ=spec.EJ,
            EC=spec.EC,
            n_cut=spec.n_cut,
            levels=spec.levels,
            drive_amplitudes=[0.02, 0.20],
            drive_frequencies=[resonant_frequency, off_frequency],
            max_multiphoton_order=1,
            detuning_width=0.08,
        )
    )
    low_resonant = scan.penalty(0.02, resonant_frequency)
    high_resonant = scan.penalty(0.20, resonant_frequency)
    high_off = scan.penalty(0.20, off_frequency)
    passed = high_resonant > 50.0 * low_resonant and high_resonant > 10.0 * high_off
    return ValidationResult(
        name="MIST multiphoton penalty",
        passed=passed,
        details={
            "resonant_frequency": resonant_frequency,
            "initial_level": float(initial_level),
            "target_level": float(target_level),
            "low_amplitude_resonant_penalty": low_resonant,
            "high_amplitude_resonant_penalty": high_resonant,
            "high_amplitude_off_resonant_penalty": high_off,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="Print detailed validation values.")
    args = parser.parse_args()

    results = [validate_clear_linear_depletion(), validate_mist_resonance_penalty()]
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {result.name}")
        if args.verbose:
            for key, value in result.details.items():
                print(f"  {key}: {value:.12g}")
    if not all(result.passed for result in results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
