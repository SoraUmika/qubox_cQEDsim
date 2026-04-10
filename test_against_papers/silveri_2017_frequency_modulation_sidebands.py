"""Literature validation for Floquet transition sidebands under sinusoidal frequency modulation.

Paper checked:
M. P. Silveri, J. A. Tuorila, E. V. Thuneberg, and G. S. Paraoanu,
"Quantum systems under frequency modulation," Reports on Progress in Physics 80,
056002 (2017). DOI: 10.1088/1361-6633/aa5170

Target result:
- Section 3.2, Eq. (45): for sinusoidal longitudinal modulation, the harmonic
  sideband amplitudes are weighted by Bessel functions J_m(xi / Omega).

What this script validates:
- A closed two-level Floquet problem with longitudinal frequency modulation of the
  excited-state projector.
- Harmonic-resolved transition strengths computed by
  cqed_sim.floquet.compute_floquet_transition_strengths(...).
- Agreement between the observed transition strengths and |J_m(xi / Omega)|^2.

Assumptions and approximations:
- Closed-system Floquet analysis only; no dissipation.
- Two-level truncation.
- Longitudinal modulation implemented through the transmon number operator.
- Probe operator taken to be the transverse x-quadrature (raising + lowering).
- Bare qubit transition chosen inside the first Floquet zone so the harmonic
    labels align directly with the sideband index in the analytical formula.
"""

from __future__ import annotations

import numpy as np
from scipy.special import jv

from cqed_sim import FloquetConfig, FloquetProblem, TransmonModeSpec, UniversalCQEDModel, solve_floquet
from cqed_sim.floquet import build_transmon_frequency_modulation_term, compute_floquet_transition_strengths


def build_model(omega_q: float = 0.4) -> UniversalCQEDModel:
    return UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=omega_q,
            dim=2,
            alpha=0.0,
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(),
    )


def evaluate_case(modulation_amplitude: float, *, harmonic_cutoff: int = 3) -> list[tuple[int, float, float, float]]:
    modulation_angular_frequency = 1.0
    model = build_model()
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            build_transmon_frequency_modulation_term(
                model,
                amplitude=modulation_amplitude,
                frequency=modulation_angular_frequency,
                waveform="cos",
            ),
        ),
        period=2.0 * np.pi / modulation_angular_frequency,
    )
    result = solve_floquet(problem, FloquetConfig(n_time_samples=256))
    probe_operator = model.transmon_lowering() + model.transmon_raising()
    strengths = compute_floquet_transition_strengths(
        result,
        probe_operator,
        harmonic_cutoff=harmonic_cutoff,
        n_time_samples=768,
        min_strength=0.0,
    )

    ground_index = int(np.flatnonzero(result.dominant_bare_state_indices == 0)[0])
    excited_index = int(np.flatnonzero(result.dominant_bare_state_indices == 1)[0])
    observed = {
        entry.harmonic: entry.strength
        for entry in strengths
        if entry.initial_mode == ground_index and entry.final_mode == excited_index
    }

    rows: list[tuple[int, float, float, float]] = []
    for harmonic in range(-harmonic_cutoff, harmonic_cutoff + 1):
        expected = float(abs(jv(harmonic, modulation_amplitude / modulation_angular_frequency)) ** 2)
        measured = float(observed[harmonic])
        rows.append((harmonic, measured, expected, abs(measured - expected)))
    return rows


def main() -> None:
    amplitudes = (0.3, 0.7, 1.1)
    harmonic_cutoff = 3
    max_abs_error = 0.0

    print("Silveri 2017 Eq. (45) sideband validation")
    print("Comparing Floquet transition strengths against |J_m(xi / Omega)|^2")
    for amplitude in amplitudes:
        rows = evaluate_case(amplitude, harmonic_cutoff=harmonic_cutoff)
        print(f"\nmodulation amplitude = {amplitude:.3f}")
        print("harmonic  measured        expected        abs_error")
        for harmonic, measured, expected, abs_error in rows:
            max_abs_error = max(max_abs_error, abs_error)
            print(f"{harmonic:>8d}  {measured:>12.6f}  {expected:>12.6f}  {abs_error:>10.3e}")

    print(f"\nmaximum absolute error: {max_abs_error:.3e}")
    if max_abs_error > 2.0e-2:
        raise SystemExit("Validation failed: sideband-strength mismatch exceeded tolerance.")


if __name__ == "__main__":
    main()