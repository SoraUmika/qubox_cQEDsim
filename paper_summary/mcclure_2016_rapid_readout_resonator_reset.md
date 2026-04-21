[1] D. T. McClure, H. Paik, L. S. Bishop, M. Steffen, J. M. Chow, and J. M. Gambetta, "Rapid Driven Reset of a Qubit Readout Resonator," Physical Review Applied 5, 011001 (2016). DOI: 10.1103/PhysRevApplied.5.011001

## Relevance to `cqed_sim`

This paper is the canonical experimental reference for piecewise-constant active cavity depletion in superconducting-qubit dispersive readout. It motivates the Phase 1 `cqed_sim.optimal_control.readout_emptying` implementation, especially the idea of using a small number of constant-amplitude pulse segments to speed ring-up and ring-down compared with a naive square readout tone.

## Key contribution

- Introduces the CLEAR-style pulse concept: a readout pulse augmented with extra constant-amplitude segments chosen to move the resonator quickly between desired populations while keeping residual photons small after the pulse.
- Demonstrates that a low-power linear cavity model predicts useful segment amplitudes before nonlinear effects become important.

## Equations / model notes

- The paper models the readout resonator as a driven damped harmonic oscillator in the linear regime.
- For `cqed_sim`, the relevant abstraction is the exact finite-duration contribution of each constant segment to the terminal cavity amplitude.
- The new `readout_emptying.py` implementation uses the exact piecewise-constant segment integral rather than a short-segment approximation.

## Convention notes

- The repository implementation follows the existing `cqed_sim.measurement.readout_chain` convention
  `dot(alpha) = -(kappa/2 + i Delta) alpha - i epsilon(t)`,
  where `Delta = omega_resonator - omega_drive`.
- The paper’s cavity-reset idea is compatible with this convention after straightforward sign translation of the drive and detuning definitions.

## Practical takeaway

- This is the right citation for the direct multi-segment readout-emptying idea.
- It justifies keeping the first implementation analytic and piecewise constant before adding hardware-aware or nonlinear refinements.
