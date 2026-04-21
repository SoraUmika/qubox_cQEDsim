[1] M. Jerger, F. Motzoi, Y. Gao, C. Dickel, L. Buchmann, A. Bengtsson, G. Tancredi, C. W. Warren, J. Bylander, D. DiVincenzo, R. Barends, and P. A. Bushev, "Dispersive Qubit Readout with Intrinsic Resonator Reset," arXiv (2024). DOI: 10.48550/arXiv.2406.04891

## Relevance to `cqed_sim`

This paper is the closest match to the new `cqed_sim` readout-emptying feature. It gives a compact analytical construction for state-dependent resonator-reset pulses and describes how to extend that construction into the nonlinear regime with Kerr-aware correction.

## Key contribution

- Formulates a universal analytic pulse-construction method that returns the cavity to its initial state at the end of the readout pulse.
- Generalizes the construction to multiple states and modes.
- Shows an iterative Kerr-aware correction strategy based on replaying the linear trajectory, estimating the nonlinear phase accumulation, and compensating that phase in the drive.

## Equations / model notes

- The paper presents the readout response through linear response theory and then builds the reset pulse by inverting the transfer function for a chosen smooth trial response.
- In the nonlinear section, the paper uses a mean-field Kerr correction and iterates the pulse design using the cavity occupations from the linear solution.
- The `cqed_sim` implementation uses the same overall idea, but starts from the exact terminal null-space construction for piecewise-constant segments and then applies a Kerr-aware phase correction on top of that segmented waveform.

## Convention notes

- The repository’s readout model again uses `Delta = omega_resonator - omega_drive` and `-i epsilon(t)` in the cavity equation of motion.
- The implemented Kerr replay uses the repo’s positive-K Hamiltonian sign convention, so the corrective drive phase must oppose the nonlinear cavity phase accumulated during replay.

## Practical takeaway

- This is the main citation for the Kerr-aware correction stage and for the claim that analytic reset pulses can remain useful beyond the strictly linear regime.
- It also supports the choice to keep the feature as a reusable analytic constructor plus replay/evaluation helpers, rather than embedding the first implementation directly into an iterative optimizer.
