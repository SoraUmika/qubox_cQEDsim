# `cqed_sim.metrics`

`cqed_sim.metrics` keeps readout quality metrics explicit and separate from the
simulators and optimizers.

## Strong-Readout Metrics

`readout_metrics.py` exposes:

- assignment fidelity from a classifier confusion matrix,
- physical QND fidelity from a dressed-projector transition matrix,
- measured two-shot QND fidelity,
- `P(0->1)` and `P(1->0)`,
- leakage probability outside the computational subspace,
- residual resonator and filter photons,
- pulse energy,
- slew penalty.

The assignment fidelity and the physical QND fidelity are intentionally separate.
High SNR is not treated as certification that the qubit state was preserved.
