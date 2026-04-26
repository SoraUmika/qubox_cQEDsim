# `cqed_sim.optimization`

`cqed_sim.optimization` contains optimization workflows that are not part of the
legacy `cqed_sim.optimal_control` gate-control API.  The first entry point is a
multi-fidelity strong-readout optimizer.

## Strong-Readout Optimizer

`StrongReadoutOptimizer` scores candidate readout pulses through staged models:

- Stage A: linear pointer model for fast seeding.
- Stage B: optional user-supplied multilevel master-equation scorer.
- Stage C: optional user-supplied trajectory-validation scorer.
- Stage D: optional MIST/Floquet penalty scorer.

The objective is

`L = wA*(1-F_assign) + wQ*(1-F_QND_phys) + wL*P_leak + wR*n_res + wE*pulse_energy + wS*slew_penalty + wM*MIST_penalty`.

Assignment fidelity and physical QND fidelity are separate fields in
`ReadoutMetricSet`; the optimizer does not treat SNR alone as final
certification.

## Constraints

`PulseConstraints` supports maximum amplitude, maximum slew rate, bandwidth,
fixed duration, fixed drive frequency, and optional drive-frequency
optimization.  The optimizer returns ranked candidates and a Pareto set.
