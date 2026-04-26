# API Reference - Strong Readout

This page documents the multi-fidelity strong-readout stack.  It is designed for
high-power superconducting-qubit readout studies where assignment fidelity,
physical QND fidelity, leakage, residual photons, Purcell/filter effects, and
measurement records must be tracked separately.

---

## Model Hierarchy

| Module | Main symbols | Purpose |
|---|---|---|
| `cqed_sim.models.transmon` | `TransmonCosineSpec`, `TransmonModel`, `diagonalize_transmon` | Full charge-basis cosine transmon and Duffing fallback |
| `cqed_sim.models.multilevel_cqed` | `MultilevelCQEDModel`, `IQPulse`, `ReadoutFrame` | Driven multilevel transmon plus readout resonator |
| `cqed_sim.models.purcell_filter` | `ExplicitPurcellFilterMode`, `FilteredMultilevelCQEDModel` | Explicit filter resonator and filter-output channel |
| `cqed_sim.models.dressed_basis` | `diagonalize_dressed_hamiltonian`, `DressedBasis` | Dressed qubit projectors, transition matrices, leakage |
| `cqed_sim.models.mist_floquet` | `MISTScanConfig`, `scan_mist` | Semiclassical MIST penalty maps |

Tensor ordering is `|q,n_r>` without a filter and `|q,n_r,n_f>` with an
explicit filter.  Frequencies, rates, Hamiltonian coefficients, and pulse
amplitudes are angular frequencies, normally rad/s; times are seconds.

---

## Solvers and Readout

| Module | Main symbols | Purpose |
|---|---|---|
| `cqed_sim.solvers.master_equation` | `solve_master_equation`, `collapse_operators_from_model` | Lindblad replay and time traces |
| `cqed_sim.solvers.trajectories` | `simulate_measurement_trajectories` | Fast homodyne/heterodyne record generation |
| `cqed_sim.readout.input_output` | `output_operator`, `linear_pointer_response` | Input-output field selection and linear pointer baseline |
| `cqed_sim.readout.classifiers` | `MatchedFilterClassifier`, `GaussianMLClassifier` | Classification and `P(predicted | prepared)` confusion matrices |

With no filter, the output field is `sqrt(kappa_r) a`.  With an explicit filter,
the output field is `sqrt(kappa_f) f`; do not add a separate Purcell qubit-decay
collapse operator for the same physical channel.

`collapse_operators_from_model(...)` preserves model-native collapse channels
and appends any supplied dressed-basis decay operators.  Matched-filter
classification uses the equal-noise score
`Re(record dot template*) - 0.5 ||template||^2`, so templates with larger norm
do not win simply because of their energy.

---

## Metrics and Optimization

| Symbol | Description |
|---|---|
| `ReadoutMetricSet` | Assignment, QND, transition, leakage, residual-photon, energy, and slew metrics |
| `compute_readout_metrics(...)` | Builds a `ReadoutMetricSet` from confusion, transition matrix, residual photons, and pulse samples |
| `StrongReadoutOptimizer` | Multi-fidelity optimizer returning ranked candidates and a Pareto set |
| `PulseConstraints` | Max amplitude, max slew, bandwidth, fixed duration, and drive-frequency constraints |
| `square_readout_seed(...)`, `clear_readout_seed(...)` | Square and CLEAR-like sampled pulse seeds |

The optimizer objective is

```text
L = wA*(1-F_assign)
  + wQ*(1-F_QND_phys)
  + wL*P_leak
  + wR*n_res
  + wE*pulse_energy
  + wS*slew_penalty
  + wM*MIST_penalty
```

The linear pointer model is used for seeding and fast ranking.  Final
certification should use the multilevel master-equation and trajectory stages.

---

## Example

Run the script:

```bash
python examples/optimize_strong_readout.py
```

It defines a cosine-transmon/readout/filter model, builds square and CLEAR
seeds, runs a short optimization, reports assignment and QND metrics separately,
prints transition and residual-photon diagnostics, and lists an `Nq`/`Nr`
convergence sweep for the best pulse.

---

## Correctness Verification

The strong-readout stack has three verification layers:

- Fast CI checks in `tests/test_65_strong_readout_stack.py` and
  `tests/test_66_strong_readout_correctness.py` cover transmon diagonalization,
  Hamiltonian dimensions and frames, Lindblad trace/positivity, linear-pointer
  agreement, dressed projectors, input-output channel selection, classifier
  decision boundaries, MIST penalties, trajectories, and optimizer stage use.
- Slow convergence checks in
  `tests/test_67_strong_readout_convergence_slow.py` sweep qubit levels,
  resonator cutoff, filter cutoff, charge-basis cutoff, and timestep for a
  compact strong-readout scenario.  They are marked `slow`.
- Paper-aligned validation in
  `test_against_papers/strong_readout_clear_mist_validation.py` checks CLEAR
  depletion against a passive linear ringdown baseline and verifies that the
  semiclassical MIST penalty is largest near a designed drive-induced
  transition.

Run the checks with:

```bash
py -3.12 -m pytest tests/test_65_strong_readout_stack.py tests/test_66_strong_readout_correctness.py -q
py -3.12 -m pytest tests/test_67_strong_readout_convergence_slow.py -m slow -q
py -3.12 test_against_papers/strong_readout_clear_mist_validation.py
```

The convergence tolerances are metric-level tolerances on assignment fidelity,
physical QND fidelity, `P(0->1)`, `P(1->0)`, leakage, and residual resonator or
filter photons.  These checks are intended to catch truncation and timestep
regressions before a pulse is treated as certified.

---

## References

[1] J. Koch et al., "Charge-insensitive qubit design derived from the Cooper
pair box," Physical Review A 76, 042319 (2007). DOI:
10.1103/PhysRevA.76.042319

[2] D. T. McClure et al., "Rapid Driven Reset of a Qubit Readout Resonator,"
Physical Review Applied 5, 011001 (2016). DOI:
10.1103/PhysRevApplied.5.011001

[3] A. Blais et al., "Circuit quantum electrodynamics," Reviews of Modern
Physics 93, 025005 (2021). DOI: 10.1103/RevModPhys.93.025005

[4] W. Dai et al., "Characterization of drive-induced unwanted state transitions
in superconducting circuits," arXiv:2506.24070 (2025). DOI:
10.48550/arXiv.2506.24070
