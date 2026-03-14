# Conventions Quick Reference

Use this checklist before trusting a new notebook result.

## Units

- Internal Hamiltonian frequencies are in `rad/s`.
- Times and pulse durations are in `s`.
- Tutorial plots may relabel axes in `MHz`, `GHz`, `ns`, or `us`, but the runtime inputs stay in SI-style internal units.

## Frames and Carriers

- `FrameSpec` sets the rotating frame used by the Hamiltonian.
- `Pulse.carrier` follows the repository waveform convention `exp(+i (omega t + phase))`.
- Because of that sign convention, `carrier_for_transition_frequency(...)` should be preferred over manually guessing the carrier sign.

## Dispersive Shift

- Runtime `chi` is the per-photon shift of the qubit transition frequency.
- Negative `chi` moves the `|g,n> <-> |e,n>` line to lower frequency as photon number increases.
- Use `manifold_transition_frequency(...)` when you want the notebook and Hamiltonian to share the same convention source.

## Kerr Terms

- Self-Kerr acts on a single bosonic mode and distorts its free evolution.
- Cross-Kerr accumulates conditional phase between subsystems.
- The notebooks label self-Kerr and cross-Kerr separately; they are not interchangeable.

## Truncation

- `n_cav`, `n_storage`, `n_readout`, and `n_tr` are numerical cutoffs.
- A cutoff is only acceptable if the observable of interest has converged for the states and drives you use.
- Tutorial `20_truncation_convergence_checks.ipynb` is the reference pattern for that validation.

## Open-System Parameters

- `T1` is relaxation time.
- `T2*` includes reversible and irreversible dephasing contributions.
- Echo-style coherence can exceed `T2*` when low-frequency dephasing is partially refocused.

For the full conventions report, see `physics_and_conventions/physics_conventions_report.tex`.
