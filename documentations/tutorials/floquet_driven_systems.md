# Floquet Driven Systems

The Floquet tutorial set shows how `cqed_sim` treats a periodically driven Hamiltonian as a first-class eigenproblem instead of a static model plus an after-the-fact perturbation argument.

These notebooks are most useful once you already understand the repository's sideband and frame conventions. They focus on two practical questions:

- where a periodic drive produces the strongest avoided crossing in the dressed spectrum
- how to interpret those crossings in terms of integer-order multiphoton resonance conditions

## Included Notebooks

### `50_floquet_driven_systems/01_sideband_quasienergy_scan.ipynb`

This notebook builds a transmon-storage sideband drive, sweeps the drive frequency across the red-sideband condition, and tracks the resulting quasienergy branches.

What it teaches:

- how to build a `FloquetProblem` from a physical model and periodic drive term
- how `run_floquet_sweep(...)` preserves branch identity across a parameter scan
- how the minimum adjacent quasienergy gap can be used as an avoided-crossing diagnostic

### `50_floquet_driven_systems/02_branch_tracking_and_multiphoton_resonances.ipynb`

This notebook solves a single driven problem near a half-frequency qubit condition and then asks which bare energy gaps align with $n\Omega$.

What it teaches:

- how to solve one driven Hamiltonian with `solve_floquet(...)`
- how to inspect folded quasienergies and compare them with bare transition gaps
- how `identify_multiphoton_resonances(...)` reports the most relevant harmonic resonance conditions

## Why This Set Exists

Many cQED control problems are genuinely periodic-drive problems. Static dressed-state intuition is often not enough once a drive is strong enough to reorganize the spectrum. These notebooks provide a compact workflow for studying that regime without dropping down to ad hoc calculations.

They also connect directly to the repository's sideband tooling, so they can be used as a bridge between the standard time-domain workflow tutorials and the more specialized Floquet API.

## Related References

- [Floquet Analysis API](../api/floquet.md)
- [User Guide: Floquet Analysis](../user_guides/floquet_analysis.md)
- [Sideband Swap tutorial](sideband_swap.md)
