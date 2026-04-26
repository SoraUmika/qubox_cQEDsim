# `cqed_sim.solvers`

`cqed_sim.solvers` contains solver wrappers used by the strong-readout stack.
It complements the existing `cqed_sim.sim` runtime rather than replacing it.

## Entry Points

- `solve_master_equation(...)`: QuTiP-backed Lindblad master-equation solve for
  `HamiltonianData`, `Qobj`, or `QobjEvo` Hamiltonians.
- `MasterEquationConfig`: solver tolerances, `max_step`, `nsteps`, state storage,
  and additional QuTiP `solver_options`.
- `build_qutip_solver_options(...)`: shared option merger used by package
  wrappers before calling native QuTiP solvers.
- `collapse_operators_from_model(...)`: helper for resonator damping, explicit
  filter damping, transmon relaxation, transmon dephasing, and dressed-basis
  collapse operators.
- `simulate_measurement_trajectories(...)`: lightweight homodyne/heterodyne
  record generator using deterministic Lindblad means plus Gaussian measurement
  noise.

## Assumptions

Hamiltonian coefficients and collapse rates are angular frequencies, normally
rad/s.  Time grids are in seconds.  The master-equation convention is

`drho/dt = -i[H(t), rho] + sum_k D[L_k] rho`.

The trajectory helper is meant for fast validation and optimizer scoring.  For
full stochastic backaction, use the existing
`cqed_sim.measurement.simulate_continuous_readout(...)` SME wrapper.

`solver_options` is the escape hatch for QuTiP options not promoted to explicit
config fields. Explicit fields such as `nsteps` cannot conflict with the same
key inside `solver_options`; such duplicates raise `ValueError`.
