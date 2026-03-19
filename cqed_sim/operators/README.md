# `cqed_sim.operators` — QuTiP Operator Primitives

## What this module does

`cqed_sim.operators` provides low-level QuTiP operator primitives for building
qubit-cavity Hamiltonians and states. These are thin wrappers around QuTiP
construction calls, with explicit Hilbert-space dimension labeling for two-mode
systems.

## Main functions

### From `operators.basic`

| Function | Description |
|---|---|
| `sigma_x()` | Pauli X operator (2x2) |
| `sigma_y()` | Pauli Y operator (2x2) |
| `sigma_z()` | Pauli Z operator (2x2) |
| `tensor_qubit_cavity(qubit_op, cav_op, n_cav)` | Tensor product of qubit and cavity operators |
| `embed_qubit_op(op, n_cav)` | Embed a qubit operator into the joint qubit-cavity space |
| `embed_cavity_op(op, n_tr)` | Embed a cavity operator into the joint qubit-cavity space |
| `as_dm(state)` | Convert a ket to a density matrix |
| `purity(rho)` | Compute purity Tr(rho^2) |
| `build_qubit_state(theta, phi)` | Build a qubit state on the Bloch sphere |
| `joint_basis_state(q_level, n, n_tr, n_cav)` | Build a joint qubit-cavity basis state |

### From `operators.cavity`

| Function | Description |
|---|---|
| `create_cavity(n_cav)` | Cavity creation operator a† |
| `destroy_cavity(n_cav)` | Cavity annihilation operator a |
| `number_operator(n_cav)` | Cavity number operator a†a |
| `fock_projector(n, n_cav)` | Projector onto the n-th Fock state |

## When to use

Use this module when you need raw QuTiP operators for custom Hamiltonian
construction outside of `UniversalCQEDModel`. For most workflows, prefer
using the model-level accessors such as:

- `model.qubit_lowering()` (or `model.transmon_lowering()`)
- `model.cavity_number()`
- `model.operators()` (returns all operators as a dict)

These model-level helpers automatically apply the correct Hilbert-space
dimensions and tensor structure for the full multi-mode system.

## Relationship to the rest of `cqed_sim`

`cqed_sim.operators` is used internally by `cqed_sim.core`, `cqed_sim.sim`, and
tests. It is rarely needed directly by users working at the model or simulation level.

The tensor-product ordering convention throughout `cqed_sim` is qubit-first:
`|q, n>` = `|q> tensor |n>`. The embedding helpers in this module follow that
same convention.

## Limitations

- These are thin wrappers; no caching, broadcasting, or automatic dimension inference.
- For multi-mode systems (three-mode models), use `UniversalCQEDModel.operators()` directly.
- Prefer `DispersiveTransmonCavityModel` or `UniversalCQEDModel` accessors for any
  production simulation code.
