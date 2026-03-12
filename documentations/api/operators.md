# API Reference — Operators (`cqed_sim.operators`)

The operators module provides fundamental quantum operators: Pauli matrices, cavity ladder operators, embedding helpers, and state construction utilities.

---

## Basic Operators (`operators.basic`)

| Function | Returns | Description |
|---|---|---|
| `sigma_x()` | `Qobj` | Pauli X |
| `sigma_y()` | `Qobj` | Pauli Y |
| `sigma_z()` | `Qobj` | Pauli Z |
| `tensor_qubit_cavity(op_q, op_c)` | `Qobj` | `qt.tensor(op_q, op_c)` |
| `embed_qubit_op(op_q, n_cav)` | `Qobj` | $\text{op}_q \otimes I_{\text{cav}}$ |
| `embed_cavity_op(op_c, n_tr=2)` | `Qobj` | $I_{\text{qubit}} \otimes \text{op}_c$ |
| `build_qubit_state(label)` | `Qobj` | `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` → ket |
| `joint_basis_state(n_cav_dim, qubit_label, n)` | `Qobj` | \|qubit_label⟩ ⊗ \|n⟩ |
| `as_dm(state)` | `Qobj` | State → density matrix |
| `purity(state)` | `float` | $\text{Tr}(\rho^2)$ |

---

## Cavity Operators (`operators.cavity`)

| Function | Returns | Description |
|---|---|---|
| `destroy_cavity(n_cav_dim)` | `Qobj` | Lowering operator $a$ |
| `create_cavity(n_cav_dim)` | `Qobj` | Raising operator $a^\dagger$ |
| `number_operator(n_cav_dim)` | `Qobj` | $a^\dagger a$ |
| `fock_projector(n_cav_dim, n)` | `Qobj` | $|n\rangle\langle n|$ |
