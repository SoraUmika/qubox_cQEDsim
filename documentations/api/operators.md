# API Reference — Operators (`cqed_sim.operators`)

The operators module provides fundamental quantum operators: Pauli matrices, cavity ladder operators, embedding helpers, and state construction utilities. All operators are cached via `lru_cache` for efficient reuse.

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
| `joint_basis_state(n_cav_dim, qubit_label, n)` | `Qobj` | \|qubit\_label⟩ ⊗ \|n⟩ |
| `as_dm(state)` | `Qobj` | State → density matrix (no-op if already `oper`) |
| `purity(state)` | `float` | $\text{Tr}(\rho^2)$ |

### Usage

```python
from cqed_sim.operators import (
    sigma_z, embed_qubit_op, embed_cavity_op,
    build_qubit_state, joint_basis_state, as_dm, purity,
)

# Build the |g, 3⟩ joint state in a 12-dim cavity
psi = joint_basis_state(n_cav_dim=12, qubit_label="g", n=3)

# Embed σ_z into the full qubit ⊗ cavity space
sz_full = embed_qubit_op(sigma_z(), n_cav=12)

# Compute expectation value
print(psi.dag() * sz_full * psi)  # 1.0 (ground state)

# Build a superposition state and check purity
plus_x = build_qubit_state("+x")
rho = as_dm(plus_x)
print(purity(rho))  # 1.0 (pure state)
```

---

## Cavity Operators (`operators.cavity`)

| Function | Returns | Description |
|---|---|---|
| `destroy_cavity(n_cav_dim)` | `Qobj` | Lowering operator $a$ |
| `create_cavity(n_cav_dim)` | `Qobj` | Raising operator $a^\dagger$ |
| `number_operator(n_cav_dim)` | `Qobj` | $\hat{n} = a^\dagger a$ |
| `fock_projector(n_cav_dim, n)` | `Qobj` | $\|n\rangle\langle n\|$ projector |

### Usage

```python
from cqed_sim.operators import destroy_cavity, number_operator, fock_projector

n_cav = 12
a = destroy_cavity(n_cav)
n_hat = number_operator(n_cav)

# Verify commutation relation [a, a†] = 1 (truncated)
commutator = a * a.dag() - a.dag() * a
print(commutator.diag()[:5])  # [1, 1, 1, 1, 1] (up to truncation)

# Project onto the |3⟩ Fock state
P3 = fock_projector(n_cav, 3)
```

---

## Embedding Convention

All operators in `cqed_sim` follow the **qubit ⊗ cavity** tensor ordering:

$$
\hat{O}_{\text{full}} = \hat{O}_{\text{qubit}} \otimes I_{\text{cavity}}
$$

Use `embed_qubit_op` and `embed_cavity_op` to construct full-space operators from subsystem operators. The `n_cav` / `n_tr` parameters must match the model's truncation dimensions.
| `fock_projector(n_cav_dim, n)` | `Qobj` | $|n\rangle\langle n|$ |
