# API Reference — Backends (`cqed_sim.backends`)

Optional dense piecewise-constant solver backends for small systems and parity checks. These backends propagate the state by computing matrix exponentials for each time step, which is accurate for small systems but slower than QuTiP's adaptive ODE solvers for larger Hilbert spaces.

!!! tip "When to use"
    Use `NumPyBackend` or `JaxBackend` for small systems (dim ≤ ~50) where you want a piecewise-constant propagation path independent of QuTiP, or for cross-checking simulation results. For production simulations on larger systems, use the default QuTiP path (`config.backend = None`).

---

## BaseBackend (ABC)

Abstract interface that all dense backends must implement:

| Method | Description |
|---|---|
| `asarray(value)` | Convert to backend array type |
| `to_numpy(value)` | Convert back to NumPy array |
| `eye(dim)` | Identity matrix |
| `zeros(shape)` | Zero array |
| `reshape(value, shape)` | Reshape array |
| `dagger(value)` | Conjugate transpose |
| `matmul(left, right)` | Matrix multiplication |
| `kron(values)` | Tensor product of sequence of matrices |
| `expm(value)` | Matrix exponential |
| `trace(value)` | Matrix trace |
| `expectation(operator, state)` | $\text{Tr}(\hat{O}\rho)$ |
| `lindbladian(hamiltonian, collapse_ops)` | Full Lindbladian superoperator (dim² × dim²) |

---

## NumPyBackend

```python
class NumPyBackend(BaseBackend):
    name = "numpy"
```

All operations use `np.complex128`. Matrix exponential via `scipy.linalg.expm`. Lindbladian constructs full superoperator for density-matrix propagation.

---

## JaxBackend

```python
class JaxBackend(BaseBackend):
    name = "jax"
    def __init__(self, device: str | None = None)
```

Requires JAX. Uses `jnp.complex128` with 64-bit precision enabled. Optional device targeting (`"cpu"`, `"gpu"`, `"tpu"`). Matrix exponential via `jax.scipy.linalg.expm`.

**Import guard:** `JaxBackend` is `None` if JAX is not installed.

---

## Usage

```python
from cqed_sim.backends import NumPyBackend
from cqed_sim.sim import SimulationConfig, simulate_sequence

# Use NumPy dense backend for a parity-check run
config = SimulationConfig(backend=NumPyBackend())
result = simulate_sequence(model, compiled, psi0, drive_ops, config=config)
```

```python
# JAX backend on GPU (requires JAX installation)
from cqed_sim.backends import JaxBackend

if JaxBackend is not None:
    config = SimulationConfig(backend=JaxBackend(device="gpu"))
```

!!! warning "Memory scaling"
    Dense backends store all matrices as full arrays — memory scales as dim² for states and dim⁴ for Liouvillians. For Hilbert space dimensions above ~50, prefer the default QuTiP path.

---

## Comparison

| Feature | Default (QuTiP) | NumPyBackend | JaxBackend |
|---|---|---|---|
| Sparse matrices | ✓ | ✗ | ✗ |
| Adaptive ODE solver | ✓ | ✗ | ✗ |
| GPU support | ✗ | ✗ | ✓ |
| Max practical dim | ~1000+ | ~50 | ~50 |
| Primary use | Production | Parity check | Parity check / GPU |

---

## Usage

```python
from cqed_sim.sim import SimulationConfig
from cqed_sim.backends import NumPyBackend

config = SimulationConfig(backend=NumPyBackend())
```

!!! note
    The dense backends implement a piecewise-constant solver intended for small-system checks and backend parity validation. They are not drop-in replacements for QuTiP's adaptive ODE solver on large systems.
