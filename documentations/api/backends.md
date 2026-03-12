# API Reference — Backends (`cqed_sim.backends`)

Optional dense piecewise-constant solver backends for small systems and parity checks.

---

## BaseBackend (ABC)

Abstract interface with methods: `asarray`, `to_numpy`, `eye`, `zeros`, `reshape`, `dagger`, `matmul`, `kron`, `expm`, `trace`, `expectation`, `lindbladian`.

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
from cqed_sim.sim import SimulationConfig
from cqed_sim.backends import NumPyBackend

config = SimulationConfig(backend=NumPyBackend())
```

!!! note
    The dense backends implement a piecewise-constant solver intended for small-system checks and backend parity validation. They are not drop-in replacements for QuTiP's adaptive ODE solver on large systems.
