# Installation

## Requirements

- **Python** ≥ 3.10
- **NumPy** ≥ 1.24
- **SciPy** ≥ 1.10
- **QuTiP** ≥ 5.0

Optional:

- **JAX** — for the dense-matrix backend (`JaxBackend`)
- **matplotlib** — for plotting functions
- **pytest** ≥ 8.0 — for running tests

---

## Development Install

The package is currently used as a development install from the repository. Clone or obtain the repository, then install in editable mode:

```bash
pip install -e .
```

This installs `cqed_sim` as a package that can be imported from anywhere on the system, while changes to the source code take effect immediately.

### From the Repository Root

```bash
cd path/to/cQED_simulation
pip install -e .
```

### Verify the Installation

```python
import cqed_sim
print(cqed_sim.__name__)  # "cqed_sim"
```

---

## Dependencies

Core dependencies are declared in `pyproject.toml` and will be installed automatically:

```toml
dependencies = [
  "numpy>=1.24",
  "scipy>=1.10",
  "qutip>=5.0",
]
```

### Optional: JAX Backend

To use the dense JAX backend for parity checks on small systems:

```bash
pip install jax jaxlib
```

Then configure:

```python
from cqed_sim.backends import JaxBackend
from cqed_sim.sim import SimulationConfig

config = SimulationConfig(backend=JaxBackend())
```

!!! note
    The JAX and NumPy backends are piecewise-constant solvers intended for small systems and backend parity validation. They are not replacements for QuTiP's adaptive ODE solver on large systems.

---

## Running Tests

Tests live under the `tests/` directory and use pytest:

```bash
pytest tests/
```

For development tests:

```bash
pip install -e ".[dev]"
pytest
```

---

## Package Not Yet on PyPI

`cqed_sim` is not currently published on PyPI. Installation is from the local repository only. The package name in `pyproject.toml` is `cqed-sim` with version `0.1.0`.
