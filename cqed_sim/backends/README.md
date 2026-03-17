# `cqed_sim.backends`

The `backends` module provides optional alternative dense-matrix solver backends for `cqed_sim`. The default runtime uses QuTiP; this module adds NumPy and JAX backends for use in small-system parity checks, fast closed-system simulations, and backend-comparison benchmarks.

## Relevance in `cqed_sim`

The QuTiP solver is the primary and well-validated simulation path. The alternative backends in this module exist for:

- validating QuTiP results against an independent dense-matrix propagator,
- enabling differentiable simulation via JAX for gradient-based optimization workflows,
- and providing a faster path for very small systems where QuTiP's overhead dominates.

Backends are selected via `SimulationConfig(backend=...)` in `cqed_sim.sim`.

## Main Capabilities

- **`BaseBackend`**: Abstract interface that all backends implement.
- **`NumPyBackend`**: Dense NumPy matrix-exponentiation solver. Solves the Schrödinger equation via piecewise-constant propagators (`expm(-i H dt)`). Does not support Lindblad/open-system evolution.
- **`JaxBackend`**: Dense JAX-based solver. Same propagator logic as `NumPyBackend`, but uses JAX `jax.scipy.linalg.expm` and supports JIT compilation and autodifferentiation. Requires JAX to be installed; falls back to `None` if unavailable.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `NumPyBackend` | Dense NumPy PWC propagator |
| `JaxBackend` | Dense JAX PWC propagator (optional) |
| `BaseBackend` | Abstract base class for custom backends |

## Usage Guidance

```python
from cqed_sim.backends import NumPyBackend
from cqed_sim.sim import SimulationConfig, simulate_sequence

# Use the NumPy backend instead of QuTiP
result = simulate_sequence(
    model, compiled, psi0, drive_ops,
    config=SimulationConfig(frame=frame, backend=NumPyBackend()),
)
```

For JAX:

```python
from cqed_sim.backends import JaxBackend
from cqed_sim.sim import SimulationConfig

if JaxBackend is not None:
    config = SimulationConfig(frame=frame, backend=JaxBackend())
```

## Important Assumptions / Conventions

- The NumPy and JAX backends implement only closed-system (unitary) evolution via the piecewise-constant propagator approximation. They do not support `NoiseSpec` or Lindblad collapse operators.
- The piecewise-constant approximation is exact when the Hamiltonian is constant within each time step, which holds when the compiled channel timeline's `dt` matches the simulation step.
- JAX backend availability is guarded at import time; `JaxBackend` is `None` if JAX is not installed.
- For large Hilbert spaces (more than ~20 levels), the dense matrix-exponentiation approach is slower than the QuTiP sparse solver.

## Relationships to Other Modules

- **`cqed_sim.sim`**: selects the backend through `SimulationConfig(backend=...)`. The QuTiP path is the default; these backends are opt-in alternatives.
- **`benchmarks/run_performance_benchmarks.py`**: uses both backends to measure parity and performance.

## Limitations / Non-Goals

- Open-system (Lindblad) simulation is not supported by either backend.
- The JAX backend is not validated for all system sizes or all gate types; treat it as an experimental path.
- GPU acceleration through JAX requires a JAX-CUDA installation and is not tested in the standard development environment.
