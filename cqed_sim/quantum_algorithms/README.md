# `cqed_sim.quantum_algorithms`

The `quantum_algorithms` module is the top-level namespace for quantum algorithm implementations built on top of the `cqed_sim` physics stack. It currently contains one submodule: `holographic_sim`.

## Submodules

### `holographic_sim`

Implements the holographic quantum algorithm architecture described in `paper_summary/holographic_quantum_algorithms.pdf`. It provides a general, reusable framework for:

- applying quantum channels iteratively to a bond register (holographic / MPS-inspired simulation),
- Monte Carlo and exact branch enumeration for correlator estimation,
- holoVQE energy decomposition,
- and holoQUADS time-sliced dynamics.

See [holographic_sim/README.md](holographic_sim/README.md) for the full documentation.

## Public API

All public symbols from `holographic_sim` are re-exported at the `quantum_algorithms` level:

```python
from cqed_sim.quantum_algorithms import (
    HolographicChannel,
    HolographicSampler,
    HolographicMPSAlgorithm,
    HoloVQEObjective,
    HoloQUADSProgram,
    MatrixProductState,
    ObservableSchedule,
    ...
)
```

## When to use this module

Use `cqed_sim.quantum_algorithms` when implementing or studying quantum algorithms that require:

- an MPS-inspired holographic simulation of a many-body system,
- iterative channel application on a bond register with physical-register measurements,
- variational energy estimation on a quantum processor via the holoVQE method,
- or QUADS-style Trotterized time evolution via the holoQUADS circuit program.

## References

- `paper_summary/holographic_quantum_algorithms.pdf`
- `holographic_sim/README.md`
