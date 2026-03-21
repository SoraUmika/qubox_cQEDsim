# Tutorial Guide: Holographic Quantum Algorithms

The reusable holographic workflow lives in:

- `cqed_sim.quantum_algorithms.holographic_sim`

Representative example scripts live in:

- `examples/quantum_algorithms/holographic_minimal_correlator.py`
- `examples/quantum_algorithms/holographic_burn_in_translation_invariant.py`
- `examples/quantum_algorithms/holographic_mps_dephasing_example.py`
- `examples/quantum_algorithms/holographic_spin_model_example.py`

---

## Conceptual Model

This package treats a many-body observable estimation problem as repeated action
of a channel on a small bond Hilbert space.

At each step:

1. prepare the physical register in a reference state
2. apply a joint unitary or equivalent Kraus update on `physical ⊗ bond`
3. optionally measure a physical observable
4. update the bond state
5. accumulate correlator weights across the schedule

`burn-in` means running the channel for several steps before starting the
observable insertions so that the bond state approaches a bulk or steady-state
regime.

---

## Minimal Usage

```python
from cqed_sim.quantum_algorithms.holographic_sim import (
    BondNoiseChannel,
    BurnInConfig,
    HolographicChannel,
    HolographicSampler,
    ObservableSchedule,
    pauli_z,
)

channel = HolographicChannel.from_unitary(U, physical_dim=2, bond_dim=4)
noise = BondNoiseChannel.dephasing(bond_dim=channel.bond_dim, probability=0.05)
# Other built-ins are available when dephasing is not the right model:
# BondNoiseChannel.amplitude_damping(...)
# BondNoiseChannel.depolarizing(...)
schedule = ObservableSchedule(
    [
        {"step": 10, "operator": pauli_z()},
        {"step": 14, "operator": pauli_z()},
    ],
    total_steps=20,
)
sampler = HolographicSampler(channel, burn_in=BurnInConfig(steps=50), bond_noise=noise)
estimate = sampler.sample_correlator(schedule, shots=5000)
```

Use `enumerate_correlator(...)` on small examples to cross-check Monte Carlo
estimates exactly.

## MPS Convenience Path

```python
from cqed_sim.quantum_algorithms.holographic_sim import (
    HolographicSampler,
    right_canonical_tensor_to_stinespring_unitary,
)

sampler = HolographicSampler.from_mps_state(psi, site=0)
unitary = right_canonical_tensor_to_stinespring_unitary(sampler.channel.mps_matrices)
```

`from_mps_state(...)` is the shortest route from a normalized dense state tensor
to a reusable holographic transfer channel. The public Stinespring helper
completes the same right-canonical tensor into the dense-unitary form expected
by the legacy `holographicSim.py` finite-sequence path.

---

## Included Workflows

`holographic_minimal_correlator.py`

- smallest ideal example
- compares Monte Carlo and exact branch enumeration

`holographic_burn_in_translation_invariant.py`

- translation-invariant channel
- nontrivial burn-in before bulk estimation

`holographic_mps_dephasing_example.py`

- builds a transfer channel directly from an MPS-compatible dense state
- compares ideal and dephased exact correlators
- demonstrates QuTiP-compatible bond-noise support through `BondNoiseChannel`
- the same API also exposes `BondNoiseChannel.amplitude_damping(...)` and
    `BondNoiseChannel.depolarizing(...)` for relaxation or isotropic mixing studies

`holographic_spin_model_example.py`

- spin-inspired transfer unitary
- demonstrates a model-flavored correlator schedule

---

## Current Limits

- the public sampler API currently assumes a repeated channel rather than a fully general per-step channel list
- optional bond-space noise maps are supported, but full hardware-aware holographic backends are not implemented yet
- `HoloVQEObjective` and `HoloQUADSProgram` are intentionally lightweight scaffolds for future work

Those limits are deliberate: the current implementation focuses on a clean,
generic foundation rather than overcommitting to paper-specific code paths.
