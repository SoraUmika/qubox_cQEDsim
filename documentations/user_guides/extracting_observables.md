# Extracting Observables

After simulation, `cqed_sim` provides a rich set of extractors for analyzing the final state (or intermediate states).

---

## Partial Traces

Extract reduced states for individual subsystems:

```python
from cqed_sim.sim import (
    reduced_qubit_state,
    reduced_cavity_state,
    reduced_storage_state,
    reduced_readout_state,
    reduced_subsystem_state,
)

state = result.final_state

rho_q = reduced_qubit_state(state)       # Qubit reduced density matrix
rho_c = reduced_cavity_state(state)       # Cavity (two-mode only)
rho_s = reduced_storage_state(state)      # Storage subsystem
rho_r = reduced_readout_state(state)      # Readout subsystem

# General: by index or string alias
rho = reduced_subsystem_state(state, "storage")
rho = reduced_subsystem_state(state, 1)   # Index 1
```

---

## Bloch Vector

```python
from cqed_sim.sim import bloch_xyz_from_joint

x, y, z = bloch_xyz_from_joint(state)
```

Computes the Bloch vector from the reduced qubit state. Requires a two-level transmon.

### Fock-Conditioned Bloch Vectors

```python
from cqed_sim.sim import conditioned_bloch_xyz, conditioned_qubit_state

x, y, z, p_n, valid = conditioned_bloch_xyz(state, n=3)
rho_q, p_n, valid = conditioned_qubit_state(state, n=3)
```

Returns the Bloch vector of the qubit conditioned on the cavity being in Fock state $|n\rangle$.

---

## Photon Numbers and Moments

```python
from cqed_sim.sim import (
    cavity_moments,
    storage_moments,
    readout_moments,
    storage_photon_number,
    readout_photon_number,
    mode_moments,
    joint_expectation,
)

moments = cavity_moments(state, n_cav=8)
# {"a": ⟨a⟩, "adag_a": ⟨a†a⟩, "n": ⟨n⟩}

n_s = storage_photon_number(state)
n_r = readout_photon_number(state)
```

---

## Multilevel Population Helpers

```python
from cqed_sim.sim import (
    subsystem_level_population,
    transmon_level_populations,
    compute_shelving_leakage,
)

# Population in a specific level of a subsystem
p_e = subsystem_level_population(state, "transmon", level=1)

# All transmon level populations
pops = transmon_level_populations(state)
# {0: p_g, 1: p_e, 2: p_f, ...}

# Shelving leakage (absolute population change)
leakage = compute_shelving_leakage(
    initial_state, final_state,
    shelved_level=1, subsystem="transmon",
)
```

---

## Qubit-Conditioned Mode Observables

```python
from cqed_sim.sim import (
    qubit_conditioned_subsystem_state,
    qubit_conditioned_mode_moments,
    readout_response_by_qubit_state,
)

# Mode state conditioned on qubit level
rho_s, p, valid = qubit_conditioned_subsystem_state(state, "storage", qubit_level=0)

# Mode moments conditioned on qubit state
moments_g = qubit_conditioned_mode_moments(state, "storage", qubit_level=0)

# Readout responses by qubit state
responses = readout_response_by_qubit_state(state)
# {0: moments_g, 1: moments_e}
```

---

## Wigner Function

```python
from cqed_sim.sim import cavity_wigner

rho_c = reduced_cavity_state(state)
xvec, yvec, W = cavity_wigner(
    rho_c,
    n_points=41,
    extent=4.0,
    coordinate="quadrature",  # or "alpha"/"coherent"
)
```

The `"quadrature"` coordinate uses natural units; `"alpha"` scales by $\sqrt{2}$.

---

## Time-Series Observables

The `result.expectations` dict contains time-series data for tracked observables:

```python
# Access expectation values over time
pe_vs_time = result.expectations["P_e"]

# Custom observables
from cqed_sim.sim import SimulationConfig

result = simulate_sequence(
    model, compiled, initial_state, drive_ops,
    config=config,
    e_ops={"my_op": my_operator},
)
my_op_vs_time = result.expectations["my_op"]
```

If `store_states=True` in the config, the full trajectory is available:

```python
config = SimulationConfig(frame=frame, store_states=True)
result = simulate_sequence(...)

for i, state in enumerate(result.states):
    x, y, z = bloch_xyz_from_joint(state)
```
