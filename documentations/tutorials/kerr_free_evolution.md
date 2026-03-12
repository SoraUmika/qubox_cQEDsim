# Tutorial: Kerr Free Evolution

This tutorial demonstrates cavity free evolution under Kerr nonlinearity — a fundamental diagnostic for verifying the self-Kerr sign and magnitude in the simulator.

---

## Physics Background

A cavity initialized in a coherent state $|\alpha\rangle$ evolves under the Kerr Hamiltonian:

$$H_K = \frac{K}{2} \, n_c(n_c - 1)$$

The Kerr nonlinearity causes photon-number-dependent phase evolution, leading to characteristic features in the Wigner function over time. At specific revival times, the state exhibits quantum interference patterns.

---

## Using the Built-in Workflow

`cqed_sim` provides a dedicated Kerr free-evolution workflow:

```python
import numpy as np
from cqed_sim.experiment import (
    run_kerr_free_evolution,
    available_kerr_parameter_sets,
)

# See available preset parameter sets
print(available_kerr_parameter_sets())
# ("phase_evolution", "value_2")

# Run free evolution with Wigner snapshots
times_us = np.linspace(0, 50, 6)  # Snapshot times in microseconds
result = run_kerr_free_evolution(
    times_us * 1e-6,              # Convert to seconds
    parameter_set="phase_evolution",
    n_cav=28,
    n_tr=3,
    use_rotating_frame=True,
    wigner_times_s=times_us * 1e-6,
)
```

---

## Inspecting Results

```python
# Access snapshots
for snap in result.snapshots:
    print(f"t = {snap.time_us:.1f} µs, ⟨n⟩ = {snap.cavity_photon_number:.3f}")
    print(f"  ⟨a⟩ = {snap.cavity_mean:.4f}")

# Model and frame used
print(result.model)
print(result.frame)
```

### Wigner Snapshots

```python
from cqed_sim.experiment import plot_kerr_wigner_snapshots

fig = plot_kerr_wigner_snapshots(result)
```

---

## Custom Initial States

```python
from cqed_sim.experiment import (
    run_kerr_free_evolution,
    StatePreparationSpec,
    qubit_state,
    coherent_state,
)

result = run_kerr_free_evolution(
    times_us * 1e-6,
    parameter_set="phase_evolution",
    state_prep=StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=coherent_state(3.0),  # Larger coherent state
    ),
    n_cav=40,
)
```

---

## Kerr Sign Verification

Verify that the documented Kerr sign matches the expected physical behavior:

```python
from cqed_sim.experiment import verify_kerr_sign

verification = verify_kerr_sign()
print(f"Documented Kerr: {verification.documented_kerr_hz:.1f} Hz")
print(f"Flipped Kerr:    {verification.flipped_kerr_hz:.1f} Hz")
print(f"Matches documented sign: {verification.matches_documented_sign}")
```

This compares the phase evolution under the documented Kerr value against a flipped-sign control run.

---

## Predefined Parameter Sets

The `KerrParameterSet` dataclass stores device parameters in Hz:

```python
from cqed_sim.experiment import KERR_FREE_EVOLUTION_PARAMETER_SETS

params = KERR_FREE_EVOLUTION_PARAMETER_SETS["phase_evolution"]
print(f"Kerr = {params.kerr_hz} Hz")
print(f"Chi  = {params.chi_hz} Hz")
```

Each parameter set has a `build_model()` method that converts Hz → rad/s and constructs a `DispersiveTransmonCavityModel`.

---

## Existing Examples

- `examples/kerr_free_evolution.py` — full Kerr free-evolution workflow
- `examples/kerr_sign_verification.py` — sign verification diagnostic
