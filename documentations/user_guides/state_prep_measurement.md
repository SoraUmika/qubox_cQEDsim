# State Preparation & Measurement

`cqed_sim` provides structured state preparation and qubit measurement abstractions for experiment-style simulation workflows.

---

## State Preparation

### StatePreparationSpec

Define the initial state of each subsystem:

```python
from cqed_sim.experiment import (
    StatePreparationSpec,
    qubit_state,
    fock_state,
    coherent_state,
    vacuum_state,
    qubit_level,
    amplitude_state,
    density_matrix_state,
    prepare_state,
    prepare_ground_state,
)

spec = StatePreparationSpec(
    qubit=qubit_state("g"),       # Qubit in |g⟩
    storage=fock_state(3),        # Storage in |3⟩
)

initial = prepare_state(model, spec)
```

### State Constructor Helpers

| Function | Description | Example |
|---|---|---|
| `qubit_state(label)` | Named qubit state | `"g"`, `"e"`, `"+x"`, `"-x"`, `"+y"`, `"-y"` |
| `qubit_level(level)` | Qubit by Fock level | `qubit_level(2)` for $\|f\rangle$ |
| `vacuum_state()` | Cavity ground state | $\|0\rangle$ |
| `fock_state(n)` | Cavity Fock state | `fock_state(5)` |
| `coherent_state(alpha)` | Coherent state | `coherent_state(2.0+1j)` |
| `amplitude_state(amps)` | Arbitrary from amplitudes | Custom superposition |
| `density_matrix_state(rho)` | From density matrix | Mixed state |

### Three-Mode Preparation

```python
spec = StatePreparationSpec(
    qubit=qubit_state("g"),
    storage=fock_state(0),
    readout=vacuum_state(),     # Include readout for three-mode models
)
```

### Convenience

```python
ground = prepare_ground_state(model)  # |g, 0⟩ or |g, 0, 0⟩
```

---

## Qubit Measurement

### QubitMeasurementSpec

```python
from cqed_sim.experiment import QubitMeasurementSpec, measure_qubit

spec = QubitMeasurementSpec(
    shots=1024,                    # Number of measurement shots (None = exact only)
    confusion_matrix=None,         # 2×2 confusion matrix, (g,e) ordering
    seed=42,                       # RNG seed for reproducibility
)

result = measure_qubit(state, spec)
```

### QubitMeasurementResult

```python
result.probabilities              # {"g": p_g, "e": p_e} — exact latent probabilities
result.observed_probabilities     # After confusion matrix application
result.expectation_z              # p_g_obs − p_e_obs
result.counts                     # {"g": n_g, "e": n_e} — shot counts (if shots > 0)
result.samples                    # ndarray of 0/1 outcomes
```

### Measurement Pipeline

1. Extract reduced qubit state via `reduced_qubit_state()`
2. Compute latent probabilities $(p_g, p_e)$, lumping higher levels
3. Apply confusion matrix: $p_{\text{obs}} = M \cdot p_{\text{latent}}$
4. If readout chain attached: apply backaction (dephasing, Purcell), generate I/Q
5. If shots requested: sample from $p_{\text{obs}}$ (or classify from I/Q)

### Confusion Matrix

Ordering: columns are latent states (g, e), rows are observed states (g, e).

```python
import numpy as np

confusion = np.array([
    [0.98, 0.03],   # P(observe g | latent g), P(observe g | latent e)
    [0.02, 0.97],   # P(observe e | latent g), P(observe e | latent e)
])

spec = QubitMeasurementSpec(shots=1024, confusion_matrix=confusion)
```

---

## Physical Readout Chain

For experiment-realistic readout modeling:

```python
import numpy as np
from cqed_sim.experiment import (
    ReadoutResonator,
    PurcellFilter,
    AmplifierChain,
    ReadoutChain,
    QubitMeasurementSpec,
    measure_qubit,
)

chain = ReadoutChain(
    resonator=ReadoutResonator(
        omega_r=2 * np.pi * 7e9,
        kappa=2 * np.pi * 8e6,
        g=2 * np.pi * 90e6,
        epsilon=2 * np.pi * 0.6e6,
        chi=2 * np.pi * 1.5e6,
    ),
    purcell_filter=PurcellFilter(bandwidth=2 * np.pi * 40e6),
    amplifier=AmplifierChain(noise_temperature=4.0, gain=12.0),
    integration_time=300e-9,
    dt=5e-9,
)

spec = QubitMeasurementSpec(
    shots=1024,
    readout_chain=chain,
    readout_duration=300e-9,
    classify_from_iq=True,
)

result = measure_qubit(state, spec)
result.iq_samples        # (shots, 2) I/Q samples
result.readout_centers   # {"g": [I, Q], "e": [I, Q]} noiseless centers
result.readout_metadata  # Dephasing rates, Purcell T₁, etc.
```

---

## SimulationExperiment Wrapper

Combine preparation, compilation, simulation, and measurement in one object:

```python
from cqed_sim.experiment import SimulationExperiment

experiment = SimulationExperiment(
    model=model,
    pulses=pulses,
    drive_ops={"q": "qubit"},
    dt=2e-9,
    frame=frame,
    state_prep=StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
    measurement=QubitMeasurementSpec(shots=2048, seed=42),
)

result = experiment.run()
result.simulation.final_state    # SimulationResult
result.measurement.probabilities # QubitMeasurementResult
```
