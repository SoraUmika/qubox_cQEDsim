# `cqed_sim.measurement`

The `measurement` module provides reusable qubit measurement primitives and a physics-based readout chain model for `cqed_sim`. It handles both lightweight exact-probability measurement and richer experiment-style readout modeling including I/Q cluster generation, measurement-induced dephasing, and Purcell-limited T1 estimation.

## Relevance in `cqed_sim`

After running a simulation with `cqed_sim.sim`, the typical next step is extracting a measurement outcome. This module provides:

- a clean interface for computing or sampling qubit measurement results from a simulation state,
- an optional physical readout chain that models the resonator response, amplifier noise, and classification,
- and estimates of back-action effects such as measurement-induced dephasing and Purcell decay.

## Main Capabilities

### Qubit measurement

- **`measure_qubit(state, spec)`**: The main entry point. Computes qubit probabilities from `state`, optionally applies a confusion matrix, and optionally samples shot outcomes. Returns a `QubitMeasurementResult`.
- **`QubitMeasurementSpec`**: Configuration dataclass. Controls number of shots, random seed, confusion matrix, and optionally attaches a `ReadoutChain` for physical readout modeling.
- **`QubitMeasurementResult`**: Holds exact probabilities, sampled outcomes (if shots requested), and readout-chain diagnostics (if a chain is attached).

### Physical readout chain

- **`ReadoutResonator`**: Models the readout resonator with frequency `omega_r`, linewidth `kappa`, coupling `g`, drive amplitude `epsilon`, and dispersive shift `chi`.
- **`PurcellFilter`**: Optional Purcell filter with a given bandwidth.
- **`AmplifierChain`**: Amplifier model parameterized by noise temperature and gain.
- **`ReadoutChain`**: Composes resonator, Purcell filter, and amplifier into a full readout model. When attached to `QubitMeasurementSpec`, `measure_qubit(...)` can:
  - generate state-conditioned resonator I/Q trajectories and clusters,
  - report measurement-induced dephasing rates,
  - estimate Purcell-limited `T1`,
  - optionally apply readout-induced dephasing and Purcell relaxation before sampling.
- **`ReadoutTrace`**: I/Q time trace output from the readout chain.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `measure_qubit(state, spec)` | Main measurement entry point |
| `QubitMeasurementSpec` | Measurement configuration |
| `QubitMeasurementResult` | Measurement result (probabilities, outcomes, diagnostics) |
| `ReadoutChain` | Physical readout model (resonator + filter + amplifier) |
| `ReadoutResonator` | Readout resonator parameters and response |
| `PurcellFilter` | Purcell filter model |
| `AmplifierChain` | Amplifier noise and gain model |
| `ReadoutTrace` | I/Q trace from the chain |

## Usage Guidance

### Lightweight measurement (exact probabilities)

```python
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

result = measure_qubit(
    final_state,
    QubitMeasurementSpec(shots=1024, seed=42),
)
print(result.p_g, result.p_e)
print(result.outcomes)  # sampled binary outcomes
```

### Measurement with confusion matrix

```python
import numpy as np

spec = QubitMeasurementSpec(
    shots=1024,
    seed=42,
    confusion_matrix=np.array([[0.98, 0.02], [0.05, 0.95]]),
)
result = measure_qubit(final_state, spec)
```

### Physical readout chain

```python
import numpy as np
from cqed_sim.measurement import (
    AmplifierChain, PurcellFilter, QubitMeasurementSpec,
    ReadoutChain, ReadoutResonator, measure_qubit,
)

chain = ReadoutChain(
    resonator=ReadoutResonator(
        omega_r=2*np.pi*7.0e9,
        kappa=2*np.pi*8.0e6,
        g=2*np.pi*90.0e6,
        epsilon=2*np.pi*0.6e6,
        chi=2*np.pi*1.5e6,
    ),
    purcell_filter=PurcellFilter(bandwidth=2*np.pi*40.0e6),
    amplifier=AmplifierChain(noise_temperature=4.0, gain=12.0),
    integration_time=300.0e-9,
    dt=5.0e-9,
)

spec = QubitMeasurementSpec(
    shots=1024,
    seed=42,
    readout_chain=chain,
    readout_duration=300.0e-9,
    classify_from_iq=True,
)
result = measure_qubit(final_state, spec)
```

## Important Assumptions / Conventions

- `measure_qubit(...)` operates on the qubit subsystem; it traces out the cavity/storage degrees of freedom before computing probabilities.
- Probabilities are exact (computed from the reduced qubit density matrix) unless shots are requested.
- The confusion matrix, when provided, should be a 2×2 matrix with `(g, e)` ordering: `p_observed = M @ p_latent`.
- The readout chain model is a semi-classical dispersive readout model. It models the homodyne response of the resonator driven at a fixed frequency; it is not a full circuit-QED input-output simulation.
- Units: all frequencies in `rad/s`, times in `s`, consistent with the rest of `cqed_sim`.

## Relationships to Other Modules

- **`cqed_sim.sim`**: provides the simulation state passed to `measure_qubit(...)`.
- **`cqed_sim.core`**: the initial state used in simulation is typically prepared with `prepare_state(...)` from `core`.
- **`cqed_sim.tomo`**: tomography routines use `measure_qubit(...)` internally for simulated readout.

## Limitations / Non-Goals

- Does not model dispersive readout photon number back-action during the pulse sequence itself (that would require including the readout drive in the simulation Hamiltonian).
- Purcell estimates are analytic first-order approximations, not derived from full simulation of the resonator + qubit + filter system.
- The readout chain model is not validated against hardware beyond the parameter ranges in the tutorials.
