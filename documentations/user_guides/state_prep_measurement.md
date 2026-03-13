# State Preparation & Measurement

This guide covers the reusable state-preparation and qubit-measurement primitives in the library.

---

## State Preparation

Import state-preparation helpers from `cqed_sim.core`:

```python
from cqed_sim.core import (
    StatePreparationSpec,
    amplitude_state,
    coherent_state,
    density_matrix_state,
    fock_state,
    prepare_ground_state,
    prepare_state,
    qubit_level,
    qubit_state,
    vacuum_state,
)
```

Example:

```python
spec = StatePreparationSpec(
    qubit=qubit_state("g"),
    storage=fock_state(3),
)

initial = prepare_state(model, spec)
ground = prepare_ground_state(model)
```

`StatePreparationSpec` follows model tensor ordering automatically:

- two-mode models: qubit, storage
- three-mode models: qubit, storage, readout

---

## Qubit Measurement

Import measurement helpers from `cqed_sim.measurement`:

```python
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

spec = QubitMeasurementSpec(
    shots=1024,
    seed=42,
)

result = measure_qubit(state, spec)
```

The returned `QubitMeasurementResult` exposes:

- `probabilities`
- `observed_probabilities`
- `expectation_z`
- `counts`, `samples`, `iq_samples` when sampling is requested

---

## Readout Chain

The physical readout-chain model also lives in `cqed_sim.measurement`:

```python
import numpy as np

from cqed_sim.measurement import (
    AmplifierChain,
    PurcellFilter,
    QubitMeasurementSpec,
    ReadoutChain,
    ReadoutResonator,
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
```

---

## Workflow Boundary

Protocol-style orchestration is example-side now. For full prepare -> compile -> simulate -> measure recipes, see:

- `examples/protocol_style_simulation.py`
- `examples/kerr_free_evolution.py`
- `examples/sequential_sideband_reset.py`
