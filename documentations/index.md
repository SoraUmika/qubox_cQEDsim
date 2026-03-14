# qubox_cqedsim

**Hardware-faithful time-domain cQED pulse simulator built on QuTiP.**

`cqed_sim` is a Python package for simulating circuit quantum electrodynamics (cQED) systems at the pulse level. It models qubit–storage and qubit–storage–readout systems in the dispersive regime with explicit drive schedules, Lindblad open-system dynamics, and calibration/tomography helpers.

---

## What It Does

- **Pulse-level simulation** of transmon–cavity systems in the dispersive regime
- **Two-mode** (qubit + storage) and **three-mode** (qubit + storage + readout) models
- **Multilevel transmon** support via `UniversalCQEDModel` with arbitrary ancilla dimension
- **Sequence compilation** with realistic hardware distortion (IQ skew, DAC quantization, crosstalk, filtering)
- **Open-system dynamics** with Lindblad collapse operators (T₁, Tφ, cavity decay, thermal photons)
- **Experiment-style workflows** with state preparation, qubit measurement, and readout chain modeling
- **Calibration** of Selective Qubit Rotation (SQR) gates with per-manifold corrections
- **Calibration targets** (spectroscopy, Rabi, Ramsey, T₁, T₂ echo, DRAG tuning)
- **Fock-resolved tomography** and all-XY calibration protocols
- **Unitary synthesis** for gate-sequence optimization in qubit–cavity subspaces
- **Kerr free-evolution** workflows with Wigner-function snapshots

---

## Quick Start

A typical simulation follows this flow:

1. **Define a model** — specify system parameters
2. **Define the frame** — choose the rotating frame
3. **Build pulses** — construct drive waveforms
4. **Compile a sequence** — sample onto a time grid with hardware processing
5. **Run the simulation** — solve the Schrödinger or Lindblad equation
6. **Extract observables** — partial traces, Bloch vectors, photon numbers, Wigner functions
7. **Inspect results** — measurement outcomes, diagnostics

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import square_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

# 1. Define a two-mode dispersive model
model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5e9,
    omega_q=2 * np.pi * 6e9,
    alpha=2 * np.pi * (-200e6),
    chi=2 * np.pi * (-2.84e6),
    kerr=2 * np.pi * (-2e3),
    n_cav=8,
    n_tr=2,
)

# 2. Define the rotating frame
frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)

# 3. Build a qubit drive pulse
pulse = Pulse("q", 0.0, 100e-9, square_envelope, amp=np.pi / 4)

# 4. Compile onto a time grid
compiled = SequenceCompiler(dt=2e-9).compile([pulse], t_end=102e-9)

# 5. Simulate
result = simulate_sequence(
    model, compiled, model.basis_state(0, 0),
    {"q": "qubit"},
    config=SimulationConfig(frame=frame),
)

# 6. Inspect final state
print(result.final_state)
print(result.expectations)
```

---

## Key Abstractions

| Concept | Primary Classes |
|---|---|
| **System model** | `DispersiveTransmonCavityModel`, `DispersiveReadoutTransmonStorageModel`, `UniversalCQEDModel` |
| **Rotating frame** | `FrameSpec` |
| **Drive pulses** | `Pulse`, envelope functions, pulse builders |
| **Sequence compilation** | `SequenceCompiler` → `CompiledSequence` |
| **Simulation** | `simulate_sequence()`, `SimulationConfig`, `SimulationResult` |
| **State preparation** | `StatePreparationSpec`, `prepare_state()` |
| **Measurement** | `QubitMeasurementSpec`, `measure_qubit()` |
| **Readout model** | `ReadoutChain`, `ReadoutResonator`, `PurcellFilter`, `AmplifierChain` |

---

## Dependencies

- **NumPy** ≥ 1.24
- **SciPy** ≥ 1.10
- **QuTiP** ≥ 5.0
- Optional: **JAX** for dense-matrix backend

---

## Navigation

- [Getting Started](getting_started.md) — mental model and first steps
- [Installation](installation.md) — how to set up the package
- [Physics & Conventions](physics_conventions.md) — Hamiltonian, signs, frames, units
- [User Guides](user_guides/defining_models.md) — step-by-step workflow documentation
- [Tutorials](tutorials/index.md) — structured notebook curriculum
- [API Reference](api/overview.md) — full public API documentation
- [Examples](examples.md) — index of example scripts and notebooks
- [Architecture](architecture.md) — package structure and design
