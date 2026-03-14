# Tutorial: Unitary Synthesis

This tutorial demonstrates how to use `cqed_sim`'s unitary synthesis module to find gate sequences that implement target unitaries within a qubit–cavity subspace.

Unlike the numbered notebook curriculum under `tutorials/`, this page documents an advanced workflow area that still primarily lives in repo-side study material.

---

## Overview

Unitary synthesis optimizes a sequence of primitive gates (Displacement, Rotation, SQR, SNAP) to approximate a desired unitary transformation. The optimization is gradient-free and operates within a defined subspace of the full Hilbert space.

---

## Step 1: Define the Subspace

```python
from cqed_sim.unitary_synthesis import Subspace

# Qubit–cavity block: |g,0⟩, |e,0⟩, |g,1⟩, |e,1⟩, |g,2⟩, |e,2⟩, |g,3⟩, |e,3⟩
sub = Subspace.qubit_cavity_block(n_match=3)
```

### Subspace Types

| Factory | Basis States | Use Case |
|---|---|---|
| `Subspace.qubit_cavity_block(n_match)` | $\|g,0\rangle$, $\|e,0\rangle$, ..., $\|g,n\rangle$, $\|e,n\rangle$ | Full qubit–cavity |
| `Subspace.cavity_only(n_match, qubit="g")` | $\|g,0\rangle$, $\|g,1\rangle$, ..., $\|g,n\rangle$ | Cavity subspace for fixed qubit |
| `Subspace.custom(full_dim, indices)` | User-specified | Arbitrary subspace |

---

## Step 2: Define the Target

```python
from cqed_sim.unitary_synthesis import make_target

target = make_target("easy", n_match=3)
```

Available targets:

| Name | Description |
|---|---|
| `"easy"` | Simple target for testing |
| `"ghz"` | GHZ-like entangling unitary |
| `"cluster"` | Cluster-state unitary |

---

## Step 3: Run the Optimizer

```python
from cqed_sim.unitary_synthesis import UnitarySynthesizer

synth = UnitarySynthesizer(
    subspace=sub,
    backend="ideal",              # "ideal" for ideal gate unitaries
    leakage_weight=0.01,          # Penalize leakage out of subspace
    optimize_times=True,          # Also optimize gate durations
)

result = synth.fit(
    target,
    init_guess="heuristic",
    multistart=4,                 # Multiple random restarts
    maxiter=300,
)
```

---

## Step 4: Inspect Results

```python
print(f"Success: {result.success}")
print(f"Objective: {result.objective:.6f}")
print(f"Total duration: {result.sequence.total_duration():.3e} s")

# Gate sequence
for gate in result.sequence.gates:
    print(f"  {gate}")
```

### Fidelity Metric

The objective is:

$$L = (1 - F_{\text{subspace}}) + \lambda_L \cdot \text{leakage}_{\text{worst}} + \lambda_t \cdot \text{time\_reg}$$

where $F_{\text{subspace}}$ is the phase-invariant subspace unitary fidelity.

---

## Drift Phase Model

The synthesis accounts for dispersive and Kerr drift phases accumulated during gate idle times:

```python
synth = UnitarySynthesizer(
    subspace=sub,
    drift_config={
        "chi": 2 * np.pi * (-2.84e6),
        "kerr": 2 * np.pi * (-2e3),
    },
)
```

The drift convention is aligned with the runtime Hamiltonian: $\chi_{\text{synth}} = \chi_{\text{runtime}}$.

---

## Constraints

### Time Grid Quantization

```python
synth = UnitarySynthesizer(
    ...,
    time_grid={"dt": 4e-9, "mode": "round"},
)
```

### Tone Spacing

```python
synth = UnitarySynthesizer(
    ...,
    constraints={
        "tone_spacing": {"domega_min": 2 * np.pi * 1e6},
    },
)
```

---

## Progress Reporting

```python
from cqed_sim.unitary_synthesis import HistoryReporter, plot_history

synth = UnitarySynthesizer(
    ...,
    progress={"reporter": "history"},
)

result = synth.fit(target)
plot_history(result.history)
```

---

## Existing Example

- `examples/unitary_synthesis_demo.py` — complete synthesis workflow
- `examples/run_snap_optimization_demo.py` — SNAP gate optimization
