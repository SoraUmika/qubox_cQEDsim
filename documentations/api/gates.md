# `cqed_sim.gates` — Gate Library

## Purpose

`cqed_sim.gates` provides ideal gate operators and gate specifications for
qubit-cavity systems. These are the mathematical reference operations (unitaries
and operators) against which pulse-level simulations are benchmarked.

## Philosophy

Gates in `cqed_sim` are ideal (unitary) operations on the joint qubit-cavity
Hilbert space. They are organized into:

- **Qubit gates**: rotations about the Bloch-sphere axes
- **Bosonic gates**: displacement, SNAP (Selective Number-dependent Arbitrary Phase)
- **Entangling gates**: SQR (Selective Qubit Rotation), conditional-phase gates
- **Composite gates**: sequential sideband operations, two-qubit gates

Gate operators are QuTiP `Qobj` instances with explicit Hilbert-space dimension
labeling. They are designed to be compared directly against simulation output
via fidelity, process fidelity, or state overlap.

## Available Gate Families

### Qubit rotations

```python
from cqed_sim.core.ideal_gates import qubit_rotation_xy, qubit_rotation_axis

# Bloch-sphere rotation by theta about axis phi
U_xy = qubit_rotation_xy(theta=np.pi/2, phi=0.0)

# Rotation about an arbitrary 3D axis
U_axis = qubit_rotation_axis(axis=(1.0, 0.0, 0.0), theta=np.pi/2)
```

### Bosonic operations

```python
from cqed_sim.core.ideal_gates import displacement_op, snap_op

# Coherent displacement
D = displacement_op(alpha=1.0+0.5j, n_cav=12)

# SNAP gate (Selective Number-dependent Arbitrary Phase)
S = snap_op(phases=[0.0, np.pi, 0.0], n_cav=12)
```

### Entangling gates

```python
from cqed_sim.core.ideal_gates import sqr_op

# SQR gate: per-Fock qubit rotation
U_sqr = sqr_op(theta=[np.pi, 0, np.pi, 0], phi=[0]*4, n_cav=12, n_tr=2)
```

### Two-qubit gates (in `gates/two_qubit.py`)

```python
from cqed_sim.gates.two_qubit import controlled_z, cnot
```

### Bosonic mode gates (in `gates/bosonic.py`)

```python
from cqed_sim.gates.bosonic import kerr_evolution, self_kerr_phase
```

## Relationship to Pulses and Simulation

Gates are **targets**, not implementations. The pulse-level workflow is:

1. Specify a target gate (from `cqed_sim.core.ideal_gates` or `cqed_sim.gates`)
2. Build pulses (`build_rotation_pulse`, `build_sqr_multitone_pulse`, etc.)
3. Simulate the pulses (`simulate_sequence`)
4. Compare the simulated unitary against the target gate

For systematic gate synthesis, see `cqed_sim.unitary_synthesis`.
For GRAPE optimization toward a target, see `cqed_sim.optimal_control`.

## Key Entry Points

| Function | Module | Purpose |
|----------|--------|---------|
| `qubit_rotation_xy(theta, phi)` | `core.ideal_gates` | Ideal qubit rotation |
| `displacement_op(alpha, n_cav)` | `core.ideal_gates` | Coherent displacement |
| `snap_op(phases, n_cav)` | `core.ideal_gates` | SNAP gate |
| `sqr_op(theta, phi, n_cav, n_tr)` | `core.ideal_gates` | SQR gate |
| `qubit_cavity_block_indices(n_cav, n)` | `core.conventions` | Index mapping |

## See Also

- `cqed_sim.core.ideal_gates` — canonical ideal gate implementations
- `cqed_sim.pulses` — pulse builders for physical gate implementation
- `cqed_sim.unitary_synthesis` — automated gate synthesis
- `cqed_sim.io` — gate I/O (loading gates from external calibration files)
