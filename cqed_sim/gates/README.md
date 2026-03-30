# `cqed_sim.gates`

The `gates` module provides a library of ideal unitary gate operators for qubit, transmon, bosonic cavity, coupled qubit-cavity, and two-qubit systems. All gates are returned as QuTiP `Qobj` instances and follow the package-wide conventions documented in `physics_and_conventions/physics_conventions_report.tex`.

## Relevance in `cqed_sim`

Ideal gates serve several roles in `cqed_sim`:

- as targets for unitary synthesis and optimal control (define what needs to be achieved),
- as building blocks for constructing ideal reference evolutions to compare against simulation results,
- and as test operators in calibration, tomography, and validation workflows.

The module is organized into subsystems so a developer can import exactly the gate types relevant to their context.

## Main Capabilities

### Qubit gates (`gates.qubit`)

Standard single-qubit gates for a two-level system:

- Pauli rotations: `rx(theta)`, `ry(theta)`, `rz(theta)`, `rphi(theta, phi)`
- Named gates: `x_gate`, `y_gate`, `z_gate`, `h_gate`, `s_gate`, `s_dag_gate`, `t_gate`, `t_dag_gate`, `identity_gate`

### Transmon gates (`gates.transmon`)

Multilevel transmon transition-selective rotations:

- `r_ge(theta, phi)`: rotation on the `|g>–|e>` transition
- `r_ef(theta, phi)`: rotation on the `|e>–|f>` transition
- `transition_rotation(lower, upper, theta, phi)`: general level-selective rotation

### Bosonic cavity gates (`gates.bosonic`)

- `displacement(alpha, dim)`: coherent displacement `D(alpha)`
- `squeeze(r, phi, dim)`: squeezing operator
- `snap(angles, dim)`: Selective-Number-dependent Arbitrary-Phase (SNAP) gate
- `parity(dim)`: photon-number parity operator `exp(iπ a†a)`
- `kerr_evolution(chi, t, dim)`: Kerr free evolution `exp(iχ t (a†a)²)`
- `oscillator_rotation(phi, dim)`: phase-space rotation `exp(iφ a†a)`

### Coupled qubit-cavity gates (`gates.coupled`)

- `conditional_rotation(angles, dim)`: Fock-sector-conditional qubit rotations
- `conditional_displacement(alpha_list, dim)`: qubit-state-conditional displacement
- `sqr(angles, dim)`: SQR gate (conditional rotation, standard cQED notation)
- `multi_sqr(angles_list, dim)`: multi-tone SQR gate
- `dispersive_phase(chi, t, dim)`: free dispersive evolution
- `controlled_parity(dim)`: controlled-parity operation
- `controlled_snap(angles, dim)`: controlled-SNAP gate
- `beam_splitter(theta, phi, dim_a, dim_b)`: beam-splitter between two bosonic modes
- `blue_sideband(g, t, dim_q, dim_c)`: blue-sideband interaction
- `jaynes_cummings(g, t, dim_q, dim_c)`: Jaynes-Cummings evolution

### Two-qubit gates (`gates.two_qubit`)

- `cnot_gate`, `cz_gate`, `swap_gate`, `iswap_gate`, `sqrt_iswap_gate`
- `controlled_phase(phi)`: controlled-phase gate

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `rx(theta)`, `ry(theta)`, `rz(theta)` | Pauli axis rotations |
| `rphi(theta, phi)` | Rotation about arbitrary equatorial axis |
| `r_ge(theta, phi)`, `r_ef(theta, phi)` | Transmon g-e and e-f rotations |
| `displacement(alpha, dim)` | Cavity displacement |
| `snap(angles, dim)` | SNAP gate |
| `sqr(angles, dim)` | SQR conditional rotation |
| `conditional_rotation(angles, dim)` | Fock-sector-conditional qubit rotation |
| `dispersive_phase(chi, t, dim)` | Dispersive evolution |
| `beam_splitter(theta, phi, ...)` | Two-mode beam splitter |
| `cnot_gate`, `cz_gate`, `swap_gate` | Two-qubit gates |

## Usage Guidance

```python
import numpy as np
from cqed_sim.gates import displacement, snap, sqr, rx

# Cavity displacement
D = displacement(alpha=1.5 + 0.5j, dim=10)

# SNAP gate (selective phases on each Fock sector)
S = snap(angles=[0, np.pi, 0, np.pi, 0, 0, 0, 0], dim=10)

# SQR gate (conditional qubit rotation per Fock sector)
U_sqr = sqr(angles=[0, np.pi/2, 0, -np.pi/2, 0, 0, 0, 0], dim=10)

# Simple qubit rotation
U_x90 = rx(np.pi / 2)
```

For transmon multilevel operations:

```python
from cqed_sim.gates import r_ge, r_ef, transition_rotation

U_ge = r_ge(np.pi, 0)     # π-pulse on g-e
U_ef = r_ef(np.pi, 0)     # π-pulse on e-f
```

## Important Assumptions / Conventions

- All operators are returned as QuTiP `qt.Qobj` instances.
- Qubit basis: `|g> = |0>`, `|e> = |1>`, consistent with `cqed_sim.core`.
- Cavity dimension `dim` must match the truncation of the model being used.
- Two-mode operators (`sqr`, `conditional_rotation`, `beam_splitter`, `jaynes_cummings`, `blue_sideband`) are constructed in the tensor product Hilbert space with qubit first, cavity second.
- Phase conventions for `rphi` and `r_ge`/`r_ef` are consistent with the drive Hamiltonian `H_drive = Ω/2 * (exp(iφ)|e><g| + h.c.)`.
- The `SNAP` gate here produces a diagonal phase operator in the Fock basis; it does not correspond to a physical SNAP pulse but to the ideal target unitary.

## Relationships to Other Modules

- **`cqed_sim.core`**: exports a subset of the same ideal gates (`qubit_rotation_xy`, `displacement_op`, `snap_op`, `sqr_op`) as convenience functions directly on the model. The `gates` module provides a broader and more explicit gate library.
- **`cqed_sim.map_synthesis`**: uses gates from `cqed_sim.core` and this module as synthesis primitives and targets.
- **`cqed_sim.tomo`**: uses ideal rotation gates as reference operations for tomography protocols.
- **`cqed_sim.optimal_control`**: ideal gates from this module are used as `UnitaryObjective` targets.

## Limitations / Non-Goals

- All gates in this module are ideal, noiseless, and analytically defined. They are not simulation results.
- Cavity-dimension-dependent gates (`displacement`, `snap`, `sqr`, etc.) are truncated; ensure `dim` is large enough to avoid truncation error for the amplitudes and evolution times used.
- This module does not implement native multi-qubit gates for more than two qubits.
