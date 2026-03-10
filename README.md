# `cqed_sim`

`cqed_sim` is the reusable cQED simulator library in this repository. It contains the core qubit-cavity Hamiltonian, frame handling, pulse primitives, sequence compilation, solver path, standard ideal gates, observables, calibration helpers, tomography utilities, and unitary-synthesis tools.

Study code, audits, paper reproductions, and workflow-specific helpers are no longer part of the core package. Those now live under `examples/`.

## Package layout

Core library:

- `cqed_sim/core`
  - Hilbert-space conventions, frames, static model, ideal gates, manifold-frequency helpers.
- `cqed_sim/pulses`
  - `Pulse`, standard envelopes, standard pulse builders, calibration formulas, hardware distortion models.
- `cqed_sim/sequence`
  - `SequenceCompiler` and compiled-channel timeline assembly.
- `cqed_sim/sim`
  - Hamiltonian assembly, solver entry points, noise model, extractors.
- `cqed_sim/calibration`
  - Reusable SQR calibration and guarded-benchmark helpers.
- `cqed_sim/observables`, `cqed_sim/operators`, `cqed_sim/tomo`, `cqed_sim/io`, `cqed_sim/plotting`
  - Diagnostics, operator helpers, tomography, gate I/O, plotting.
- `cqed_sim/unitary_synthesis`
  - Subspace-aware synthesis, drift-phase abstractions, reporting, constraints, optimization.

Example-side code:

- `examples/workflows`
  - Sequential workflow helpers and SQR transfer artifacts.
- `examples/audits`
  - Convention-audit utilities.
- `examples/studies`
  - SNAP optimization studies and speed-limit studies.
- `examples/paper_reproductions`
  - PRL-133-specific reproduction code.
- `examples/smoke_tests`
  - Smoke and integration checks for the moved example paths.

## Core conventions

- Tensor ordering is qubit first, cavity second: `|q,n> = |q> tensor |n>`.
- Computational basis is `|g> = |0>`, `|e> = |1>`.
- Internal Hamiltonian and frame frequencies are in `rad/s`; times are in `s`.
- Complex drive envelopes use `exp(+i * (omega * t + phase))`.
- The runtime Hamiltonian uses `n_q = |e><e|` rather than writing the qubit term directly as `sigma_z / 2`.
- Runtime `chi` means the per-photon downward pull of the `|g,n> <-> |e,n>` transition frequency.

The canonical reference for physics conventions and caveats is:

- `physics_and_conventions/physics_conventions_report.tex`

## Common API

### Build a model and frame

```python
import numpy as np

from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec

model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.0e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=2.0 * np.pi * (-200.0e6),
    chi=2.0 * np.pi * (-2.84e6),
    kerr=2.0 * np.pi * (-2.0e3),
    n_cav=8,
    n_tr=2,
)

frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)
```

Useful entry points:

- `DispersiveTransmonCavityModel.static_hamiltonian(frame=...)`
- `DispersiveTransmonCavityModel.basis_state(q_level, cavity_level)`
- `manifold_transition_frequency(model, n, frame=...)`

### Build standard pulses

```python
import numpy as np

from cqed_sim.io import RotationGate
from cqed_sim.pulses import build_rotation_pulse

gate = RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0)
pulses, drive_ops, meta = build_rotation_pulse(
    gate,
    {
        "duration_rotation_s": 100.0e-9,
        "rotation_sigma_fraction": 0.18,
    },
)
```

Common builders:

- `build_rotation_pulse(...)`
- `build_displacement_pulse(...)`
- `build_sqr_multitone_pulse(...)`

If you need a custom pulse family, construct `Pulse(...)` directly.

### Compile a sequence

```python
from cqed_sim.sequence import SequenceCompiler

compiled = SequenceCompiler(dt=2.0e-9).compile(
    pulses,
    t_end=max(pulse.t1 for pulse in pulses) + 2.0e-9,
)
```

`SequenceCompiler` is the normal entry point for:

- overlap summation on a global time grid,
- timing quantization,
- IF/LO and IQ distortion,
- crosstalk mixing,
- optional compile caching.

### Run a simulation

```python
from cqed_sim.sim import SimulationConfig, simulate_sequence

result = simulate_sequence(
    model,
    compiled,
    model.basis_state(0, 0),
    drive_ops,
    config=SimulationConfig(frame=frame, max_step=2.0e-9),
)
```

Useful runtime entry points:

- `simulate_sequence(...)`
- `hamiltonian_time_slices(...)`
- `SimulationConfig(...)`
- `NoiseSpec(...)`

### Extract common observables

```python
from cqed_sim.sim import reduced_qubit_state, reduced_cavity_state, conditioned_bloch_xyz

rho_q = reduced_qubit_state(result.final_state)
rho_c = reduced_cavity_state(result.final_state)
bloch_n0 = conditioned_bloch_xyz(result.final_state, n=0)
```

Also see:

- `bloch_xyz_from_joint(...)`
- `cavity_moments(...)`
- `cavity_wigner(...)`
- `cqed_sim.observables.*` for higher-level phase, trajectory, and weakness diagnostics

### Use standard ideal gates

```python
import numpy as np

from cqed_sim.core import displacement_op, qubit_rotation_xy, sqr_op

u_rot = qubit_rotation_xy(np.pi / 2.0, 0.0)
u_disp = displacement_op(8, 0.3 + 0.1j)
u_sqr = sqr_op(
    theta=[np.pi / 2.0, 0.0, 0.0],
    phi=[0.0, 0.0, 0.0],
)
```

### Calibration and tomography helpers

Reusable calibration entry points:

- `calibrate_sqr_gate(...)`
- `load_or_calibrate_sqr_gate(...)`
- `calibrate_guarded_sqr_target(...)`
- `benchmark_random_sqr_targets_vs_duration(...)`

Reusable tomography entry points:

- `run_all_xy(...)`
- `autocalibrate_all_xy(...)`
- `selective_pi_pulse(...)`
- `run_fock_resolved_tomo(...)`
- `calibrate_leakage_matrix(...)`

### Unitary synthesis

Main synthesis entry points:

- `Subspace.qubit_cavity_block(...)`
- `make_target(...)`
- `UnitarySynthesizer(...).fit(...)`

Important caveat: `cqed_sim/unitary_synthesis` has its own drift-phase abstraction. Its `sigma_z` sign and `chi` normalization are not identical to the runtime Hamiltonian. Use the conventions report before translating parameters between the runtime and synthesis layers.

## Normal usage patterns

Typical library use is:

1. Construct `DispersiveTransmonCavityModel` and `FrameSpec`.
2. Create `Pulse` objects directly or use the standard builders in `cqed_sim.pulses`.
3. Compile the pulses with `SequenceCompiler`.
4. Run `simulate_sequence`.
5. Extract reduced states, Bloch vectors, or phase diagnostics with `cqed_sim.sim` and `cqed_sim.observables`.

For gate-sequence JSON workflows, use:

- `cqed_sim.io.load_gate_sequence(...)`
- the pulse builders in `cqed_sim.pulses`
- the example-side sequential workflow helpers in `examples/workflows/sequential`

## Examples

Example entry points are intentionally outside the core library:

- `examples/workflows/sequential`
  - Case-style sequential workflows and gate-by-gate trajectories.
- `examples/workflows/sqr_transfer.py`
  - SQR transfer artifact construction and replay.
- `examples/audits/experiment_convention_audit.py`
  - Convention-audit utilities and sign scans.
- `examples/studies/snap_opt`
  - SNAP optimization study code.
- `examples/studies/sqr_speedlimit_multitone_gaussian.py`
  - SQR speed-limit study code.
- `examples/paper_reproductions/snap_prl133`
  - PRL-133 reproduction code.

These are example or study APIs, not part of the canonical `cqed_sim` library surface.

## Tests

Core reusable-library suite:

```bash
pytest tests -q
```

Example-side verification:

```bash
pytest examples/audits/tests \
  examples/workflows/tests \
  examples/studies/tests \
  examples/paper_reproductions/tests \
  examples/smoke_tests/tests -q
```

The core suite is the canonical validation path for the reusable library. The example-side suites validate the moved workflow, study, audit, and reproduction code separately.
