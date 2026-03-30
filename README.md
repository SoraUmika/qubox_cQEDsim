# `cqed_sim`

`cqed_sim` is the reusable cQED simulator library in this repository. It is intended to cover low-level Hamiltonian and pulse simulation, plus reusable preparation and measurement primitives, for:

- qubit/transmon + storage-cavity systems,
- storage + transmon + readout-resonator systems,
- pulse-level schedules compiled onto explicit drive channels,
- periodically driven closed-system Floquet analysis,
- deterministic open-system evolution,
- lightweight state-preparation and measurement wrappers.

## Installation

Install from the repository root, which is the standard location for the package metadata:

```bash
cd path/to/cQED_simulation
pip install -e .
```

For a non-editable install:

```bash
pip install .
```

The install metadata lives in `pyproject.toml` at the repository root. Do not create a second setup file inside `cqed_sim/`; that directory is the import package, not the build/install entry point.

The wheel also bundles `physics_and_conventions`, which is used by parts of the public `cqed_sim` API at runtime.

Structured tutorial notebooks live under `tutorials/`. Study code, audits, paper reproductions, and workflow-specific helpers remain outside the core package under `examples/`.

## Package layout

Core library:

- `cqed_sim/core`
  - Hilbert-space conventions, frames, the universal subsystem-based cQED model, thin two-mode and three-mode compatibility wrappers, ideal gates, manifold-frequency helpers.
- `cqed_sim/pulses`
  - `Pulse`, standard envelopes, standard pulse builders, calibration formulas, hardware distortion models.
- `cqed_sim/sequence`
  - `SequenceCompiler` and compiled-channel timeline assembly.
- `cqed_sim/sim`
  - Hamiltonian assembly, solver entry points, noise model, extractors, readout-conditioned response helpers.
- `cqed_sim/floquet`
  - Periodic-drive Floquet analysis, quasienergies, one-period propagators, harmonic-space Sambe builders, resonance helpers, and branch-tracking utilities for driven cQED models.
- `cqed_sim/measurement`
  - Reusable qubit measurement primitives and readout-chain modeling.
- `cqed_sim/analysis`, `cqed_sim/calibration_targets`, `cqed_sim/backends`
  - Parameter translation, calibration-target surrogates, and optional dense NumPy/JAX backend support.
- `cqed_sim/calibration`, `cqed_sim/observables`, `cqed_sim/operators`, `cqed_sim/tomo`, `cqed_sim/io`, `cqed_sim/plotting`, `cqed_sim/map_synthesis`, `cqed_sim/unitary_synthesis`, `cqed_sim/optimal_control`
  - Reusable calibration, diagnostics, tomography, gate I/O, plotting, map-synthesis, and direct optimal-control helpers that remain part of the library surface, including reduced conditioned-qubit multitone reachability checks, full targeted-subspace multitone audits for dispersive SQR-style studies, model-backed GRAPE optimization, and structured hardware-aware pulse-family optimization with pulse export back into the standard runtime stack. `cqed_sim.map_synthesis` is now the preferred synthesis namespace, while `cqed_sim.unitary_synthesis` remains as a backward-compatible compatibility facade during the transition.
- `cqed_sim/rl_control`, `cqed_sim/system_id`
  - RL-facing hybrid control environments, action/observation/reward abstractions, benchmark tasks, domain randomization, diagnostics, and calibration-informed prior hooks for future fit-then-randomize workflows.

Calibration-side multitone helpers now cover two complementary validation modes for dispersive SQR-style studies:

- reduced conditioned-qubit validation via `cqed_sim.calibration.conditioned_multitone`
- full logical-subspace validation and optimization via `cqed_sim.calibration.targeted_subspace_multitone`

The reduced path asks whether each conditioned qubit state can be reached. The targeted-subspace path asks the stronger question of whether the same common waveform implements the intended coherent operator on a chosen joint qubit-cavity logical subspace while preserving cavity-block structure and suppressing leakage.

Example-side code:

- `tutorials`
  - Numbered notebook curriculum for guided learning, API onboarding, and physically annotated walkthroughs.
- `agent_workflow`
  - Repo-side semi-autonomous task orchestration, prompt templates, backend profiles, validation fixtures, and resumable run artifacts.
- `examples/workflows`
  - Standalone workflow helpers and specialized repo-side orchestration that is not part of the numbered tutorial curriculum.
- `examples/audits`
  - Convention-audit utilities.
- `examples/studies`
  - SNAP and related study code.
- `examples/paper_reproductions`
  - Paper-specific reproduction code.
- `examples/smoke_tests`
  - Smoke and integration checks for the moved example paths.
- `test_against_papers`
  - Notebook-style literature checks and paper-alignment diagnostics that are retained outside the reusable package and outside the automated `examples/*` workflows.

## Core conventions

- Two-mode tensor ordering is qubit first, storage second: `|q,n> = |q> tensor |n>`.
- Three-mode tensor ordering is qubit first, storage second, readout third: `|q,n_s,n_r>`.
- Computational basis is `|g> = |0>`, `|e> = |1>`.
- The library is unit-coherent: it does not enforce specific physical units for frequencies or times. Any internally consistent unit system is valid (for example, rad/s with times in seconds, or rad/ns with times in nanoseconds). The recommended convention used in the main examples and calibration function naming is rad/s and seconds.
- Low-level complex drive envelopes use `exp(+i * (omega * t + phase))`.
- The low-level compatibility field `Pulse.carrier` is therefore the negative of the rotating-frame transition frequency it addresses.
- User-facing positive drive-tone frequencies should be translated through `drive_frequency_for_transition_frequency(...)`, `transition_frequency_from_drive_frequency(...)`, `internal_carrier_from_drive_frequency(...)`, and `drive_frequency_from_internal_carrier(...)` rather than by reasoning directly about the raw carrier sign.
- Runtime dispersive terms use the excitation projector `n_q = b^\dagger b`; for a two-level qubit, `n_q = |e><e| = (I - sigma_z) / 2`.
- Runtime `chi` means the per-photon shift of the `|g,n> <-> |e,n>` transition frequency.
- Negative `chi` lowers the qubit transition frequency with photon number; positive `chi` raises it.
- Measurement helpers return exact probabilities by default and sampled outcomes only when `shots` is requested.

The canonical source of truth for physics conventions, caveats, and verified test coverage is:

- `physics_and_conventions/physics_conventions_report.tex`

## Common API

`UniversalCQEDModel` is the generalized model-layer entry point. The existing
`DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel`
remain supported as convenience wrappers around that shared core.

### Build a two-mode model and frame

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
- `carrier_for_transition_frequency(...)`, `transition_frequency_from_carrier(...)`

### Build a three-mode storage-transmon-readout model

```python
import numpy as np

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec

model = DispersiveReadoutTransmonStorageModel(
    omega_s=2.0 * np.pi * 5.0e9,
    omega_r=2.0 * np.pi * 7.5e9,
    omega_q=2.0 * np.pi * 6.0e9,
    alpha=2.0 * np.pi * (-220.0e6),
    chi_s=2.0 * np.pi * (-2.8e6),
    chi_r=2.0 * np.pi * (-1.2e6),
    chi_sr=2.0 * np.pi * 15.0e3,
    kerr_s=2.0 * np.pi * (-2.0e3),
    kerr_r=2.0 * np.pi * (-30.0e3),
    n_storage=10,
    n_readout=12,
    n_tr=2,
)

frame = FrameSpec(
    omega_c_frame=model.omega_s,
    omega_q_frame=model.omega_q,
    omega_r_frame=model.omega_r,
)
```

Useful entry points:

- `DispersiveReadoutTransmonStorageModel.static_hamiltonian(frame=...)`
- `DispersiveReadoutTransmonStorageModel.basis_state(q_level, storage_level, readout_level)`
- `DispersiveReadoutTransmonStorageModel.qubit_transition_frequency(...)`
- `DispersiveReadoutTransmonStorageModel.storage_transition_frequency(...)`
- `DispersiveReadoutTransmonStorageModel.readout_transition_frequency(...)`

### Build a generalized multilevel model directly

```python
import numpy as np

from cqed_sim.core import (
    BosonicModeSpec,
    DispersiveCouplingSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
)

model = UniversalCQEDModel(
    transmon=TransmonModeSpec(
        omega=2.0 * np.pi * 6.0e9,
        dim=5,
        alpha=2.0 * np.pi * (-200.0e6),
        label="qubit",
        aliases=("qubit", "transmon"),
        frame_channel="q",
    ),
    bosonic_modes=(
        BosonicModeSpec(
            label="storage",
            omega=2.0 * np.pi * 5.0e9,
            dim=12,
            kerr=2.0 * np.pi * (-2.0e3),
            aliases=("storage", "cavity"),
            frame_channel="c",
        ),
        BosonicModeSpec(
            label="readout",
            omega=2.0 * np.pi * 7.5e9,
            dim=10,
            aliases=("readout",),
            frame_channel="r",
        ),
    ),
    dispersive_couplings=(
        DispersiveCouplingSpec(mode="storage", chi=2.0 * np.pi * (-2.8e6)),
        DispersiveCouplingSpec(mode="readout", chi=2.0 * np.pi * (-1.1e6)),
    ),
)
```

Useful entry points:

- `UniversalCQEDModel.hamiltonian(frame=...)`
- `UniversalCQEDModel.energy_spectrum(frame=..., levels=...)`
- `UniversalCQEDModel.basis_state(...)`
- `UniversalCQEDModel.transmon_lowering()`
- `UniversalCQEDModel.mode_annihilation("storage")`
- `UniversalCQEDModel.transmon_transition_frequency(...)`
- `UniversalCQEDModel.mode_transition_frequency(...)`

### Inspect dressed energy levels

```python
import numpy as np

from cqed_sim import FrameSpec, compute_energy_spectrum
from cqed_sim.plotting import plot_energy_levels

lab_spectrum = compute_energy_spectrum(model, frame=FrameSpec(), levels=12)
fig = plot_energy_levels(
    lab_spectrum,
    max_levels=12,
    energy_scale=1.0 / (2.0 * np.pi * 1.0e6),
    energy_unit_label="MHz",
)
```

The returned energies are always shifted so the vacuum basis state has energy `0`.
For physically interpretable ladder plots, prefer `frame=FrameSpec()` unless you
explicitly want rotating-frame energies.

### Analyze a periodically driven Hamiltonian

```python
import numpy as np

from cqed_sim import FloquetConfig, FloquetProblem, PeriodicDriveTerm
from cqed_sim.core import DispersiveTransmonCavityModel
from cqed_sim.floquet import solve_floquet

model = DispersiveTransmonCavityModel(
  omega_c=2.0 * np.pi * 5.0,
  omega_q=2.0 * np.pi * 6.0,
  alpha=2.0 * np.pi * (-0.22),
  chi=2.0 * np.pi * (-0.015),
  n_cav=4,
  n_tr=4,
)

drive = PeriodicDriveTerm(
  target="qubit",
  amplitude=2.0 * np.pi * 0.08,
  frequency=2.0 * np.pi * 6.0,
  waveform="cos",
)

problem = FloquetProblem(
  model=model,
  periodic_terms=[drive],
  period=2.0 * np.pi / drive.frequency,
)

result = solve_floquet(problem, FloquetConfig(n_time_samples=128))
print(result.quasienergies)
```

Use the Floquet path when the drive is genuinely periodic and you want quasienergies, dressed-state structure, or resonance analysis instead of a finite-duration trajectory.

### Build pulses

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
- `build_sideband_pulse(...)`
- `build_sqr_multitone_pulse(...)`

If you need a custom drive family, construct `Pulse(...)` directly and map its channel to a target such as `"qubit"`, `"storage"`, `"cavity"`, `"readout"`, or `"sideband"`. For multilevel ancilla control, `drive_ops` can also use structured targets such as `TransmonTransitionDriveSpec(lower_level=0, upper_level=2)` and `SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)`.

### Compile and run low-level simulations

```python
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

compiled = SequenceCompiler(dt=2.0e-9).compile(
    pulses,
    t_end=max(pulse.t1 for pulse in pulses) + 2.0e-9,
)

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
- `transmon_transition_frequency(...)`
- `sideband_transition_frequency(...)`
- `effective_sideband_rabi_frequency(...)`

For multilevel ancilla decay, `NoiseSpec(transmon_t1=(T1_ge, T1_fe, ...))` builds explicit ladder collapse operators `|j-1><j|` instead of a single aggregate lowering operator. The extractor path also now includes `subsystem_level_population(...)`, `transmon_level_populations(...)`, and `compute_shelving_leakage(...)` for shelving-style sideband benchmarks.

### State preparation and measurement

```python
from cqed_sim.core import StatePreparationSpec, fock_state, prepare_state, qubit_state
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

initial = prepare_state(
    model,
    StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
)

measurement = measure_qubit(
    initial,
    QubitMeasurementSpec(shots=1024, seed=123),
)
```

Reusable preparation helpers:

- `prepare_state(...)`
- `prepare_ground_state(...)`
- `qubit_state(...)`, `qubit_level(...)`
- `fock_state(...)`, `coherent_state(...)`
- `amplitude_state(...)`, `density_matrix_state(...)`

Measurement helpers:

- `measure_qubit(...)`
- `QubitMeasurementSpec(...)`

`measure_qubit(...)` remains lightweight by default. It computes exact qubit probabilities from the runtime state, optionally applies a confusion matrix via `p_observed = M @ p_latent` with `(g, e)` ordering, and optionally samples repeated-shot outcomes.

For experiment-style readout, it can now also use a physical readout chain:

- `ReadoutResonator(...)`
- `PurcellFilter(...)`
- `AmplifierChain(...)`
- `ReadoutChain(...)`

For continuous-measurement replay and high-power readout studies, the same measurement layer also exposes:

- `ContinuousReadoutSpec(...)`
- `simulate_continuous_readout(...)`
- `StrongReadoutMixingSpec(...)`
- `build_strong_readout_disturbance(...)`

When a `ReadoutChain` is attached to `QubitMeasurementSpec`, `measure_qubit(...)` can:

- generate state-conditioned resonator responses and I/Q clusters,
- report measurement-induced dephasing rates,
- estimate Purcell-limited `T1`,
- optionally apply readout-induced dephasing and Purcell relaxation before sampling.

Example:

```python
import numpy as np

from cqed_sim.measurement import AmplifierChain, PurcellFilter, QubitMeasurementSpec, ReadoutChain, ReadoutResonator, measure_qubit

readout_chain = ReadoutChain(
    resonator=ReadoutResonator(
        omega_r=2.0 * np.pi * 7.0e9,
        kappa=2.0 * np.pi * 8.0e6,
        g=2.0 * np.pi * 90.0e6,
        epsilon=2.0 * np.pi * 0.6e6,
        chi=2.0 * np.pi * 1.5e6,
    ),
    purcell_filter=PurcellFilter(bandwidth=2.0 * np.pi * 40.0e6),
    amplifier=AmplifierChain(noise_temperature=4.0, gain=12.0),
    integration_time=300.0e-9,
    dt=5.0e-9,
)

measurement = measure_qubit(
    initial,
    QubitMeasurementSpec(
        shots=1024,
        seed=123,
        readout_chain=readout_chain,
        readout_duration=300.0e-9,
        classify_from_iq=True,
    ),
)
```

## Tutorials and example workflows

The recommended library path is:

1. Prepare an initial state with `StatePreparationSpec` and `prepare_state(...)`.
2. Build pulses or gate-derived pulse segments.
3. Compile onto a global timeline with `SequenceCompiler`.
4. Simulate with `simulate_sequence(...)`.
5. Extract reduced states, photon numbers, conditioned responses, or tomography observables.
6. Optionally sample qubit measurement outcomes with `measure_qubit(...)`.

For the guided learning path, start in `tutorials/`:

- `tutorials/README.md`
- `tutorials/00_getting_started/01_protocol_style_simulation.ipynb`
- `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
- `tutorials/10_core_workflows/02_kerr_free_evolution.ipynb`
- `tutorials/20_bosonic_and_sideband/01_sideband_swap.ipynb`
- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`
- `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
- `tutorials/40_validation_and_conventions/01_kerr_sign_and_frame_checks.ipynb`

The GRAPE tutorial now covers both closed-system optimization and simulator-backed noisy replay, and it shows how to launch the dedicated benchmark harness for larger cases.

The earlier flat numbered curriculum under `tutorials/*.ipynb` is still available as a broader API/conventions primer, but the categorized workflow suite above is now the notebook-first entry point for the migrated example programs.

Standalone repo-side scripts still live in `examples/`:

- `examples/protocol_style_simulation.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`
- `examples/sequential_sideband_reset.py`
- `examples/grape_storage_subspace_gate_demo.py`
- `examples/rl_hybrid_control_rollout.py`

Optimal-control benchmark artifacts live under `benchmarks/`:

- `benchmarks/run_optimal_control_benchmarks.py`

## Performance-oriented usage

For high-throughput workloads, the recommended fast path is:

1. Compile once with `SequenceCompiler`.
2. Prepare the runtime once with `prepare_simulation(...)`.
3. Reuse the returned `SimulationSession` across many initial states or parameter loops.
4. Pass `e_ops={}` when you only need the final state and want to avoid expectation-value bookkeeping.

Example:

```python
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, prepare_simulation, simulate_batch

compiled = SequenceCompiler(dt=0.01).compile(pulses, t_end=1.1)
session = prepare_simulation(
    model,
    compiled,
    drive_ops,
    config=SimulationConfig(frame=frame),
    e_ops={},  # Fast path when only final states matter.
)

single = session.run(initial_state)
many = simulate_batch(session, [initial_state_a, initial_state_b], max_workers=1)
```

Practical guidance:

- Use `simulate_sequence(...)` for one-off runs.
- Use `prepare_simulation(...)` for repeated runs with the same model, compiled schedule, and drive mapping.
- Use `simulate_batch(...)` or `SimulationSession.run_many(...)` for sweep-style execution over many initial states.
- Keep `store_states=False` unless you actually need the full trajectory.
- Prefer serial prepared sessions for inner loops; on this Windows environment, multiprocessing with `spawn` only pays off for much coarser jobs than the small and medium benchmarks in `benchmarks/performance_audit.md`.

Performance artifacts:

- `benchmarks/run_performance_benchmarks.py`
- `benchmarks/run_optimal_control_benchmarks.py`
- `benchmarks/performance_audit.md`

GPU note:

- The default runtime path is still the QuTiP solver stack.
- `SimulationConfig(backend=NumPyBackend())` and `SimulationConfig(backend=JaxBackend())` now enable an optional dense piecewise-constant solver path for small systems and backend parity checks.
- The current architecture still benefits more from cache/session reuse and coarse CPU parallelism than from GPU acceleration on large QuTiP-style workloads.

## Parameter Translation and Calibration Targets

Parameter translators:

- `from_transmon_params(...)`
- `from_measured(...)`
- `HamiltonianParams`

These helpers live in `cqed_sim.analysis.parameter_translation` and translate between bare transmon inputs, measured dispersive parameters, and the shared runtime/synthesis dispersive convention used inside the library.

Calibration-target helpers:

- `run_spectroscopy(...)`
- `run_rabi(...)`
- `run_ramsey(...)`
- `run_t1(...)`
- `run_t2_echo(...)`
- `run_drag_tuning(...)`

These live in `cqed_sim.calibration_targets` and return fitted parameters, uncertainties, and raw simulated data for standard calibration sweeps.

## Optional Coupling Terms

The runtime models now accept additional nonlinear or exchange terms through:

- `CrossKerrSpec(...)`
- `SelfKerrSpec(...)`
- `ExchangeSpec(...)`
- `cross_kerr(...)`
- `self_kerr(...)`
- `exchange(...)`
- `TunableCoupler(...)`

This makes it possible to add bosonic self-Kerr, cross-Kerr, or direct exchange couplings without replacing the existing dispersive-model APIs.

## Extract common observables

Two-mode extractors:

```python
from cqed_sim.sim import conditioned_bloch_xyz, reduced_cavity_state, reduced_qubit_state

rho_q = reduced_qubit_state(result.final_state)
rho_c = reduced_cavity_state(result.final_state)
bloch_n0 = conditioned_bloch_xyz(result.final_state, n=0)
```

Three-mode and readout-facing helpers:

```python
from cqed_sim.sim import (
    readout_moments,
    readout_response_by_qubit_state,
    reduced_readout_state,
    reduced_storage_state,
)

rho_storage = reduced_storage_state(result.final_state)
rho_readout = reduced_readout_state(result.final_state)
moments = readout_moments(result.final_state)
conditioned = readout_response_by_qubit_state(result.final_state)
```

Also see:

- `mode_moments(...)`
- `storage_moments(...)`
- `storage_photon_number(...)`
- `readout_photon_number(...)`
- `joint_expectation(...)`
- `cqed_sim.observables.*` for phase, trajectory, and weakness diagnostics

## Ideal gates, calibration, tomography, and synthesis

Standard ideal gates:

- `qubit_rotation_xy(...)`
- `qubit_rotation_axis(...)`
- `displacement_op(...)`
- `snap_op(...)`
- `sqr_op(...)`

Reusable calibration entry points:

- `calibrate_sqr_gate(...)`
- `load_or_calibrate_sqr_gate(...)`
- `calibrate_guarded_sqr_target(...)`
- `benchmark_random_sqr_targets_vs_duration(...)`
- `run_conditioned_multitone_validation(...)`
- `optimize_conditioned_multitone(...)`

Conditioned multitone reachability helpers:

- `ConditionedQubitTargets.from_spec(...)` accepts list/array/dict target specifications for per-Fock Bloch angles.
- `build_conditioned_multitone_tones(...)` and `build_conditioned_multitone_waveform(...)` construct the common multitone qubit drive using the same carrier-sign and additive-amplitude conventions as the rest of the library.
- `evaluate_conditioned_multitone(...)` reports per-sector conditioned qubit fidelities, Bloch vectors, angle errors, and a weighted aggregate cost.
- `optimize_conditioned_multitone(...)` tunes `d_lambda`, `d_alpha`, and `d_omega` against the reduced conditioned-qubit objective without imposing full joint-unitary correctness.

Reusable tomography entry points:

- `run_all_xy(...)`
- `autocalibrate_all_xy(...)`
- `selective_pi_pulse(...)`
- `run_fock_resolved_tomo(...)`
- `calibrate_leakage_matrix(...)`

Unitary-synthesis entry points:

- `Subspace.qubit_cavity_block(...)`
- `QuantumSystem`
- `CQEDSystemAdapter(...)`
- `PrimitiveGate(...)`
- `TargetUnitary(...)`
- `TargetStateMapping(...)`
- `SynthesisConstraints(...)`
- `LeakagePenalty(...)`
- `MultiObjective(...)`
- `ParameterDistribution(...)`
- `make_target(...)`
- `QuantumMapSynthesizer(...).fit(...)`
- `QuantumMapSynthesizer(...).explore_pareto(...)`

`cqed_sim/map_synthesis` now uses the same projector-based dispersive and Kerr semantics as the runtime Hamiltonian. Matrix-defined primitives, target-state mappings, and model-backed waveform primitives all route through that same convention set. The synthesizer now talks to a backend `QuantumSystem` interface, with `CQEDSystemAdapter(...)` preserving the existing cQED model workflow while preparing the architecture for future non-cQED systems. Phase 2 adds constraint-aware objectives, leakage-aware/noisy synthesis, robust parameter-distribution sampling, Pareto exploration, warm starts, and result export on top of that same runtime model stack. The remaining low-level sign distinction is the pulse waveform rule `Pulse.carrier = -omega_transition(frame)`, but user-facing workflows should stay in positive physical drive frequencies and convert only at the low-level pulse boundary through the shared core frequency helpers.

## RL-ready hybrid control

`cqed_sim.rl_control` adds a first-pass simulator-trained control layer on top of the existing physics stack rather than beside it.

The new environment surface centers on:

- `HybridCQEDEnv(...)` with `reset(...)`, `step(...)`, `render_diagnostics()`, and `estimate_metrics(...)`
- two model regimes through `HybridSystemConfig`: a fast reduced dispersive model and a fuller multilevel pulse model
- action-space families for low-dimensional parametric pulses, primitive/hierarchical controls, and waveform-level scaffolding
- ideal and measurement-like observation encoders, including IQ, counts, classifier-logit, and one-hot outcome views, plus domain-randomized hidden-parameter sampling
- simulator-shaped and measurement-proxy reward builders, including explicit ancilla-assignment rewards for partially observed studies
- a benchmark task ladder spanning vacuum preservation, coherent-state preparation, Fock-state preparation, storage-basis superpositions, even/odd cat preparation, ancilla-storage entanglement, and a reduced conditional-phase gate task
- richer rollout diagnostics that expose compiled channels together with segment metadata, pulse summaries, and frame/regime metadata for debugging

The intended public entry points are:

- `HybridCQEDEnv`, `HybridEnvConfig`, `HybridSystemConfig`
- `ReducedDispersiveModelConfig`, `FullPulseModelConfig`
- `ParametricPulseActionSpace`, `PrimitiveActionSpace`, `WaveformActionSpace`
- `build_observation_model(...)`, `build_reward_model(...)`
- `benchmark_task_suite()` and the individual benchmark task constructors
- `DomainRandomizer`, `FixedPrior`, `UniformPrior`, `NormalPrior`, `ChoicePrior`
- `CalibrationEvidence`, `randomizer_from_calibration(...)`

For a repo-side script template, see `examples/rl_hybrid_control_rollout.py`. For an interactive notebook walkthrough that builds the environment, runs scripted and random actions, and plots diagnostics directly, see `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`.

## What stays in `examples/`

The core library intentionally does not contain:

- the numbered tutorial curriculum,
- paper reproductions,
- one-off optimization studies,
- experiment-specific reconciliation scripts,
- legacy workflow glue,
- smoke/integration scripts for moved example paths.

Use `examples/` for those:

- `examples/workflows/sequential`
- `examples/workflows/sqr_transfer.py`
- `examples/audits/experiment_convention_audit.py`
- `examples/studies/*`
- `examples/paper_reproductions/*`
- `examples/sideband_swap_demo.py`
- `examples/shelving_isolation_demo.py`
- `examples/detuned_sideband_sync_demo.py`
- `examples/open_system_sideband_degradation.py`
- `examples/kerr_sign_verification.py`
- `examples/multimode_crosskerr_demo.py`

These are example or study APIs, not part of the canonical `cqed_sim` library surface.

Use `tutorials/` when you want the supported user-facing learning path.

For the migrated workflow-tutorial suite, the main starting points are:

- `tutorials/00_getting_started/01_protocol_style_simulation.ipynb`
- `tutorials/10_core_workflows/01_displacement_then_qubit_spectroscopy.ipynb`
- `tutorials/20_bosonic_and_sideband/03_sequential_sideband_reset.ipynb`
- `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`
- `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
- `tutorials/TUTORIAL_MIGRATION_PLAN.md`

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

The `tutorials/` notebooks are intentionally not treated as replacements for automated tests.
