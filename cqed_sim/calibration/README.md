# `cqed_sim.calibration`

The `calibration` module provides reusable calibration and validation routines for cQED gate operations, with particular focus on Selective-Number-dependent Arbitrary-Phase (SNAP) Qubit Rotation (SQR) gates and multi-tone qubit drives for dispersive quantum control. It offers two complementary validation paths for dispersive SQR-style studies.

## Relevance in `cqed_sim`

Calibration lives between the simulation layer (`cqed_sim.sim`) and experiment-facing use cases. The module is relevant when:

- calibrating an SQR gate (finding the tone amplitudes and phases that implement a target per-Fock rotation),
- validating that a multitone waveform implements the intended operation on the joint qubit-cavity subspace,
- or studying the reachability and fidelity of conditional qubit operations as a function of drive parameters.

The two included paths ask complementary questions:

1. **Conditioned multitone** (`conditioned_multitone`): Does each conditioned-qubit branch reach the intended Bloch-vector target? This is a reduced-state validation that does not require defining a full joint unitary.
2. **Targeted-subspace multitone** (`targeted_subspace_multitone`): Does the common waveform implement the intended coherent operator on a chosen joint qubit-cavity logical subspace, preserving cavity-block structure and suppressing leakage? This is the stronger full-unitary validation.

## Main Capabilities

### SQR gate calibration (`sqr`)

- **`calibrate_sqr_gate(model, target, config)`**: Calibrates a single SQR gate by optimizing tone amplitudes and phases to match a target per-Fock rotation array.
- **`load_or_calibrate_sqr_gate(...)`**: Loads a cached calibration result or runs calibration if no cache is found.
- **`calibrate_guarded_sqr_target(...)`**: Calibrates a guarded SQR target with joint optimization over gate time and tone parameters.
- **`benchmark_random_sqr_targets_vs_duration(...)`**: Runs a duration sweep over random SQR targets and returns a fidelity-vs-duration table.
- **`evaluate_sqr_gate_levels(...)`**, `evaluate_guarded_sqr_target(...)`: Evaluate a calibrated result without re-running optimization.
- **`extract_sqr_gates(...)`, `extract_multitone_effective_qubit_unitary(...)`, `extract_effective_qubit_unitary(...)`**: Extract the effective per-Fock unitary from a simulation result.
- **`conditional_process_fidelity(...)`, `conditional_loss(...)`**: Fidelity and loss metrics for conditioned processes.
- **`SQRCalibrationResult`, `GuardedBenchmarkResult`, `RandomSQRTarget`**: Result and specification types.

### Conditioned multitone reachability (`conditioned_multitone`)

- **`run_conditioned_multitone_validation(model, targets, waveform, config)`**: Runs a full conditioned-qubit validation: simulates the multitone drive in each Fock sector and computes per-sector fidelities and Bloch-vector errors.
- **`optimize_conditioned_multitone(model, targets, config)`**: Optimizes `d_lambda`, `d_alpha`, and `d_omega` against the reduced conditioned-qubit objective.
- **`evaluate_conditioned_multitone(model, waveform, targets)`**: Evaluates a fixed waveform without optimization.
- **`build_conditioned_multitone_tones(...)`, `build_conditioned_multitone_waveform(...)`**: Construct the multitone qubit drive.
- **`ConditionedQubitTargets.from_spec(...)`**: Accepts list/array/dict target specifications for per-Fock Bloch angles.
- Result types: `ConditionedValidationResult`, `ConditionedSectorMetrics`, `ConditionedMultitoneWaveform`, `ConditionedOptimizationResult`.

### Targeted-subspace multitone validation (`targeted_subspace_multitone`)

- **`run_targeted_subspace_multitone_validation(model, transfer_set, waveform, config)`**: Full joint-subspace validation and optimization for a targeted multitone drive.
- **`optimize_targeted_subspace_multitone(model, transfer_set, config)`**: Optimizes the waveform to implement the target operator on the chosen logical subspace.
- **`evaluate_targeted_subspace_multitone(model, waveform, transfer_set)`**: Evaluates fidelity and block metrics for a fixed waveform.
- **`build_spanning_state_transfer_set(...)`, `build_block_rotation_target_operator(...)`**: Helper constructors for the logical subspace target.
- **`analyze_targeted_subspace_operator(...)`**: Decomposes and diagnoses the effective operator on the logical subspace.
- Result types: `TargetedSubspaceValidationResult`, `TargetedSubspaceTransferMetric`, `TargetedSubspaceBasisMetric`, `TargetedSubspaceOptimizationResult`.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `calibrate_sqr_gate(...)` | Calibrate a single SQR gate |
| `load_or_calibrate_sqr_gate(...)` | Cached SQR calibration |
| `calibrate_guarded_sqr_target(...)` | Guarded SQR calibration |
| `benchmark_random_sqr_targets_vs_duration(...)` | Duration-fidelity sweep |
| `run_conditioned_multitone_validation(...)` | Reduced conditioned-qubit validation |
| `optimize_conditioned_multitone(...)` | Optimize conditioned-qubit waveform |
| `ConditionedQubitTargets.from_spec(...)` | Build per-Fock Bloch-angle targets |
| `build_conditioned_multitone_waveform(...)` | Construct the multitone waveform |
| `run_targeted_subspace_multitone_validation(...)` | Full joint-subspace validation |
| `optimize_targeted_subspace_multitone(...)` | Optimize targeted-subspace waveform |

## Usage Guidance

### Conditioned multitone validation

```python
from cqed_sim.calibration import (
    ConditionedQubitTargets, ConditionedMultitoneRunConfig,
    build_conditioned_multitone_waveform, run_conditioned_multitone_validation,
)

targets = ConditionedQubitTargets.from_spec(
    {0: (np.pi / 2, 0), 1: (np.pi, 0), 2: (np.pi / 2, np.pi)},
    n_sectors=8,
)
waveform = build_conditioned_multitone_waveform(model, targets, config=...)
result = run_conditioned_multitone_validation(model, targets, waveform, config=ConditionedMultitoneRunConfig(...))
print(result.weighted_fidelity)
```

### SQR gate calibration

```python
from cqed_sim.calibration import calibrate_sqr_gate, SQRCalibrationResult

result = calibrate_sqr_gate(model, target_angles=[0, np.pi, 0, np.pi, 0, np.pi, 0, 0])
print(result.fidelity)
```

## Important Assumptions / Conventions

- The conditioned multitone validation uses the same additive-amplitude convention and the same hybrid frequency split as the rest of the library: user-facing boundaries should stay in positive physical drive frequencies, while emitted low-level `Pulse.carrier` values still satisfy `Pulse.carrier = -omega_transition(frame)`.
- Chi convention: per-photon qubit-frequency shift, consistent with `cqed_sim.core`. The multitone tone frequencies are set relative to the qubit transition in each Fock sector using `manifold_transition_frequency(model, n, frame)`.
- The targeted-subspace path enforces cavity-block structure: the joint unitary is required to preserve the cavity Fock-number sectors up to a specified tolerance.
- SQR calibration optimizes over per-tone amplitudes and phases; the pulse duration is a fixed input parameter.

## Relationships to Other Modules

- **`cqed_sim.sim`**: calibration routines call `simulate_sequence(...)` internally to evaluate the simulated unitary.
- **`cqed_sim.core`**: uses `manifold_transition_frequency(...)` for frequency targeting and `DispersiveTransmonCavityModel` for the physical Hamiltonian.
- **`cqed_sim.pulses`**: `build_conditioned_multitone_waveform(...)` uses the SQR pulse builders internally.
- **`cqed_sim.calibration_targets`**: separate module for standard calibration sweeps (Rabi, Ramsey, T1, etc.) that does not overlap with the SQR/multitone routines here.

## Limitations / Non-Goals

- The conditioned multitone optimization tunes three scalar parameters (`d_lambda`, `d_alpha`, `d_omega`); it does not perform full waveform optimization. For full waveform optimization, use `cqed_sim.optimal_control`.
- The SQR calibration assumes a flat-top or Gaussian multitone pulse family; arbitrary waveform shapes require custom implementation.
- These routines are single-qubit, single-cavity. Generalization to multi-qubit or multi-cavity systems is not currently implemented.

## References

- Root `README.md` — describes both validation modes and their complementary scope.
- `cqed_sim.calibration_targets` — for standard single-qubit calibration protocols (Rabi, Ramsey, T1, T2, DRAG).
