# `cqed_sim.unitary_synthesis`

The `unitary_synthesis` module provides a gate-sequence synthesis framework for cQED systems. Given a target unitary or state mapping and a set of primitive gates, it searches for a sequence of parametrized primitives that realizes the target on a specified logical subspace, subject to leakage penalties, physical constraints, and optional robustness objectives.

## Relevance in `cqed_sim`

Synthesis occupies the layer between abstract gate specifications and physically executable pulse sequences. It is relevant when:

- a target operation (e.g. a SNAP gate, a conditional-phase gate, or a multi-photon unitary) needs to be decomposed into a calibrated sequence of elementary control primitives,
- leakage into levels outside the computational subspace must be suppressed as part of the synthesis,
- or when Pareto exploration over fidelity vs. total gate duration is needed.

The synthesizer uses the same dispersive Hamiltonian and Kerr conventions as `cqed_sim.sim` and `cqed_sim.core`, so the outputs are directly compatible with the simulation and pulse-build pipeline.

## Main Capabilities

### Synthesis engine

- **`UnitarySynthesizer`**: The main synthesis class. Implements `.fit(target, constraints)` for single-run optimization and `.explore_pareto(...)` for multi-objective Pareto exploration over fidelity/duration trade-offs.
- **`SynthesisResult`**: Result of `.fit(...)`: the optimized gate sequence, final fidelity, and per-objective values.
- **`ParetoFrontResult`**: Result of `.explore_pareto(...)`: a set of Pareto-optimal solutions.

### System interface

- **`QuantumSystem`**: Abstract interface for the quantum system the synthesizer talks to.
- **`CQEDSystemAdapter`**: Concrete adapter that wraps a `cqed_sim` model and exposes it as a `QuantumSystem`. The preferred way to connect a `cqed_sim` model to the synthesizer.

### Gate primitives

- **`PrimitiveGate`**: Base class for a parametrized elementary gate (e.g. `QubitRotation`, `Displacement`, `SNAP`, `SQR`, `ConditionalPhaseSQR`, `CavityBlockPhase`).
- **`GateSequence`**: Ordered list of `PrimitiveGate` instances forming a synthesis ansatz.
- **`GateTimeParam`**: Parametrized gate duration, used when duration is a free variable in the synthesis.
- **`DriftPhaseModel`**, `FreeEvolveCondPhase`**: Handles drift-phase accumulation between discrete gates.

### Targets

- **`TargetUnitary`**: Optimization target is a full unitary on the joint Hilbert space.
- **`TargetStateMapping`**: Target is a set of input-to-output state mappings.
- **`TargetReducedStateMapping`**: Target defined on a reduced subspace.
- **`TargetIsometry`**, **`TargetChannel`**, **`ObservableTarget`**: Additional target types for isometries, quantum channels, and observable expectations.
- **`make_target(...)`**: Convenience constructor.

### Constraints and objectives

- **`SynthesisConstraints`**: Hard and soft constraints including leakage bounds, amplitude limits, and tone-spacing requirements.
- **`LeakagePenalty`**: Additive penalty for population outside the computational subspace.
- **`MultiObjective`**: Combines fidelity and additional penalty terms into a single optimization objective.
- **`ParameterDistribution`** (`Normal`, `Uniform`): Parameter-uncertainty distributions for robust synthesis.

### Subspace

- **`Subspace`**: Defines the logical subspace over which the target unitary is specified. `Subspace.qubit_cavity_block(...)` constructs the standard qubit ⊗ Fock-sector subspace.

### Metrics

- **`subspace_unitary_fidelity(...)`**: Fidelity of a unitary restricted to a subspace.
- **`leakage_metrics(...)`**: Leakage diagnostics.
- **`logical_block_phase_diagnostics(...)`**: Phase diagnostics for block-diagonal targets.
- **`operator_truncation_sanity_metrics(...)`**, `truncation_sanity_metrics(...)`: Sanity checks for operator truncation.

### Progress reporting

- **`HistoryReporter`**, **`JupyterLiveReporter`**, **`NullReporter`**: Plug-in progress reporters for tracking synthesis iteration history.
- **`plot_history(...)`**, `history_to_dataframe(...)`: Visualization and analysis of synthesis convergence history.

### Constraints

- **`enforce_slew_limit(...)`**, `evaluate_tone_spacing(...)`, `project_tone_frequencies(...)`: Physical constraint projection applied after each optimization step.

### Waveform bridge

- **`waveform_bridge`**: Exports a synthesized gate sequence as a sequence of `Pulse` objects compatible with `SequenceCompiler` and `simulate_sequence(...)`.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `UnitarySynthesizer` | Main synthesis class |
| `CQEDSystemAdapter(model)` | Connect a `cqed_sim` model to the synthesizer |
| `QuantumSystem` | Abstract system interface |
| `PrimitiveGate` | Elementary parametrized gate |
| `GateSequence` | Ansatz sequence of primitives |
| `TargetUnitary` | Unitary target spec |
| `TargetStateMapping` | State-mapping target spec |
| `SynthesisConstraints` | Constraint specification |
| `LeakagePenalty` | Leakage penalty term |
| `Subspace.qubit_cavity_block(...)` | Logical subspace definition |
| `subspace_unitary_fidelity(...)` | Fidelity on a subspace |
| `UnitarySynthesizer.fit(...)` | Single optimization run |
| `UnitarySynthesizer.explore_pareto(...)` | Pareto-front exploration |

## Usage Guidance

```python
from cqed_sim.unitary_synthesis import (
    CQEDSystemAdapter, UnitarySynthesizer,
    GateSequence, QubitRotation, SNAP, TargetUnitary,
    SynthesisConstraints, LeakagePenalty, Subspace,
)
from cqed_sim.core import snap_op

# Define the system
adapter = CQEDSystemAdapter(model)

# Define the target
target_U = snap_op(model, angles=[0, np.pi, 0, 0, 0, 0, 0, 0])
subspace = Subspace.qubit_cavity_block(n_cav=model.n_cav, n_tr=model.n_tr, fock_levels=range(8))
target = TargetUnitary(unitary=target_U, subspace=subspace)

# Define the synthesis ansatz
ansatz = GateSequence([
    QubitRotation(theta=0.0, phi=0.0),
    SNAP(angles=[0.0] * 8),
    QubitRotation(theta=0.0, phi=0.0),
])

# Run synthesis
synthesizer = UnitarySynthesizer(system=adapter)
result = synthesizer.fit(
    target=target,
    sequence=ansatz,
    constraints=SynthesisConstraints(penalties=[LeakagePenalty(weight=0.1)]),
)
print(result.fidelity)
```

For a full notebook walkthrough: `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`

## Important Assumptions / Conventions

- The synthesizer uses the same projector-based dispersive and Kerr semantics as the runtime Hamiltonian in `cqed_sim.sim`.
- Subspace fidelity is computed using the normalized Hilbert–Schmidt inner product on the specified logical subspace.
- Drift phases between gates are tracked via `DriftPhaseModel` using the static Hamiltonian in the specified rotating frame.
- The pulse waveform sign convention applies to output pulses: `Pulse.carrier = -omega_transition(frame)`.
- Leakage is defined as population outside the logical subspace at the end of the gate sequence.

## Relationships to Other Modules

- **`cqed_sim.core`**: provides the model and `FrameSpec` that `CQEDSystemAdapter` wraps.
- **`cqed_sim.sim`**: the `backends.py` file in this module re-exports `simulate_sequence` for internal use; converged sequences can be validated against the full solver.
- **`cqed_sim.optimal_control`**: GRAPE can refine synthesized waveforms at the pulse level; `objective_from_unitary_synthesis_target(...)` bridges the two.
- **`cqed_sim.pulses`** and **`cqed_sim.sequence`**: `waveform_bridge` exports synthesized sequences as `Pulse` lists for compilation and simulation.

## Limitations / Non-Goals

- Synthesis is local optimization; convergence to the global optimum is not guaranteed. Use `explore_pareto(...)` or multiple random restarts for difficult targets.
- The current primitive gate library covers qubit rotations, displacements, SNAP, SQR, and phase gates. Custom primitives require subclassing `PrimitiveGate`.
- Synthesis does not account for finite-bandwidth pulse shaping unless hard constraint projection (`enforce_slew_limit`) is enabled.
- Phase-2 features (robust synthesis over parameter distributions, full Pareto exploration, warm starts) are implemented; phase-3 hardware-level AWG integration is not.

## References

- Tutorial notebook: `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`
- Root `README.md` — lists the full public API surface for synthesis.
