# Logical Block-Phase Primitive Design

## Goal

Add a first-class cavity-only logical block-phase primitive

$$
U_{\mathrm{blockphase}} = I_q \otimes \sum_n e^{i\phi_n} |n\rangle\langle n|
$$

to the upstream `cqed_sim` stack so that targeted-subspace multitone workflows can treat block-phase control as an explicit ansatz layer rather than as a notebook-only post-processing correction.

## Existing Insertion Points

1. `cqed_sim.core.ideal_gates` already contains the ideal cavity-diagonal helper `snap_op(phases)` and the joint qubit-cavity selective rotation helper `sqr_op(...)`.
2. `cqed_sim.unitary_synthesis.sequence` already models reusable ideal primitives such as `SQR`, `SNAP`, and `FreeEvolveCondPhase`.
3. `cqed_sim.unitary_synthesis.systems`, `fast_eval`, and `optim` already know how to optimize parameterized ideal primitives inside `GateSequence` objects.
4. `cqed_sim.calibration.targeted_subspace_multitone` already provides the physically relevant targeted-subspace evaluation path, but it currently optimizes only waveform-side multitone correction knobs.
5. The study scripts `examples/studies/sqr_block_phase_study.py` and `examples/studies/sqr_block_phase_followup.py` already contain the block-phase diagnostics and ideal post-correction logic that should be upstreamed.

## Design Decisions

### 1. Primitive Placement

The new primitive belongs in two upstream layers:

- `cqed_sim.core.ideal_gates` for exact operator construction.
- `cqed_sim.unitary_synthesis.sequence` for optimizer-facing primitive semantics.

This keeps the mathematical operator independent from any specific optimizer while still letting synthesis treat the primitive as a real ansatz component.

### 2. Ideal Versus Control-Backed Status

The initial implementation is intentionally **ideal-only but synthesis-compatible**.

- It is not compiled from an existing physical pulse family.
- It is not represented as a fake surrogate built from conditional qubit phases plus global qubit rotations.
- In pulse-mode synthesis evaluation it will continue to act as an ideal matrix primitive, and the documentation will state that explicitly.

This preserves an honest distinction between:

- ideal logical block-phase support,
- effective drift-backed conditional phases already present in the repo,
- future pulse-backed or hardware-compiled cavity block-phase realizations.

### 3. Reused Abstractions

- Reuse `snap_op(...)` semantics for contiguous cavity-diagonal phase application.
- Reuse `GateBase` / `GateSequence` / `UnitarySynthesizer` infrastructure for parameter management and optimization.
- Reuse `TargetedSubspaceValidationResult` as the main carrier for restricted-process and state-transfer metrics.
- Reuse study-proven block-phase extraction logic, but move it into reusable package diagnostics rather than leaving it in example scripts.

### 4. Targeted-Subspace Extension

The targeted-subspace path should support an explicit post-layer ansatz

$$
U_{\mathrm{ansatz}} = U_{\mathrm{blockphase}}(\phi)\, U_{\mathrm{multitone}}(\theta)
$$

by adding:

- a block-phase correction container for logical cavity sectors,
- optional application of that correction during targeted-subspace evaluation,
- optional optimization of block-phase parameters alongside waveform-side multitone correction parameters.

This is the minimal architectural change needed to make the new capability part of the physical ansatz rather than a notebook-only analysis step.

### 5. Diagnostics

Reusable diagnostics should live in `cqed_sim.unitary_synthesis.metrics` and be consumed by the targeted-subspace calibration layer. The package-level diagnostics will expose:

- gauge-fixed block-overlap phases,
- per-block phase residuals,
- RMS block-phase error,
- best-fit logical block-phase correction vectors,
- block-gauge fidelity and basic phase/fidelity correlation helpers.

### 6. Tests, Example, and Docs

The implementation should update:

- ideal primitive tests,
- unitary-synthesis primitive tests,
- targeted-subspace validation tests,
- API docs in `API_REFERENCE.md` and `documentations/api/*`,
- `physics_and_conventions/physics_conventions_report.tex`,
- a clean example under `examples/` showing the baseline multitone result and the improved block-phase-augmented ansatz.

## Planned Files

- `cqed_sim/core/ideal_gates.py`
- `cqed_sim/core/__init__.py`
- `cqed_sim/__init__.py`
- `cqed_sim/unitary_synthesis/sequence.py`
- `cqed_sim/unitary_synthesis/systems.py`
- `cqed_sim/unitary_synthesis/fast_eval.py`
- `cqed_sim/unitary_synthesis/metrics.py`
- `cqed_sim/unitary_synthesis/__init__.py`
- `cqed_sim/calibration/targeted_subspace_multitone.py`
- `cqed_sim/calibration/__init__.py`
- `tests/test_16_ideal_primitives_and_extractors.py`
- `tests/test_36_targeted_subspace_multitone.py`
- `tests/unitary_synthesis/test_primitives_and_backends.py`
- `tests/unitary_synthesis/test_metrics.py`
- `examples/logical_block_phase_targeted_subspace_demo.py`
- `API_REFERENCE.md`
- `documentations/api/core.md`
- `documentations/api/unitary_synthesis.md`
- `documentations/api/calibration.md`
- `documentations/examples.md`
- `physics_and_conventions/physics_conventions_report.tex`

## Limitations Of The First Pass

- No new pulse family or hardware compilation path is introduced yet.
- The new primitive is exact in the simulator and optimizer, but its physical realization remains future work.
- The targeted-subspace optimizer will support the single post-layer form first; alternating `SQR` and block-phase layers remain naturally expressible in the generic synthesis stack but are not required in the calibration shortcut API.