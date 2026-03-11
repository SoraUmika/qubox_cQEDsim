# Experimental-Protocol Alignment Note

Date: March 10, 2026

## What Still Needed to Change

- The cleaned library still exposed only low-level pulse and solver primitives for most experiment-style workflows.
- State preparation logic lived mostly in example-side helpers instead of a canonical reusable API.
- There was no measurement layer separating exact expectations from sampled outcomes.
- The solver path assumed a qubit plus exactly one bosonic mode, which blocked a clean readout-resonator extension.
- The README described the abstract simulator well enough, but it did not show a recommended experiment-style path from preparation through measurement.

## What Remained Correct As-Is

- The two-mode runtime Hamiltonian, frame conventions, and drive-envelope sign conventions were already internally consistent.
- The package boundary between reusable library code and examples/studies was still the right boundary and should not be reopened.
- `SequenceCompiler`, `simulate_sequence`, and the existing pulse builders remained the correct execution backbone.
- The conventions report already served as the right home for sign, unit, and Hamiltonian caveats.

## What Was Added In This Follow-Up

- `cqed_sim/experiment`
  - `StatePreparationSpec` and reusable preparation helpers for basis, coherent, amplitude-defined, and density-matrix initial states.
  - `QubitMeasurementSpec` and `measure_qubit(...)` for exact probabilities, optional confusion-matrix application, optional repeated-shot sampling, and optional synthetic Gaussian I/Q clouds.
  - `SimulationExperiment` as a light protocol wrapper over `SequenceCompiler` plus `simulate_sequence`.
- `cqed_sim/core/DispersiveReadoutTransmonStorageModel`
  - three-mode storage + transmon + readout Hamiltonian with explicit storage, readout, and qubit drives.
- Generalized runtime support
  - mode-specific loss channels in `NoiseSpec`,
  - three-mode default observables and extractor helpers,
  - qubit-conditioned readout-response helpers.

## What Should Be Added Next

- Protocol templates for a small set of canonical calibrations that are reusable enough to belong in the library:
  - amplitude Rabi,
  - Ramsey,
  - spectroscopy,
  - readout ring-up/ring-down.
- A slightly richer measurement layer for readout-mode observables:
  - basic thresholded classification from synthetic I/Q,
  - optional POVM-style measurement wrappers if they can stay simple.
- Shared metadata containers for protocol provenance, targets, and extracted fit results.

## What Should Stay In `examples/`

- Paper-specific calibrations and reproductions.
- Large workflow orchestration code.
- One-off studies and optimization sweeps.
- Experiment-specific reconciliation utilities.

The intended rule is: reusable building blocks stay in `cqed_sim`; case-specific workflows stay in `examples/`.
