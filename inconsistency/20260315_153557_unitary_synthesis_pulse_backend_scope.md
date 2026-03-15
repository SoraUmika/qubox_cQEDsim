# Unitary Synthesis Pulse-Backend Scope Audit

Created: 2026-03-15 15:35:57 local time  
Status: fixed

## Confirmed issue

### `cqed_sim.unitary_synthesis` labeled a backend as `"pulse"` even when the path did not generally execute waveform-driven model simulation

- What it is:
  - The existing `cqed_sim.unitary_synthesis.backends.simulate_sequence(...)` dispatches to `GateSequence.unitary(...)`.
  - For the built-in synthesis primitives, the `"pulse"` branch mostly applies synthesis-local unitary formulas such as time-scaled ideal gates or analytic drift propagation.
  - That behavior is useful for fast optimization, but the backend name suggests direct reuse of the repository's runtime `Pulse -> SequenceCompiler -> cqed_sim.sim` workflow.
- Where it appears:
  - `cqed_sim/unitary_synthesis/backends.py`
  - `cqed_sim/unitary_synthesis/sequence.py`
  - user-facing docs that describe a pulse backend without clearly separating analytic synthesis approximations from model-backed waveform simulation
- Affected components:
  - unitary-synthesis API expectations
  - waveform primitive extensibility
  - documentation accuracy
  - advanced tutorial workflow expectations
- Why inconsistent:
  - Elsewhere in the repository, "pulse simulation" refers to compiling explicit `Pulse` objects and simulating them under a model Hamiltonian through `cqed_sim.sequence` and `cqed_sim.sim`.
  - The synthesis module used the same label for a narrower approximation path, which made the API harder to extend cleanly to arbitrary `cQED_model` waveform-driven primitives.
- Consequences:
  - users could reasonably expect model-aware waveform simulation when selecting `backend="pulse"` inside unitary synthesis and instead receive a synthesis-local approximation
  - the old naming made the requested primitive-gate and arbitrary-model extension ambiguous

## Resolution update

Updated: 2026-03-15 local time

- Fixed:
  - Extended the synthesis layer with explicit matrix-defined and waveform-defined primitive support.
  - Routed waveform primitives through the repository's runtime simulation stack based on `Pulse`, `SequenceCompiler`, and `cqed_sim.sim`.
  - Updated synthesis docs and tutorial material to distinguish matrix-defined primitives from model-backed waveform primitives clearly.

## Remaining follow-up items

- None for this refactor pass.
