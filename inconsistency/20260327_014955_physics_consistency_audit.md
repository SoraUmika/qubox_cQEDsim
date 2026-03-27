# 2026-03-27 01:49:55 Physics Consistency Audit

## Executive Summary

This audit focused on whether the repository is physically self-consistent across
its documented conventions, core runtime implementation, calibration layers,
waveform export path, and public API surfaces.

Update on 2026-03-27 later in the day:

- The built-in conditional-phase primitive discussed below was subsequently
  removed from the active unitary-synthesis and Gate I/O public API by refactor.
- References to that primitive in this report remain as historical audit
  findings for the state of the codebase at audit time.

The core physics layer is largely internally consistent on the main conventions
that matter most for simulation correctness:

- Hilbert-space ordering is qubit-first for qubit-cavity systems.
- The dispersive convention is the excitation-projector form with positive
  Hamiltonian contribution `+chi * n * n_q`.
- The Kerr convention in the universal model is internally consistent.
- Pulse sampling uses `exp(+i * (carrier * t + phase))`.
- Frame helpers consistently use `carrier = -omega_transition(frame)`.
- Measurement confusion-matrix semantics are consistent between code and docs:
  `p_obs = M @ p_latent` in `(g, e)` ordering.

Two live inconsistencies were confirmed:

1. A high-severity semantic mismatch in the waveform bridge for
   `ConditionalPhaseSQR`: nonzero `phases_n` are accepted as parameters but are
   discarded when converting the gate to pulse-level waveforms.
2. A medium-severity convention mismatch between the reduced conditioned
   multitone path and the targeted-subspace multitone path: the targeted
   diagnostics reconstruct target Bloch components using a different meaning of
   `phi` than the rest of the qubit-rotation stack.

One additional API/documentation inconsistency was confirmed:

3. The public Gate I/O surface documents `ConditionalPhaseSQRGate`, but the
   JSON loader still rejects `type="ConditionalPhaseSQR"` entries.

I did not find a current live inconsistency in the core Hamiltonian sign,
Hilbert ordering, confusion-matrix semantics, or pulse carrier sign.

## Convention Map

### Core conventions confirmed from source

- Hilbert ordering:
  - `cqed_sim.core.conventions` and the coupled-gate helpers use qubit-first
    ordering for qubit-cavity systems.
- Qubit basis:
  - `|g> = |0>`, `|e> = |1>` throughout the core and gate layers.
- Dispersive term:
  - `cqed_sim.core.universal_model` uses `+chi * number * n_q`.
- Kerr term:
  - implemented with the documented positive coefficient convention in the
    static Hamiltonian builder.
- Pulse convention:
  - `cqed_sim.pulses.pulse` samples analytic waveforms as
    `amp * envelope * exp(i * (carrier * t + phase))`.
- Carrier convention:
  - `cqed_sim.core.frequencies.carrier_for_transition_frequency(...)` returns
    `-transition_frequency(...)`.
- Rotation convention:
  - `cqed_sim.core.ideal_gates.qubit_rotation_xy(theta, phi)` uses `phi` as the
    in-plane rotation-axis angle.
- Units:
  - the core runtime is unit-coherent rather than intrinsically tied to seconds;
    the recommended library-wide convention remains rad/s and seconds.

## Issue Table

| ID | Severity | Area | Summary |
|---|---|---|---|
| PHY-01 | High | Unitary synthesis / waveform export | `ConditionalPhaseSQR` waveform export drops explicit `phases_n` and produces identical pulse payloads for different requested phase vectors. |
| PHY-02 | Medium | Calibration diagnostics | `targeted_subspace_multitone` computes target Bloch components with spherical-azimuth formulas that conflict with the repo's rotation-axis `phi` convention. |
| API-01 | Medium | Gate I/O / docs | `ConditionalPhaseSQRGate` is documented and typed publicly, but `validate_gate_entry(...)` still rejects `ConditionalPhaseSQR` JSON entries. |

## Confirmed Issues

### 1. `ConditionalPhaseSQR` waveform export discards the requested conditional phases

- What:
  - `cqed_sim.unitary_synthesis.sequence.ConditionalPhaseSQR` explicitly uses
    `phases_n` to build a Fock-selective qubit-Z phase layer in both
    `ideal_unitary(...)` and `pulse_unitary(...)`.
  - `cqed_sim.unitary_synthesis.waveform_bridge.waveform_primitive_from_gate(...)`
    accepts a `phases` parameter for the same gate type but ignores it when
    creating the pulse payload.
- Where:
  - `cqed_sim/unitary_synthesis/sequence.py`
  - `cqed_sim/unitary_synthesis/waveform_bridge.py`
  - regression coverage gap in:
    - `tests/unitary_synthesis/test_time_policy_and_condphase.py`
    - `tests/unitary_synthesis/test_free_evolve_condphase.py`
- Affected components:
  - waveform bridge
  - pulse-backed synthesis workflows that rely on the bridge
  - exported or replayed pulse representations of conditional-phase gates
- Why this is inconsistent:
  - the ideal gate model says `phases_n` is part of the gate semantics.
  - the waveform bridge produces the same zero-theta / zero-phi SQR pulse for
    any `phases` vector at fixed duration, so the exported pulse no longer
    represents the requested gate except in the special drift-only subset.
- Consequences:
  - nonzero `phases_n` silently vanish on the waveform-export path.
  - optimization or replay through the bridged pulse path can optimize a gate
    whose declared parameters do not match the actual pulse-level operation.
  - current tests miss this because they cover only `phases_n = 0` cases.
- Evidence:
  - In `sequence.py`, `_drive_unitary(...)` builds block-diagonal
    `diag(exp(-i p/2), exp(+i p/2))` factors from `phases_n`.
  - In `waveform_bridge.py`, the `ConditionalPhaseSQR` branch creates an
    `SQRGate` with `theta=(0, ..., 0)` and `phi=(0, ..., 0)` and never maps the
    requested `phases` into the emitted pulse representation.
  - Direct runtime check:
    - requested phases A: `[0.1, -0.2, 0.3]`
    - requested phases B: `[1.3, 0.7, -2.1]`
    - bridge output summaries for both were identical:
      `[(1.0, 0.0, 0.0, 1e-07)]`
- Likely correct convention:
  - Either:
    - reject `ConditionalPhaseSQR` in the waveform bridge unless it is the
      drift-only zero-phase subset, or
    - implement a pulse/export representation that actually carries the
      requested conditional phase parameters.
  - Until fixed, public docs should explicitly say that only the drift-only
    subset is representable on the waveform bridge.

### 2. `targeted_subspace_multitone` uses the wrong Bloch reconstruction for the repo's `phi` convention

- What:
  - `cqed_sim.calibration.conditioned_multitone` defines `phi` as the
    rotation-axis angle for `R_phi(theta)|g>`.
  - `cqed_sim.calibration.targeted_subspace_multitone._conditioned_metric(...)`
    reconstructs target Bloch components as if `phi` were the ordinary
    spherical azimuth of the final Bloch vector.
- Where:
  - `cqed_sim/calibration/conditioned_multitone.py`
  - `cqed_sim/calibration/targeted_subspace_multitone.py`
- Affected components:
  - targeted-subspace conditioned-sector diagnostics
  - reported target Bloch coordinates and `bloch_distance`
  - developer interpretation of calibration reports
- Why this is inconsistent:
  - `conditioned_multitone.py` states and implements:
    - `X = sin(theta) * sin(phi)`
    - `Y = -sin(theta) * cos(phi)`
  - `targeted_subspace_multitone.py` instead uses:
    - `X = sin(theta) * cos(phi)`
    - `Y = sin(theta) * sin(phi)`
  - The latter is not the same convention used by `qubit_rotation_xy(...)` or
    by the reduced conditioned-multitone path.
- Consequences:
  - target Bloch coordinates reported in targeted-subspace analysis are rotated
    relative to the actual convention used to build the qubit target state.
  - `bloch_distance` is computed against the wrong target vector, so that
    diagnostic is physically mislabelled.
  - fidelity-based objective terms remain mostly protected because they compare
    against the correct target density matrix, which is why this bug can remain
    silent in optimization tests.
- Evidence:
  - `conditioned_multitone.py` documents the convention inline and in code.
  - `targeted_subspace_multitone.py` uses the incompatible formulas in
    `_conditioned_metric(...)`.
  - Direct runtime check at `theta = 0.37*pi`, `phi = 0.61*pi`:
    - reduced-path convention: `[0.863498, 0.310878, 0.397148]`
    - targeted-path formula: `[-0.310878, 0.863498, 0.397148]`
    - difference norm: `1.2979010385729293`
- Likely correct convention:
  - `targeted_subspace_multitone._conditioned_metric(...)` should use the same
    Bloch reconstruction as `conditioned_multitone.bloch_vector_from_angles(...)`
    and the `qubit_rotation_xy(...)` definition.
  - Tests should assert that the reduced and targeted diagnostics report the
    same target Bloch vector for the same `(theta, phi)` input.

### 3. Gate I/O documents `ConditionalPhaseSQRGate`, but the loader rejects it

- What:
  - `cqed_sim.io.gates` defines a `ConditionalPhaseSQRGate` dataclass and the
    public docs expose it as part of the `Gate` union.
  - `validate_gate_entry(...)` does not implement a parsing branch for
    `type == "ConditionalPhaseSQR"`.
- Where:
  - `cqed_sim/io/gates.py`
  - `documentations/api/gate_io.md`
  - `API_REFERENCE.md`
- Affected components:
  - JSON gate loading
  - public Gate I/O API documentation
  - any workflow expecting round-trip support for documented gate records
- Why this is inconsistent:
  - the public API and docs say the gate type exists.
  - the actual loader rejects that same documented type as unsupported.
- Consequences:
  - users can construct or serialize the dataclass in Python but cannot load
    the equivalent JSON gate record through the documented loader path.
  - documentation overstates support.
- Evidence:
  - `validate_gate_entry(...)` has branches for `Displacement`, `Rotation`, and
    `SQR`, then falls through to `ValueError`.
  - Direct runtime check with
    `{"type": "ConditionalPhaseSQR", "target": "qubit", "params": {"phases": [...]}}`
    raised:
    `ValueError: Gate 0 has unsupported type 'ConditionalPhaseSQR'.`
  - `API_REFERENCE.md` and `documentations/api/gate_io.md` both document
    `ConditionalPhaseSQRGate` as part of the public Gate union.
- Likely correct convention:
  - Either add JSON parsing support for `ConditionalPhaseSQR`, or remove it
    from the documented Gate union until the loader is implemented.

## Suspected / Follow-up Questions

- The tomography helper is locally self-consistent in its Hz-to-rad/ns usage,
  but some wording in `cqed_sim/tomo/README.md` can be read as if the model type
  itself has an intrinsic nanosecond time unit. The core runtime is actually
  unit-coherent. This currently looks like documentation phrasing drift rather
  than a live simulation bug, but it should be clarified.
- `API_REFERENCE.md` contains an internal contradiction about waveform-bridge
  coverage:
  - one section documents `ConditionalPhaseSQR` as supported by the synthesis
    and Gate I/O surface,
  - another still states that waveform bridging supports only
    `QubitRotation`, `Displacement`, and `SQR` and that
    `ConditionalPhaseSQR` is not bridged.
  This is documentation drift regardless of how the bridge is ultimately fixed.
- The conditional-phase tests should add at least one nonzero-phase case on the
  waveform path. Current coverage is concentrated on drift-only or ideal-unitary
  behavior and will not catch the bridge mismatch described in PHY-01.

## Status

- Current status:
  - Partially fixed on 2026-03-27, then superseded later that day by removal of
    the built-in conditional-phase primitive from the active API surface.
- Resolution summary:
  - `targeted_subspace_multitone._conditioned_metric(...)` now reuses
    `bloch_vector_from_angles(...)`, so targeted-subspace diagnostics follow the
    same rotation-axis `phi` convention as the reduced conditioned-multitone
    path and the core qubit-rotation helpers.
  - The earlier bridge/API mitigation for the built-in conditional-phase
    primitive was later superseded by removal of that primitive and its Gate I/O
    representation from the active public API. Users are expected to use custom
    waveform primitives for that workflow instead.
  - Focused regressions passed after the fix:
    `tests/test_36_targeted_subspace_multitone.py`,
    `tests/unitary_synthesis/test_waveform_bridge.py`, and
    `tests/unitary_synthesis/test_primitives_and_backends.py`.

## Proposed Corrections

1. `ConditionalPhaseSQR` waveform bridge:
   - short term: reject nonzero-phase `ConditionalPhaseSQR` on the waveform
     bridge with an explicit error message.
   - medium term: either implement a real pulse/export encoding for `phases_n`
     or narrow the public contract to the drift-only subset.
   - add regression tests that compare distinct nonzero `phases_n` inputs and
     assert distinct emitted pulse semantics or explicit rejection.
2. Targeted-subspace Bloch diagnostics:
   - replace the target Bloch formulas in `_conditioned_metric(...)` with the
     same convention used by `bloch_vector_from_angles(...)`.
   - add a regression test that compares reduced and targeted target-Bloch
     diagnostics for the same input angles.
3. Gate I/O support/doc sync:
   - either implement a `ConditionalPhaseSQR` loader branch in
     `validate_gate_entry(...)` or remove the type from the documented JSON API.
   - update `API_REFERENCE.md` and `documentations/api/gate_io.md` so that the
     supported loader surface is unambiguous.

## Fix Record

- Fixed by:
  - `cqed_sim/calibration/targeted_subspace_multitone.py` now uses
    `bloch_vector_from_angles(...)` for target Bloch diagnostics.
  - `cqed_sim/unitary_synthesis/waveform_bridge.py` now narrows
    `ConditionalPhaseSQR` waveform support to the drift-only zero-phase subset
    and rejects unsupported nonzero explicit phase vectors.
  - `cqed_sim/io/gates.py` now parses `ConditionalPhaseSQR` JSON entries and
    reports them in gate summaries.
  - Regression coverage added in:
    - `tests/test_36_targeted_subspace_multitone.py`
    - `tests/unitary_synthesis/test_waveform_bridge.py`
    - `tests/unitary_synthesis/test_primitives_and_backends.py`
  - Public docs updated in:
    - `documentations/api/gate_io.md`
    - `API_REFERENCE.md`
- Remaining concerns:
  - The waveform bridge still does not encode nonzero explicit
    `ConditionalPhaseSQR.phases_n` into a pulse/export representation. That is
    now an explicit limitation rather than a silent semantic bug.
  - The tomo helper wording noted in the follow-up section remains worth a
    separate documentation cleanup pass.