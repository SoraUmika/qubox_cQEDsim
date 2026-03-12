# Multilevel Transmon Model Refactor Plan

Created: 2026-03-11

## Phase 0: Survey / Audit

### Current model classes and responsibilities

- `cqed_sim/core/model.py::DispersiveTransmonCavityModel`
  - Two-mode dispersive transmon-storage model.
  - Owns tensor embedding, operator caching, static Hamiltonian assembly, basis helpers, and transmon/sideband frequency helpers.
- `cqed_sim/core/readout_model.py::DispersiveReadoutTransmonStorageModel`
  - Three-mode storage-transmon-readout model.
  - Re-implements the same responsibilities with a larger tensor product.
- `cqed_sim/core/hamiltonian.py`
  - Generic additive cross-Kerr, self-Kerr, and exchange coupling helpers.
- `cqed_sim/sim/runner.py`
  - Assumes the model exposes `operators()`, `static_hamiltonian(...)`, `subsystem_dims`, and optionally structured transition/sideband helpers.

### Current assumptions that hard-code a 2-level qubit

- `cqed_sim/core/ideal_gates.py`
  - `qubit_rotation_xy`, `qubit_rotation_axis`, and `sqr_op` are explicitly SU(2) helpers.
- `cqed_sim/tomo/protocol.py`
  - Calibration and tomography helpers intentionally target the `|g>`,`|e>` manifold and use Pauli-based pre-rotations.
- `examples/workflows/sequential/common.py`
  - The workflow builder still fixes `n_tr=2` because that workflow is a qubit-only SQR pipeline.
- `cqed_sim/sim/extractors.py`
  - Bloch-vector helpers are intentionally restricted to a two-level reduced qubit state, though the same module already supports multilevel subsystem populations.

These are physics- or workflow-specific SU(2) helpers rather than model-layer blockers. They should remain explicit rather than being silently generalized.

### Current operator construction strategy

- Each concrete model class manually embeds annihilation/lowering operators with `qt.tensor(...)`.
- Each class constructs its own operator cache and its own label set (`a`, `a_s`, `a_r`, `b`, `n_q`, `n_c`, `n_s`, `n_r`).
- Each class also re-implements transition projectors, transition operators, and sideband operators.

### Current tensor ordering and basis ordering

- Two-mode: transmon first, bosonic mode second, `|q,n>`.
- Three-mode: transmon first, storage second, readout third, `|q,n_s,n_r>`.
- Flat index helpers live in `cqed_sim/core/conventions.py`.

### Current frame conventions

- `FrameSpec` stores:
  - `omega_c_frame` for storage/cavity-like modes,
  - `omega_q_frame` for the transmon,
  - `omega_r_frame` for readout.
- The concrete models subtract these values directly when assembling `static_hamiltonian(...)`.
- Current frame semantics are correct and already used consistently by spectroscopy and sideband helpers.

### Current places where transmon dimension is assumed to be 2

- The model layer itself already supports `n_tr >= 2`.
- The main remaining dimension-2 assumptions are in SU(2) gate/tomography helpers and example workflows built around qubit rotations.
- `cqed_sim/sim/runner.py::default_observables(...)` assumes subsystem 0 is a transmon-like subsystem and reports `P_g`, `P_e`, and `P_f` where available.

### Current tests and examples impacted by a model refactor

Core model coverage:

- `tests/test_11_model_invariants.py`
- `tests/conventions/test_conventions.py`
- `tests/test_27_three_mode_model.py`
- `tests/test_31_multilevel_sideband_extension.py`
- `tests/sim/test_couplings.py`
- `tests/test_10_chi_convention.py`

Examples and workflows that instantiate the existing models:

- `examples/displacement_qubit_spectroscopy.py`
- `examples/sideband_swap_demo.py`
- `examples/shelving_isolation_demo.py`
- `examples/detuned_sideband_sync_demo.py`
- `examples/open_system_sideband_degradation.py`
- `examples/multimode_crosskerr_demo.py`
- `examples/workflows/cqed_sim_usage_examples.ipynb`

### Gaps in the current architecture

1. The two concrete model classes duplicate nearly all operator-generation and Hamiltonian logic.
2. The runtime only has specialized concrete models, not a shared subsystem/coupling abstraction.
3. Generic helper code in `frequencies.py` branches on `len(model.subsystem_dims)` instead of using a canonical subsystem specification.
4. The public API has no clean entry point for a generalized transmon-plus-modes system even though the internals already support multilevel transmons.

## Phase 1: Design Proposal

### Should there be one universal model?

Yes, but with conservative scope:

- Introduce one reusable internal/public model abstraction for:
  - an optional transmon-like subsystem,
  - one or more bosonic modes,
  - dispersive couplings between the transmon and bosonic modes,
  - generic cross-Kerr / self-Kerr / exchange terms.
- Preserve the existing specialized model classes as compatibility wrappers that delegate to the universal model.

### Proposed new core types

- `TransmonModeSpec`
  - label, Hilbert dimension, bare frequency, anharmonicity, aliases, frame channel.
- `BosonicModeSpec`
  - label, Hilbert dimension, bare frequency, self-Kerr hierarchy, aliases, frame channel.
- `DispersiveCouplingSpec`
  - bosonic mode label, transmon label, first-order `chi`, and higher-order `chi_k`.
- `UniversalCQEDModel`
  - optional transmon spec,
  - tuple of bosonic mode specs,
  - tuple of dispersive coupling specs,
  - existing `CrossKerrSpec`, `SelfKerrSpec`, `ExchangeSpec`.

### Invariants

- Tensor order is fixed by subsystem declaration order:
  - transmon first when present,
  - then bosonic modes in the order supplied.
- Operator labels and aliases are stable:
  - transmon: `b`, `bdag`, `n_q`,
  - storage/cavity-compatible single-mode aliases: `a`, `adag`, `n_c`,
  - storage/readout-compatible aliases: `a_s`, `adag_s`, `n_s`, `a_r`, `adag_r`, `n_r`,
  - per-mode canonical aliases using the declared mode label.
- The canonical physics convention remains unchanged:
  - positive `chi` raises the transmon transition with bosonic occupancy,
  - positive `K` raises adjacent bosonic ladder spacings,
  - frame subtraction remains `omega - omega_frame`.

### How old models map onto it

- `DispersiveTransmonCavityModel`
  - wraps `UniversalCQEDModel(transmon=..., bosonic_modes=(storage,), dispersive_couplings=(storage-chi, ...))`
- `DispersiveReadoutTransmonStorageModel`
  - wraps `UniversalCQEDModel(transmon=..., bosonic_modes=(storage, readout), dispersive_couplings=(chi_s, chi_r), cross/self/exchange terms...)`

### How operators are generated

- The universal model constructs one tensor-embedded lowering operator per subsystem via QuTiP.
- Derived operators (`dag`, number operator, projectors, transition operators) are cached centrally.
- Existing concrete models delegate `operators()`, `transmon_level_projector(...)`, `transmon_transition_operators(...)`, `mode_operators(...)`, and `sideband_drive_operators(...)` to the shared implementation.

### How dimensions and truncations are represented

- Directly in the subsystem specs:
  - `TransmonModeSpec.dim`
  - `BosonicModeSpec.dim`
- Existing wrapper fields (`n_tr`, `n_cav`, `n_storage`, `n_readout`) remain the user-facing truncation parameters.

### How couplings are represented

- Dispersive couplings move into explicit `DispersiveCouplingSpec`.
- Generic cross-Kerr, self-Kerr, and exchange terms continue using the existing `CrossKerrSpec`, `SelfKerrSpec`, and `ExchangeSpec`.

### How rotating frames are represented

- Keep `FrameSpec` unchanged for backward compatibility.
- Each subsystem spec declares which `FrameSpec` channel it uses:
  - transmon -> `omega_q_frame`
  - storage/cavity -> `omega_c_frame`
  - readout -> `omega_r_frame`
- This keeps current public semantics intact while moving the model internals toward an explicit subsystem registry.

### Migration strategy

1. Add the universal subsystem/model layer.
2. Re-implement the current concrete models as thin delegating wrappers.
3. Keep existing constructors and method signatures alive.
4. Add new tests that prove wrapper equivalence against the universal model and prove the `n_tr=2` reduction.
5. Update docs to present the universal model as the new generalized path, while keeping the old models as convenience constructors.

### What remains unchanged

- Existing constructors:
  - `DispersiveTransmonCavityModel(...)`
  - `DispersiveReadoutTransmonStorageModel(...)`
- Existing frame convention and sign convention.
- Existing spectroscopy helpers and sideband helpers.

### What is new

- `UniversalCQEDModel`
- `TransmonModeSpec`
- `BosonicModeSpec`
- `DispersiveCouplingSpec`
- Convenience operator accessors such as `transmon_lowering()`, `cavity_annihilation()`, and `hamiltonian(...)`.

## Phase 2 target

Preferred implementation for this task:

- introduce the universal model now,
- keep the existing models as wrappers,
- do not attempt to generalize every SU(2)-specific helper in the repo in the same patch.

That gives the model layer a clean universal foundation without turning the rest of the repository into a risky, all-at-once rewrite.

## Implementation status

Implemented on 2026-03-11:

- `cqed_sim/core/universal_model.py` now provides:
  - `TransmonModeSpec`
  - `BosonicModeSpec`
  - `DispersiveCouplingSpec`
  - `UniversalCQEDModel`
- `DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` now delegate to that shared universal layer while preserving their previous public constructors.
- `cqed_sim/core/frequencies.py` now uses explicit model capabilities rather than branching on `len(model.subsystem_dims)`.
- `cqed_sim/sim/runner.py` and `cqed_sim/sim/noise.py` were updated so cavity-only universal models behave consistently.
- Regression coverage was added in `tests/test_34_universal_cqed_model.py`.

The intentionally unchanged boundary is that SU(2)-specific gate, tomography, and workflow helpers remain explicit rather than being silently generalized as part of this model-layer refactor.
