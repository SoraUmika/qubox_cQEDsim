# Holographic Quantum Algorithms

This subpackage provides a generic, reusable foundation for the holographic
quantum-algorithm architecture described in
`paper_summary/holographic_quantum_algorithms.pdf`.

## Core vocabulary

- `physical register`: the small register prepared, coupled to the bond system,
  and optionally measured at each step
- `bond register`: the persistent Hilbert space that carries the effective
  many-body or MPS boundary information
- `holographic channel`: the CPTP map induced on the bond register by preparing
  the physical register in a reference state, applying a joint unitary, and
  tracing or conditioning on the physical subsystem
- `burn-in`: repeated channel application before measurements, used to approach
  bulk or steady-state behavior
- `observable schedule`: explicit insertion plan for where physical observables
  are measured during a holographic run

## Main public entry points

- `BondNoiseChannel`
- `HolographicChannel`
- `HolographicChannelSequence`
- `ObservableSchedule`
- `HolographicSampler`
- `HolographicMPSAlgorithm`
- `HoloVQEObjective`
- `HoloQUADSProgram`
- `StepUnitarySpec`
- `right_canonical_tensor_to_stinespring_unitary`

## Terminology mapping from the report

| Report concept | Implementation |
|---|---|
| bond-space transfer channel | `HolographicChannel` |
| optional bond-only noise | `BondNoiseChannel` |
| purified / Stinespring embedding | `PurifiedChannelStep` |
| finite per-step holographic program | `HolographicChannelSequence`, `StepUnitarySpec` |
| measurement insertion pattern | `ObservableSchedule` |
| holographic Monte Carlo estimator | `HolographicSampler.sample_correlator(...)` |
| exact small-system branch table | `HolographicSampler.enumerate_correlator(...)` |
| right-canonical MPS tensor / transfer matrix | `MatrixProductState`, `HolographicChannel.from_right_canonical_mps(...)`, `HolographicChannel.from_mps_state(...)` |
| dense Stinespring completion | `right_canonical_tensor_to_stinespring_unitary(...)`, `MatrixProductState.site_stinespring_unitary(...)` |
| holoVQE energy decomposition | `HoloVQEObjective` + `EnergyTerm` |
| holoQUADS time slicing | `HoloQUADSProgram` + `TimeSlice` |

## Scope implemented now

- dense-unitary, Kraus, and right-canonical MPS channel construction
- direct dense-state to channel / sampler convenience via `from_mps_state(...)`
- finite non-translation-invariant step sequences via `HolographicChannelSequence.from_unitaries(...)`, `HolographicSampler.from_unitary_sequence(...)`, and `HolographicSampler.from_mps_sequence(...)`
- explicit subsystem embedding for per-step unitaries through `StepUnitarySpec(acts_on="joint" | "physical" | "bond")`
- Monte Carlo and exact branch enumeration
- explicit burn-in and right-boundary postselection
- optional bond-space noise maps, including computational-basis dephasing,
  target-state amplitude damping, and depolarizing noise
- QuTiP-superoperator import for bond-noise channels via `BondNoiseChannel.from_qutip_super(...)`
- diagnostics for unitarity, completeness, trace preservation, and normalization
- example channel constructors and spin-inspired helpers
- backward-compatible wrappers in `holographicSim.py`

Built-in `BondNoiseChannel` constructors now cover three common wrapped cases:

- `BondNoiseChannel.dephasing(...)` for computational-basis coherence damping
- `BondNoiseChannel.amplitude_damping(...)` for computational-basis relaxation
  into a designated target basis state; for `bond_dim=2` this reduces to the
  standard qubit amplitude-damping channel
- `BondNoiseChannel.depolarizing(...)` for mixing toward the maximally mixed
  bond state

## Intentionally deferred

- full holographic variational optimization loops
- hardware-aware holographic backends beyond the current optional bond-noise layer
- large-scale MPS tooling beyond the included right-canonical helpers

Finite per-step sequences are now first-class public objects. The remaining
translation-invariant restriction is narrower: `burn_in` still means repeated
application of one fixed channel before measurements begin, so it is supported
for single-channel workflows but not for finite step sequences whose steps are
all explicitly specified.

The current design keeps those extensions natural without hardcoding the package
to any one hardware model or paper benchmark.

## Development Notes

The holographic code in this package was developed from a prototype that hardcoded
qubit-system assumptions (`SYS_DIM = 2`) and returned low-level tuples/arrays rather
than structured API objects. The current architecture decomposes into five layers:

1. Physics/model helpers — lightweight spin-style model helpers and example channels
2. MPS/channel layer — `HolographicChannel` from unitary, Kraus, or right-canonical MPS data; `PurifiedChannelStep`
3. Schedule/observable layer — explicit observable insertions, burn-in config, optional right-boundary postselection
4. Execution/inference layer — Monte Carlo sampler, exact branch enumeration, structured result objects
5. Diagnostics layer — CPTP/completeness/normalization checks, Monte Carlo vs exact comparison helpers

Future extension scaffolding includes `HoloVQEObjective` and `HoloQUADSProgram` for variational
energy objectives and time-slice/causal-cone work respectively.

### Integration with `cqed_sim`

Reused from the broader library: documentation style, public API export conventions, and
dataclass/result-object patterns. The holographic channel and MPS abstractions are
intentionally kept self-contained because the module is conceptually generic and should
not inherit cQED-only model or pulse abstractions. Integration is at the packaging and
documentation level; internal math remains backend-agnostic and hardware-agnostic.

### Deferred work

- Hardware-target circuit IR / backend compilation layers
- Noisy hardware execution models and leakage-aware hardware mitigation
- Full optimizer stack for holoVQE
- Full time-evolution engine for holoQUADS
- Advanced entropy estimation protocols (replica-SWAP workflows)
- Large-scale benchmark reproduction against the paper
