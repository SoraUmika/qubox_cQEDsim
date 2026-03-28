# Generalized Holographic Step-Unitary Report

## Scope

This report documents the March 27, 2026 generalization of the holographic
simulation workflow in `cqed_sim` from a repeated single-channel public API to a
finite explicit per-step unitary workflow that still remains mathematically
consistent with the existing MPS and Stinespring constructions.

## Old Limitation

The public sampler path in `cqed_sim.quantum_algorithms.holographic_sim` was
centered on one translation-invariant channel:

- `HolographicSampler` stored a single `HolographicChannel`.
- `sample_correlator(...)` and `enumerate_correlator(...)` reused that same
  channel at every step.
- the legacy `holographicSim.py` compatibility wrapper already accepted a list
  of full joint unitaries, but that capability was not exposed through a
  first-class public sequence abstraction.

In other words, the implementation already knew how to process a finite list of
resolved joint unitaries, but the reusable public API stopped at one repeated
channel.

## Code Paths Changed

The main implementation changes are:

- `cqed_sim/quantum_algorithms/holographic_sim/step_unitary.py`
  - adds `StepUnitarySpec`
  - defines explicit `joint`, `physical`, and `bond` embeddings in `physical ⊗ bond` order
- `cqed_sim/quantum_algorithms/holographic_sim/step_sequence.py`
  - adds `HolographicChannelSequence`
  - supports construction from explicit unitary lists and from MPS-derived site tensors
- `cqed_sim/quantum_algorithms/holographic_sim/channel.py`
  - extends `HolographicChannel.from_unitary(...)` with `acts_on`
- `cqed_sim/quantum_algorithms/holographic_sim/sampler.py`
  - generalizes the public sampler and exact enumerator to accept either one
    repeated channel or a validated finite sequence
  - adds `from_unitary_sequence(...)` and `from_mps_sequence(...)`
- `cqed_sim/quantum_algorithms/holographic_sim/mps.py`
  - adds `site_stinespring_unitaries(...)`
  - adds `to_holographic_channel_sequence(...)`
- `tests/quantum_algorithms/test_holographic_sim.py`
  - adds regression coverage for subsystem embedding, finite-sequence
    validation, and sampled-vs-exact sequence agreement
- `tests/quantum_algorithms/test_holographic_mps_validation.py`
  - adds public MPS-sequence validation against dense expectations

## New Public Abstraction

The generalized interface is now:

1. `StepUnitarySpec`
   - wraps one step unitary and states whether it acts on the full joint space,
     only the physical register, or only the bond register.
2. `HolographicChannelSequence`
   - stores a finite validated ordered list of step channels.
3. `HolographicSampler`
   - accepts either one `HolographicChannel` or one
     `HolographicChannelSequence`.

The subsystem embedding rule is fixed and explicit:

$$
U_{\mathrm{joint}} = U_{\mathrm{physical}} \otimes I_{\mathrm{bond}}
\quad \text{or} \quad
U_{\mathrm{joint}} = I_{\mathrm{physical}} \otimes U_{\mathrm{bond}},
$$

with the tensor ordering always taken as

$$
|\sigma\rangle_{\mathrm{physical}} \otimes |b\rangle_{\mathrm{bond}}.
$$

## Mathematical Assumptions

The implementation preserves the existing holographic conventions:

- physical register first, bond register second
- default physical reference state $|0\rangle$
- Kraus extraction

$$
K_\sigma = \langle \sigma | U | 0 \rangle
$$

- right-canonical MPS tensors mapped through

$$
K_\sigma = V_\sigma^\dagger
$$

- dense Stinespring completion consistent with the same physical-input
  convention.

`burn_in` keeps its original meaning: repeated application of the same channel
before measurement insertions. Because of that definition, nonzero `burn_in` is
allowed for translation-invariant single-channel workflows and intentionally
disallowed for finite explicit step sequences.

## Worked Example

The canonical worked example lives in:

- `examples/quantum_algorithms/holographic_generalized_unitary_workflow.py`

### Exact Initial State

The exact computational-basis seed state is

$$
|1011\rangle.
$$

### Exact Random-MPS Construction

The example does not use random sampling to define the MPS. Instead, the state
structure itself is randomized deterministically:

1. Start from $|1011\rangle$.
2. Apply four site-local Haar-random $SU(2)$ rotations with seed `12345`.
3. Apply nearest-neighbor partial-swap entanglers with angles
   $(0.41, -0.33, 0.27)$.
4. Convert the resulting normalized dense state into a right-canonical MPS.
5. Complete each site tensor to the square right-isometry form and then to the
   dense extended unitary used by the finite holographic sequence.

The completed tensors all have shape `(4, 2, 4)`, and the lifted dense step
unitaries all have shape `(8, 8)`.

### Observables and Correlations Tested

The example checks:

- single-site expectations: `Z0`, `X1`, `Z2`
- two-site correlations: `Z0Z2`, `X1Z3`
- connected correlator:

$$
C_{Z_0,Z_2} = \langle Z_0 Z_2 \rangle - \langle Z_0 \rangle \langle Z_2 \rangle
$$

### Shot Count

- primary validation run: `100000` shots
- mixed-embedding stress test: `80000` shots

## Validation Results

### Structural Correctness

The per-site structural checks were:

| Site | Tensor shape | Joint unitary shape | Kraus completeness error | Right-isometry error | Unitary error |
|---|---|---|---:|---:|---:|
| 0 | `(4, 2, 4)` | `(8, 8)` | `4.65e-16` | `4.65e-16` | `5.65e-16` |
| 1 | `(4, 2, 4)` | `(8, 8)` | `8.96e-16` | `8.96e-16` | `1.04e-15` |
| 2 | `(4, 2, 4)` | `(8, 8)` | `1.05e-15` | `1.05e-15` | `1.17e-15` |
| 3 | `(4, 2, 4)` | `(8, 8)` | `8.78e-17` | `8.78e-17` | `2.65e-16` |

These values are all at numerical roundoff level.

### Observable Agreement

The dense, MPS, and exact extended-unitary values agreed to machine precision.
The many-shot sampled values were:

| Observable | Exact | Sampled | Std. error | Absolute error |
|---|---:|---:|---:|---:|
| `Z0` | `-0.021091` | `-0.025180` | `0.003161` | `0.004089` |
| `X1` | `0.567432` | `0.570240` | `0.002598` | `0.002808` |
| `Z2` | `-0.610576` | `-0.618100` | `0.002486` | `0.007524` |
| `Z0Z2` | `-0.072077` | `-0.073260` | `0.003154` | `0.001183` |
| `X1Z3` | `0.054934` | `0.055300` | `0.003157` | `0.000366` |
| `Connected(Z0,Z2)` | `-0.084954` | `-0.088824` | `0.003711` | `0.003870` |

The worst normalized deviation in the primary example was the `Z2` observable,
whose sampled estimate differed from the exact value by roughly `3.03` standard
errors. The rest of the listed observables were closer.

### Correlation Agreement

The connected correlator remained clearly nonzero:

$$
C_{Z_0,Z_2}^{\mathrm{exact}} = -0.084954,
\qquad
C_{Z_0,Z_2}^{\mathrm{sampled}} = -0.088824 \pm 0.003711.
$$

That confirms the workflow is not only reproducing local one-point functions; it
also preserves nontrivial correlation structure through the MPS,
right-isometry, extended-unitary, exact holographic, and sampled holographic
representations.

### Stress Test

The second run used genuinely mixed step semantics:

1. `physical_rx` acting only on the physical register
2. `bond_rz` acting only on the bond register
3. `joint_partial_swap` acting on the full `physical ⊗ bond` space
4. `physical_rx_tail` acting only on the physical register

The exact sequence result was `1.4745e-17`, the legacy exact branch table gave
the same value, and the sampled estimate was

$$
0.001725 \pm 0.003536.
$$

This is the clearest direct evidence that the generalized public interface is
working as intended rather than only for the special repeated-channel case.

## Residual Limitations

- `burn_in` still applies only to repeated single-channel workflows.
- the public sequence API is finite and explicit; it does not yet introduce a
  higher-level symbolic circuit IR for holographic programs.
- the current validation is dense and small-scale by design; it is intended as
  a correctness reference, not as a large-scale tensor-network benchmark.