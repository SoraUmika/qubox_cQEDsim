# Phase 1 Design Note: Hybrid cQED RL Architecture

**Date:** 2026-03-17
**Scope:** `cqed_sim/rl_control/` and its integration with the broader `cqed_sim` stack
**Purpose:** Architectural rationale for the hybrid digital-twin RL pipeline

---

## Overview

Phase 1 establishes a modular digital-twin pipeline in which a physics-accurate cQED simulator serves as the environment for reinforcement learning agents. The central design principle is a strict separation between the physics layer (Hamiltonian construction, open-system dynamics, measurement) and the RL wrapper layer (action spaces, observation encoding, reward shaping, episode management). Neither layer has explicit knowledge of the other's internals; they communicate through well-defined interfaces.

---

## Architecture: Modular Digital-Twin Pipeline

The pipeline is organized into five sequential stages, each realized as a named component in `rl_control/runtime.py`:

1. **HamiltonianModelFactory** — constructs the system Hamiltonian and collapse operators from a `HybridSystemConfig`. Accepts domain-randomized parameter samples at episode reset. Decoupled from pulse details.
2. **PulseGenerator** — maps RL actions (parametric, primitive, or waveform) to `Pulse` objects and compiles them through `SequenceCompiler`. Action-space logic lives here, not in the environment.
3. **DistortionModel** — applies hardware transfer functions to compiled waveforms before integration. Isolates hardware-imperfection modeling from both the RL policy and the physics solver.
4. **OpenSystemEngine** — wraps `simulate_sequence` / `mesolve` and returns `SimulationResult`. The only component that calls QuTiP directly; the rest of the pipeline is QuTiP-agnostic.
5. **MeasurementModel** and **ClassicalProcessor** — apply confusion matrices, IQ sampling, and shot aggregation to simulation outputs; feed processed results back to the observation encoder.

`HybridCQEDEnv` in `env.py` orchestrates these stages. Its `reset` and `step` methods are the only public surface visible to an external RL agent.

---

## Physics Models vs. RL Wrappers

The physics stack (`cqed_sim/core/`, `cqed_sim/pulses/`, `cqed_sim/sim/`) is completely independent of `rl_control/`. This separation has three practical consequences:

- Physics models can be tested, calibrated, and validated without any RL machinery present.
- GRAPE (`cqed_sim/optimal_control/`) and RL share the same underlying physics models, so optimal-control results can be used as demonstrations or warm-start policies.
- Swapping the physics backend (e.g., replacing `mesolve` with a stochastic trajectory solver) requires changes only in `OpenSystemEngine` and `sim/solver.py`.

---

## Reduced and Full Models

Two model tiers coexist through configuration objects rather than code branching:

- `ReducedDispersiveModelConfig` (N_transmon=3, N_cavity=6): fast enough for high-episode-count RL training. Uses `DispersiveTransmonCavityModel`.
- `FullPulseModelConfig` (N_transmon=4, N_cavity=10): slower, includes higher-order Kerr and chi terms, used for policy validation and GRAPE. Uses `UniversalCQEDModel`.

`HybridSystemConfig` selects the active tier at construction time. The RL environment code does not branch on model type; the factory handles all differences. This allows the same policy to be evaluated under both tiers without modification.

---

## Measurement Realism Insertion Points

Measurement realism is inserted at two independently configurable points:

- **Readout chain** (`measurement/readout_chain.py`): models the physical resonator, Purcell filter, and amplifier chain. Output is an IQ signal distribution.
- **QubitMeasurementSpec** (`measurement/qubit.py`): applies a confusion matrix to map ideal projective outcomes to noisy binary results, including shot-by-shot sampling.

`MeasurementIQObservation` in `observations.py` can present raw IQ clusters to the agent, enabling agents that learn to be robust to readout noise. `IdealSummaryObservation` and `ReducedDensityObservation` bypass readout noise for rapid prototyping or ablation studies. The `build_observation_model` factory selects among these at configuration time.

---

## Domain Randomization and Reproducibility

`DomainRandomizer` in `domain_randomization.py` holds separate prior distributions for training and evaluation. Supported prior types are `FixedPrior`, `UniformPrior`, `NormalPrior`, and `ChoicePrior`, composable per parameter. At each episode reset, `HybridCQEDEnv` draws a `RandomizationSample` from the active prior using a seeded RNG. The seed is recorded in episode metadata, making any episode exactly reproducible. Training and evaluation priors are independently specified, which allows controlled distribution-shift studies without touching environment or policy code.

---

## System Identification Hooks

`cqed_sim/system_id/` is positioned between real-device calibration data and domain randomization:

- `CalibrationEvidence` holds calibration measurements (Rabi rates, chi shifts, T1/T2 values).
- `randomizer_from_calibration` is intended to construct a `DomainRandomizer` whose priors reflect posterior uncertainty over physical parameters inferred from that evidence.

Once an inference engine (Bayesian or MCMC) is connected, the RL training distribution will automatically tighten around experimentally constrained parameter regions. No changes to `HybridCQEDEnv` or policy code are required; only `system_id/` needs to be populated.

---

## Scaffolded vs. Fully Realized

| Component | Status |
|---|---|
| `HybridCQEDEnv` (reset/step/rollout) | Fully realized |
| 8 benchmark tasks (Tiers 1-4) | Fully realized |
| Parametric and primitive action spaces | Fully realized |
| All observation model variants | Fully realized |
| Composite reward with all penalty terms | Fully realized |
| Domain randomization (train/eval priors) | Fully realized |
| Scripted demonstrations and GRAPE integration | Fully realized |
| Tier 5 adaptive drift/feedback tasks | Scaffolded only |
| SME / quantum jump trajectory backend | Scaffolded only |
| Posterior inference in system_id | Scaffolded only |
| IQ imbalance and LO leakage in hardware model | Not yet started |

---

## Performance and Numerical Strategy

- The Hilbert space dimension is the dominant cost. The reduced model (dim = 3 x 6 = 18) keeps per-episode wall time tractable for standard RL episode budgets. The full model (dim = 4 x 10 = 40) is reserved for validation.
- `SequenceCompiler` caching is disabled in the RL runtime (`enable_cache=False`) to prevent waveform reuse across episodes with different domain-randomization draws.
- Tensor ordering is qubit-first (qubit tensor-product storage) throughout, consistent with the QuTiP convention adopted in `cqed_sim/core/`. All extractors and reward metrics assume this ordering; new components must respect it.
- Internal units are rad/s for all frequencies and seconds for all times. Conversion to/from lab-frame MHz happens at the configuration boundary, not inside physics or RL modules.
