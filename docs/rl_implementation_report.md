# RL-Ready Hybrid cQED Control — Implementation Report

**Module:** `cqed_sim.rl_control`, `cqed_sim.system_id`
**Date:** 2026-03-17
**Status:** First-pass MVP complete; scaffolding in place for future extensions.

---

## 1. Summary

This report documents the design and implementation of `cqed_sim.rl_control` and `cqed_sim.system_id`, which extend the `cqed_sim` library with a **simulator-trained reinforcement learning stack** for hybrid bosonic-ancilla control in circuit QED.

The implementation provides:

- A Gym-like RL environment (`HybridCQEDEnv`) with reset/step/rollout/diagnostics APIs.
- Two physics regimes: reduced dispersive and full pulse-level.
- Three action-space modes: parametric pulses, primitive actions, and waveform control.
- Multiple observation and reward modes, including measurement-realistic options.
- Episode-level domain randomization with separate train/eval distributions.
- Eight benchmark tasks spanning Tiers 1–4 of a hybrid cQED control ladder.
- Comprehensive unit, physics, and regression tests.
- Example script, tutorial notebook, and documentation.

The extension integrates cleanly with the existing `cqed_sim` simulation infrastructure, reusing models, solvers, pulse builders, frame conventions, measurement utilities, and calibration routines without duplicating them.

---

## 2. Phase 0 Audit Results

See [docs/phase0_audit_note.md](phase0_audit_note.md) for the full audit.

**Key findings:**

| Category | Status Before Extension |
|---|---|
| Core dispersive physics models | Complete (`DispersiveTransmonCavityModel`, `UniversalCQEDModel`) |
| Pulse building and sequence compilation | Complete |
| Open-system QuTiP mesolve solver | Complete |
| Qubit measurement with confusion matrix | Complete |
| Ideal gate library | Complete |
| Calibration targets (Rabi, Ramsey, T1, etc.) | Complete |
| Tomography and observables | Complete |
| RL environment | Missing — this extension |
| Domain randomization | Missing — this extension |
| System identification hooks | Missing — this extension |
| GRAPE optimal control | Added in recent commit |

---

## 3. Design Decisions

### 3.1 Modular digital-twin pipeline

The extension implements a five-stage pipeline, each with a clean interface:

```
action → PulseGenerator → DistortionModel → OpenSystemEngine
       → MeasurementModel → ClassicalProcessor → observation / reward
```

Each stage is a separate class in `runtime.py`, parameterized by `EpisodeModelBundle`. This allows any stage to be replaced independently for hardware-facing extensions.

### 3.2 Two physics regimes through configuration, not code branching

`HamiltonianModelFactory.build()` selects between:
- **Reduced dispersive** (`ReducedDispersiveModelConfig`): builds `DispersiveTransmonCavityModel` — fast, ideal for RL iteration.
- **Full pulse-level** (`FullPulseModelConfig`): builds `UniversalCQEDModel` with exchange coupling — slower, more physics-complete.

The RL environment code is identical for both regimes; the only change is the config object.

### 3.3 Action, observation, and reward as independently configurable layers

These three components are fully orthogonal. The environment accepts any action space with a `parse` and `flatten` interface, any observation model with an `encode` interface, and any reward model with a `compute` interface. The `build_observation_model` and `build_reward_model` factories provide the standard options.

### 3.4 Measurement realism at two levels

The environment inserts measurement realism at two independent points:
1. **Observations**: `MeasurementIQObservation` exposes integrated IQ, classifier logits, counts, or outcome one-hot vectors derived from `measure_qubit`.
2. **State collapse**: `collapse_on_measurement=True` applies projective collapse to the joint state, enabling measurement-feedback protocols.

These are independently configurable; an agent can receive IQ-like observations without collapsing the state.

### 3.5 Domain randomization via seeded prior sampling

`DomainRandomizer` holds separate train/eval priors for four categories: model parameters, noise rates, hardware distortion coefficients, and measurement spec fields. At each `reset()`, it samples a `RandomizationSample` using the episode seed, applying the sampled overrides to the config via `dataclasses.replace`. Per-episode metadata is recorded in `info["randomization"]`.

### 3.6 No new simulator: RL wraps the existing stack

The RL layer does not introduce a parallel simulation path. It reuses:
- `simulate_sequence` and `prepare_simulation` from `cqed_sim.sim`
- `SequenceCompiler` from `cqed_sim.sequence`
- `NoiseSpec` / `collapse_operators` from `cqed_sim.sim.noise`
- `measure_qubit` from `cqed_sim.measurement`
- Frame conventions from `cqed_sim.core.frame`

---

## 4. File-by-File Change List

### New files added

| File | Purpose |
|---|---|
| `cqed_sim/rl_control/__init__.py` | Public API exports for the rl_control subpackage |
| `cqed_sim/rl_control/env.py` | `HybridCQEDEnv` — Gym-like environment |
| `cqed_sim/rl_control/configs.py` | `ReducedDispersiveModelConfig`, `FullPulseModelConfig`, `HybridSystemConfig`, `HybridEnvConfig` |
| `cqed_sim/rl_control/action_spaces.py` | `ParametricPulseActionSpace`, `PrimitiveActionSpace`, `WaveformActionSpace` and action dataclasses |
| `cqed_sim/rl_control/observations.py` | All observation models and `build_observation_model` factory |
| `cqed_sim/rl_control/rewards.py` | All reward terms and `build_reward_model` factory |
| `cqed_sim/rl_control/tasks.py` | `HybridBenchmarkTask` and 8 task constructors |
| `cqed_sim/rl_control/domain_randomization.py` | `DomainRandomizer`, all prior types, `RandomizationSample` |
| `cqed_sim/rl_control/runtime.py` | `ControlSegment`, `EpisodeModelBundle`, `HamiltonianModelFactory`, `PulseGenerator`, `DistortionModel`, `OpenSystemEngine`, `MeasurementModel`, `ClassicalProcessor` |
| `cqed_sim/rl_control/metrics.py` | `state_fidelity`, `photon_number_distribution`, `parity_expectation`, `evaluate_state_task_metrics`, `evaluate_unitary_task_metrics`, and more |
| `cqed_sim/rl_control/diagnostics.py` | `build_rollout_diagnostics` |
| `cqed_sim/rl_control/demonstrations.py` | `scripted_demonstration`, `rollout_records`, `DemonstrationRollout` |
| `cqed_sim/rl_control/README.md` | Module-level README per AGENTS.md policy |
| `cqed_sim/system_id/__init__.py` | Public API for system identification |
| `cqed_sim/system_id/priors.py` | Re-exports prior types from rl_control |
| `cqed_sim/system_id/calibration_hooks.py` | `CalibrationEvidence`, `randomizer_from_calibration` |
| `tests/test_38_rl_control_env.py` | Core RL environment tests (6 tests) |
| `tests/test_39_rl_control_extensions.py` | RL extensions tests: tasks, observations, proxies (3 tests) |
| `tests/test_43_rl_physics_validation.py` | Physics validation tests: Hermiticity, Kerr, dispersive phase, leakage (10 tests) |
| `tests/test_44_rl_sanity_and_regression.py` | RL sanity and regression tests (18 tests) |
| `examples/rl_hybrid_control_rollout.py` | Standalone rollout example script |
| `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb` | Tutorial notebook |
| `docs/phase0_audit_note.md` | Phase 0 audit note |
| `docs/phase1_design_note.md` | Phase 1 architecture design note |
| `docs/rl_implementation_report.md` | This document |

### Modified files

| File | Change |
|---|---|
| `cqed_sim/__init__.py` | Added rl_control and system_id exports |
| `API_REFERENCE.md` | Added Section 17B covering RL control and system_id |
| `README.md` | Added `rl_control` / `system_id` to package layout section |
| `physics_and_conventions/physics_conventions_report.tex` | Added RL Wrapper Conventions section |

---

## 5. Benchmark Tasks

| Task | Tier | Kind | Success Threshold | Notes |
|---|---|---|---|---|
| `vacuum_preservation` | 1 | state_preparation | 0.999 | Trivial baseline: WaitAction(0) |
| `coherent_state_preparation` | 1 | state_preparation | 0.99 | Displacement baseline |
| `fock_1_preparation` | 1 | state_preparation | 0.92 | No baseline (SNAP needed) |
| `storage_superposition_preparation` | 1 | state_preparation | 0.92 | No baseline |
| `even_cat_preparation` | 2 | state_preparation | 0.90 | Requires entanglement and reset |
| `odd_cat_preparation` | 2 | state_preparation | 0.88 | Requires entanglement and reset |
| `ancilla_storage_bell` | 3 | state_preparation | 0.85 | Blue-sideband baseline |
| `conditional_phase_gate` | 4 | unitary_synthesis | 0.99 | Chi-dependent wait baseline |

---

## 6. Tests and Validation

### Test summary

| File | Tests | Status | Coverage |
|---|---|---|---|
| `test_38_rl_control_env.py` | 6 | All pass | Core env: reset, step, baseline, metrics, observation shapes |
| `test_39_rl_control_extensions.py` | 3 | All pass | Task structures, proxy rewards, diagnostics |
| `test_43_rl_physics_validation.py` | 10 | All pass | Physics: no-drive stationarity, photon loss, dispersive phase, Kerr, leakage, cat parity |
| `test_44_rl_sanity_and_regression.py` | ~18 | All pass | Sanity: rollouts finite, metadata, termination, regression fidelity |

**Total new tests:** ~37 tests added across all four files.

### Numerical tolerances

| Test | Tolerance |
|---|---|
| Vacuum state fidelity after free evolution | > 0.9999 |
| Photon loss after κt >> 1 | mean photon < 0.1 |
| Trace preservation under mesolve | error < 1e-6 |
| Dispersive pi-shift: overlap with initial | < 0.1 |
| Kerr pi-shift: fidelity with initial | < 0.1 |
| Cross-regime weak displacement fidelity (reduced) | > 0.95 |
| Cross-regime weak displacement fidelity (full) | > 0.90 |
| Vacuum preservation baseline regression | > 0.9999 |
| Coherent state baseline regression (alpha=0.2) | > 0.95 |

---

## 7. Known Limitations

| Limitation | Notes |
|---|---|
| **No Tier 5 tasks** | Drift-robust, adaptive, and feedback-conditioned tasks are not yet implemented. |
| **No SME / quantum jump trajectories** | Only Lindblad mesolve is available. Stochastic master equation (measurement-conditioned) evolution is scaffolded conceptually but not implemented. |
| **No posterior system identification** | `cqed_sim.system_id` provides only a lightweight `randomizer_from_calibration` hook. A full inference engine (e.g. Hamiltonian learning) must be connected externally. |
| **No IQ imbalance, LO leakage, or image sideband** | Hardware nonidealities beyond amplitude miscalibration, detuning drift, and confusion matrix assignment are not modeled. |
| **Single-process rollouts only** | `OpenSystemEngine.propagate_states` runs sequentially (`max_workers=1`). Parallelized or batched rollouts require external vectorization. |
| **Dense matrices only** | The solver uses dense QuTiP Qobj throughout. For large Hilbert spaces (N_cav > ~20), sparse or matrix-product-state backends would be necessary. |
| **No behavior-cloning pipeline** | `demonstrations.py` provides a record format, but a training-ready behavior cloning pipeline (dataset creation, policy training) must be built externally. |

---

## 8. Recommended Next Steps

### Near-term (high value, low effort)

1. **Add Tier 5 benchmark tasks**: implement a `drift_robust_calibration_task()` and a simple `feedback_reset_task()` using the `MeasurementAction` + `ResetAction` primitives.
2. **Improve noise coverage**: add `kappa_storage_only` tests and thermal photon tests to the physics validation suite.
3. **Add observation history to more tasks**: the `HistoryObservationWrapper` is implemented but not used in any default task configuration. Wire it in for the Tier 3–4 tasks.

### Medium-term

4. **Stochastic trajectories (SME)**: expose QuTiP `mcsolve` as an alternative backend in `OpenSystemEngine`. This requires adding `jump_ops` support and a trajectory-averaging wrapper.
5. **Posterior system identification**: connect `cqed_sim.calibration_targets` outputs (Ramsey, T1, dispersive spectroscopy) to `CalibrationEvidence` to enable the full fit-then-randomize workflow.
6. **Parallelized rollouts**: wrap `estimate_metrics` with a process-pool executor for faster sweep evaluation.

### Longer-term

7. **JAX-compatible backend**: add a JAX-differentiable dynamics backend for gradient-through-dynamics RL (e.g. analytic policy gradients).
8. **Sim-to-real transfer workflow**: connect the RL environment to a hardware execution backend to run trained policies on physical hardware.
9. **SNAP-family hierarchical action space**: build a higher-level action space that selects among validated SNAP + displacement + SQR composite blocks.

---

## 9. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                        HybridCQEDEnv                             │
│                                                                  │
│  reset(seed, options)                                            │
│    └─ HamiltonianModelFactory.build(system_config, randomization)│
│         ├─ ReducedDispersiveModelConfig → DispersiveTransmonCavityModel
│         └─ FullPulseModelConfig → UniversalCQEDModel            │
│                                                                  │
│  step(action)                                                    │
│    ├─ action_space.parse(action)                                 │
│    ├─ PulseGenerator.generate(parsed, bundle)→ ControlSegment   │
│    ├─ DistortionModel.compile(segment) → CompiledTimeline        │
│    ├─ OpenSystemEngine.propagate_state(state, compiled, ...)     │
│    │    └─ simulate_sequence (cqed_sim.sim)                      │
│    ├─ MeasurementModel.observe(state, ...)                       │
│    │    └─ measure_qubit (cqed_sim.measurement)                  │
│    ├─ reward_model.compute(state, metrics, measurement, ...)     │
│    └─ observation_model.encode(state, metrics, measurement, ...) │
│                                                                  │
│  diagnostics() → build_rollout_diagnostics(...)                  │
│  estimate_metrics(policy, n_rollouts) → distribution summaries   │
└──────────────────────────────────────────────────────────────────┘
```

---

*End of implementation report.*
