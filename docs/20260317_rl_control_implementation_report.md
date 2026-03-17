# RL Control Implementation Report

## Summary

This implementation extends `cqed_sim` with a first-pass RL-ready hybrid bosonic-ancilla control stack built on top of the existing simulator architecture rather than as a disconnected side project. The new surface supports both state-preparation and gate-oriented tasks, exposes a Gym-like environment API, reuses the package's existing pulse and simulation layers, and adds benchmark tasks, measurement-facing observation and reward modes, richer diagnostics, tests, examples, and documentation.

## Phase 0 Audit Results

The initial repository survey found that the simulator already had most of the low-level pieces needed for a serious RL-facing control layer:

- `UniversalCQEDModel`, `DispersiveTransmonCavityModel`, and `DispersiveReadoutTransmonStorageModel` already covered the core Hamiltonian regimes.
- `Pulse`, `SequenceCompiler`, `HardwareConfig`, `SimulationSession`, `simulate_sequence(...)`, and `NoiseSpec` already provided the control-compilation and propagation seam.
- `QubitMeasurementSpec`, `measure_qubit(...)`, and the readout-chain code already provided a reusable measurement abstraction for measurement-like observations.
- The unitary-synthesis and calibration layers already had growing support for benchmark-style objective evaluation, logical-subspace handling, and reusable pulse-level primitives.

The gaps were mainly at the orchestration layer: action parsing, observation encoding, reward composition, episode state management, domain randomization, task registries, and calibration-informed hooks.

## Design Decisions

The implementation follows these design choices:

- One environment interface, two physics regimes. `HybridCQEDEnv` accepts a `HybridSystemConfig` that selects either a reduced dispersive path or a fuller multilevel pulse path.
- Clear layering. Action decoding, pulse generation, distortion/compilation, evolution, measurement, observations, rewards, metrics, and diagnostics are separate modules.
- Native simulator reuse. The RL layer calls the same model, pulse, and measurement abstractions already used elsewhere in `cqed_sim`.
- Measurement-aware but not trajectory-heavy. The first pass supports measurement-like observations and optional collapse-on-measurement, but does not yet implement full stochastic master-equation trajectories.
- Measurement-proxy objectives are explicit. In addition to simulator-shaped fidelity rewards, the public reward builder now exposes a measurement-assignment proxy mode together with classifier-logit and one-hot observation views.
- Domain randomization as a first-class interface. The environment samples hidden episode parameters from train/eval priors and exposes the sampled metadata.

## File-Level Change Summary

### New RL-facing modules

- `cqed_sim/rl_control/action_spaces.py`
- `cqed_sim/rl_control/configs.py`
- `cqed_sim/rl_control/domain_randomization.py`
- `cqed_sim/rl_control/runtime.py`
- `cqed_sim/rl_control/observations.py`
- `cqed_sim/rl_control/rewards.py`
- `cqed_sim/rl_control/metrics.py`
- `cqed_sim/rl_control/diagnostics.py`
- `cqed_sim/rl_control/tasks.py`
- `cqed_sim/rl_control/demonstrations.py`
- `cqed_sim/rl_control/env.py`
- `cqed_sim/rl_control/__init__.py`

### Calibration and synthesis support added alongside the RL stack

- `cqed_sim/calibration/conditioned_multitone.py`
- `cqed_sim/calibration/targeted_subspace_multitone.py`
- `cqed_sim/unitary_synthesis/fast_eval.py`
- related tests and examples for logical-block phase, relevance-aware objectives, and flexible target actions

### System-identification scaffolding

- `cqed_sim/system_id/priors.py`
- `cqed_sim/system_id/calibration_hooks.py`
- `cqed_sim/system_id/__init__.py`

### Public package export updates

- `cqed_sim/__init__.py` now reexports the RL and system-ID public surface.

### Examples, tutorials, and reports

- `examples/rl_hybrid_control_rollout.py`
- `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`
- `docs/20260317_rl_control_phase0_audit.md`
- `docs/20260317_rl_control_phase1_design_plan.md`
- this implementation report

### Follow-on benchmark and observation hardening

- `cqed_sim/rl_control/tasks.py` now also provides Fock-state, storage-superposition, and odd-cat benchmarks.
- `cqed_sim/rl_control/runtime.py` and `cqed_sim/rl_control/observations.py` now expose classifier-logit and one-hot outcome observation encodings.
- `cqed_sim/rl_control/rewards.py` now exposes a measurement-assignment proxy reward mode.
- `cqed_sim/rl_control/diagnostics.py` now reports segment metadata and pulse summaries alongside compiled channels.

## Important Fixes During Validation

Three issues were found and fixed during the validation pass:

1. The RL observation-model builder had a keyword collision around `mode`, which broke measurement-like observation construction. The selector argument was renamed to `observation_mode`.
2. The full-model RL runtime path double-counted transmon-storage coupling structure by inserting both dispersive-style and cross-Kerr-style terms. The duplicate insertion was removed.
3. The environment API exposed `diagnostics()` and `estimate_task_metrics(...)`, but the requested API shape wanted `render_diagnostics()` and `estimate_metrics(...)`. Compatibility aliases were added.

## Tests And Validation

The following targeted validation steps were completed during implementation:

- The script example `examples/rl_hybrid_control_rollout.py` was executed successfully.
- `tests/test_37_api_export_completeness.py` passed after the top-level export update.
- `tests/test_38_rl_control_env.py` passed, covering reduced-regime rollout behavior, deterministic seeding, action clipping, train/eval randomization separation, unitary-task metrics, and a full-pulse measurement-observation smoke path.
- `tests/test_39_rl_control_extensions.py` adds regression coverage for the extended benchmark ladder, measurement-facing observation aliases, proxy rewards, and richer diagnostics payloads.

The validated script example produced representative output showing:

- baseline total reward of about `0.8225`
- baseline state fidelity of about `0.99995`
- held-out evaluation mean state fidelity of about `0.99782`

Those figures are not a formal benchmark claim. They are a smoke-level confirmation that the environment, proxy reward, and diagnostic stack are numerically wired correctly on the reference coherent-state task.

## Documentation Updates

The following documentation surfaces were updated to include the new RL stack:

- `README.md`
- `API_REFERENCE.md`
- `documentations/api/overview.md`
- `documentations/api/rl_control.md`
- `documentations/examples.md`
- `documentations/tutorials/index.md`
- `documentations/tutorials/rl_hybrid_control.md`
- `documentations/architecture.md`
- `documentations/physics_conventions.md`
- `physics_and_conventions/physics_conventions_report.tex`
- `mkdocs.yml`

## Known Limitations

- The first pass does not yet provide a full training loop, replay buffer, or optimizer integration. It focuses on the environment and physics interface layer.
- Measurement-conditioned stochastic trajectories and richer feedback-control semantics are not yet implemented.
- The waveform action path is present as scaffolding and interface support, but the intended first training mode is still the lower-dimensional parametric or primitive action space.
- The full multilevel regime is validated as a smoke path, not yet as an aggressively benchmarked production-scale RL workload.

## Recommended Next Steps

1. Add a policy-training harness layer that can plug into stable external RL trainers without forcing a hard dependency into `cqed_sim`.
2. Add benchmark suites for longer-horizon bosonic code tasks, feedback-assisted tasks, and robustness sweeps over broader posterior samples.
3. Extend the measurement layer toward trajectory-level feedback and partially observed control workflows.
4. Add performance profiling and solver-path selection for larger full-pulse RL studies.