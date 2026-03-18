# Phase 0 Audit Note: cQED Simulation Codebase

**Date:** 2026-03-17
**Scope:** `cqed_sim` package — circuit-QED simulation library built on QuTiP 5
**Purpose:** Baseline inventory before RL extension development

---

## Summary

The `cqed_sim` package is a well-structured, physics-complete simulation library for superconducting circuit-QED systems. It covers the full stack from Hamiltonian construction through pulse compilation, open-system dynamics, measurement modeling, and reinforcement-learning environment wrapping. As of this audit, 448 tests pass (2 skipped).

---

## What Exists

### Core Physics

`cqed_sim/core/` provides three Hamiltonian models of increasing generality:

- `model.py` — `DispersiveTransmonCavityModel`: two-mode dispersive model (qubit + cavity), parameters omega_c, omega_q, alpha, chi, kerr, and higher-order corrections.
- `universal_model.py` — `UniversalCQEDModel`: multilevel transmon with arbitrary bosonic modes, CrossKerrSpec, ExchangeSpec.
- `readout_model.py` — `DispersiveReadoutTransmonStorageModel`: three-mode (qubit + storage + readout).

Supporting infrastructure: rotating frame definitions (`frame.py`), dressed-frequency computations (`frequencies.py`), ideal gate unitaries for displacement, SNAP, SQR, and qubit rotations (`ideal_gates.py`), state preparation utilities (`state_prep.py`), and drive-target specifications (`drive_targets.py`).

### Pulse and Simulation Stack

`cqed_sim/pulses/` defines the `Pulse` dataclass with full envelope, carrier, DRAG, and phase fields. Over 20 envelope families are implemented (`envelopes.py`). Builder functions (`builders.py`) construct rotation, displacement, and sideband pulses. `HardwareConfig` models distortion.

`cqed_sim/sequence/` compiles pulse sequences into timestamped timelines with hardware distortions applied. `cqed_sim/sim/` wraps QuTiP `mesolve` with `SimulationConfig`, `SimulationResult`, and `SimulationSession` abstractions, batch execution support, Lindblad collapse operators for T1/T2/cavity loss/thermal noise, and 25+ state extractors (Wigner function, Bloch vector, conditioned observables, photon-number distributions).

### Measurement and Readout

`cqed_sim/measurement/` implements confusion-matrix-based qubit measurement with IQ sampling and shot statistics. `ReadoutChain` models the resonator, Purcell filter, and amplifier chain.

### Calibration, Analysis, and Tomography

Calibration targets exist for Rabi, Ramsey, T1, T2, spectroscopy, and DRAG. `cqed_sim/tomo/` provides Fock tomography, all-XY sequences, and leakage calibration. `cqed_sim/analysis/` handles bare-to-dressed parameter translation.

### RL Control Module

`cqed_sim/rl_control/` is fully scaffolded and substantially implemented:

- `env.py` — `HybridCQEDEnv` Gym-compatible environment with reset, step, rollout, diagnostics, and metric estimation.
- `configs.py` — `ReducedDispersiveModelConfig`, `FullPulseModelConfig`, `HybridSystemConfig`, `HybridEnvConfig`.
- `action_spaces.py` — parametric (hybrid_block, Gaussian, displacement, sideband), primitive, and waveform action spaces.
- `observations.py` — ideal summary, reduced density matrix, IQ measurement, and gate-metric observation models with a history wrapper.
- `rewards.py` — state fidelity, process fidelity, Wigner samples, parity, leakage penalty, ancilla-return penalty, control cost, and composite reward composition.
- `tasks.py` — `HybridBenchmarkTask` plus 8 concrete tasks spanning vacuum preservation through conditional phase gates.
- `domain_randomization.py` — fixed, uniform, normal, and choice priors; seeded `DomainRandomizer` for reproducible train/eval splits.
- `runtime.py` — modular pipeline: `HamiltonianModelFactory`, `PulseGenerator`, `DistortionModel`, `OpenSystemEngine`, `MeasurementModel`, `ClassicalProcessor`.
- `metrics.py`, `demonstrations.py`, `diagnostics.py` — evaluation metrics, scripted demonstrations, and rollout diagnostics.

### Optimal Control and System Identification

`cqed_sim/optimal_control/` implements GRAPE (`GrapeSolver`, `GrapeConfig`, `GrapeResult`) with objectives and penalties. `cqed_sim/system_id/` scaffolds `CalibrationEvidence` and `randomizer_from_calibration` for posterior-informed domain randomization.

---

## Gaps and Missing Pieces

| Gap | Notes |
|---|---|
| Tier 5 benchmark tasks (adaptive drift/feedback) | Scaffolded in `tasks.py`; logic not yet implemented |
| Stochastic master equation / quantum jump solver | Only Lindblad `mesolve`; no SME or Monte Carlo trajectories |
| Posterior sampling in system_id | Interface present; no inference engine behind it |
| IQ imbalance, LO leakage, image sideband | Not modeled in `HardwareConfig` |
| Additional physics unit tests | Hermiticity, trace preservation, Kerr/chi evolution, leakage bounds |
| Additional RL sanity tests | Random rollout smoke tests, termination logic, episode metadata checks |
| Baseline regression tests | No numerical threshold fixtures yet |

---

## Clean Extension Points

- **New tasks:** add a `TaskSpec` subclass to `tasks.py` and register with `HybridBenchmarkTask`.
- **New observation models:** implement the `ObservationModel` protocol in `observations.py`.
- **New reward terms:** subclass `RewardTerm` in `rewards.py` and compose with `CompositeReward`.
- **SME solver:** add a trajectory backend to `sim/solver.py` behind the existing `solve_with_backend` interface.
- **Posterior system-id:** populate `system_id/` with a Bayesian or MCMC inference engine; `DomainRandomizer` already accepts the resulting prior objects.
- **Hardware imperfections:** extend `HardwareConfig` and `DistortionModel` with IQ/LO/sideband terms without touching higher layers.

---

## Performance Notes

- Primary bottleneck: QuTiP `mesolve` on the full density matrix (dim = N_transmon x N_cavity).
- `ReducedDispersiveModelConfig` (N_tr=3, N_cav=6) is fast enough for practical RL episode budgets.
- `FullPulseModelConfig` (N_tr=4, N_cav=10) is slower but more physics-complete; suitable for validation and GRAPE.
- `SequenceCompiler` cache is disabled (`enable_cache=False`) in the RL runtime to avoid stale waveforms across episodes.
