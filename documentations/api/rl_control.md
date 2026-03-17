# API Reference: RL Control And System Identification (`cqed_sim.rl_control`, `cqed_sim.system_id`)

This page documents the first-pass RL-ready hybrid bosonic-ancilla control layer built on top of the existing `cqed_sim` runtime.

!!! note "Design boundary"
    The RL package does not introduce a second simulator stack. It reuses the same model, pulse, sequence, noise, and measurement layers already documented elsewhere in the API reference.

---

## Core Environment Surface

```python
env = HybridCQEDEnv(config)
obs, info = env.reset(seed=..., options=...)
obs, reward, terminated, truncated, info = env.step(action)
diagnostics = env.render_diagnostics()
metrics = env.estimate_metrics(policy_or_actions, ...)
```

```python
@dataclass
class HybridEnvConfig:
    system: HybridSystemConfig
    task: Any
    action_space: Any
    observation_model: Any
    reward_model: Any
    randomizer: Any | None = None
    randomization_mode: Literal["train", "eval"] = "train"
    episode_horizon: int = 4
    measurement_spec: QubitMeasurementSpec | None = None
    collapse_on_measurement: bool = False
    auto_measurement: bool = False
```

Key points:

- `reset(...)` seeds both the environment and the episode-level randomization path deterministically.
- `step(...)` follows the same physical order every time: action parsing -> pulse generation -> distortion/compilation -> propagation -> optional measurement -> observation/reward.
- `render_diagnostics()` intentionally exposes simulator-only information that is richer than the policy observation.
- `estimate_metrics(...)` rolls out either a supplied policy, a supplied action sequence, or the task baseline across multiple seeds and summarizes the resulting distributions.

---

## Physics Regimes

```python
@dataclass(frozen=True)
class ReducedDispersiveModelConfig:
    omega_c: float
    omega_q: float
    alpha: float
    chi: float
    kerr: float = 0.0
    n_cav: int = 10
    n_tr: int = 3

@dataclass(frozen=True)
class FullPulseModelConfig:
    omega_c: float
    omega_q: float
    alpha: float
    exchange_g: float
    kerr: float = 0.0
    cross_kerr: float = 0.0
    n_cav: int = 10
    n_tr: int = 4
```

```python
@dataclass
class HybridSystemConfig:
    regime: Literal["reduced_dispersive", "full_pulse"] = "reduced_dispersive"
    reduced_model: ReducedDispersiveModelConfig | None = None
    full_model: FullPulseModelConfig | None = None
    frame: FrameSpec = FrameSpec()
    use_model_rotating_frame: bool = True
    noise: NoiseSpec | None = None
    hardware: dict[str, HardwareConfig] = field(default_factory=dict)
    crosstalk_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    dt: float = 4.0e-9
    max_step: float | None = None
```

Key points:

- The reduced regime is the fast RL iteration path for dispersive/Kerr tasks.
- The full regime is the richer multilevel pulse path and uses the same `UniversalCQEDModel` infrastructure as the rest of the package.
- If `use_model_rotating_frame=True` and the `FrameSpec` is left at zero, the environment convenience layer adopts the model's bare storage/qubit frequencies as the working frame.

---

## Action Spaces

### Parametric actions

```python
ParametricPulseActionSpace(family="qubit_gaussian")
ParametricPulseActionSpace(family="cavity_displacement")
ParametricPulseActionSpace(family="sideband")
ParametricPulseActionSpace(family="hybrid_block")
```

### Primitive actions

```python
PrimitiveActionSpace(
    primitives=("qubit_gaussian", "cavity_displacement", "sideband", "wait", "measure", "reset")
)
```

### Waveform actions

```python
WaveformActionSpace(segments=16, channels=("qubit", "storage"))
```

Key points:

- Parametric actions are the intended default because they remain low-dimensional and physically interpretable.
- Primitive actions align with validated control blocks and make scripted baselines simple to express.
- Waveform actions are available as a scaffold for future higher-bandwidth control studies and are not the recommended first training mode.

---

## Observations And Rewards

```python
build_observation_model("ideal_summary")
build_observation_model("reduced_density")
build_observation_model("measurement_iq", mode="iq_mean")
build_observation_model("measurement_classifier_logits")
build_observation_model("measurement_outcome")
build_observation_model("gate_metrics")
```

```python
build_reward_model("state")
build_reward_model("gate")
build_reward_model("cat")
build_reward_model("measurement_proxy")
```

Key points:

- The ideal observation modes are intended for debugging and simulator-side algorithm development.
- Measurement-like observations are built from `QubitMeasurementSpec` and inherit the same measurement conventions as `cqed_sim.measurement`, including IQ summaries, counts, classifier logits, and one-hot noisy outcomes.
- Reward builders combine shaped task terms with leakage, ancilla-return, and control-cost penalties, and can also expose explicit measurement-assignment proxy rewards.

---

## Diagnostics And Metrics

The environment metrics layer includes:

- `evaluate_state_task_metrics(...)`
- `evaluate_unitary_task_metrics(...)`
- `state_fidelity(...)`
- `ancilla_return_metric(...)`
- `parity_expectation(...)`
- `photon_number_distribution(...)`
- `sparse_wigner_samples(...)`
- `summarize_distribution(...)`

`render_diagnostics()` packages those metrics together with simulator-side state, reduced states, ancilla populations, compiled-channel payloads, segment metadata, pulse summaries, frame metadata, and optional Wigner diagnostics.

---

## Benchmark Tasks

The first-pass task registry currently includes:

- `vacuum_preservation_task()`
- `coherent_state_preparation_task()`
- `fock_state_preparation_task()`
- `storage_superposition_task()`
- `even_cat_preparation_task()`
- `odd_cat_preparation_task()`
- `ancilla_storage_bell_task()`
- `conditional_phase_gate_task()`
- `benchmark_task_suite()`

Each task defines:

- initial state
- target state or target operator
- optional logical subspace
- episode horizon
- success threshold
- baseline action sequence
- recommended action/observation mode

---

## Domain Randomization

```python
DomainRandomizer(
    model_priors_train={"chi": NormalPrior(...), "kerr": UniformPrior(...)},
    measurement_priors_train={"iq_sigma": UniformPrior(...)},
    drift_priors_train={"storage_amplitude_scale": UniformPrior(...)},
    model_priors_eval={...},
)
```

Supported prior helpers:

- `FixedPrior`
- `UniformPrior`
- `NormalPrior`
- `ChoicePrior`

Key points:

- Train and eval priors can be different.
- Episode metadata records the sampled parameter values.
- Drift-state overrides are kept separate from static model overrides.

---

## System Identification Hooks

`cqed_sim.system_id` currently provides lightweight fit-then-randomize scaffolding:

```python
CalibrationEvidence(
    model_posteriors={...},
    noise_posteriors={...},
    hardware_posteriors={...},
    measurement_posteriors={...},
)

randomizer_from_calibration(evidence)
```

These helpers are intentionally simple. They are meant to bridge future calibration/posterior workflows into the RL randomization layer, not to replace a full inference package.

---

## Example Entry Points

- Script template: `examples/rl_hybrid_control_rollout.py`
- Notebook walkthrough: `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`

The script and notebook use the same pattern: build a reduced dispersive environment, exercise measurement-facing observations and proxy rewards, inspect diagnostics, and evaluate a baseline controller under held-out randomization.