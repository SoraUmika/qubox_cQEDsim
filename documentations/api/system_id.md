# `cqed_sim.system_id` — System Identification and Calibration Inference

## Purpose

`cqed_sim.system_id` is the inverse-calibration layer for common cQED workflows.
It fits measured calibration traces, converts fit summaries into posterior-like
prior objects, and bridges those priors into the RL/domain-randomization stack.

## Key Classes and Functions

### `CalibrationEvidence`

Collects posteriors from calibration measurements:

```python
from cqed_sim.system_id import CalibrationEvidence, FixedPrior, NormalPrior

evidence = CalibrationEvidence(
    model_posteriors={
        "chi": NormalPrior(mean=-2.84e6, sigma=50e3),
        "kerr": NormalPrior(mean=-28.8e3, sigma=1e3),
    },
    noise_posteriors={
        "t1": NormalPrior(mean=10e-6, sigma=0.5e-6),
    },
    measurement_posteriors={
        "p_e_given_g": FixedPrior(value=0.02),
    },
)
```

Fields:
- `model_posteriors` — posteriors over Hamiltonian parameters such as `chi`, `kerr`, and dressed frequencies
- `noise_posteriors` — posteriors over decoherence parameters such as `t1`, `t2_star`, and `t2_echo`
- `hardware_posteriors` — per-channel hardware posteriors
- `measurement_posteriors` — readout or classifier error posteriors
- `notes` — free-form metadata, including optional fit provenance

### Measured-trace fit helpers

The fit helpers all return `CalibrationResult`, the same lightweight result object used by `cqed_sim.calibration_targets`.

| Function | Input trace | Main fitted parameters |
|---|---|---|
| `fit_spectroscopy_trace(drive_frequencies, response)` | single-peak spectroscopy trace | `omega_peak`, `linewidth` |
| `fit_rabi_trace(amplitudes, excited_population, duration=...)` | driven Rabi trace | `omega_scale` |
| `fit_ramsey_trace(delays, excited_population)` | Ramsey fringe trace | `delta_omega`, `t2_star` |
| `fit_t1_trace(delays, excited_population)` | relaxation trace | `t1` |
| `fit_t2_echo_trace(delays, excited_population)` | Hahn-echo trace | `t2_echo` |

Each fit also returns nuisance parameters such as amplitude, offset, and phase where relevant.

Example:

```python
from cqed_sim.calibration_targets import run_spectroscopy, run_t1
from cqed_sim.system_id import fit_spectroscopy_trace, fit_t1_trace

spectroscopy_target = run_spectroscopy(model, drive_frequencies)
t1_target = run_t1(model, delays, t1=24.0e-6)

spectroscopy_fit = fit_spectroscopy_trace(
    spectroscopy_target.raw_data["drive_frequencies"],
    spectroscopy_target.raw_data["ground_response"],
)
t1_fit = fit_t1_trace(
    t1_target.raw_data["delays"],
    t1_target.raw_data["excited_population"],
)
```

### `prior_from_fit(...)`

Converts a fitted value and uncertainty into either:

- `NormalPrior(mean, sigma, low, high)` when uncertainty is nonzero, or
- `FixedPrior(value)` when the effective uncertainty collapses to zero.

`min_sigma` and `sigma_scale` let you enforce a floor or inflate narrow fit covariances before they enter the training distribution.

### `evidence_from_fit(...)`

Maps selected fitted parameters into one `CalibrationEvidence` category:

```python
from cqed_sim.system_id import evidence_from_fit

model_evidence = evidence_from_fit(
    spectroscopy_fit,
    category="model",
    parameter_map={"omega_peak": "omega_q"},
    bounds={"omega_q": (0.0, None)},
)

noise_evidence = evidence_from_fit(
    t1_fit,
    category="noise",
    bounds={"t1": (0.0, None)},
    min_sigma={"t1": 0.5e-6},
)
```

Categories:
- `model`
- `noise`
- `measurement`
- `hardware` with an explicit `channel=...`

### `merge_calibration_evidence(...)`

Combines multiple evidence blocks while rejecting duplicate posterior keys:

```python
from cqed_sim.system_id import CalibrationEvidence, NormalPrior, merge_calibration_evidence

evidence = merge_calibration_evidence(
    model_evidence,
    noise_evidence,
    CalibrationEvidence(
        model_posteriors={
            "chi": NormalPrior(mean=-2.84e6, sigma=50e3),
        },
    ),
)
```

### `randomizer_from_calibration(evidence)`

Converts `CalibrationEvidence` into a `DomainRandomizer` for RL training:

```python
from cqed_sim.system_id import randomizer_from_calibration

randomizer = randomizer_from_calibration(evidence)
```

The resulting randomizer uses the same posteriors for both training and evaluation.
To separate train and eval distributions, construct `DomainRandomizer` directly.

### Prior Types (re-exported)

| Type | Description |
|------|-------------|
| `FixedPrior(value)` | No randomization — always returns `value` |
| `NormalPrior(mean, sigma)` | Gaussian distribution with optional clipping bounds |
| `UniformPrior(low, high)` | Uniform distribution |
| `ChoicePrior(values)` | Discrete uniform or weighted categorical prior |

## Integration with RL Workflows

The shortest path is:

1. Generate or measure calibration traces.
2. Fit them with `fit_*_trace(...)`.
3. Convert selected fitted parameters with `evidence_from_fit(...)`.
4. Merge those blocks into `CalibrationEvidence`.
5. Call `randomizer_from_calibration(...)` and pass the result into `HybridEnvConfig`.

An end-to-end repo example lives in `examples/calibration_systemid_rl_pipeline.py`.

## Scope and Limitations

- The module now performs lightweight measured-trace fitting, but it is still not a joint Bayesian inference engine.
- `fit_spectroscopy_trace(...)` assumes a single dominant Lorentzian peak or dip in the supplied response trace.
- The `randomizer_from_calibration(...)` function sets train and eval distributions identically. For separate train/eval splits, construct `DomainRandomizer` directly.
