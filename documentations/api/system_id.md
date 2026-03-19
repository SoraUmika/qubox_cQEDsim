# `cqed_sim.system_id` — System Identification and Calibration Bridge

## Purpose

`cqed_sim.system_id` provides the bridge between experimental calibration evidence
and robust-control or RL training workflows. It converts measured posteriors over
device parameters into `DomainRandomizer` priors that can be used to train
policies robust to device uncertainty.

## Key Classes and Functions

### `CalibrationEvidence`

Collects posteriors from calibration measurements:

```python
from cqed_sim.system_id import CalibrationEvidence, NormalPrior, FixedPrior

evidence = CalibrationEvidence(
    model_posteriors={
        "chi": NormalPrior(mean=-2.84e6, std=50e3),
        "kerr": NormalPrior(mean=-28.8e3, std=1e3),
    },
    noise_posteriors={
        "t1_s": NormalPrior(mean=10e-6, std=0.5e-6),
    },
)
```

Fields:
- `model_posteriors` — posteriors over Hamiltonian parameters (chi, kerr, omega, etc.)
- `noise_posteriors` — posteriors over noise/decoherence parameters (T1, T2, etc.)
- `hardware_posteriors` — per-channel posteriors (drive amplitudes, offsets, etc.)
- `measurement_posteriors` — measurement error posteriors (confusion matrices, etc.)
- `notes` — free-form metadata

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
| `NormalPrior(mean, std)` | Gaussian distribution |
| `UniformPrior(low, high)` | Uniform distribution |
| `ChoicePrior(choices)` | Discrete uniform over a list |

## Integration with RL Workflows

```python
from cqed_sim.rl_control import HybridCQEDEnv, HybridEnvConfig, HybridSystemConfig
from cqed_sim.system_id import CalibrationEvidence, NormalPrior, randomizer_from_calibration

evidence = CalibrationEvidence(
    model_posteriors={"chi": NormalPrior(mean=-2.84e6, std=50e3)},
)
randomizer = randomizer_from_calibration(evidence)

config = HybridEnvConfig(
    system=HybridSystemConfig(...),
    domain_randomizer=randomizer,
)
env = HybridCQEDEnv(config=config)
```

## Scope and Limitations

- This module is currently a thin bridge layer. For richer calibration workflows,
  see `cqed_sim.calibration_targets` (which produces calibration measurements)
  and `cqed_sim.rl_control` (which consumes randomizers).
- The `randomizer_from_calibration` function sets train and eval distributions
  identically. For separate train/eval splits, construct `DomainRandomizer` directly.
- Priors must be expressed as `ParameterPrior` objects; raw floats must be wrapped
  in `FixedPrior`.
