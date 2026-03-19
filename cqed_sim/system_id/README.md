# `cqed_sim.system_id` — System Identification and Calibration Bridge

## What this module does

`cqed_sim.system_id` bridges calibration evidence (measured posteriors over device
parameters) to model randomization for robust control and RL training workflows.
It converts measured posteriors into `DomainRandomizer` priors that can be used to
train policies robust to device uncertainty.

## Main classes and functions

| Name | Type | Description |
|---|---|---|
| `CalibrationEvidence` | dataclass | Collects posterior distributions over model, noise, hardware, and measurement parameters |
| `randomizer_from_calibration(evidence)` | function | Converts `CalibrationEvidence` into a `DomainRandomizer` |
| `FixedPrior(value)` | prior | No randomization — always returns `value` |
| `NormalPrior(mean, std)` | prior | Gaussian distribution |
| `UniformPrior(low, high)` | prior | Uniform distribution |
| `ChoicePrior(choices)` | prior | Discrete uniform over a list |

The prior types are re-exported from `cqed_sim.rl_control.domain_randomization`.

## How it works

`CalibrationEvidence` collects posteriors from calibration measurements, organized
by category:

```python
from cqed_sim.system_id import CalibrationEvidence, NormalPrior, FixedPrior

evidence = CalibrationEvidence(
    model_posteriors={
        "chi": NormalPrior(mean=-2.84e6, std=50e3),
        "kerr": NormalPrior(mean=-28.8e3, std=1e3),
    },
    noise_posteriors={
        "t1": NormalPrior(mean=10e-6, std=0.5e-6),
    },
    hardware_posteriors={
        "drive_q": {"amp_scale": FixedPrior(1.0)},
    },
    measurement_posteriors={
        "p_e_given_g": FixedPrior(0.02),
    },
)
```

`randomizer_from_calibration(evidence)` converts this into a `DomainRandomizer`,
setting both the train and eval distributions to the same posteriors:

```python
from cqed_sim.system_id import randomizer_from_calibration

randomizer = randomizer_from_calibration(evidence)
```

## When to use

Use this module when you have measured device parameters with uncertainty and want
to train an RL policy (via `HybridCQEDEnv`) that is robust to that uncertainty. The
typical workflow is:

1. Run calibration sweeps (`cqed_sim.calibration_targets`) to measure device parameters.
2. Fit posteriors and wrap them in prior objects.
3. Build a `CalibrationEvidence` and convert it to a `DomainRandomizer`.
4. Pass the randomizer to `HybridEnvConfig` for RL training.

## Relationship to the rest of `cqed_sim`

- **Depends on**: `cqed_sim.rl_control.domain_randomization` (for `DomainRandomizer` and prior types)
- **Feeds into**: `HybridCQEDEnv` via `HybridEnvConfig(domain_randomizer=randomizer)`
- **Upstream**: `cqed_sim.calibration_targets` produces the calibration measurements that become posteriors

## Limitations

- This module is a thin bridge layer. It does not perform fitting or inference.
- `randomizer_from_calibration` sets train and eval distributions identically.
  For separate train/eval splits, construct `DomainRandomizer` directly.
- Priors must be expressed as `ParameterPrior` objects; raw floats must be wrapped
  in `FixedPrior`.
