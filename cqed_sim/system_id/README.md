# `cqed_sim.system_id` — System Identification and Calibration Inference

## What this module does

`cqed_sim.system_id` covers the lightweight inverse-calibration layer of the
repository. It fits common measured calibration traces, converts the fitted values
and uncertainties into prior objects, and bridges those priors into the RL/domain-
randomization stack.

## Main classes and functions

| Name | Type | Description |
|---|---|---|
| `CalibrationEvidence` | dataclass | Collects posterior distributions over model, noise, hardware, and measurement parameters |
| `fit_spectroscopy_trace(...)` | function | Fits a single Lorentzian spectroscopy peak or dip |
| `fit_rabi_trace(...)` | function | Fits an offset Rabi trace and returns `omega_scale` plus nuisance terms |
| `fit_ramsey_trace(...)` | function | Fits Ramsey detuning, `t2_star`, offset, amplitude, and phase |
| `fit_t1_trace(...)` | function | Fits an offset exponential `T1` decay |
| `fit_t2_echo_trace(...)` | function | Fits an offset exponential Hahn-echo decay |
| `prior_from_fit(...)` | function | Converts a fitted value and uncertainty into a bounded `NormalPrior` or `FixedPrior` |
| `evidence_from_fit(...)` | function | Maps selected fitted parameters into one `CalibrationEvidence` category |
| `merge_calibration_evidence(...)` | function | Combines multiple evidence blocks while rejecting duplicate posterior keys |
| `randomizer_from_calibration(evidence)` | function | Converts `CalibrationEvidence` into a `DomainRandomizer` |
| `FixedPrior(value)` | prior | No randomization — always returns `value` |
| `NormalPrior(mean, sigma)` | prior | Gaussian distribution with optional clipping bounds |
| `UniformPrior(low, high)` | prior | Uniform distribution |
| `ChoicePrior(values)` | prior | Discrete uniform or weighted categorical prior |

The prior types are re-exported from `cqed_sim.rl_control.domain_randomization`.

## Typical workflow

```python
import numpy as np

from cqed_sim.calibration_targets import run_spectroscopy, run_t1
from cqed_sim.system_id import (
    CalibrationEvidence,
    NormalPrior,
    evidence_from_fit,
    fit_spectroscopy_trace,
    fit_t1_trace,
    merge_calibration_evidence,
    randomizer_from_calibration,
)

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

evidence = merge_calibration_evidence(
    evidence_from_fit(
        spectroscopy_fit,
        category="model",
        parameter_map={"omega_peak": "omega_q"},
        bounds={"omega_q": (0.0, None)},
    ),
    evidence_from_fit(
        t1_fit,
        category="noise",
        bounds={"t1": (0.0, None)},
        min_sigma={"t1": 0.5e-6},
    ),
    CalibrationEvidence(
        model_posteriors={
            "chi": NormalPrior(mean=-2.84e6, sigma=50e3),
        },
    ),
)

randomizer = randomizer_from_calibration(evidence)
```

## Relationship to the rest of `cqed_sim`

- **Upstream**: `cqed_sim.calibration_targets` generates synthetic or simulated calibration traces that can be re-fit through this module.
- **Downstream**: `randomizer_from_calibration(...)` feeds `cqed_sim.rl_control.DomainRandomizer` and then `HybridCQEDEnv`.
- **Shared result type**: The fit helpers return `CalibrationResult`, the same lightweight result container used by `cqed_sim.calibration_targets`.

## Current scope and limitations

- Fits are independent trace-by-trace curve fits, not a joint Bayesian inference engine.
- `fit_spectroscopy_trace(...)` assumes one dominant Lorentzian peak or dip in the supplied trace.
- `randomizer_from_calibration(...)` sets train and eval distributions identically. For separate train/eval splits, construct `DomainRandomizer` directly.
