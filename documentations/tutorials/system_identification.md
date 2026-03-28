# System Identification & Randomization

This tutorial set explains how `cqed_sim` connects calibration summaries to robust-control workflows. The emphasis is not on a single fitting routine in isolation, but on the handoff from fitted experimental summaries to structured uncertainty models that a control environment can consume.

These notebooks are the right entry point if you want to understand why the repository includes both calibration-target helpers and domain-randomized control environments.

## Included Notebooks

### `31_system_identification_and_domain_randomization/01_calibration_targets_and_fitting.ipynb`

This notebook generates effective spectroscopy, Rabi, and T1 targets and inspects the fitted parameter summaries that come back from each helper.

What it teaches:

- how `run_spectroscopy(...)`, `run_rabi(...)`, and `run_t1(...)` package synthetic calibration traces
- which fitted parameters and uncertainty estimates are exposed for downstream use
- how to think about these outputs as workflow inputs for later system-identification and control stages

### `31_system_identification_and_domain_randomization/02_evidence_to_randomizer_and_env.ipynb`

This notebook packages fitted summaries into `CalibrationEvidence`, converts that evidence into a `DomainRandomizer`, and then wires the resulting priors into a hybrid RL environment.

What it teaches:

- how calibration posteriors become train-time randomization priors
- how `randomizer_from_calibration(...)` bridges the calibration and control subsystems
- what metadata and observation products are produced at `env.reset(...)`

## Why This Set Exists

Robust control should not be disconnected from calibration. If the device characterization changes, the uncertainty model used for controller training should change with it. These notebooks make that dependency explicit.

They also provide a stable workflow surface for understanding the repository's system-identification abstractions before moving on to policy training or more hardware-aware control studies.

## Related References

- [System Identification API](../api/system_id.md)
- [Calibration Targets API](../api/calibration_targets.md)
- [RL Hybrid Control tutorial](rl_hybrid_control.md)
