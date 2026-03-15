# Unitary Synthesis Phase 2 Runtime Audit

Created: 2026-03-15 16:22:17 local time  
Status: fixed

## Confirmed issues

### Open-system target preparation flattened subsystem dimensions in some synthesis paths

- What it is:
  - `TargetStateMapping` and the new open-system `TargetUnitary` probe-state path could coerce model-backed states into flat `[[full_dim], [1]]` or `[[full_dim], [full_dim]]` dims even when the runtime model was a tensor-product system such as `(2, 2)`.
- Where it appeared:
  - `cqed_sim/unitary_synthesis/targets.py`
  - `cqed_sim/unitary_synthesis/optim.py`
- Affected components:
  - open-system waveform synthesis
  - Lindblad/master-equation state propagation
  - noisy unitary-target optimization
- Why inconsistent:
  - The runtime simulator and QuTiP solver stack expect composite subsystem dims to remain aligned with the model Hamiltonian and collapse operators.
  - Flattening those dims breaks otherwise valid composite-model synthesis tasks.
- Consequences:
  - open-system synthesis could fail with dimension-mismatch errors
  - noisy unitary targets were not reliably usable for composite cQED models

### Robust optimization accepted custom sampler objects at runtime but could not serialize them in result reports

- What it is:
  - `ParameterDistribution` already accepted any object exposing `sample(...)` / `nominal(...)`, but the report/export path still assumed non-built-in specs could be converted directly to `float`.
- Where it appeared:
  - `cqed_sim/unitary_synthesis/config.py`
- Affected components:
  - robust optimization result export
  - reproducibility reports for custom deterministic samplers used in tests or studies
- Why inconsistent:
  - The runtime API permitted a broader sampler surface than the reporting layer documented or encoded.
- Consequences:
  - successful robust runs could fail during report generation or result export

## Resolution update

Updated: 2026-03-15 local time

- Fixed:
  - Preserved or restored model-native subsystem dims for open-system state mappings and unitary-target probe states before runtime simulation.
  - Added a safe serialization fallback for custom parameter-distribution samplers so reports and saved results remain exportable.

## Remaining follow-up items

- None for this Phase 2 pass.
