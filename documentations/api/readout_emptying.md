# API Reference: Readout Emptying (`cqed_sim.optimal_control.readout_emptying`)

The readout-emptying helpers add a physics-driven segmented pulse family for dispersive readout resonators. The implementation now has two layers:

- solve a terminal emptying constraint exactly for piecewise-constant segments,
- select a useful null-space waveform by maximizing branch separation,
- optionally apply a Kerr-aware phase correction,
- export the result to the standard `Pulse` / replay stack,
- expose the same family as a reduced `CallableParameterization`,
- and qualify or refine the pulse under measurement, Lindblad, and hardware-aware replay through `cqed_sim.optimal_control.readout_emptying_eval`.

This keeps the feature aligned with the existing `cqed_sim.optimal_control` architecture instead of introducing a second readout-only control stack.

---

## Core objects

```python
from cqed_sim.optimal_control import (
    ReadoutEmptyingConstraints,
    ReadoutEmptyingRefinementConfig,
    ReadoutEmptyingRefinementResult,
    ReadoutEmptyingReplay,
    ReadoutEmptyingResult,
    ReadoutEmptyingSpec,
    ReadoutEmptyingVerificationConfig,
    ReadoutEmptyingVerificationReport,
    ReadoutResonatorBranch,
)
```

### `ReadoutEmptyingSpec`

Key fields:

- `kappa`
- `chi`
- `tau`
- `n_segments`
- `detuning_center`
- `segment_times`
- `allow_complex_segments`
- `target_states`
- `kerr`
- `include_kerr_phase_correction`
- `kerr_correction_strategy`

Convention note:

- `detuning_center` is the drive-frequency offset relative to the midpoint of the dressed `g/e` resonator frequencies.
- The replay model follows the existing measurement-layer sign convention
  `dot(alpha) = -(kappa/2 + i Delta) alpha - i epsilon(t)`,
  with `Delta = omega_resonator - omega_drive`.
- The exported pulse envelope still follows the repository baseband rule `c(t) = I(t) + i Q(t)`.

### `ReadoutEmptyingConstraints`

Useful controls:

- `amplitude_max`
- `enforce_zero_start`
- `enforce_zero_end`
- `favor_real_waveform`
- `smoothness_weight`
- `min_average_photons`
- `preferred_solution`

The current implementation supports:

- `preferred_solution="max_separation"`: the default practical choice.
- `preferred_solution="min_norm"`: a projected square-pulse-like null-space choice.

### `ReadoutResonatorBranch`

Explicit branch override:

- `label`
- `detuning`

Use this when the default two-state `g/e` dispersive branch construction is not the right abstraction for a study.

### `ReadoutEmptyingReplay`

Returned by the direct replay helpers:

- `time_grid_s`
- `command_waveform`
- `trajectories`
- `photon_numbers`
- `final_alpha`
- `final_n`

### `ReadoutEmptyingResult`

Returned by `synthesize_readout_emptying_pulse(...)`:

- `segment_amplitudes`
- `segment_edges_s`
- `time_grid_s`
- `command_waveform`
- `trajectories`
- `final_alpha`
- `final_n`
- `metrics`
- `diagnostics`

The `diagnostics` payload stores the constraint matrix, null-space basis, linear replay metrics, and optional Kerr-correction traces used during synthesis.

### `ReadoutEmptyingVerificationConfig`

Controls the deployment-style verification path:

- optional `measurement_chain`
- optional `hardware_model`
- optional `readout_model`, `frame`, and `noise`
- `compiler_dt_s`
- `shots_per_branch`
- mismatch sweeps for `chi`, `kappa`, `kerr`, amplitude scale, and timing
- optional `hardware_variants`

### `ReadoutEmptyingVerificationReport`

Returned by `verify_readout_emptying_pulse(...)`:

- `baseline_results`
- `baseline_metrics`
- `measurement_metrics`
- `lindblad_metrics`
- `hardware_metrics`
- `robustness`
- `comparison_table`
- `artifacts`
- `diagnostics`

### `ReadoutEmptyingRefinementConfig`

Controls the reduced refinement harness:

- derivative-free outer-loop settings such as `maxiter` and `method`
- objective weights for residual, separation, measurement error, leakage, robustness, and bandwidth sensitivity
- optional `measurement_chain`, `hardware_model`, `readout_model`, `frame`, and `noise`
- optional chirp-scale, segment-duration, and endpoint-ramp controls
- uncertainty levels used for robustness-aware scoring

### `ReadoutEmptyingRefinementResult`

Returned by `refine_readout_emptying_pulse(...)`:

- `seed_result`
- `refined_result`
- `objective_value`
- `initial_objective_value`
- `parameter_names`
- `parameter_values`
- `metrics`
- `history`
- `diagnostics`
- optional `verification_report`

---

## Main entry points

```python
from cqed_sim.optimal_control import (
    apply_phase_chirp,
    build_emptying_constraint_matrix,
    build_kerr_phase_correction,
    build_readout_emptying_parameterization,
    compute_emptying_null_space,
    default_readout_emptying_branches,
    evaluate_readout_emptying_with_chain,
    export_readout_emptying_to_pulse,
    refine_readout_emptying_pulse,
    replay_kerr_readout_branches,
    replay_linear_readout_branches,
    synthesize_readout_emptying_pulse,
    verify_readout_emptying_pulse,
)
```

### `synthesize_readout_emptying_pulse(...)`

Primary constructor for the feature.

It:

1. builds the exact terminal constraint matrix,
2. extracts its null space with SVD,
3. selects a segmented waveform,
4. rescales it to the requested amplitude / occupancy regime,
5. optionally applies the Kerr-aware phase correction,
6. and returns replay diagnostics plus summary metrics.

### `export_readout_emptying_to_pulse(...)`

Returns a standard repository `Pulse` with an analytic piecewise-constant envelope, so the result can be compiled and replayed through the usual pulse/runtime stack.

### `build_readout_emptying_parameterization(...)`

Returns a reduced `CallableParameterization` whose variables are null-space coordinates rather than raw waveform samples. Any waveform generated by this parameterization still satisfies the terminal emptying constraints.

This is the recommended bridge into the rest of `cqed_sim.optimal_control`.

### `evaluate_readout_emptying_with_chain(...)`

Routes the synthesized waveform through `ReadoutChain.simulate_waveform(...)` and reports:

- noiseless traces,
- I/Q centers,
- measurement separation,
- and a simple synthetic classification proxy.

### `verify_readout_emptying_pulse(...)`

Builds the qualification-first comparison table around one seed result.

By default it compares:

- matched-duration square pulse
- analytic seed
- Kerr-corrected seed when Kerr is active
- any extra baselines supplied through `comparison_results=...`

The report replays each baseline through the configured measurement chain, optional hardware model, optional dispersive Lindblad model, and mismatch sweeps.

### `refine_readout_emptying_pulse(...)`

Runs a reduced-dimensional outer-loop refinement over the existing null-space family plus optional shared-chirp, segment-duration, and endpoint-ramp controls.

This is intentionally solver-adjacent rather than solver-native: it reuses the simulator and hardware stack, but it does not introduce new `ControlProblem` objective types in this phase.

---

## Minimal example

```python
import numpy as np

from cqed_sim.optimal_control import (
    ReadoutEmptyingConstraints,
    ReadoutEmptyingSpec,
    build_readout_emptying_parameterization,
    export_readout_emptying_to_pulse,
    synthesize_readout_emptying_pulse,
)

spec = ReadoutEmptyingSpec(
    kappa=2.0 * np.pi * 2.0e6,
    chi=2.0 * np.pi * 1.0e6,
    tau=320e-9,
    n_segments=4,
    kerr=2.0 * np.pi * 0.08e6,
    include_kerr_phase_correction=True,
)
constraints = ReadoutEmptyingConstraints(
    amplitude_max=2.0 * np.pi * 8.0e6,
)

result = synthesize_readout_emptying_pulse(spec, constraints)
pulse = export_readout_emptying_to_pulse(result, channel="readout")
parameterization = build_readout_emptying_parameterization(spec, constraints)
```

Qualification / refinement example:

```python
from cqed_sim.optimal_control import (
    ReadoutEmptyingRefinementConfig,
    ReadoutEmptyingVerificationConfig,
    refine_readout_emptying_pulse,
    verify_readout_emptying_pulse,
)

verification = verify_readout_emptying_pulse(
    result,
    ReadoutEmptyingVerificationConfig(shots_per_branch=64),
)
refined = refine_readout_emptying_pulse(
    result,
    ReadoutEmptyingRefinementConfig(maxiter=12),
)
```

Concrete runnable example:

- `examples/readout_emptying_demo.py`
- `examples/studies/readout_emptying/linear_seed_validation.py`
- `examples/studies/readout_emptying/kerr_replay_and_chirp.py`
- `examples/studies/readout_emptying/dispersive_lindblad_validation.py`
- `examples/studies/readout_emptying/reduced_refinement.py`
- `examples/studies/readout_emptying/hardware_sensitivity.py`
- `examples/studies/readout_emptying/summary_benchmark.py`

Generated evidence from a fresh run in this repository:

- `outputs/readout_emptying_demo/waveform.png`
- `outputs/readout_emptying_demo/phase_space.png`
- `outputs/readout_emptying_demo/photons.png`
- `outputs/readout_emptying_demo/residuals.png`
- `outputs/readout_emptying_demo/separation.png`
- `outputs/readout_emptying_demo/iq_clusters.png`
- `outputs/readout_emptying_demo/metrics.json`
- `outputs/readout_emptying_qualification/`

---

## Caveats

- The Kerr correction is an approximate mean-field compensation, not an exact simultaneous cancellation for both branches when their occupations differ.
- Perfect terminal emptying is not enough on its own; a useful readout pulse also needs strong branch separation.
- High photon number can leave the regime where the simple dispersive/Kerr model remains quantitatively reliable.
- Hardware bandwidth and mixer distortion can spoil exact cancellation, so hardware-aware replay or later refinement is still recommended for deployment studies.
- The reduced refinement harness is not yet a native open-system `ControlProblem` solver path; it is a study-oriented outer loop built on top of the existing simulator and hardware APIs.

---

## References

[1] D. T. McClure, H. Paik, L. S. Bishop, M. Steffen, J. M. Chow, and J. M. Gambetta, "Rapid Driven Reset of a Qubit Readout Resonator," Physical Review Applied 5, 011001 (2016). DOI: 10.1103/PhysRevApplied.5.011001

[2] M. Jerger, F. Motzoi, Y. Gao, C. Dickel, L. Buchmann, A. Bengtsson, G. Tancredi, C. W. Warren, J. Bylander, D. DiVincenzo, R. Barends, and P. A. Bushev, "Dispersive Qubit Readout with Intrinsic Resonator Reset," arXiv (2024). DOI: 10.48550/arXiv.2406.04891

[3] A. Blais, A. L. Grimsmo, S. M. Girvin, and A. Wallraff, "Circuit quantum electrodynamics," Reviews of Modern Physics 93, 025005 (2021). DOI: 10.1103/RevModPhys.93.025005
