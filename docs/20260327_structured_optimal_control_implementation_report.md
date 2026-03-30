# 2026-03-27 Structured Optimal-Control Implementation Report

## Scope

This task added a first-class structured, hardware-aware optimal-control workflow to `cqed_sim` without replacing the existing GRAPE backend.

Implemented deliverables:

- new module: `cqed_sim/optimal_control/structured.py`
- shared pulse-export helper update in `cqed_sim/optimal_control/parameterizations.py`
- custom objective hook in `cqed_sim/optimal_control/objectives.py`
- shared evaluator extension in `cqed_sim/optimal_control/grape.py`
- serialization fix in `cqed_sim/optimal_control/utils.py`
- new regression coverage in `tests/test_52_structured_optimal_control.py`
- new study script in `examples/structured_optimal_control_demo.py`
- public API export updates in `cqed_sim/optimal_control/__init__.py` and `cqed_sim/__init__.py`
- documentation synchronization across `README.md`, `API_REFERENCE.md`, `documentations/`, `mkdocs.yml`, and `cqed_sim/optimal_control/README.md`

## Audit References

This implementation extends the earlier optimal-control work rather than creating a parallel stack.

- `docs/20260317_optimal_control_implementation_report.md`
- `docs/20260318_optimal_control_hardware_constraints_design.md`
- `inconsistency/20260318_194251_optimal_control_hardware_pipeline_refactor.md`

The design goal from those earlier tasks was preserved: keep `ControlProblem`, `HardwareModel`, and runtime pulse export as the stable shared abstractions, then layer new optimization modes on top.

## Architecture Summary

The structured workflow introduces a low-dimensional parameter-space solver on top of the existing direct-control pipeline.

Added layers:

- `PulseParameterSpec`
  - named bounds/default metadata for one structured parameter
- `StructuredPulseFamily`
  - base abstraction for smooth pulse families that can return both waveforms and parameter Jacobians
- `GaussianDragPulseFamily`
  - compact DRAG-style pulse family with amplitude, sigma, center, phase, and derivative-weight parameters
- `FourierSeriesPulseFamily`
  - truncated Fourier basis over I and Q quadratures
- `StructuredControlChannel`
  - binds one pulse family to either one scalar/I/Q control term or one I/Q pair
- `StructuredPulseParameterization`
  - maps a flat parameter vector to command waveforms on the propagation grid and provides the gradient pullback from waveform space back to parameter space
- `StructuredControlConfig` and `StructuredControlSolver`
  - SciPy-based parameter optimizer using the shared optimal-control evaluator
- `build_structured_control_problem_from_model(...)`
  - model-backed builder that reuses the existing control-term generation path
- `StructuredControlArtifacts` and `save_structured_control_artifacts(...)`
  - study-oriented artifact export for parameters, waveforms, spectra, and optimization history

The implementation intentionally reuses the existing infrastructure instead of bypassing it:

- `ControlProblem`
- `HardwareModel`
- `ControlResult`
- `_evaluate_schedule(...)`
- `_prepare_objective(...)`
- `_prepare_leakage_penalties(...)`
- runtime pulse export through `waveform_values_to_pulses(...)`

## Mathematical Form

The structured solver exposes the control pipeline explicitly:

$$
\theta \rightarrow u_{\mathrm{cmd}}(t; \theta) \rightarrow u_{\mathrm{phys}}(t) = \mathcal{H}[u_{\mathrm{cmd}}] \rightarrow H(t) = H_0 + \sum_k u_{\mathrm{phys},k}(t) H_k.
$$

This separates:

- optimization variables `theta`,
- command waveforms on the solver grid,
- physical waveforms after the hardware model,
- Hamiltonian coefficients used for propagation.

### Gaussian-DRAG family

Let $\tau = t / T$, $u = \tau - c$, and

$$
g(\tau; \sigma, c) = \exp\left[-\frac{1}{2}\left(\frac{u}{\sigma}\right)^2\right].
$$

The implemented complex envelope is

$$
u_{\mathrm{cmd}}(t) = A \left[g(\tau; \sigma, c) + i \alpha \frac{d g}{d \tau}\right] e^{i\phi},
$$

with

$$
\frac{d g}{d \tau} = -\frac{u}{\sigma^2} g(\tau; \sigma, c).
$$

The code provides an analytic Jacobian with respect to:

- amplitude,
- sigma fraction,
- center fraction,
- phase,
- DRAG weight.

### Fourier family

For $\tau = t / T$ and truncation order $N$, the implemented family uses

$$
I(\tau) = \sum_{n=0}^{N-1} a_n^{(I,c)} \cos(2\pi n \tau) + \sum_{n=1}^{N-1} a_n^{(I,s)} \sin(2\pi n \tau),
$$

$$
Q(\tau) = \sum_{n=0}^{N-1} a_n^{(Q,c)} \cos(2\pi n \tau) + \sum_{n=1}^{N-1} a_n^{(Q,s)} \sin(2\pi n \tau),
$$

$$
u_{\mathrm{cmd}}(t) = I(\tau) - i Q(\tau).
$$

Because the basis is linear in the coefficients, the Jacobian is the basis matrix itself.

### Gradient pullback

The shared evaluator returns gradients in command-waveform coordinates. `StructuredPulseParameterization.pullback(...)` contracts that waveform gradient with the family Jacobian to obtain a parameter-space gradient:

$$
\nabla_\theta J = \left(\frac{\partial u_{\mathrm{cmd}}}{\partial \theta}\right)^\top \nabla_{u_{\mathrm{cmd}}} J.
$$

This is what lets the structured solver reuse the existing evaluator without changing the Hamiltonian propagation logic.

## Transfer Model and Waveform Mapping

The structured parameterization samples pulse families at the time-grid midpoints and emits one command waveform per resolved control term.

Channel binding modes are:

- one I/Q pair sharing the same export channel,
- one I-only control term,
- one Q-only control term,
- one scalar control term.

The runtime sign convention is preserved exactly:

$$
c(t) = I(t) + i Q(t).
$$

The example study attaches a `HardwareModel` with three maps on the exported `qubit` channel:

- `GainHardwareMap(gain=0.93, export_channels=("qubit",))`
- `DelayHardwareMap(delay_samples=1, export_channels=("qubit",))`
- `FirstOrderLowPassHardwareMap(cutoff_hz=28.0e6, export_channels=("qubit",))`

That makes the optimization explicitly hardware-aware rather than treating distortion as a post hoc replay step.

## Objective Surface and Optimization

The structured solver reuses the shared objective preparation and schedule evaluation path, so the new backend supports the same physical-model objectives as the GRAPE path.

Reused objective families:

- `StateTransferObjective`
- `UnitaryObjective`
- leakage penalties prepared through `_prepare_leakage_penalties(...)`

New extension point:

- `CustomControlObjective`
- `CustomObjectiveContext`
- `CustomObjectiveEvaluation`

The custom objective hook returns both a scalar cost and a gradient in physical-waveform coordinates, allowing structured studies to add hardware- or metric-specific objectives without forking the solver.

`StructuredControlConfig` defaults to:

- `optimizer_method="L-BFGS-B"`
- `apply_hardware_in_forward_model=True`
- `report_command_reference=True`
- `use_gradients=True`

The solver rescales variables by finite bound magnitudes before passing them to SciPy so that optimizer convergence is not dominated by raw `rad/s` magnitudes.

## Example Study Configuration

The delivered study script is `examples/structured_optimal_control_demo.py`.

Shared study configuration:

- model: one cavity level plus a two-level transmon
- frame: cavity and qubit rotating frame
- propagation grid: 32 slices at 4 ns each
- total duration: 128 ns
- target channel: qubit I/Q pair with bounds `(-8.0e7, 8.0e7)` in `rad/s`
- objective: state transfer `|g,0> -> |e,0>`

Family-specific configuration:

- Gaussian DRAG study
  - amplitude bounds `(0.0, 7.0e7)`
  - sigma-fraction bounds `(0.1, 0.24)`
  - center-fraction bounds `(0.42, 0.58)`
  - phase bounds `(-pi, pi)`
  - drag bounds `(-0.3, 0.3)`
- Fourier study
  - `n_modes=3`
  - coefficient bound `4.0e7`

Observed example outputs from the executed study:

- Gaussian DRAG
  - success: `True`
  - objective: `1.320277e-12`
  - nominal command fidelity: `0.979873`
  - nominal physical fidelity: `1.000000`
- Fourier basis
  - success: `True`
  - objective: `1.487699e-14`
  - nominal command fidelity: `0.809058`
  - nominal physical fidelity: `1.000000`

The gap between command-reference fidelity and physical-reference fidelity is the expected signature of optimizing through the hardware model instead of against the undistorted waveform.

## Artifact Export

`save_structured_control_artifacts(...)` writes a study bundle containing:

- `result.json`
- `parameters.csv`
- `waveforms.csv`
- `history.csv`
- `waveforms.png`
- `spectrum.png`
- `optimization_history.png`

The example produces:

- `outputs/structured_optimal_control_demo/gaussian_drag/`
- `outputs/structured_optimal_control_demo/fourier_basis/`

This satisfies the requested visualization and study-artifact deliverables without requiring a notebook-only workflow.

## Integration with the Existing Workflow

The new backend remains inside the existing `cqed_sim.optimal_control` package and participates in the same runtime/export path as the rest of the repository.

Integration points:

- top-level exports through `cqed_sim/__init__.py`
- module exports through `cqed_sim/optimal_control/__init__.py`
- pulse export back into the standard runtime stack
- `ControlResult.evaluate_with_simulator(...)` for replay against the time-domain simulator
- shared hardware reporting and command-vs-physical diagnostics

This means the structured backend is simulator-first today, but it is not simulator-only by design. The exposed parameter vector, artifact bundle, and custom-objective hook are compatible with a future hardware-closed-loop path where measured costs replace or augment the current simulator-derived objective.

## Validation

Added validation:

- `tests/test_52_structured_optimal_control.py`
  - Gaussian-DRAG Jacobian check
  - Fourier Jacobian check
  - gradient pullback check
  - end-to-end structured hardware-aware solve and artifact export
  - custom objective support

Regression coverage re-run during this task:

- `tests/test_40_optimal_control_grape.py`
- `tests/test_41_optimal_control_followup.py`
- `tests/test_51_optimal_control_hardware_constraints.py`
- `tests/test_52_structured_optimal_control.py`

The structured example script also ran successfully and generated the expected artifact directories.

## Physics and Convention Impact

This task did not introduce a new physics convention. It reused existing repository conventions for:

- tensor ordering,
- rotating-frame sign conventions,
- drive quadrature mapping,
- runtime pulse export,
- units (`rad/s` for coefficients, `s` for time).

Because the physical meaning of the Hamiltonian, frames, and sign conventions did not change, no update to `physics_and_conventions/physics_conventions_report.tex` was required for this task.

## Remaining Extensions

The current structured backend is intentionally focused:

- smooth pulse families are supported, but derivative-free or measurement-only outer loops are not yet implemented,
- the solver remains a dense closed-system optimizer internally,
- the public tutorial notebook is still GRAPE-centric even though the docs now describe the broader optimal-control surface.

Those are extension opportunities, not architectural blockers. The main outcome of this task is that `cqed_sim` now has a first-class structured-control layer built on the same physical and runtime conventions as the rest of the library.