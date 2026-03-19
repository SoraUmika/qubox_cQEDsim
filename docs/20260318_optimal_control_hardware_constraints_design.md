# Hardware-Constrained GRAPE Design Note

## Scope

This note describes the first hardware-aware extension of `cqed_sim.optimal_control`.
The goal is to add realistic control-delivery constraints without breaking the
existing model-backed GRAPE workflow.

The implementation targets the following first-value subset:

- explicit separation between optimization parameters, command waveforms,
  physical waveforms, and Hamiltonian coefficients,
- coarse held-sample parameterization for AWG-style update constraints,
- differentiable first-order low-pass filtering in the forward model,
- differentiable or structured enforcement hooks for amplitude, IQ-radius, and
  boundary constraints,
- hardware-aware diagnostics in solve results and simulator replay.

Advanced basis families such as spline and Fourier controls, explicit
quantization-aware optimization, and pre-emphasis inversion are intentionally
deferred until the core pipeline is stable.

## Current Limitation

The pre-extension solver uses one array for all of the following meanings:

- optimization variables,
- command waveform samples,
- physical Hamiltonian coefficients.

That is sufficient for plain piecewise-constant GRAPE, but it blocks realistic
hardware modeling because coarse sample-and-hold, transfer-function filtering,
and paired I/Q constraints require at least one intermediate waveform layer.

## Proposed Pipeline

The new control flow is:

1. optimization parameters `theta`
2. parameterization map `P(theta)` producing a command waveform on the solver
   time grid
3. hardware map `H(u_cmd)` producing the physical waveform used by propagation
4. propagation under `H0 + sum_k u_phys,k(t) H_k`
5. objective and penalty evaluation

Structurally, the code should expose those layers directly rather than hiding
them inside the solver.

## Architecture

### Parameterization layer

Generalize the parameterization interface so it can:

- store parameter-space values in `ControlSchedule`,
- expand those values into a command waveform on the propagation grid,
- apply the reverse-mode pullback from command-waveform gradients back to the
  parameter-space variables,
- export either parameter-grid or time-grid waveforms back into repository
  `Pulse` objects.

The first concrete parameterizations are:

- `PiecewiseConstantParameterization`: identity map on the propagation grid,
- `HeldSampleParameterization`: coarse sample grid with zero-order hold onto the
  propagation grid.

### Hardware layer

Introduce a composable `HardwareModel` built from sequential `HardwareMap`
objects. Each map:

- transforms a command waveform into a new waveform on the same propagation
  grid,
- provides a pullback operation for reverse-mode gradient propagation,
- reports diagnostics relevant to that map.

The first concrete maps are:

- `FirstOrderLowPassHardwareMap`
- `BoundaryWindowHardwareMap`
- `SmoothIQRadiusLimitHardwareMap`

Amplitude bounds for scalar controls remain available through existing optimizer
bounds on `ControlTerm.amplitude_bounds`; the new layer makes the resulting
command and physical amplitudes explicit in diagnostics and replay.

### Penalty layer

Extend the additive penalty framework so penalties can act on a selected control
domain instead of assuming the raw schedule array is always the propagated
waveform.

The first additions are:

- `BoundPenalty`
- `BoundaryConditionPenalty`
- `IQRadiusPenalty`

Existing penalties keep their public names for backward compatibility.

### Solver integration

The solver should:

- optimize the schedule parameters only,
- build the command waveform through the parameterization,
- optionally apply the hardware model in the forward propagation path,
- propagate adjoint gradients back through hardware maps and parameterization
  pullbacks,
- record diagnostics for both command and physical waveforms.

Backward compatibility rule:

- if no hardware model is attached and the parameterization is piecewise
  constant, the behavior should match the previous implementation.

## Gradient Strategy

The GRAPE adjoint already provides gradients with respect to propagated control
coefficients on the solver time grid. The extension keeps that part unchanged
and adds two reverse-mode stages:

1. hardware pullback: `dJ/du_cmd = (dH/du_cmd)^T dJ/du_phys`
2. parameterization pullback: `dJ/dtheta = (dP/dtheta)^T dJ/du_cmd`

Implemented analytic or closed-form pullbacks in the first phase:

- identity / piecewise-constant map,
- held-sample accumulation pullback,
- boundary-window multiplication pullback,
- first-order low-pass filter reverse recurrence,
- smooth IQ radial saturation pullback.

Non-differentiable quantization and post-update projections remain deferred.

## Diagnostics

The result object should report, at minimum:

- parameter-grid schedule values,
- command waveform values,
- physical waveform values,
- per-map hardware diagnostics,
- aggregate hardware metrics such as clipping fraction, max command amplitude,
  max physical amplitude, max slew, effective update period, and IQ-radius
  violations.

If a hardware model is active, the final result should also expose the nominal
reference fidelity obtained by propagating the command waveform directly without
the hardware transform. That makes command-vs-physical degradation explicit.

## Replay and Validation

`evaluate_control_with_simulator(...)` should support replaying either:

- the command waveform, or
- the physical waveform after the attached hardware model.

This keeps the runtime validation path aligned with the new internal pipeline
without forcing the user to rebuild pulses manually.

## Phase 5 Extensions (2026-03-18)

The following items previously listed as deferred have been implemented.

### FourierParameterization

Parameters are cosine and sine amplitudes for `K` frequency modes.
The command waveform is a linear map:

```text
u_cmd[c, n] = sum_{k=0}^{K-1} a[c,k] cos(2π k t_n / T)
            + sum_{k=0}^{K-1} b[c,k] sin(2π k t_n / T)
```

Stored as a `(n_controls, 2K)` parameter array.  The forward pass is
`values @ B` where `B` is the `(2K, N)` precomputed basis matrix.  The
pullback is `grad_cmd @ B.T` — exact, no approximation.

**Use case**: smoothly band-limited pulses with an explicit frequency cap at
`n_modes / T`.

### LinearInterpolatedParameterization

Parameters are amplitudes at `K` uniformly-spaced knots.  The command waveform
is obtained by linear interpolation onto the propagation grid using an
`(N, K)` sparse matrix `M`.  The forward pass is `values @ M.T` and the pullback
is `grad_cmd @ M`.  The boundary knots are pinned to the first and last
propagation midpoints.

**Use case**: smooth pulses with a small number of free parameters while
preserving boundary behaviour.

### QuantizationHardwareMap

Quantizes the physical waveform to `2^n_bits` DAC levels within
`amplitude_bounds`.  The forward pass is a round-to-nearest operation.
Gradients use the straight-through estimator (identity pullback), which makes
the surrounding optimization gradient well-defined.  Requires finite
`amplitude_bounds`.

**Use case**: understanding DAC resolution impact; co-optimizing under discrete
amplitude constraints.

### FIRHardwareMap

Applies a causal FIR filter with coefficient vector `h`:

```text
y[n] = sum_{l=0}^{L-1} h[l] x[n-l]
```

The pullback is the transposed convolution:

```text
dx[m] = sum_{l=0}^{L-1} h[l] dy[m+l]
```

This is exact for any finite FIR kernel and verified by finite-difference
gradient checks.

**Use case**: pre-emphasis modeling; AWG impulse-response correction.  The
kernel `[1/α, -(1-α)/α]` approximately inverts a first-order IIR low-pass
filter with step coefficient `α = dt / (τ + dt)`.

### Gradient verification

All four new maps/parameterizations pass finite-difference central-difference
gradient checks at relative tolerance 1 × 10⁻⁵.  The quantization STE pullback
is identity and is verified separately.

## Remaining Deferred Items

- Spline (B-spline / cubic-spline) parameterizations,
- explicit post-update projection steps inside a custom optimizer loop,
- mixer imbalance correction hooks,
- hardware-aware open-system (Lindblad) optimization.
