# Optimal Control Tutorial

`cqed_sim` now supports two direct-control workflows built on the same physical model and simulation stack:

- structured, hardware-aware parameter-space optimization over smooth pulse families,
- GRAPE for slice-level waveform refinement.

Use the structured workflow when the experiment loop is naturally:

1. propose a pulse family,
2. pass it through a hardware or transfer model,
3. simulate the effective waveform,
4. update a small parameter vector.

Use GRAPE when you need fine-grained piecewise control once the high-level pulse family is no longer the right abstraction.

Primary entry points:

- `examples/structured_optimal_control_demo.py`
- `examples/hardware_constrained_grape_demo.py`
- `examples/grape_storage_subspace_gate_demo.py`
- `tutorials/30_advanced_protocols/06_grape_optimal_control_workflow.ipynb`

---

## Physics Background

Given a quantum system with Hamiltonian

$$
H(t) = H_0 + \sum_j u_j(t) H_j,
$$

the direct-control problem is to choose control waveforms that optimize a figure of merit such as state fidelity or unitary fidelity.

For state preparation,

$$
\mathcal{F}_{\mathrm{state}} = \left|\langle \psi_{\mathrm{target}} | U(T) | \psi_0 \rangle\right|^2.
$$

For unitary synthesis,

$$
\mathcal{F}_{\mathrm{unitary}} = \frac{1}{d^2}\left|\operatorname{Tr}\left[U_{\mathrm{target}}^\dagger U(T)\right]\right|^2.
$$

In cQED, `cqed_sim` evaluates these objectives on the full truncated physical Hilbert space, not on an abstracted qubit-only toy model.

---

## Structured Workflow

The structured backend optimizes a low-dimensional parameter vector

$$
\theta = (\theta_1, \ldots, \theta_P)
$$

instead of one independent amplitude per time sample.

The internal pipeline is explicit:

$$
\theta \rightarrow u_{\mathrm{cmd}}(t; \theta) \rightarrow u_{\mathrm{phys}}(t) = \mathcal{H}[u_{\mathrm{cmd}}] \rightarrow H(t; u_{\mathrm{phys}}).
$$

That separation matters because realistic hardware introduces:

- bandwidth limits,
- gain errors,
- delays or skews,
- boundary smoothing,
- I/Q distortions.

For model-backed I/Q controls, the exported rotating-frame envelope uses

$$
c(t) = I(t) + i Q(t)
$$

and the model builder derives the Hermitian `Q` quadrature as `+i(raising - lowering)`, so replay through the standard runtime stays consistent with the optimizer Hamiltonian.

### Built-in pulse families

- `GaussianDragPulseFamily`

  $$
  u(t; \theta) = A \left[g(t; \sigma, c) + i\,\alpha\,\frac{dg}{d\tau}\right] e^{i\phi}
  $$

- `FourierSeriesPulseFamily`

  $$
    u(t; \theta) = I(t; \theta_I) + i Q(t; \theta_Q)
  $$

  with truncated cosine and sine bases on each quadrature.

### Minimal structured example

```python
import numpy as np
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FirstOrderLowPassHardwareMap,
    FrameSpec,
    GaussianDragPulseFamily,
    GainHardwareMap,
    HardwareModel,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    StructuredControlChannel,
    StructuredControlConfig,
    build_structured_control_problem_from_model,
    solve_structured_control,
    state_preparation_objective,
)

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9,
    omega_q=2*np.pi*6e9,
    alpha=0.0,
    chi=0.0,
    kerr=0.0,
    n_cav=1,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

problem = build_structured_control_problem_from_model(
    model,
    frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=32, dt_s=4e-9),
    channel_specs=(
        ModelControlChannelSpec(
            name="qubit",
            target="qubit",
            quadratures=("I", "Q"),
            amplitude_bounds=(-8e7, 8e7),
            export_channel="qubit",
        ),
    ),
    structured_channels=(
        StructuredControlChannel(
            name="gaussian_drag",
            pulse_family=GaussianDragPulseFamily(default_phase=-0.5*np.pi),
            export_channel="qubit",
        ),
    ),
    objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
    hardware_model=HardwareModel(maps=(
        GainHardwareMap(gain=0.93, export_channels=("qubit",)),
        FirstOrderLowPassHardwareMap(cutoff_hz=28e6, export_channels=("qubit",)),
    )),
)

result = solve_structured_control(
    problem,
    config=StructuredControlConfig(maxiter=60, seed=7, initial_guess="random"),
)
```

### Example study artifacts

Running `examples/structured_optimal_control_demo.py` generates two study directories:

- `outputs/structured_optimal_control_demo/gaussian_drag/`
- `outputs/structured_optimal_control_demo/fourier_basis/`

Each directory contains:

- `result.json`
- `parameters.csv`
- `waveforms.csv`
- `history.csv`
- `waveforms.png`
- `spectrum.png`
- `optimization_history.png`

These files provide the requested time-domain waveform view, frequency-domain view, and objective/fidelity progression without requiring a notebook-only workflow.

### Generated GRAPE Plot

The figure below shows the GRAPE convergence curve and optimized waveform for a qubit $|g,0\rangle \to |e,0\rangle$ state-preparation problem, produced by `tools/generate_tutorial_plots.py`:

![GRAPE Optimal Control](../assets/images/tutorials/grape_optimal_control.png)

---

## GRAPE Workflow

GRAPE remains useful when you want direct slice-level control on the propagation grid.

For piecewise-constant controls $u_j^{(k)}$, the GRAPE gradient is computed exactly from forward and backward propagation on the same grid used for the dense propagator.

### Minimal GRAPE example

```python
import numpy as np
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    build_control_problem_from_model,
    state_preparation_objective,
)

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9,
    omega_q=2*np.pi*6e9,
    alpha=0.0,
    chi=0.0,
    kerr=0.0,
    n_cav=1,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

problem = build_control_problem_from_model(
    model,
    frame=frame,
    time_grid=PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=20e-9),
    channel_specs=(
        ModelControlChannelSpec(
            name="qubit",
            target="qubit",
            quadratures=("I", "Q"),
            amplitude_bounds=(-8e7, 8e7),
            export_channel="qubit",
        ),
    ),
    objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
)

result = GrapeSolver(GrapeConfig(maxiter=80, seed=42)).solve(problem)
```

---

## Choosing Between Them

Choose structured control when:

- smoothness and hardware realism are the main constraints,
- you want meaningful parameters rather than raw samples,
- you plan to move toward hardware-closed-loop refinement.

Choose GRAPE when:

- you need the extra expressivity of a slice-wise schedule,
- the command waveform itself is the object you want to refine,
- you already have a hardware-aware parameterization such as held-sample controls and want local refinement on top of it.

---

## Practical Notes

### Hardware-aware optimization

Attach a `HardwareModel` to either workflow to optimize through the physical transfer path. The same command-vs-physical distinction is available through `resolve_control_schedule(...)` and the solve result metrics.

### Multi-start GRAPE

GRAPE restarts are handled through `solve_grape_multistart(...)` and `GrapeMultistartConfig(...)`, not through a `GrapeConfig(n_random_restarts=...)` field.

### Leakage penalties

Use `LeakagePenalty(...)` to suppress population outside the retained subspace. There is no `LeakagePenaltyObjective` class in the current public API.

### Replay through the full simulator

After either solve path, validate the final command or physical waveform through the time-domain runtime:

```python
replay = result.evaluate_with_simulator(
    problem,
    model=model,
    frame=frame,
    compiler_dt_s=1e-9,
    waveform_mode="physical",
)
```

---

## See Also

- [Optimal Control API](../api/optimal_control.md)
- [Hardware & Control API](../api/hardware.md)
- [Hardware-Aware Control Tutorial](hardware_context.md)
- [Unitary Synthesis](unitary_synthesis.md)
