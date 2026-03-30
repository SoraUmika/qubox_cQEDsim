---
hide:
  - navigation
  - toc
---

# cqed_sim

<div class="grid" markdown>

<div markdown>

**Hardware-faithful time-domain cQED pulse simulator built on QuTiP.**

`cqed_sim` is a Python library for pulse-level simulation of circuit quantum electrodynamics systems. It models transmon–cavity hardware with explicit drive schedules, realistic hardware distortion, open-system dynamics, and integrated tools for calibration, optimal control, and gate synthesis.

Designed for lab-scale hardware-faithful simulation where physical conventions, unit safety, and reproducibility matter.

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[API Reference](api/overview.md){ .md-button }

</div>

<div markdown>

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.pulses import Pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9, omega_q=2*np.pi*6e9,
    alpha=2*np.pi*(-200e6), chi=2*np.pi*(-2.84e6),
    kerr=2*np.pi*(-2e3), n_cav=8, n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c,
                  omega_q_frame=model.omega_q)
pulse = Pulse("q", 0.0, 80e-9, gaussian_envelope, amp=np.pi/2)
compiled = SequenceCompiler(dt=2e-9).compile([pulse], t_end=100e-9)
result = simulate_sequence(
    model, compiled, model.basis_state(0, 0),
    {"q": "qubit"}, config=SimulationConfig(frame=frame),
)
```

</div>

</div>

---

## What `cqed_sim` Does

<div class="grid cards" markdown>

-   :material-cube-outline:{ .lg .middle } **System Modeling**

    ---

    Two-mode (qubit + storage), three-mode (qubit + storage + readout), and generalized N-mode models via `UniversalCQEDModel`. Supports multilevel transmon, dispersive coupling, cross-Kerr, self-Kerr, and exchange interactions.

    [:octicons-arrow-right-24: Defining models](user_guides/defining_models.md)

-   :material-waveform:{ .lg .middle } **Pulse-Level Simulation**

    ---

    Build drive schedules from `Pulse` objects with standard envelopes (Gaussian, DRAG, square, cosine-filtered). Compile onto a time grid with `SequenceCompiler`, then run through QuTiP's ODE solver or a dense piecewise-constant backend.

    [:octicons-arrow-right-24: Pulse construction](user_guides/pulse_construction.md)

-   :material-chip:{ .lg .middle } **Hardware Distortion**

    ---

    Model realistic DAC/AWG effects: low-pass filtering, IQ skew, quantization, crosstalk, and arbitrary FIR filters. The `HardwareContext` pipeline connects programmed controls to physical waveforms via explicit calibration maps.

    [:octicons-arrow-right-24: Hardware & control](api/hardware.md)

-   :material-sine-wave:{ .lg .middle } **Floquet Analysis**

    ---

    Compute quasienergies, one-period propagators, and Fourier components for periodically driven closed-system Hamiltonians. Includes branch tracking for multiphoton resonance identification.

    [:octicons-arrow-right-24: Floquet API](api/floquet.md)

-   :material-tune:{ .lg .middle } **Optimal Control (GRAPE)**

    ---

    Model-backed GRAPE with piecewise-constant or held-sample parameterization. Optional hardware maps apply FIR filters, IQ-radius limits, and boundary windows inside the gradient loop. Supports hardware-aware replay and open-system evaluation.

    [:octicons-arrow-right-24: Optimal control](api/optimal_control.md)

-   :material-robot:{ .lg .middle } **RL Hybrid Control**

    ---

    Gym-compatible reinforcement learning environment wrapping the cQED simulator. Measurement-like observations, configurable reward functions, domain randomization, and hardware-representative action spaces.

    [:octicons-arrow-right-24: RL control](api/rl_control.md)

-   :material-ruler:{ .lg .middle } **Calibration & Tomography**

    ---

    SQR gate calibration, Rabi/Ramsey/T₁/T₂ echo calibration targets, Fock-resolved tomography, DRAG tuning, and all-XY protocols. Parameter translation between bare and dressed frequencies.

    [:octicons-arrow-right-24: Calibration API](api/calibration.md)

-   :material-chart-scatter-plot:{ .lg .middle } **Measurement & Readout**

    ---

    `QubitMeasurementSpec` for synthetic measurement (probabilities, counts, I/Q samples). Full physical readout chain modeling: resonator, Purcell filter, amplifier, and stochastic master equation replay.

    [:octicons-arrow-right-24: Measurement API](api/measurement.md)

</div>

---

## Physics & Conventions at a Glance

| Quantity | Convention |
|---|---|
| Hamiltonian coefficients | **rad/s** — `omega_q = 2π × 6 GHz` |
| Time arguments | **seconds** — `duration = 80e-9` |
| Tensor ordering | **qubit first**, cavity second: `\|q, n⟩` |
| Dispersive term | `+χ n_c n_q` in Hamiltonian (`χ < 0` is typical) |
| Drive waveform | `amp × envelope(t_rel) × exp(+i(carrier·t + phase))` |
| Carrier convention | raw `Pulse.carrier = −ω_transition` in rotating frame; prefer positive drive-tone helpers for user-facing code |
| Computational basis | `\|g⟩ = \|0⟩`, `\|e⟩ = \|1⟩` |

!!! warning "Read the conventions page before writing pulses"
    The carrier sign and dispersive-shift sign differ from some other cQED packages.
    See [Physics & Conventions](physics_conventions.md) for the complete reference.

---

## Navigation

<div class="grid cards" markdown>

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    First simulation in under 5 minutes — install, build a model, run a qubit π-pulse, and read out the result.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

-   :material-book-open-variant:{ .lg .middle } **User Guides**

    ---

    Step-by-step documentation for every part of the simulation workflow: models, frames, pulses, sequences, solvers, observables, and open-system dynamics.

    [:octicons-arrow-right-24: User guides](user_guides/defining_models.md)

-   :material-school:{ .lg .middle } **Tutorials**

    ---

    Structured Jupyter notebook curriculum organized by topic. Covers core workflows, bosonic physics, advanced protocols (GRAPE, RL, unitary synthesis), and convention validation.

    [:octicons-arrow-right-24: Tutorials](tutorials/index.md)

-   :material-code-braces:{ .lg .middle } **Examples**

    ---

    Standalone scripts and study helpers for common simulation tasks — with links to companion tutorial notebooks.

    [:octicons-arrow-right-24: Examples](examples.md)

-   :material-book-alphabet:{ .lg .middle } **API Reference**

    ---

    Complete public API organized by submodule with signatures, parameters, and usage notes.

    [:octicons-arrow-right-24: API reference](api/overview.md)

-   :material-atom:{ .lg .middle } **Physics & Conventions**

    ---

    Hamiltonian definitions, sign conventions, tensor ordering, unit conventions, gate library conventions, and the dissipation model.

    [:octicons-arrow-right-24: Conventions](physics_conventions.md)

</div>
