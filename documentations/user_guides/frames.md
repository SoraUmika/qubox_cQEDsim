# Frames

The rotating frame determines which fast oscillations are removed from the simulation dynamics. A well-chosen frame simplifies the Hamiltonian and makes time-dependent behavior easier to interpret.

---

## FrameSpec

```python
from cqed_sim.core import FrameSpec

frame = FrameSpec(
    omega_c_frame=2 * np.pi * 5.0e9,   # Cavity/storage frame frequency (rad/s)
    omega_q_frame=2 * np.pi * 6.0e9,   # Qubit frame frequency (rad/s)
    omega_r_frame=2 * np.pi * 7.5e9,   # Readout frame frequency (rad/s)
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `omega_c_frame` | `float` | `0.0` | Cavity/storage rotating-frame frequency |
| `omega_q_frame` | `float` | `0.0` | Qubit rotating-frame frequency |
| `omega_r_frame` | `float` | `0.0` | Readout rotating-frame frequency |

!!! note
    `omega_s_frame` is a read-only alias for `omega_c_frame`, provided for clarity in three-mode workflows.

---

## Common Frame Choices

### Lab Frame

No rotation. All bare frequencies appear in the Hamiltonian:

```python
lab_frame = FrameSpec()  # All zeros
```

### Resonant Frame

Set frame frequencies equal to model frequencies. Removes bare rotations—only interaction terms and detunings remain:

```python
frame = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q,
)
```

For three-mode models:

```python
frame = FrameSpec(
    omega_c_frame=model.omega_s,
    omega_q_frame=model.omega_q,
    omega_r_frame=model.omega_r,
)
```

### Single-Mode Frame

Rotate only one mode into its resonant frame:

```python
frame = FrameSpec(omega_q_frame=model.omega_q)
```

---

## Effect on the Hamiltonian

In the rotating frame, the static Hamiltonian contains detuning terms:

$$\delta_c = \omega_c - \omega_c^{\text{frame}}, \quad \delta_q = \omega_q - \omega_q^{\text{frame}}$$

When frame frequencies match model frequencies, $\delta_c = \delta_q = 0$, and the static Hamiltonian contains only interaction terms (χ, Kerr, anharmonicity).

---

## Effect on Transition Frequencies

Transition frequencies reported by model methods (e.g., `manifold_transition_frequency`, `transmon_transition_frequency`) shift by the frame:

```python
# Lab-frame transition
omega_lab = model.manifold_transition_frequency(n=0, frame=None)

# Rotating-frame transition (shifted by frame)
omega_rot = model.manifold_transition_frequency(n=0, frame=frame)
```

---

## Frame and Pulse Carriers

For user-facing code, the preferred physical drive tone in a frame with frequency $\omega_{\text{frame}}$ is:

$$\omega_{\text{drive}} = \omega_{\text{frame}} + \omega_{\text{transition}}$$

The raw low-level `Pulse.carrier` compatibility field is still specified in the rotating frame. A pulse resonant with a transition at rotating-frame frequency $\omega_{\text{transition}}$ must therefore have:

$$\text{Pulse.carrier} = -\omega_{\text{transition}}$$

Use the helper functions:

```python
from cqed_sim.core import (
    drive_frequency_for_transition_frequency,
    internal_carrier_from_drive_frequency,
    transition_frequency_from_drive_frequency,
)

drive_frequency = drive_frequency_for_transition_frequency(omega_transition, frame.omega_q_frame)
carrier = internal_carrier_from_drive_frequency(drive_frequency, frame.omega_q_frame)
omega_back = transition_frequency_from_drive_frequency(drive_frequency, frame.omega_q_frame)
```
