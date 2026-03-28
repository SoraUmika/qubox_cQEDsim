# Tutorial: Frame Sanity Checks & Common Failure Modes

Diagnose common mistakes in rotating-frame definitions, carrier frequencies, and sign conventions that can silently produce incorrect simulation results.

**Notebook:** `tutorials/26_frame_and_convention_sanity_checks.ipynb`

---

## Why Frames Matter

Every simulation in `cqed_sim` runs in a **rotating frame** defined by `FrameSpec`. If the frame frequencies don't match the carrier frequencies of the drives, the dynamics will show spurious oscillations or wrong evolution rates. Getting the frame wrong is one of the most common sources of simulation errors.

---

## Common Failure Modes

### 1. Mismatched Frame and Drive Frequency

If `omega_q_frame` differs from the qubit drive frequency, the qubit sees a detuning:

$$\Delta = \omega_d - \omega_{q,\text{frame}}$$

This produces fast oscillations in the lab frame instead of smooth Rabi nutation.

**Symptom:** $P_e(t)$ oscillates at GHz frequencies instead of MHz Rabi rate.

**Fix:** Ensure `FrameSpec(omega_q_frame=model.omega_q)` and `Pulse(frequency=model.omega_q)`.

### 2. Wrong Sign Convention for χ

The dispersive shift $\chi$ defines how the qubit frequency depends on cavity photon number:

$$\omega_q(n) = \omega_q + n\chi$$

If the sign of $\chi$ is flipped, the frequency shift goes the wrong way. This matters especially for:

- Number-selective pulses (SNAP gates)
- Readout optimization
- Dispersive spectroscopy simulations

**Symptom:** SNAP gates address the wrong Fock-state transition.

**Fix:** Check `model.chi` sign against the convention in [Units, Frames & Conventions](units_frames_conventions.md).

### 3. Missing Rotating-Wave Approximation Artifacts

Running with a too-large time step $dt$ can alias fast counter-rotating terms that should average to zero under the RWA.

**Symptom:** Simulation results depend on $dt$.

**Fix:** Use $dt \leq 1/(10\, f_\text{max})$ where $f_\text{max}$ is the highest frequency difference in the problem.

### 4. Inconsistent Units

All frequencies in `cqed_sim` are in **angular frequency** (rad/s). A common mistake is passing $f$ in Hz instead of $\omega = 2\pi f$:

```python
# WRONG — passes frequency in Hz
model = DispersiveTransmonCavityModel(omega_q=6e9, ...)

# CORRECT — passes angular frequency
model = DispersiveTransmonCavityModel(omega_q=2*np.pi*6e9, ...)
```

---

## Diagnostic Code

```python
import numpy as np
from cqed_sim.core import (
    DispersiveTransmonCavityModel, FrameSpec,
    StatePreparationSpec, qubit_state, fock_state, prepare_state,
)
from cqed_sim.sim import SimulationConfig, simulate_sequence, reduced_qubit_state
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.pulses import Pulse

model = DispersiveTransmonCavityModel(
    omega_c=2*np.pi*5e9, omega_q=2*np.pi*6e9,
    alpha=2*np.pi*(-220e6), chi=2*np.pi*(-2.5e6),
    kerr=2*np.pi*(-2e3), n_cav=4, n_tr=2,
)

# ✅ Correct frame
frame_ok = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

# ❌ Wrong frame — 100 MHz offset
frame_bad = FrameSpec(
    omega_c_frame=model.omega_c,
    omega_q_frame=model.omega_q + 2*np.pi*100e6,  # off by 100 MHz!
)

psi0 = prepare_state(model, StatePreparationSpec(
    qubit=qubit_state("g"), storage=fock_state(0),
))
pulse = Pulse(channel="qubit", t0=0.0, duration=40e-9,
              frequency=model.omega_q, amplitude=2*np.pi*12.5e6, phase=0.0)
compiler = SequenceCompiler(dt=1e-9)

for label, frame in [("correct", frame_ok), ("wrong", frame_bad)]:
    compiled = compiler.compile([pulse])
    result = simulate_sequence(model, compiled, psi0, {},
                               config=SimulationConfig(frame=frame))
    rho_q = reduced_qubit_state(result.final_state)
    pe = float(np.real(rho_q[1, 1]))
    print(f"Frame={label:8s}: P(e) = {pe:.4f}")
# correct → P(e) ≈ 1.0 (π pulse)
# wrong   → P(e) ≈ 0.02 (detuned, barely excites)
```

---

## Checklist

Before running any simulation, verify:

- [ ] All model frequencies include $2\pi$ (angular frequency, rad/s)
- [ ] `FrameSpec` frequencies match the corresponding model frequencies
- [ ] Pulse carrier frequencies match the intended transition
- [ ] $dt$ is small enough to resolve the fastest dynamics
- [ ] $\chi$ sign is consistent with the expected dispersive shift direction
- [ ] Hilbert-space dimensions are sufficient (see [Truncation Convergence](truncation_convergence.md))

---

## See Also

- [Units, Frames & Conventions](units_frames_conventions.md) — full convention reference
- [Minimal Dispersive Model](minimal_dispersive_model.md) — correct model construction
- [Truncation Convergence](truncation_convergence.md) — dimension selection
