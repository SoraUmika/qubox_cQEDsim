# Detuning Convention Implementation Note

## Root Cause
- Rotation path (`Pulse._sample_analytic`) used `exp(+i*(carrier*t + phase))`.
- SQR multitone path used `exp(+i*phase)*exp(-i*omega*t)`.
- This made positive numeric detuning map to opposite IQ phase slope between Rotation and SQR.

## Minimal Fix Applied
- Updated `cqed_sim/pulses/envelopes.py::multitone_gaussian_envelope` to use `exp(+i*omega*t)`.
- Updated `cqed_sim/pulses/calibration.py::build_sqr_tone_specs` to set `omega_waveform = -manifold_transition_frequency(...)` so resonant physical tones remain unchanged while parameter semantics are unified.
- Updated `examples/sqr_multitone_study.py::multitone_envelope` and spectrum marker sign to the same canonical convention.

## Post-fix Verification
- Rotation sign match vs lab waveform: phase_sign=-1, omega_sign=+1.
- SQR sign match vs lab waveform: phase_sign=-1, omega_sign=+1.
- Detuning axis sign comparison: rotation(+delta)=-0.421987, sqr(+delta)=-0.421987, relative=same.

## Canonical Convention
| Quantity | Canonical meaning |
|---|---|
| Waveform phasor | `w(t)=I(t)+iQ(t)` |
| Pulse phase knob | implemented as `exp(+i*phase)` so matching lab `exp(-i*phi_eff)` requires `phase=-phi_eff` |
| Detuning knob `d_omega` | increases IQ phase slope via `exp(+i*d_omega*t)` |
| Effective detuning sign in rotating-frame block extraction | frame-dependent; use the block-unitary sign check, not IQ slope alone |
| Target axis mapping | `R_xy(theta,phi_axis)` with `phi_axis=arg(w)` and `I->+x`, `Q->+y` |