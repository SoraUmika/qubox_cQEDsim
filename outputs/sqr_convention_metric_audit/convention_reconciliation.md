# Convention Reconciliation: qubox vs cQED_sim

## Lab-side Reference
- QubitRotation waveform: `w = s*(I+iQ)*exp(-i*phi_eff)*exp(+i*omega*t)`
- SQR tone waveform: `w_n = s_n*w0*exp(-i*phi_eff_n)*exp(+i*omega_n*t)` and `w = sum_n w_n`

## Derived Mapping
- Positive `phi_eff` rotates the IQ phasor clockwise (negative complex-plane angle).
- With `H_drive = Re[w]*sigma_x/2 + Im[w]*sigma_y/2`, axis phase is `phi_axis = arg(w) = -phi_eff`.
- Therefore to implement `R_xy(theta, phi_axis)`, lab waveform needs `phi_eff = -phi_axis`.

## Direct Numerical Equivalence (Two-level)
- Best Rotation match to qubox waveform: phase_sign=-1, omega_sign=+1, fidelity=1.000000000
- Best SQR(single-tone) match to qubox waveform: phase_sign=-1, omega_sign=+1, fidelity=1.000000000
- Interpretation: cQED Rotation and SQR now share the same phase and frequency sign convention.

## Detuning Sign Check
- Rotation-like drive: +delta axis_z=-0.421987, -delta axis_z=0.421987
- SQR-like drive: +delta axis_z=-0.421987, -delta axis_z=0.421987, relative_to_rotation=same

## Convention Table
| Quantity | lab qubox | cQED Rotation | cQED SQR | Required mapping |
|---|---|---|---|---|
| waveform phase factor | exp(-i phi_eff) | exp(+i phase) | exp(+i phase) | phase_cqed = -phi_eff |
| time modulation | exp(+i omega t) | exp(+i carrier t) | exp(+i omega t) | carrier=omega and tone_omega=omega (same sign semantics) |
| detuning knob sign | +domega -> positive IQ phase slope | +carrier -> negative residual Z sign in this frame | +domega -> same sign response as rotation | shared positive detuning convention |