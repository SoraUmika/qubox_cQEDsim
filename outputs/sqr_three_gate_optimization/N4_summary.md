# N=4 Support-Aware Three-SQR Optimization Summary

## Setup
- Active support: S={0,1,2,3}
- Objective: support-aware
- Duration-aware strategy score enabled (penalizes longer gates)
- Tone manifold restriction: support_plus_boundary (0..4)
- Solver grid: dt_eval=1 ns, dt_opt=4 ns

## Best 3-gate run (shared strategy)
- Best coarse strategy: D_1.0
- Final mode/duration: amp_phase_detuning @ 1.00 us
- Mean active-weighted process fidelity: 0.976023
- Minimum active block process fidelity: 0.285822
- Mean support-state fidelity: 0.886996
- Mean support leakage max: 2.307e-08

### Per-gate (final run)
- Gate 2 SQR_527e17733de113ecaead4876b2bcafeb: active_mean=0.980803, active_min=0.314842, support_state_mean=0.885228, support_leak_max=2.259e-08
- Gate 5 SQR_bc8194a85a260be4c0959c6c4d8d7ae1: active_mean=0.960278, active_min=0.285822, support_state_mean=0.883376, support_leak_max=3.103e-08
- Gate 8 SQR_789278ba62949511b06e9d76a4bb8437: active_mean=0.986989, active_min=0.313292, support_state_mean=0.892383, support_leak_max=1.560e-08

## Single-gate best combo (for troubleshooting)
- E_g_1.5 (amp_phase_detuning_ramp, 1.50 us): active_mean=0.959610, active_min=0.592676

## Interpretation
- For this decomposition (only n=0 target rotation, n=1..3 identity on support), the optimizer can drive high active-weighted mean fidelity but worst active block remains limited by selectivity/crosstalk.
- Duration-aware scoring currently favors 1.0 us; longer durations tested did not consistently improve the worst active block in this parameterization.
