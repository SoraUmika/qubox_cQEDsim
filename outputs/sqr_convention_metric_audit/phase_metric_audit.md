# Phase Metric Audit

## Legacy Code Path
- Legacy phase metric used only `relative_block_phase_rad` from simulated blocks.
- That quantity did not reference target block phase, so it can remain nearly unchanged across optimized cases.

## Audit Result
- Legacy metric span across cases: 8.891470e-02 rad
- Current study `phase_rms_rad` span across cases: 0.000000e+00 rad
- Corrected metric span across cases: 2.073659e-02 rad
- Corrected metric recomputes phase-sensitive mismatch from `U_err = U_target^dag U_sim` with a gauge-fixed `Z-Rxy-Z` decomposition.

## Corrected Case Metrics
| Case | legacy phase_rms_rad | current phase_rms_rad | new phase_sensitive_rms_rad | mean process fidelity | residual pre-Z rms | residual post-Z rms |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.694718 | 1.160674 | 2.056051 | 0.629869 | 1.351344 | 1.026670 |
| C | 0.663472 | 1.160674 | 2.039819 | 0.624270 | 1.216478 | 1.154936 |
| D | 0.664259 | 1.160674 | 2.039578 | 0.624415 | 1.216410 | 1.154581 |
| E | 0.752386 | 1.160674 | 2.035315 | 0.644203 | 1.349260 | 0.987338 |