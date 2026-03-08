# Phase Metric Audit

## Legacy Code Path
- Legacy phase metric used only `relative_block_phase_rad` from simulated blocks.
- That quantity did not reference target block phase, so it can remain nearly unchanged across optimized cases.

## Audit Result
- Legacy metric span across cases: 1.768729e-06 rad
- Current study `phase_rms_rad` span across cases: 7.271754e-01 rad
- Corrected metric span across cases: 8.113381e-01 rad
- Corrected metric recomputes phase-sensitive mismatch from `U_err = U_target^dag U_sim` with a gauge-fixed `Z-Rxy-Z` decomposition.

## Corrected Case Metrics
| Case | legacy phase_rms_rad | current phase_rms_rad | new phase_sensitive_rms_rad | mean process fidelity | residual pre-Z rms | residual post-Z rms |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.870572 | 0.860911 | 1.928817 | 0.540414 | 1.275730 | 1.162618 |
| C | 0.870574 | 0.153055 | 1.290263 | 0.609644 | 1.085537 | 0.680414 |
| D | 0.870572 | 0.143806 | 1.117479 | 0.712010 | 0.925656 | 0.609295 |
| E | 0.870574 | 0.133735 | 1.186441 | 0.718437 | 0.980888 | 0.653924 |