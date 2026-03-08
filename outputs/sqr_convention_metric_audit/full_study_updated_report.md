# Updated Full SQR Study (After Convention + Metric Audit)

## Profile: structured_seed17
| Case | mean process fidelity | min process fidelity | phase_sensitive_rms | phase_axis_rms | pre-Z rms | post-Z rms | state fidelity mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 0.540414 | 0.018021 | 1.928817 | 0.860911 | 1.275730 | 1.162618 | 0.765920 |
| C | 0.607552 | 0.026365 | 1.302291 | 0.163561 | 1.101673 | 0.674927 | 0.821018 |
| D | 0.711553 | 0.018978 | 1.121142 | 0.150412 | 0.928545 | 0.610033 | 0.857353 |
| E | 0.717944 | 0.020424 | 1.187678 | 0.132594 | 0.978519 | 0.659924 | 0.865643 |
- Case B mismatch example (n=6): process infidelity=0.0607, phi-axis error=-0.1992, pre-Z=-1.0697, post-Z=1.0670.
- Cross-talk (neighbor mean): Case B=0.8179, Case D=0.6742
- Best phase-sensitive case: D (phase_sensitive_rms=1.121142).

## Profile: hard_random_seed36
| Case | mean process fidelity | min process fidelity | phase_sensitive_rms | phase_axis_rms | pre-Z rms | post-Z rms | state fidelity mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| B | 0.487021 | 0.065697 | 1.412519 | 0.854535 | 0.985684 | 0.541669 | 0.591713 |
| C | 0.658373 | 0.280529 | 0.969809 | 0.134818 | 0.618556 | 0.734671 | 0.761527 |
| D | 0.649137 | 0.263291 | 0.995826 | 0.176108 | 0.653529 | 0.730449 | 0.777464 |
| E | 0.699804 | 0.254597 | 1.000797 | 0.171267 | 0.689149 | 0.705220 | 0.822139 |
- Case B mismatch example (n=2): process infidelity=0.6377, phi-axis error=0.9160, pre-Z=-2.0942, post-Z=0.2923.
- Cross-talk (neighbor mean): Case B=0.8756, Case D=0.8365
- Best phase-sensitive case: C (phase_sensitive_rms=0.969809).

## Conclusion
- The updated metrics are phase-sensitive and differ across cases, unlike the previous frozen metric.
- Optimized pulses can improve blockwise fidelity on selected targets; phase cancellation quality is case-dependent.
- Positive detuning sign must be interpreted carefully: current SQR internal sign is opposite to Rotation for equal numeric detuning.