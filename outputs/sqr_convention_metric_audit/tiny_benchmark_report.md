# Tiny SQR Benchmark (n=0,1)

## Targets
- x90_x90: theta=[1.5707963267948966, 1.5707963267948966], phi=[0.0, 0.0]
- x90_y90: theta=[1.5707963267948966, 1.5707963267948966], phi=[0.0, 1.5707963267948966]
- x180_identity: theta=[3.141592653589793, 0.0], phi=[0.0, 0.0]

## Results
| Target | Case | mean process fidelity | phase_sensitive_rms_rad |
|---|---|---:|---:|
| x90_x90 | B | 0.453299 | 1.458488 |
| x90_x90 | C | 0.446175 | 1.657309 |
| x90_x90 | D | 0.462295 | 1.606632 |
| x90_x90 | E | 0.466140 | 1.600545 |
| x90_y90 | B | 0.463059 | 1.722209 |
| x90_y90 | C | 0.446656 | 1.854572 |
| x90_y90 | D | 0.467902 | 1.699844 |
| x90_y90 | E | 0.470554 | 1.680904 |
| x180_identity | B | 0.436699 | 1.200978 |
| x180_identity | C | 0.393805 | 1.276097 |
| x180_identity | D | 0.499435 | 0.897431 |
| x180_identity | E | 0.499466 | 0.897087 |

- Best optimized tiny-case fidelity: target=x180_identity, case=E, fidelity=0.499466, phase_sensitive_rms=0.897087
- Conclusion: tiny benchmark does not reach 0.95; likely limited by pulse ansatz/optimizer settings in this configuration.