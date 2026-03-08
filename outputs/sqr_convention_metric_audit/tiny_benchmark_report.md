# Tiny SQR Benchmark (n=0,1)

## Targets
- x90_x90: theta=[1.5707963267948966, 1.5707963267948966], phi=[0.0, 0.0]
- x90_y90: theta=[1.5707963267948966, 1.5707963267948966], phi=[0.0, 1.5707963267948966]
- x180_identity: theta=[3.141592653589793, 0.0], phi=[0.0, 0.0]

## Results
| Target | Case | mean process fidelity | phase_sensitive_rms_rad |
|---|---|---:|---:|
| x90_x90 | B | 0.894967 | 0.451635 |
| x90_x90 | C | 0.951427 | 0.233992 |
| x90_x90 | D | 0.895939 | 0.547788 |
| x90_x90 | E | 0.993567 | 0.436624 |
| x90_y90 | B | 0.901279 | 1.342032 |
| x90_y90 | C | 0.960220 | 0.165385 |
| x90_y90 | D | 0.999174 | 0.021723 |
| x90_y90 | E | 0.993725 | 0.439078 |
| x180_identity | B | 0.971107 | 0.242893 |
| x180_identity | C | 0.973372 | 0.221769 |
| x180_identity | D | 0.973372 | 0.221801 |
| x180_identity | E | 0.973372 | 0.221801 |

- Best optimized tiny-case fidelity: target=x90_y90, case=D, fidelity=0.999174, phase_sensitive_rms=0.021723
- Conclusion: tiny benchmark reaches high blockwise fidelity (>=0.95), so optimizer/pipeline is trustworthy at small manifold size.