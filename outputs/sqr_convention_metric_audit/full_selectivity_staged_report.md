# Selectivity-Focused Full SQR Report

- Baseline settings: chi = -2.84 MHz, duration = 1.0 us, Gaussian envelope.
- Cases: B (naive), C (amp+phase), D (amp+phase+detuning), E (with chirp).

## Profile: structured_seed61
| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.531834 | 0.025970 | 1.783537 | 0.857402 | 1.275750 | 0.904615 | 0.9650/3.4317 | 0.537254 |
| C | 0.625387 | 0.034889 | 1.144241 | 0.140206 | 0.946972 | 0.626796 | nan/nan | 0.696740 |
| D | 0.740713 | 0.041940 | 1.151902 | 0.121690 | 0.927897 | 0.671623 | 0.7636/2.3665 | 0.724842 |
| E | 0.754390 | 0.036366 | 1.167387 | 0.104462 | 0.941993 | 0.681563 | nan/nan | 0.740050 |
- Case D loss terms: infidelity=2.5929e-01, phase_axis=3.7021e-03, theta=2.9718e-01, residual_z=2.8837e-01, state=5.1939e-02, off_block=0.0000e+00, selectivity_mean=3.2691e-04, selectivity_max=6.1761e-04, regularization=3.8634e-02
- Case E loss terms: infidelity=2.4561e-01, phase_axis=2.7281e-03, theta=2.8964e-01, residual_z=2.9314e-01, state=4.8749e-02, off_block=0.0000e+00, selectivity_mean=4.5153e-05, selectivity_max=5.4012e-05, regularization=6.8787e-03

## Profile: moderate_random_seed84
| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.508306 | 0.007037 | 1.543693 | 0.872610 | 0.994911 | 0.794790 | 0.9567/2.3102 | 0.616579 |
| C | 0.594679 | 0.316177 | 1.192525 | 0.120420 | 1.077193 | 0.497263 | nan/nan | 0.726508 |
| D | 0.691676 | 0.357113 | 1.183044 | 0.121726 | 1.051171 | 0.528978 | 0.8070/1.6253 | 0.738079 |
| E | 0.701171 | 0.355505 | 1.024155 | 0.114116 | 0.667100 | 0.768666 | nan/nan | 0.755608 |
- Case D loss terms: infidelity=3.0832e-01, phase_axis=3.7043e-03, theta=2.7614e-01, residual_z=1.6629e-01, state=5.9368e-02, off_block=0.0000e+00, selectivity_mean=4.9886e-04, selectivity_max=1.8908e-03, regularization=4.4226e-02
- Case E loss terms: infidelity=2.9883e-01, phase_axis=3.2556e-03, theta=2.6690e-01, residual_z=1.7051e-01, state=5.4530e-02, off_block=0.0000e+00, selectivity_mean=3.6328e-05, selectivity_max=4.4964e-05, regularization=1.0630e-02

## Profile: hard_random_seed107
| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.531279 | 0.005422 | 1.623889 | 0.866565 | 0.755236 | 1.147039 | 0.8613/2.1240 | 0.716047 |
| C | 0.611319 | 0.148497 | 1.200285 | 0.179376 | 0.781955 | 0.892779 | nan/nan | 0.720444 |
| D | 0.537802 | 0.104143 | 1.434192 | 0.424848 | 0.930986 | 1.004827 | 0.8901/2.1240 | 0.522622 |
| E | 0.611335 | 0.147123 | 1.359961 | 0.384394 | 0.868234 | 0.973604 | nan/nan | 0.659501 |
- Case D loss terms: infidelity=4.6220e-01, phase_axis=4.5124e-02, theta=4.7913e-01, residual_z=2.1510e-01, state=1.4200e-01, off_block=0.0000e+00, selectivity_mean=8.1010e-03, selectivity_max=5.3889e-02, regularization=8.5066e-03
- Case E loss terms: infidelity=3.8866e-01, phase_axis=3.6940e-02, theta=3.8556e-01, residual_z=2.2982e-01, state=9.7558e-02, off_block=0.0000e+00, selectivity_mean=7.8672e-03, selectivity_max=5.2323e-02, regularization=7.2823e-03

## Baseline Limitation
- At T=1.0 us and chi=-2.84 MHz, the dominant limiter remains manifold selectivity (neighbor leakage / off-target action), not only phase bookkeeping.