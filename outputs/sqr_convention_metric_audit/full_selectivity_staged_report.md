# Selectivity-Focused Full SQR Report

- Baseline settings: chi = -2.84 MHz, duration = 1.0 us, Gaussian envelope.
- Cases: B (naive), C (amp+phase), D (amp+phase+detuning), E (with chirp).

## Profile: structured_seed61
| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.614371 | 0.318287 | 2.003510 | 1.120278 | 1.203545 | 1.144775 | 1.8850/6.1938 | 0.537254 |
| C | 0.608037 | 0.318292 | 2.026244 | 1.120278 | 1.233845 | 1.152506 | nan/nan | 0.602147 |
| D | 0.608095 | 0.318292 | 2.026656 | 1.120278 | 1.234638 | 1.152380 | 1.8526/6.2311 | 0.612375 |
| E | 0.614740 | 0.318283 | 2.027092 | 1.120278 | 1.224621 | 1.163779 | nan/nan | 0.587408 |
- Case D loss terms: infidelity=3.9191e-01, phase_axis=3.1376e-01, theta=8.3494e-01, residual_z=1.9064e-01, state=1.1373e-01, off_block=9.2932e-03, selectivity_mean=2.9614e-05, selectivity_max=8.4952e-05, regularization=1.3609e-03
- Case E loss terms: infidelity=3.8526e-01, phase_axis=3.1376e-01, theta=8.2093e-01, residual_z=1.8609e-01, state=1.1644e-01, off_block=2.3982e-02, selectivity_mean=9.5708e-05, selectivity_max=5.9432e-04, regularization=2.7366e-03

## Profile: moderate_random_seed84
| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.684908 | 0.515815 | 2.245542 | 1.244771 | 1.254168 | 1.385664 | 2886163.6362/33584513.6132 | 0.616579 |
| C | 0.680320 | 0.515881 | 2.152076 | 1.244771 | 1.182368 | 1.297684 | nan/nan | 0.697604 |
| D | 0.680226 | 0.515871 | 2.163653 | 1.244771 | 1.193176 | 1.307008 | 1.8525/6.2313 | 0.683803 |
| E | 0.696946 | 0.515845 | 1.671073 | 1.244771 | 0.697379 | 0.869881 | nan/nan | 0.676585 |
- Case D loss terms: infidelity=3.1977e-01, phase_axis=3.8736e-01, theta=8.8823e-01, residual_z=5.1392e-01, state=9.9032e-02, off_block=4.4759e-03, selectivity_mean=2.9129e-06, selectivity_max=6.6155e-06, regularization=3.4071e-03
- Case E loss terms: infidelity=3.0305e-01, phase_axis=3.8736e-01, theta=1.0173e+00, residual_z=8.0446e-02, state=9.0522e-02, off_block=6.7736e-02, selectivity_mean=4.8480e-05, selectivity_max=3.1859e-04, regularization=8.8814e-03

## Profile: hard_random_seed107
| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 0.597921 | 0.011886 | 1.881139 | 1.657507 | 0.627514 | 0.630542 | 1.8573/6.2284 | 0.716047 |
| C | 0.594657 | 0.012284 | 1.862377 | 1.657507 | 0.587858 | 0.612817 | nan/nan | 0.652647 |
| D | 0.595341 | 0.012289 | 1.862141 | 1.657507 | 0.586738 | 0.613171 | 1.8524/6.2312 | 0.647566 |
| E | 0.616235 | 0.012009 | 1.878392 | 1.657507 | 0.577462 | 0.669005 | nan/nan | 0.638856 |
- Case D loss terms: infidelity=4.0466e-01, phase_axis=6.8683e-01, theta=5.2016e-01, residual_z=5.4204e-02, state=1.0573e-01, off_block=1.7159e-02, selectivity_mean=4.9099e-04, selectivity_max=3.2324e-03, regularization=2.5810e-03
- Case E loss terms: infidelity=3.8376e-01, phase_axis=6.8683e-01, theta=5.0300e-01, residual_z=3.3919e-02, state=1.0613e-01, off_block=2.3088e-02, selectivity_mean=2.2112e-04, selectivity_max=1.4687e-03, regularization=4.0587e-03

## Baseline Limitation
- At T=1.0 us and chi=-2.84 MHz, the dominant limiter remains manifold selectivity (neighbor leakage / off-target action), not only phase bookkeeping.