# Envelope Ansatz Comparison

- Compared Case D (`amp+phase+detuning`) for Gaussian vs flat-top Gaussian at baseline and best longer duration per profile.

## Profile: structured_seed41
| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | gaussian | 0.674749 | 0.024049 | 1.237418 | 1.2954/8.1655 | 0.743165 |
| 1.0 | flat_top | 0.495923 | 0.008928 | 1.828361 | 1.1515/5.2352 | 0.587875 |
| 1.5 | gaussian | 0.548496 | 0.131610 | 1.254848 | 2.4010/17.4996 | 0.569171 |
| 1.5 | flat_top | 0.663025 | 0.258757 | 0.981651 | 1.6278/10.0883 | 0.563666 |

## Profile: moderate_random_seed64
| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | gaussian | 0.724546 | 0.052113 | 0.927676 | 0.7145/1.9989 | 0.784948 |
| 1.0 | flat_top | 0.476480 | 0.005133 | 1.299033 | 0.6767/1.9979 | 0.708194 |
| 1.5 | gaussian | 0.486212 | 0.115007 | 1.310218 | 0.9098/2.1177 | 0.632847 |
| 1.5 | flat_top | 0.504762 | 0.101127 | 1.262784 | 0.7669/2.2990 | 0.635029 |

## Profile: hard_random_seed87
| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | gaussian | 0.741815 | 0.207398 | 0.966843 | 0.6212/1.2071 | 0.757325 |
| 1.0 | flat_top | 0.494695 | 0.072461 | 1.348978 | 0.5939/1.2093 | 0.711526 |
| 2.5 | gaussian | 0.581919 | 0.205518 | 1.105203 | 0.7542/1.1852 | 0.819921 |
| 2.5 | flat_top | 0.658441 | 0.067771 | 1.108785 | 0.7510/1.1670 | 0.889878 |
