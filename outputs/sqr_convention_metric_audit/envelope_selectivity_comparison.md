# Envelope Ansatz Comparison

- Compared Case D (`amp+phase+detuning`) for Gaussian vs flat-top Gaussian at baseline and best longer duration per profile.

## Profile: structured_seed41
| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | gaussian | 0.690566 | 0.371823 | 1.947906 | 1.8527/6.2312 | 0.681681 |
| 1.0 | flat_top | 0.709811 | 0.369837 | 2.082492 | 1.7147/5.9444 | 0.695867 |
| 2.5 | gaussian | 0.729017 | 0.334134 | 1.371955 | 14.8611/162.8458 | 0.722853 |
| 2.5 | flat_top | 0.728989 | 0.371789 | 2.482502 | 1856700.4197/22280372.5376 | 0.726418 |

## Profile: moderate_random_seed64
| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | gaussian | 0.592202 | 0.349965 | 1.818208 | 1.8526/6.2312 | 0.706955 |
| 1.0 | flat_top | 0.600311 | 0.346097 | 1.800762 | 2717641.8360/32573382.7281 | 0.705794 |
| 2.0 | gaussian | 0.468061 | 0.041410 | 2.124479 | 1.6156/6.0431 | 0.609346 |
| 2.0 | flat_top | 0.462805 | 0.132247 | 2.285176 | 1.6063/6.0513 | 0.621400 |

## Profile: hard_random_seed87
| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |
|---:|---|---:|---:|---:|---:|---:|
| 1.0 | gaussian | 0.558340 | 0.073095 | 2.042816 | 1.8526/6.2309 | 0.658588 |
| 1.0 | flat_top | 0.581990 | 0.071318 | 2.426395 | 1.8370/5.9526 | 0.692054 |
| 2.0 | gaussian | 0.396744 | 0.020042 | 2.095105 | 12046765.5153/144330298.2758 | 0.556829 |
| 2.0 | flat_top | 0.406472 | 0.022783 | 2.614604 | 17.6281/148.4057 | 0.270940 |
