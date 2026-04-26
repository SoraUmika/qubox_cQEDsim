# `cqed_sim.readout`

`cqed_sim.readout` contains reusable readout-analysis helpers for the
strong-readout stack.

## Entry Points

- `input_output.py`: output-field selection and integration.  The convention is
  `sqrt(kappa_r) a` without an explicit filter and `sqrt(kappa_f) f` when a
  filter mode is part of the Hamiltonian.
- `classifiers.py`: matched-filter classification, Gaussian maximum-likelihood
  classification, and a small adapter interface for time-resolved path/HMM-style
  classifiers.

## Units and Records

Input-output amplitudes use the same angular-frequency/time convention as the
Hamiltonian simulation.  Measurement records are arrays whose last axis is time.
The helper `confusion_matrix(...)` returns `P(predicted | prepared)` with rows
ordered by predicted state and columns ordered by prepared state.

Transfer functions can be applied in the frequency domain by passing a callable
that receives angular frequencies and returns a complex response.
