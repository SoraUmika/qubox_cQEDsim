# Tutorial Physics Verification Audit

Created: 2026-03-27 17:35 local time

## Confirmed Issues

### 1. Tutorial 04 used the wrong square-pulse `pi`-time formula

- What:
  - `tutorials/04_qubit_drive_and_basic_population_dynamics.ipynb` described the resonant square-pulse population transfer as if `t_pi = pi / Omega`.
  - In the runtime convention documented elsewhere in the repo, the resonant two-level drive enters as `H_drive = Omega sigma_x`, so the correct population law is `P_e(t) = sin^2(Omega t)` and therefore `t_pi = pi / (2 Omega)`.
- Where:
  - source generator: `tutorials/_generate_tutorials.py`
  - generated notebook: `tutorials/04_qubit_drive_and_basic_population_dynamics.ipynb`
- Affected components:
  - foundational drive tutorial
  - downstream user intuition for Tutorials 09 and 10
- Why inconsistent:
  - `documentations/PHYSICS_CORRECTNESS_EVALUATION.md` already states the repo's square-pulse convention as `P_e(t) = sin^2(Omega t)` with `t_pi = pi / (2 Omega)`.
  - The tutorial used the older/common alternate parameterization without adapting it to the repo's actual drive normalization.
- Consequences:
  - the notebook labeled a near-identity `2pi` pulse as a `pi` pulse.
  - users reading the notebook could infer the wrong amplitude-duration calibration law for the rest of the tutorial track.

### 2. Tutorial 13 claimed Hahn-echo refocusing while simulating only Markovian dephasing and reading out the wrong final basis

- What:
  - `tutorials/13_spin_echo_and_dephasing_mitigation.ipynb` compared a Ramsey-style sequence and a Hahn echo under the same Lindblad `NoiseSpec(tphi=...)` model and claimed that the echo suppressed static phase accumulation.
  - The notebook also used a final `x90` pulse for the echo sequence, which produced a complementary population readout rather than a like-for-like refocused population signal.
- Where:
  - source generator: `tutorials/_generate_tutorials.py`
  - generated notebook: `tutorials/13_spin_echo_and_dephasing_mitigation.ipynb`
- Affected components:
  - coherence / echo tutorial
  - user-facing explanation of what Hahn echo actually refocuses
- Why inconsistent:
  - quasi-static detuning refocusing is not the same physical process as Markovian `T_phi` decay.
  - the original tutorial did not include quasi-static detuning disorder at all, so it could not honestly demonstrate the claimed mechanism.
  - the readout basis made the two traces non-comparable as population returns.
- Consequences:
  - the tutorial could be interpreted as evidence that a Lindblad pure-dephasing model is reversed by echo, which is not the intended or generally correct message.
  - the final plot compared two different readout mappings rather than one common coherence observable.

## Suspected / Follow-up Questions

- The broader tutorial curriculum still contains conceptual/API notebooks that do not need closed-form overlays, but the repo should keep distinguishing those from physics-verification notebooks explicitly.
- Workflow notebooks under `tutorials/00_getting_started/`, `tutorials/10_core_workflows/`, and later folders should continue to be audited case by case when they make sign- or frame-sensitive claims.

## Status

- Fixed on 2026-03-27.

## Fix Record

- `tutorials/_generate_tutorials.py`
  - corrected Tutorial 04 to use `t_pi = pi / (2 Omega)` and added an explicit `sin^2(Omega t)` comparison.
  - rebuilt Tutorial 13 around a pulse-level quasi-static detuning ensemble with a final `-x90` readout and closed-form Ramsey/echo comparisons.
  - added explicit theory checks and overlays to Tutorials 06, 09, 10, 11, 12, 14, and 15.
- `tutorials/tutorial_support.py`
  - added reusable analytical helper functions for resonant Rabi, `T1`, Ramsey, quasi-static Ramsey/echo, and cross-Kerr phase predictions.
- `tests/test_56_tutorial_physics_validation.py`
  - added focused regressions for the corrected `pi`-pulse normalization, Ramsey/T1 formulas, quasi-static echo refocusing, and cross-Kerr phase sign.