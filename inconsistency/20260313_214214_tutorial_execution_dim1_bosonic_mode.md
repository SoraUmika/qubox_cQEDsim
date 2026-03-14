# Tutorial Execution Audit

Created: 2026-03-13 21:42:14 local time
Status: fixed

## Confirmed issues

### 1. Minimal bosonic truncations (`dim = 1`) are accepted by the model spec but fail during operator/default-observable construction

- What it is:
  - `BosonicModeSpec.dim` explicitly allows positive dimensions, so a trivial bosonic subsystem with `dim = 1` is currently a valid model configuration.
  - Several tutorials use `DispersiveTransmonCavityModel(..., n_cav=1, ...)` to suppress cavity dynamics when the notebook is really about a qubit-only or spectroscopy-only slice.
  - During execution, `UniversalCQEDModel.operators()` calls `qt.destroy(1)` for that bosonic mode, and the current QuTiP version raises `ValueError: number of diagonals does not match number of offsets`.
- Where it appears:
  - `cqed_sim/core/universal_model.py`
  - generated tutorials including at least `tutorials/04_qubit_drive_and_basic_population_dynamics.ipynb`
- Affected components:
  - notebook execution
  - default observable construction
  - any simulation path that relies on `model.operators()` for a bosonic subsystem with `dim = 1`
- Why inconsistent:
  - the public model API accepts `dim = 1` bosonic modes, but the operator-construction path does not treat the corresponding annihilation operator as the trivial zero operator.
- Consequence:
  - top-to-bottom execution of the tutorial curriculum fails even though the model specification itself looks valid.

### 2. Tutorial 07 shadowed its detuning-scan array with the loop variable

- What it is:
  - `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb` used `for detuning_mhz in detuning_mhz:`, which overwrote the scan array with the last scalar value.
- Affected components:
  - notebook execution
  - printed peak-location summary
- Consequence:
  - later cells tried to index a scalar as though it were the original scan array.

### 3. Tutorials 07 and 22 left the storage mode in the lab frame during long fixed-Fock qubit probes

- What it is:
  - Both tutorials used long, weak qubit spectroscopy-style probes on fixed cavity Fock states while matching only the qubit frame.
- Affected components:
  - `tutorials/07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb`
  - `tutorials/22_parameter_sweeps_and_batch_simulation.ipynb`
- Why inconsistent:
  - the cavity state is static for the pedagogical point of those notebooks, so keeping the large bare storage rotation in the Hamiltonian only makes the integration stiffer without adding physical insight.
- Consequence:
  - QuTiP raised `IntegratorException: Excess work done on this call` during the tutorial runs.

### 4. Tutorial 19 treated the solver time axis like a NumPy array before conversion

- What it is:
  - `weak_result.solver_result.times * 1.0e9` assumed the time list supported scalar multiplication.
- Consequence:
  - the visualization cell raised `TypeError: can't multiply sequence by non-int of type 'float'`.

### 5. Tutorial 26 requested more energy levels than the model dimension and initially referenced an undefined helper variable

- What it is:
  - The notebook asked `compute_energy_spectrum(..., levels=6)` for a two-state Hilbert space and, after the first fix, also needed the `spectrum_levels` helper defined in the model-construction cell.
- Consequence:
  - the notebook failed with a spectrum-size `ValueError` and then with `NameError` until the helper variable lived in the correct cell.

## Resolution update

Updated: 2026-03-13 local time

- Issue 1 fixed:
  - Patched `cqed_sim/core/universal_model.py` so bosonic `dim = 1` uses an explicit zero annihilation operator instead of `qt.destroy(1)`.
  - Patched `cqed_sim/sim/extractors.py` so singleton-mode moment helpers return zero moments directly.
  - Added regression coverage in `tests/test_11_model_invariants.py` and `tests/test_16_ideal_primitives_and_extractors.py`.
- Issue 2 fixed:
  - Renamed the inner scan variable in the tutorial generator and regenerated Tutorial 07.
  - Updated `tests/test_33_usage_examples_spectroscopy_sign.py` to match the corrected notebook text.
- Issue 3 fixed:
  - Updated Tutorials 07 and 22 to use a matched cavity-plus-qubit frame for the fixed-Fock qubit-probe setup, which removes unnecessary stiffness.
- Issue 4 fixed:
  - Converted the Tutorial 19 time axis to `np.asarray(..., dtype=float)` before scaling.
- Issue 5 fixed:
  - Made Tutorial 26 derive `spectrum_levels` from `model.subsystem_dims` and use that same value consistently in both the spectrum call and the plot helper.
- Validation:
  - Regenerated the tutorial notebooks from `tutorials/_generate_tutorials.py`.
  - Executed all 27 notebooks top-to-bottom with a plain-Python harness against the notebook JSON and confirmed `ALL_NOTEBOOKS_OK 27`.
