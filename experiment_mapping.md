# Experiment-to-Simulation Mapping

## Source inspected
- `C:\Users\dazzl\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\JJL_Experiments\post_cavity_calibrations.ipynb`
- `...\JJL_Experiments\calibrations\fock_resolved_pulse_train_tomo.py`

## What the experiment does (extracted)

### SQR calibration mode 1 (direct per-Fock)
- The notebook includes a **single-step, per-Fock direct objective** (`N_values=[1]`) using fock-resolved tomography over multiple qubit prep states (`g,e,+x,-x,+y,-y`).
- The direct cost is the mean squared direction mismatch between measured and target Bloch vectors per Fock manifold.
- Parameters are updated with **SPSA** (two evaluations per iteration: plus/minus perturbation).
- Main optimized knobs are `d_alpha` and `d_lambda`; `d_omega` is optional and can be frozen.

### SQR calibration mode 2 (pulse-train)
- The notebook runs iterative calibration via `iterate_sqr_calibration(...)` with pulse-train tomography (`run_fock_resolved_pulse_train_tomo`).
- For each iteration:
  1. acquire fock-resolved pulse-train data at selected `N_values`,
  2. fit per-Fock residuals (`amp_err`, `phase_err`, `delta`) using rotation/linear/physics models,
  3. map residuals to knob increments with:
     - `Δd_alpha[n] = -phase_err[n]`
     - `Δd_lambda[n] = lam0 * (1/(1+amp_err[n]) - 1)`
     - `Δd_omega[n] = delta[n] / T_sel` (up to sign/unit convention),
  4. apply damped update `d += gain * Δd`.
- The experiment uses scheduling (`n_avg_schedule`, `prep_schedule`) and optional staged enabling of `d_omega`.

## How this repository simulation implements the same logic

## Device and units
- Uses the provided cQED parameter set exactly as the device config source.
- Uses conversion
  - `omega(rad/ns) = 2*pi*f(Hz)*1e-9`
  - `t(s) = t(ns)*1e-9`.
- Uses provided `fock_fqs` as source-of-truth selective frequencies for number-selective tagging.
- Qubit decoherence uses Lindblad `T1`, `T2` with
  `1/T2 = 1/(2T1) + 1/Tphi`.

## Part I tomography
- OFF/ON subtraction is simulated explicitly per axis and per Fock manifold:
  - OFF: pre-rotation + matched idle,
  - ON: same pre-rotation + number-selective π pulse at `fock_fqs[n]`.
- Reconstructs `⟨σ_a Π_n⟩`, conditioned Bloch vectors, and independently estimates `P(n)` from selective-tag `P_e` after qubit reset to `|g⟩`.
- Includes leakage-matrix calibration (`W,b`) using coherent-state calibration data and linear unmixing via regularized pseudo-inverse.

## Part II SQR modes
- **Mode 1 direct (SPSA):** reproduces single-step (`N=1`) direct per-Fock objective from measured directions and SPSA updates.
- **Mode 2 pulse-train iterative:** reproduces iterative acquisition/fit/update loop and the same residual-to-knob mapping equations listed above.
- Pulse-train extracted quantities are reported (phase slope vs train length and XY contrast), and mapped to correction updates through fitted residuals.

## Files
- Workflow implementation: `examples/simulate_fock_tomo_and_sqr_calibration.py`
- Summary output: `outputs/fock_tomo_sqr_summary.json`
- Figures saved in: `outputs/figures/`
