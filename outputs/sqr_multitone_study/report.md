# SQR Multi-tone Study Report

## Phase 0: Gaussian Convention Audit
- Verdict: Gaussian convention matches required R_xy(theta,phi) with I->+x and Q->+y.
- Envelope interpretation: `epsilon(t) = 0.5 * (I + i Q)`
- Pure I x90 from |g>: Bloch=[0.0, -0.9999999956426483, 9.335257610687542e-05]
- Pure Q y90 from |g>: Bloch=[0.9999999956426483, -6.123233969055667e-17, 9.335257610687542e-05]
- Phase sweep axes (phi, nx, ny, nz): (0.000, 1.000, 0.000, 0.000), (1.571, 0.000, 1.000, 0.000), (3.142, -1.000, 0.000, 0.000), (4.712, -0.000, -1.000, 0.000)
- Detuning sign check: +delta nz=-0.772, -delta nz=0.772, flip=True
- Multi-tone vs Gaussian convention consistency: process fidelity=1.000000

## Main SQR Cases
### Profile: structured_seed17
- Target mode: structured
- Seed: 17
- Case B: mean fidelity=0.540414, mean infidelity=0.459586, phase RMS=0.8706 rad, residual Z RMS=1.2878 rad, state fidelity mean=0.765920
  population-vs-phase mismatch example (n=3): population error=0.0007, process infidelity=0.9820, relative block phase=1.4544 rad
- Case C: mean fidelity=0.607265, mean infidelity=0.392735, phase RMS=0.8706 rad, residual Z RMS=1.2106 rad, state fidelity mean=0.811148
  population-vs-phase mismatch example (n=3): population error=0.0282, process infidelity=0.9790, relative block phase=1.4544 rad
- Case D: mean fidelity=0.553234, mean infidelity=0.446766, phase RMS=0.8706 rad, residual Z RMS=1.2752 rad, state fidelity mean=0.807208
  population-vs-phase mismatch example (n=3): population error=0.0017, process infidelity=0.9782, relative block phase=1.4544 rad
- Case E: mean fidelity=0.655133, mean infidelity=0.344867, phase RMS=0.8706 rad, residual Z RMS=1.3026 rad, state fidelity mean=0.837225
  population-vs-phase mismatch example (n=3): population error=0.0228, process infidelity=0.9857, relative block phase=1.4544 rad
- Neighbor cross-talk (mean |theta_{n+/-1}|/|theta_n|): Case B=0.8179, Case D=0.8169
- Final relative phase error Case B: {'n0': 0.0, 'n2': 0.17908695914799022, 'n4': 0.10035536968628378}
- Final relative phase error Case C: {'n0': 0.0, 'n2': 0.213677704527937, 'n4': 0.18076911768419635}
- Final relative phase error Case D: {'n0': 0.0, 'n2': 0.23096213381690722, 'n4': 0.12397759452336254}
- Final relative phase error Case E: {'n0': 0.0, 'n2': 0.4047704789269466, 'n4': 0.053319835398001114}

### Profile: hard_random_seed36
- Target mode: hard_random
- Seed: 36
- Case B: mean fidelity=0.487021, mean infidelity=0.512979, phase RMS=0.8706 rad, residual Z RMS=1.0796 rad, state fidelity mean=0.591713
  population-vs-phase mismatch example (n=4): population error=0.0225, process infidelity=0.9343, relative block phase=-1.2024 rad
- Case C: mean fidelity=0.654551, mean infidelity=0.345449, phase RMS=0.8706 rad, residual Z RMS=0.8400 rad, state fidelity mean=0.753111
  population-vs-phase mismatch example (n=1): population error=0.0026, process infidelity=0.2866, relative block phase=0.4848 rad
- Case D: mean fidelity=0.625667, mean infidelity=0.374333, phase RMS=0.8706 rad, residual Z RMS=0.7537 rad, state fidelity mean=0.707418
  population-vs-phase mismatch example (n=4): population error=0.0294, process infidelity=0.5968, relative block phase=-1.2024 rad
- Case E: mean fidelity=0.666815, mean infidelity=0.333185, phase RMS=0.8706 rad, residual Z RMS=0.7744 rad, state fidelity mean=0.680228
- Neighbor cross-talk (mean |theta_{n+/-1}|/|theta_n|): Case B=0.8756, Case D=0.7973
- Final relative phase error Case B: {'n0': 0.0, 'n2': -0.16543898806791457, 'n4': 0.41243710250250487}
- Final relative phase error Case C: {'n0': 0.0, 'n2': 0.018046869865732784, 'n4': 0.3716213577746741}
- Final relative phase error Case D: {'n0': 0.0, 'n2': -0.4804576463734618, 'n4': 0.3840936187711459}
- Final relative phase error Case E: {'n0': 0.0, 'n2': 0.1066531137138016, 'n4': 0.16955038652170984}

## Duration and Dispersive-Strength Scans
- Duration scan (Case B naive):
  - short: T=0.350 us, mean infidelity=0.234939, phase RMS=0.0036
  - nominal: T=1.000 us, mean infidelity=0.459315, phase RMS=0.8706
  - long: T=1.700 us, mean infidelity=0.424509, phase RMS=0.8626
- Chi scan (Case B naive):
  - easy: chi=-5.200 MHz, mean infidelity=0.426446, phase RMS=0.8001
  - nominal: chi=-2.840 MHz, mean infidelity=0.459315, phase RMS=0.8706
  - hard: chi=-1.000 MHz, mean infidelity=0.232709, phase RMS=0.0227

## Convention Regression (Case H)
- Re-run verdict: Gaussian convention matches required R_xy(theta,phi) with I->+x and Q->+y.
- Re-run envelope interpretation: `epsilon(t) = 0.5 * (I + i Q)`

## Parameter Snapshot
```json
{
  "seed": 17,
  "theta_max_rad": 2.8902652413026098,
  "coherent_alpha": "(1.1+0.25j)",
  "include_case_e": true,
  "run_profiles": [
    "structured",
    "hard_random"
  ],
  "output_dir": "outputs\\sqr_multitone_study",
  "system": {
    "n_max": 6,
    "omega_c_hz": 0.0,
    "omega_q_hz": 0.0,
    "qubit_alpha_hz": 0.0,
    "chi_nominal_hz": -2840000.0,
    "chi_easy_hz": -5200000.0,
    "chi_hard_hz": -1000000.0,
    "chi2_hz": 0.0,
    "chi3_hz": 0.0,
    "kerr_hz": 0.0,
    "use_rotating_frame": true
  },
  "pulse": {
    "duration_nominal_s": 1e-06,
    "duration_short_s": 3.5e-07,
    "duration_long_s": 1.7e-06,
    "sigma_fraction": 0.16666666666666666,
    "envelope_kind": "gaussian",
    "flat_top_rise_fraction": 0.12,
    "theta_cutoff": 1e-09,
    "dt_eval_s": 2e-09,
    "dt_opt_s": 5e-09,
    "max_step_eval_s": 2e-09,
    "max_step_opt_s": 4e-09,
    "qutip_nsteps": 200000
  },
  "optimization": {
    "method_stage1": "Powell",
    "method_stage2": "L-BFGS-B",
    "maxiter_stage1_basic": 12,
    "maxiter_stage2_basic": 20,
    "maxiter_stage1_extended": 14,
    "maxiter_stage2_extended": 24,
    "maxiter_stage1_chirp": 14,
    "maxiter_stage2_chirp": 20,
    "amp_delta_bounds": [
      -0.8,
      0.8
    ],
    "phase_delta_bounds": [
      -3.141592653589793,
      3.141592653589793
    ],
    "detuning_hz_bounds": [
      -2400000.0,
      2400000.0
    ],
    "phase_ramp_hz_bounds": [
      -1800000.0,
      1800000.0
    ],
    "w_infid": 1.0,
    "w_phase": 0.25,
    "w_theta": 0.1,
    "w_residual_z": 0.22,
    "w_state": 0.35,
    "reg_amp": 0.001,
    "reg_phase": 0.001,
    "reg_detuning": 0.0006,
    "reg_phase_ramp": 0.0004
  }
}
```