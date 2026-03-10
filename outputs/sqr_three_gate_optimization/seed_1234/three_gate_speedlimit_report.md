# Three-Gate SQR Optimization Audit

## Original Execution Path
- Notebook entrypoints: `cqed_sim.analysis.run_speedlimit_sweep_point` and `cqed_sim.analysis.evaluate_nominal_case`.
- Pre-refactor optimization internals: `cqed_sim.calibration.sqr` used a custom reduced two-level QuTiP propagation path per manifold.

## Current Shared-Stack Path
- Waveform generation: `cqed_sim.pulses.calibration.build_sqr_tone_specs`.
- Hamiltonian construction: `cqed_sim.sim.runner.hamiltonian_time_slices`.
- Propagation: coherent runs use `qutip.propagator`; dissipative runs use `qutip.mesolve` with the shared collapse-operator convention.
- Objective: `cqed_sim.calibration.evaluate_guarded_sqr_target` computes logical process fidelity and guard selectivity.

## Convention Audit Findings
- Tensor ordering is qubit-major (`|q> ⊗ |n>`), matching the main `cqed_sim` convention.
- Tone and phase conventions now come from the shared pulse builders instead of notebook-local reduced-model logic.
- The main mismatch found in the original path was architectural: the notebook already imported `cqed_sim.analysis`, but that path still delegated to duplicated reduced-model Hamiltonian/propagation logic inside `cqed_sim.calibration.sqr`.

## Validation Summary
- Dissipation enabled in this run: False.
- Parallel candidate evaluation enabled in this run: False (n_jobs=2).
- Representative simulation mode reported by the selected points: unitary.
- Summary artifact: `outputs\sqr_three_gate_optimization\seed_1234\three_gate_summary.json`.

## Recommended Follow-Up
1. Increase `MULTISTART` and optimizer iteration caps once the shared-stack path is validated at the desired physical parameters.
2. Enable `QB_T1_RELAX_NS` and `QB_T2_RAMSEY_NS` or `QB_T2_ECHO_NS` for open-system optimization studies.
3. Use `PARALLEL_ENABLED = True` only when running independent candidate starts on a machine with enough CPU and RAM to support spawned workers.