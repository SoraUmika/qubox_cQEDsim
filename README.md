# cQED Time-Domain Pulse Simulator (`cqed_sim`)

Hardware-faithful transmon+cavity pulse simulation in the dispersive regime using QuTiP for state evolution, while keeping pulse compilation and hardware waveform realism explicit in Python.

## Features
- Dispersive transmon+cavity model with Duffing nonlinearity, Kerr, dispersive `chi`, and higher-order `kerr_higher` / `chi_higher`.
- Explicit pulse timeline compiler with overlap summation on a global sample grid.
- Hardware pipeline controls: IQ gain/skew/DC offsets, LO/IF, image leakage, channel gain, ZOH, simple low-pass filtering, timing quantization, detuning.
- Open-system Lindblad noise via `NoiseSpec`: qubit `T1`, qubit `Tphi`, cavity `kappa`, thermal occupancy `nth`.
- Sideband interaction primitive channel (`"sideband"`) implementing `H_sb = g(t) (a^\dagger sigma_- + a sigma_+)`.
- QuTiP-based solver wrapper with explicit tolerances and frame specification.
- Deterministic pytest suite covering physics signatures, timeline correctness, convergence/regression, and runtime budgets.

## Install
```bash
pip install -e .[dev]
```

## Quick Start
```python
import numpy as np
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.core.frame import FrameSpec
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import simulate_sequence, SimulationConfig

def square(x):
    return np.ones_like(x, dtype=np.complex128)

model = DispersiveTransmonCavityModel(
    omega_c=0.0, omega_q=0.0, alpha=-2*np.pi*0.25, chi=2*np.pi*0.02, kerr=-2*np.pi*0.005,
    n_cav=12, n_tr=3
)
pulses = [Pulse("q", 0.0, 1.0, square, amp=np.pi/4)]
compiled = SequenceCompiler(dt=0.02).compile(pulses, t_end=1.2)
res = simulate_sequence(
    model, compiled, model.basis_state(0, 0), {"q": "qubit"},
    config=SimulationConfig(frame=FrameSpec())
)
print(res.expectations["P_e"][-1])
```

## Sequential Simulation Notebook
- Notebook: `sequential_simulation.ipynb`
- Generator: `outputs/generate_sequential_simulation_notebook.py`
- Refactored reusable APIs live under:
  - `cqed_sim.io.gates`
  - `cqed_sim.simulators`
  - `cqed_sim.observables`
  - `cqed_sim.plotting`
  - `cqed_sim.tests.test_sanity`
- Regenerate the notebook after changing the generator:
```bash
python outputs/generate_sequential_simulation_notebook.py
```
- Run the notebook top-to-bottom in a kernel with `qutip`, `matplotlib`, and the editable package installed.
- The notebook includes:
  - Case A ideal gates
  - Case B pulse-level unitary simulation
  - Case C pulse-level dissipative simulation
  - compact Wigner grids, relative phase tracking, weakness metrics, and a baseline-vs-refactor sanity check

## API Summary
- `DispersiveTransmonCavityModel`: static Hamiltonian and basis state utilities.
- `FrameSpec(omega_c_frame, omega_q_frame)`: per-mode rotating frame frequencies.
- `Pulse(channel, t0, duration, envelope, carrier, phase, amp, drag, ...)`: analytic or sampled envelope pulse.
- `HardwareConfig`: per-channel distortion and mixer settings.
- `SequenceCompiler(dt, hardware).compile(...)`: timeline + overlap + hardware distortion output.
- `simulate_sequence(...)`: QuTiP `sesolve` / `mesolve` integration wrapper.
- `NoiseSpec(t1, tphi, kappa, nth)`: optional Lindblad noise specification.
- Sideband pulse convention: for constant `g`, swap `|e,0> -> |g,1>` at `T_pi = pi / (2g)`.
- Sequential notebook entry points:
  - `cqed_sim.io.gates.load_gate_sequence(...)`
  - `cqed_sim.simulators.run_case_a(...)`
  - `cqed_sim.simulators.run_case_b(...)`
  - `cqed_sim.simulators.run_case_c(...)`
  - `cqed_sim.observables.attach_weakness_metrics(...)`
  - `cqed_sim.plotting.plot_bloch_track(...)`
  - `cqed_sim.plotting.plot_wigner_grid(...)`
  - `cqed_sim.plotting.plot_relative_phase_track(...)`

## Numerical Settings
- Solver defaults: `atol=1e-8`, `rtol=1e-7`, optional `max_step`.
- Piecewise sampled drives are passed as explicit coefficient arrays on the compiled time grid.
- Determinism: tests pin NumPy random seed; no nondeterministic noise path is enabled by default.
- Units: `t1/tphi` in seconds, `kappa` in `1/s`, Hamiltonian couplings in angular-rate units compatible with time grid.

## Tests
Run all:
```bash
pytest -q
```

Run only the notebook refactor sanity coverage:
```bash
pytest -q cqed_sim/tests/test_sanity.py
```

Skip slow tests:
```bash
pytest -q -m "not slow"
```

Runtime budget policy:
- Fast tests: target `< 1 s` each.
- Integration tests: target `< 2-5 s` each.
- Full suite target on laptop: `< 60-120 s`.

## Trust Checklist (Requirement -> Test)
- Hermiticity / operator sanity -> `tests/test_01_sanity_and_free.py::test_hamiltonian_hermitian_over_grid`
- Free evolution / conserved populations -> `tests/test_01_sanity_and_free.py`
- Kerr-only analytic phase -> `tests/test_01_sanity_and_free.py::test_kerr_only_phase_matches_analytic`
- Linear cavity coherent signature -> `tests/test_02_cavity_drive_and_kerr.py::test_linear_cavity_drive_coherent_signature`
- Kerr distortion + timestep refinement -> `tests/test_02_cavity_drive_and_kerr.py::test_kerr_signature_and_timestep_refinement`
- Dispersive conditional phase scaling -> `tests/test_03_dispersive_and_ramsey.py::test_chi_conditional_phase_scales_with_photon_number`
- Ramsey pull with photons -> `tests/test_03_dispersive_and_ramsey.py::test_ramsey_pull_with_cavity_photons`
- XY phase coherence and overlap cancellation -> `tests/test_04_xy_phase_and_overlap.py`
- Detuning response and frame invariance -> `tests/test_05_detuning_and_frames.py`
- Multi-level leakage and DRAG directionality -> `tests/test_06_leakage_drag_and_higher_order.py::test_multilevel_leakage_and_drag_directionality`
- Higher-order smoke test -> `tests/test_06_leakage_drag_and_higher_order.py::test_higher_order_terms_smoke`
- Convergence + golden regression -> `tests/test_07_convergence_regression.py::test_numerical_convergence_and_regression`
- Timeline compiler alignment/overlap -> `tests/test_08_timeline_and_hardware.py::test_timeline_overlap_boundaries_alignment`
- IQ imbalance image and LO leakage from DC -> `tests/test_08_timeline_and_hardware.py`
- Runtime policy enforcement -> `tests/test_09_runtime_policy.py`

## Updating Regression Golden Data
- File: `tests/golden/hard_sequence.json`
- Recompute expected values by running the helper in `tests/test_07_convergence_regression.py` at the finest timestep and updating JSON only when a physics-model change is intentional.

## Diagnostics
- `cqed_sim.sim.diagnostics.channel_norms(...)`
- `cqed_sim.sim.diagnostics.instantaneous_phase_frequency(...)`
- Example sanity script: `examples/sanity_run.py`
- Example cavity ringdown with noise: `examples/ringdown_noise.py`
- Example sideband swap: `examples/sideband_swap.py`

## Fock-Resolved Tomography
- Driver: `cqed_sim.tomo.protocol.run_fock_resolved_tomo(...)`
- ALL_XY tools: `run_all_xy(...)`, `autocalibrate_all_xy(...)`
- Leakage calibration: `calibrate_leakage_matrix(...)` returns `(W, b, condition_number)` where:
  - `W` is the manifold-mixing matrix in `v_hat ~= W v_true + b`
  - `b` is axis-wise additive bias
- Unmixing uses pseudoinverse by default: `v_rec = pinv(W) @ (v_hat - b)`.
- Example workflow and saved diagnostics/plots: `examples/fock_tomo_workflow.py`.

## SNAP Optimization (Landgraf-Inspired)
- Module: `cqed_sim.snap_opt`
- Implements an interpretable multi-tone selective-stage SNAP ansatz with only per-tone:
  - amplitude `A_n`
  - detuning `delta_n`
  - phase `phi_n`
- Optimization loop updates `(A_n, delta_n, phi_n)` iteratively using coherent-error components:
  - phase-like error `dtheta_n`
  - longitudinal amplitude error `dlambda_n`
  - transversal leakage error `dalpha_n`
- Uses backtracking for stable convergence.
- References:
  - J. Landgraf et al., “Fast quantum control of cavities using an improved protocol without coherent errors”, arXiv:2310.10498, PRL 133, 260802 (2024).
- Example scripts:
  - `examples/run_snap_optimization_demo.py`
  - `examples/make_figures_like_paper.py`

## PRL 133 Reproduction Pipeline
- Module: `cqed_sim.snap_prl133`
- Paper mapping/citations: `cqed_sim/snap_prl133/paper_notes.md` (PRL DOI, arXiv, supplemental reference).
- Reproduction entrypoint:
```bash
python -m cqed_sim.snap_prl133.reproduce
```
- Outputs:
  - Figures/data: `outputs/figures/`
  - Summary JSON: `outputs/reproduce_summary.json`
  - Markdown report: `outputs/report.md`
  - Auto-generated PDF report: `outputs/report/report.pdf`
- Headline metric:
  - `F` (mean squared overlap, bounded in `[0,1]`)
  - `epsilon_coh = 1 - F` (bounded in `[0,1]`)
  - Legacy `error_vector_norm` is retained only as a secondary diagnostic.
