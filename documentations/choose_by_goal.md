# Choose by Goal

Use this page when you know what you want to do, but do not yet know which
`cqed_sim` model, documentation surface, tutorial, or example is the right
entry point.

This page is task-first on purpose. Each goal below gives you one default path
to start with, one exact repo artifact to open, and the next page to read after
that.

---

## User Goals

### Run My First Pulse-Level Simulation

- Recommended starting page: [Quickstart](quickstart.md)
- Exact example or notebook: `examples/protocol_style_simulation.py`
- Next page to open: [Protocol-Style Simulation](tutorials/getting_started_simulation.md)
- Choose this when: you want the shortest path from model -> pulses -> compile -> simulate -> measure.
- Do not start here if: you already know you need a readout resonator, Floquet analysis, or optimization tooling.

### Do Dispersive Spectroscopy or Extract `chi`

- Recommended starting page: [Displacement & Spectroscopy](tutorials/displacement_spectroscopy.md)
- Exact example or notebook: `examples/displacement_qubit_spectroscopy.py`
- Next page to open: [Dispersive Shift & Dressed States](tutorials/dispersive_shift_dressed.md)
- Choose this when: you want an experiment-like path to frequency sweeps, peak spacing, and dispersive-shift interpretation.
- Do not start here if: you mainly want time-domain pulse control rather than spectroscopy.

### Prepare or Diagnose Cavity States

- Recommended starting page: [Observables & Visualization](tutorials/observables_visualization.md)
- Exact example or notebook: `tutorials/03_cavity_displacement_basics.ipynb`
- Next page to open: [Phase Space Conventions](tutorials/phase_space_conventions.md)
- Choose this when: you need Wigner plots, reduced cavity states, coherent-state intuition, or state-diagnostic tooling.
- Do not start here if: your main question is photon-number spectroscopy or sideband transfer.

### Run Sideband or Bosonic Transfer Workflows

- Recommended starting page: [Sideband Swap & Bosonic Control](tutorials/sideband_swap.md)
- Exact example or notebook: `examples/sideband_swap_demo.py`
- Next page to open: [Sideband Interactions](tutorials/sideband_interactions.md)
- Choose this when: you want state transfer, bosonic exchange, detuned sidebands, reset, or shelving-style workflows.
- Do not start here if: you need a quasienergy or multiphoton-resonance view of a periodic drive.

### Analyze a Periodically Driven System with Floquet Tools

- Recommended starting page: [Floquet Driven Systems](tutorials/floquet_driven_systems.md)
- Exact example or notebook: `examples/floquet/transmon_cavity_sideband_floquet_scan.py`
- Next page to open: [Floquet Analysis](user_guides/floquet_analysis.md)
- Choose this when: your question is about quasienergies, avoided crossings, branch tracking, or multiphoton resonances.
- Do not start here if: a direct time-domain simulation with fixed pulses is enough.

### Model Readout, IQ Separation, or Readout Emptying

- Recommended starting page: [Readout Resonator](tutorials/readout_resonator.md)
- Exact example or notebook: `tutorials/17_readout_resonator_response.ipynb`
- Next page to open: [Readout Emptying Qualification](tutorials/readout_emptying_qualification.md)
- Choose this when: you need pointer-state separation, IQ trajectories, readout-chain intuition, or a path toward readout emptying.
- Do not start here if: you only need synthetic qubit measurement from a final state and not an explicit readout model.

### Convert Cryogenic Wiring Into Bath Occupations

- Recommended starting page: [Microwave Thermal Noise](user_guides/microwave_noise.md)
- Exact example or notebook: `examples/microwave_noise_fridge_chain.py`
- Next page to open: [Microwave Thermal Noise API](api/microwave_noise.md)
- Choose this when: you need normally ordered thermal photon occupations from attenuators, cables, filters, isolators, or passive S-matrix components before setting `NoiseSpec(nth=...)`.
- Do not start here if: you are trying to change solver collapse operators rather than supply calibrated bath occupations.

### Optimize Control Pulses

- Recommended starting page: [Optimal Control](tutorials/optimal_control.md)
- Exact example or notebook: `examples/structured_optimal_control_demo.py`
- Next page to open: [Optimal Control API](api/optimal_control.md)
- Choose this when: you have a state-transfer or gate target and want deterministic waveform optimization.
- Do not start here if: you want logic-level primitive sequencing, channel matching, or RL training loops first.

### Use RL or System ID Workflows

- Recommended starting page: [Calibration & Domain Randomization](tutorials/system_identification.md)
- Exact example or notebook: `examples/calibration_systemid_rl_pipeline.py`
- Next page to open: [RL Hybrid Control](tutorials/rl_hybrid_control.md)
- Choose this when: you want calibration evidence, fitted priors, domain randomization, or a training environment tied to model uncertainty.
- Do not start here if: you only need one deterministic controller and do not need an environment or prior-fitting workflow.

---

## Decision Tables

### Which Model Class Should I Use?

| Model class | Start here when | Default doc to open | Do not choose first when |
|---|---|---|---|
| `DispersiveTransmonCavityModel` | You need the standard qubit + storage workflow and no explicit readout resonator | [Defining Models](user_guides/defining_models.md) | Your physics question depends on explicit readout-mode occupancy or three-mode cross-Kerr structure |
| `DispersiveReadoutTransmonStorageModel` | You need qubit + storage + readout in one explicit model | [Readout Resonator](tutorials/readout_resonator.md) | You are still at the first-simulation stage and do not need readout-mode dynamics yet |
| `UniversalCQEDModel` | You need more than the standard two- or three-mode wrappers, or need generalized subsystem structure | [Defining Models](user_guides/defining_models.md) | A standard wrapper already matches your hardware and you want the shortest setup path |

### Which Documentation Surface Should I Use?

| Surface | Use this when | Typical output |
|---|---|---|
| [Quickstart](quickstart.md) | You want a runnable first success in minutes | one minimal end-to-end simulation |
| [Getting Started](getting_started.md) | You want the mental model of the full pipeline before diving into details | workflow overview and library orientation |
| [User Guides](user_guides/defining_models.md) | You know the subsystem and want focused explanations of one API area | reference-quality workflow guidance |
| [Tutorials](tutorials/index.md) | You want physics background, generated artifacts, and a structured learning path | guided topic walkthroughs |
| [Examples](examples.md) | You want runnable scripts and workflow helpers without tutorial narration | task-focused scripts and study helpers |
| [API Reference](api/overview.md) | You already know which subsystem you need and want exact signatures | callable-level API details |

### Which Control Stack Should I Use?

| Control stack | Choose this when | Start here |
|---|---|---|
| direct pulses | You already know the pulse shape and want explicit schedule control | [Quickstart](quickstart.md) |
| structured optimal control | You want low-parameter ansatz families such as Gaussian/DRAG/Fourier with interpretable knobs | [Optimal Control](tutorials/optimal_control.md) |
| GRAPE | You want gradient-based waveform optimization over a discretized control grid | [Optimal Control](tutorials/optimal_control.md) |
| map or unitary synthesis | You want primitive-sequence optimization, channel matching, reduced-state targets, or leakage-aware sequence design | [Unitary Synthesis](tutorials/unitary_synthesis.md) |
| RL | You want a training environment, measurement-like observations, or domain-randomized control policies | [RL Hybrid Control](tutorials/rl_hybrid_control.md) |

---

## Contributor Routing

Use this table when you are extending the repo and need to know where new work belongs.

| If you are adding... | Put it here | Companion updates to check |
|---|---|---|
| new reusable library behavior | `cqed_sim/...` | update public docs and `API_REFERENCE.md` if user-facing; update physics docs if conventions or physical meaning change |
| user-facing runnable workflow | `examples/...` | add or update the linked tutorial/docs page when the workflow is intended to teach or demonstrate behavior |
| structured learning material | `tutorials/...` plus `documentations/tutorials/...` | regenerate any checked-in tutorial assets and rebuild `site/` |
| literature reproduction or benchmark | `test_against_papers/...` | add references and `paper_summary/` coverage when the paper is important to the feature |
| regression or correctness coverage | `tests/...` | keep tests aligned with the public behavior or documented convention they protect |

### Required Companion Updates

| Change type | Also update |
|---|---|
| public API or workflow surface changed | docs pages, API reference, and checked-in `site/` output |
| physical meaning, sign, frame, units, or Hamiltonian assumptions changed | `physics_and_conventions/physics_conventions_report.tex` and related public docs |
| tutorial or physics claim added from literature | tutorial references and, when high value, `paper_summary/` |
| maintenance work uncovers drift or contradiction | a timestamped report under `inconsistency/` |

---

## Fast Picks

If you want the shortest possible routing:

- first runnable simulation -> [Quickstart](quickstart.md)
- spectroscopy and `chi` extraction -> [Displacement & Spectroscopy](tutorials/displacement_spectroscopy.md)
- cavity-state diagnostics -> [Observables & Visualization](tutorials/observables_visualization.md)
- sideband transfer -> [Sideband Swap & Bosonic Control](tutorials/sideband_swap.md)
- periodic-drive analysis -> [Floquet Driven Systems](tutorials/floquet_driven_systems.md)
- readout and emptying -> [Readout Resonator](tutorials/readout_resonator.md)
- deterministic pulse optimization -> [Optimal Control](tutorials/optimal_control.md)
- fitted priors and RL workflows -> [Calibration & Domain Randomization](tutorials/system_identification.md)
