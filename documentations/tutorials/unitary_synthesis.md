# Tutorial: Unitary Synthesis

The main workflow notebook is:

- `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`

The unitary-synthesis material now covers six realistic synthesis patterns in the main notebook, plus one standalone relevance-aware example script:

1. constraint-limited optimization
2. leakage-aware synthesis
3. robust synthesis under model uncertainty
4. export, warm start, and Pareto exploration
5. flexible target-action matching for reduced states, isometries, and channels
6. leakage-aware relevant-map optimization and visualization

The separate `examples/unitary_synthesis_relevance_aware_optimizer.py` script still covers observable and trajectory objectives with accelerated ideal evaluation.

---

## Workflow 1: Constraint-Limited Unitary Synthesis

```python
from cqed_sim.unitary_synthesis import (
    MultiObjective,
    PrimitiveGate,
    SynthesisConstraints,
    TargetUnitary,
    UnitarySynthesizer,
)

primitive = PrimitiveGate(
    name="ry",
    duration=40e-9,
    matrix=lambda params, model: build_unitary(params["theta"]),
    parameters={"theta": 0.2, "duration": 40e-9},
    parameter_bounds={"theta": (-np.pi, np.pi), "duration": (20e-9, 100e-9)},
    hilbert_dim=2,
)

synth = UnitarySynthesizer(
    primitives=[primitive],
    target=TargetUnitary(U_target, ignore_global_phase=True),
    synthesis_constraints=SynthesisConstraints(max_duration=60e-9),
    objectives=MultiObjective(fidelity_weight=1.0, duration_weight=0.1),
)

result = synth.fit(maxiter=200)
```

This is the right entry point when you want a hard or penalized bound on total duration, amplitude, or forbidden parameter regions.

---

## Workflow 2: Leakage-Aware and Open-System State Mapping

```python
from cqed_sim.sim import NoiseSpec
from cqed_sim.unitary_synthesis import (
    CQEDSystemAdapter,
    LeakagePenalty,
    MultiObjective,
    PrimitiveGate,
    TargetStateMapping,
    UnitarySynthesizer,
)

system = CQEDSystemAdapter(model=my_cqed_model)

synth = UnitarySynthesizer(
    system=system,
    backend="pulse",
    primitives=[waveform_primitive],
    target=TargetStateMapping(initial_state=psi0, target_state=phi0),
    leakage_penalty=LeakagePenalty(weight=0.2),
    objectives=MultiObjective(fidelity_weight=1.0, leakage_weight=0.2),
    simulation_options={"noise": NoiseSpec(t1=40e-6, tphi=30e-6), "dt": 2e-9},
)

result = synth.fit(maxiter=200)
```

This path is the recommended notebook interface for dissipative state preparation and leakage-sensitive bosonic control problems.

---

## Workflow 3: Robust Optimization Under Parameter Uncertainty

```python
from cqed_sim.unitary_synthesis import (
    CQEDSystemAdapter,
    MultiObjective,
    Normal,
    ParameterDistribution,
    UnitarySynthesizer,
)

system = CQEDSystemAdapter(model=my_cqed_model)

synth = UnitarySynthesizer(
    system=system,
    backend="pulse",
    primitives=[waveform_primitive],
    target=target,
    objectives=MultiObjective(fidelity_weight=1.0, robustness_weight=1.0),
    parameter_distribution=ParameterDistribution(
        sample_count=4,
        aggregate="mean",
        chi=Normal(-2.8e6, 0.05e6),
    ),
)

result = synth.fit(maxiter=200)
```

The synthesizer evaluates sampled model variants during optimization and records the robustness summary in `result.report["robustness"]`.

---

## Workflow 4: Export, Warm Start, and Pareto Exploration

```python
front = synth.explore_pareto(
    [
        MultiObjective(fidelity_weight=1.0, duration_weight=0.0),
        MultiObjective(fidelity_weight=1.0, duration_weight=0.2),
        MultiObjective(fidelity_weight=1.0, duration_weight=0.5),
    ],
    maxiter=120,
)
```

`ParetoFrontResult.results` contains every weighted run, and `ParetoFrontResult.nondominated()` returns the nondominated subset.

---

## Workflow 5: Flexible Target-Action Matching

```python
from cqed_sim.unitary_synthesis import (
    ExecutionOptions,
    PrimitiveGate,
    TargetChannel,
    TargetIsometry,
    TargetReducedStateMapping,
    UnitarySynthesizer,
)

channel_synth = UnitarySynthesizer(
    primitives=[rotation_primitive],
    target=TargetChannel(unitary=target_qubit_gate, enforce_cptp=True),
    execution=ExecutionOptions(engine="numpy"),
)

reduced_state_synth = UnitarySynthesizer(
    primitives=[two_qubit_primitive],
    target=TargetReducedStateMapping(
        initial_states=[psi00, psi10],
        target_states=[qubit_g, qubit_e],
        retained_subsystems=(0,),
        subsystem_dims=(2, 2),
    ),
)

isometry_synth = UnitarySynthesizer(
    primitives=[encoding_primitive],
    target=TargetIsometry(encoding_columns),
)
```

Use this path when the experiment only cares about a logical reduced state, an encoding/isometry, or a full process action rather than a square unitary on the entire retained Hilbert space. `TargetChannel` records Choi/superoperator-style diagnostics, while every target type also reports truncation-edge and outside-tail population summaries to help validate the chosen cutoff.

---

## Workflow 6: Leakage-Aware Relevant-Map Optimization and Visualization

```python
import matplotlib.pyplot as plt

from examples.unitary_synthesis_leakage_aware_visualization import (
    comparison_summary,
    plot_case_overview,
    run_comparison,
)

results = run_comparison(maxiter_penalized=8)
summary = comparison_summary(results)
figures = plot_case_overview(results)
plt.show()
```

This workflow keeps the task definition on the relevant reduced map while adding explicit regularizers and diagnostics for where discarded amplitude goes. The comparison uses three cases:

- no leakage penalty
- final logical leakage penalty
- logical leakage plus an edge-projector penalty that steers the residual leakage away from the highest retained ancillary level

The example is intentionally small and uses bounded iteration counts so the notebook remains responsive while still showing the qualitative difference between hidden ancilla motion, total logical leakage, and edge-of-truncation occupancy.

---

## Notebook Outputs

The notebook demonstrates:

- defining a cQED model and primitive set
- wrapping that model in `CQEDSystemAdapter(...)`
- running a constrained unitary-target optimization
- running leakage-aware noisy state-mapping synthesis
- adding a `ParameterDistribution` for robustness
- exporting and warm-starting a saved result while inspecting a small Pareto front
- running channel-first and isometry-style target-action matching
- comparing leakage-aware relevant-map solutions with operator, density-matrix, and leakage-profile visualizations

![Unitary synthesis convergence](../assets/images/tutorials/unitary_synthesis_convergence.png)

The repository also includes standalone example scripts that complement the notebook:

- `examples/unitary_synthesis_relevance_aware_optimizer.py`
- `examples/unitary_synthesis_flexible_target_actions.py`
- `examples/unitary_synthesis_leakage_aware_visualization.py`
