# Tutorial: Unitary Synthesis

The main workflow notebook is:

- `tutorials/30_advanced_protocols/03_unitary_synthesis_workflow.ipynb`

Phase 2 extends that notebook beyond the original matrix-vs-waveform examples and adds four realistic synthesis patterns:

1. constraint-limited optimization
2. leakage-aware synthesis
3. robust synthesis under model uncertainty
4. multi-objective / Pareto exploration

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

## Workflow 4: Pareto Exploration

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

## Notebook Outputs

The updated notebook now demonstrates:

- defining a cQED model and primitive set
- wrapping that model in `CQEDSystemAdapter(...)`
- running a constrained unitary-target optimization
- running leakage-aware noisy state-mapping synthesis
- adding a `ParameterDistribution` for robustness
- exporting and warm-starting a saved result
- plotting convergence and inspecting a small Pareto front
