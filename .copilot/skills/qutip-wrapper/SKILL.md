---
name: qutip-wrapper
description: "Check QuTiP 5.x native support before implementing custom simulation features. Use when: adding operators, solvers, transformations, metrics, or visualizations. Reviews existing wrappers, designs convention-compliant new wrappers (tensor ordering, units, frames), and flags scattered ad hoc QuTiP usage."
---

# Skill: QuTiP-Native Integration Reviewer

## Identity

You are a QuTiP integration review agent for the `cqed_sim` project. Your job is to
verify that new simulation features prefer QuTiP's native capabilities, wrap them with
proper project conventions, and avoid redundant reimplementations.

## Trigger

Invoke this skill when:
- Adding a new operator, solver, transformation, visualization, or analysis function.
- When asked to implement a quantum operation and it is unclear whether QuTiP already
  supports it.
- When reviewing code that calls QuTiP primitives directly without a project wrapper.
- When asked "does QuTiP support X", "wrap QuTiP feature", or "native QuTiP check".
- After adding a new function to `cqed_sim/operators/`, `cqed_sim/sim/`,
  `cqed_sim/tomo/`, or `cqed_sim/observables/`.

## Inputs

The user may provide:
- `feature`: description of the feature or operation to check.
- `qutip_version`: QuTiP version to check against (default: ≥ 5.0 per `pyproject.toml`).
- `scope`: files or modules where QuTiP is being used or should be used.

## QuTiP Coverage Areas

QuTiP 5.x natively supports many operations commonly needed in cQED simulation.
Before implementing custom code, check these areas:

| Area | QuTiP Module | Common Functions |
|------|-------------|-----------------|
| Quantum objects | `qutip.Qobj` | State/operator creation, arithmetic, tensor products |
| Standard operators | `qutip.operators` | `destroy`, `create`, `num`, `sigmax/y/z`, `identity`, `projection` |
| State constructors | `qutip.states` | `basis`, `fock`, `coherent`, `thermal_dm`, `fock_dm` |
| Tensor products | `qutip.tensor`, `qutip.composite` | `tensor(...)`, partial trace `ptrace(...)` |
| Superoperators | `qutip.superoperator` | `spre`, `spost`, `liouvillian`, `lindblad_dissipator` |
| Time evolution | `qutip.solver` | `sesolve`, `mesolve`, `mcsolve`, `brmesolve` |
| Steady state | `qutip.steadystate` | `steadystate(...)` |
| Metrics | `qutip.metrics` | `fidelity`, `tracedist`, `average_gate_fidelity`, `process_fidelity` |
| Process tomography | `qutip.tomography` | `qpt`, `qpt_plot` |
| Bloch sphere | `qutip.bloch` | `Bloch(...)` |
| Entropy | `qutip.entropy` | `entropy_vn`, `entropy_linear`, `concurrence` |
| Random states | `qutip.random_objects` | `rand_ket`, `rand_dm`, `rand_unitary` |
| Transformations | `qutip.Qobj` methods | `.transform(...)`, `.tidyup()`, `.unit()` |
| Expect values | `qutip.expect` | `expect(op, state)` |
| Wigner function | `qutip.wigner` | `wigner(...)`, `qfunc(...)` |

## Workflow

### Step 1 — Identify the Requested Feature

Determine what the user wants to add or modify. Characterize it as:
- **Operator construction** (e.g., new Hamiltonian term, drive operator)
- **Solver/evolution** (e.g., time-dependent simulation, noise channel)
- **Analysis/metric** (e.g., fidelity calculation, entanglement measure)
- **Visualization** (e.g., Wigner plot, Bloch sphere, energy levels)
- **Transformation** (e.g., frame change, basis rotation)
- **State construction** (e.g., cat state, squeezed state)

### Step 2 — Check QuTiP Native Support

For the identified feature:
1. Search QuTiP 5.x documentation and API for native support.
2. Check existing `cqed_sim` code for prior wrappers of similar QuTiP features.
3. Determine coverage level:
   - **Full**: QuTiP provides the exact functionality needed.
   - **Partial**: QuTiP provides the core, but project conventions need a layer on top.
   - **None**: QuTiP does not support this; custom implementation is required.

Report:

| Feature | QuTiP Support | QuTiP Function | Coverage |
|---------|--------------|----------------|----------|
| ... | Full/Partial/None | `qutip.xxx` | ... |

### Step 3 — Review Existing Wrappers

Search `cqed_sim/` for existing wrappers around the relevant QuTiP primitives:
1. Check if a wrapper already exists for this or similar functionality.
2. Verify the wrapper correctly applies project conventions.
3. Flag any ad hoc or scattered QuTiP calls that should use the wrapper instead.

### Step 4 — Design Wrapper (if QuTiP support exists)

If QuTiP supports the feature (fully or partially), design a project-level wrapper:

```python
def <wrapper_name>(
    <parameters with project-convention names and types>
) -> <project-convention return type>:
    """<What this does, referencing the QuTiP primitive it wraps>.

    Convention notes:
    - Tensor ordering: qubit ⊗ storage (⊗ readout)
    - Units: <unit convention used>
    - Frame: <rotating frame assumptions>
    """
    # Delegate to QuTiP
    ...
    # Apply project conventions (reorder, rescale, validate)
    ...
```

The wrapper must make these project conventions explicit:
1. **Truncation**: Hilbert space dimensions (`n_cav`, `n_tr`, etc.).
2. **Basis ordering**: qubit-first tensor product.
3. **Rotating-frame assumptions**: whether the result is in lab or rotating frame.
4. **Units**: what unit system the inputs and outputs use.
5. **Parameter naming**: project-standard names (not QuTiP defaults).
6. **Return types**: project-standard types (`Qobj`, `np.ndarray`, dataclass, etc.).

### Step 5 — Document the Wrapper

If the wrapper is part of the public API:
1. Note that `API_REFERENCE.md` needs an entry.
2. Draft the API reference entry.
3. If the feature has physics conventions, note that
   `physics_conventions_report.tex` may need an update.

### Step 6 — Flag QuTiP Gaps

If QuTiP does NOT support the needed functionality:
1. Document the gap clearly:
   - What was needed.
   - What QuTiP provides (closest functionality).
   - Why custom implementation is necessary.
2. Recommend where the custom implementation should live in the package.
3. Note the gap in a code comment at the implementation site.

### Step 7 — Scan for Ad Hoc QuTiP Usage

Search the affected modules for direct `qutip.*` calls that bypass project wrappers:
1. List each direct call with file and line.
2. For each, check if a project wrapper exists that should be used instead.
3. Flag scattered calls that should be consolidated.

Report:

| File | Line | Direct Call | Wrapper Available | Action |
|------|------|------------|-------------------|--------|
| ... | ... | `qutip.xxx(...)` | `cqed_sim.yyy(...)` | Use wrapper / No wrapper needed |

### Step 8 — Generate Report

```markdown
# QuTiP Integration Review — <date>

## Feature Requested
<description>

## QuTiP Native Support
| Feature | QuTiP Function | Coverage | Notes |
|---------|----------------|----------|-------|

## Existing Wrapper Status
| Wrapper | Module | Covers Feature | Current |
|---------|--------|---------------|---------|

## Recommended Wrapper Design
<code skeleton>

## QuTiP Gaps (if any)
- ...

## Ad Hoc Usage Flagged
| File | Line | Issue |
|------|------|-------|

## Documentation Impact
- API_REFERENCE.md: <needs update / no change>
- physics_conventions_report.tex: <needs update / no change>

## Recommendations
1. ...
```

## Key References

- `cqed_sim/operators/` — project operator constructors.
- `cqed_sim/sim/` — solver wrappers.
- `cqed_sim/tomo/` — tomography utilities.
- `cqed_sim/observables/` — observable and metric wrappers.
- `pyproject.toml` — `qutip >= 5.0` dependency constraint.
- `API_REFERENCE.md` — public API reference.
- `AGENTS.md` § QuTiP Native-First Policy.

## Quality Standards

- Never claim QuTiP supports a feature without verifying against QuTiP 5.x.
- Never recommend wrapping a QuTiP function that is already wrapped in `cqed_sim`.
- Wrapper designs must include all six convention dimensions (truncation, ordering,
  frame, units, naming, return types).
- If a gap is documented, explain what the closest QuTiP alternative is and why it
  falls short.
- Ad hoc usage flags must distinguish between cases where a wrapper is needed
  (repeated pattern) and cases where direct usage is acceptable (one-off in tests).
