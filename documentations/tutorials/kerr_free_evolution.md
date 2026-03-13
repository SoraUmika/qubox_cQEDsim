# Tutorial: Kerr Free Evolution

This tutorial points to the repo-side Kerr workflow, which now lives under `examples/` rather than inside the installed `cqed_sim` package.

---

## Workflow Location

Use the example-side helpers and scripts:

- `examples/workflows/kerr_free_evolution.py`
- `examples/kerr_free_evolution.py`
- `examples/kerr_sign_verification.py`

Run the script directly from the repository root:

```bash
python examples/kerr_free_evolution.py
python examples/kerr_sign_verification.py
```

---

## Library Building Blocks Used By The Workflow

The Kerr workflow is built from reusable library primitives:

- `DispersiveTransmonCavityModel` and `FrameSpec` from `cqed_sim.core`
- `StatePreparationSpec`, `coherent_state`, and `prepare_state(...)` from `cqed_sim.core`
- `reduced_cavity_state(...)` and `cavity_wigner(...)` from `cqed_sim.sim`

That means you can either:

1. run the repo example as-is, or
2. compose the same behavior manually from the stable low-level modules.

---

## Minimal Manual Pattern

```python
from cqed_sim.core import FrameSpec, StatePreparationSpec, coherent_state, prepare_state, qubit_state
from cqed_sim.sim import cavity_wigner, reduced_cavity_state

initial_state = prepare_state(
    model,
    StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=coherent_state(2.0),
    ),
)

rho_c = reduced_cavity_state(initial_state)
xvec, yvec, wigner = cavity_wigner(rho_c)
```

For the full time-evolution recipe, use the example workflow module.
