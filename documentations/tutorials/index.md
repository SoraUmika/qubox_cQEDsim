# Tutorial Curriculum

The primary guided-learning material for `cqed_sim` now lives in the repository's top-level `tutorials/` directory as a numbered notebook curriculum.

Start with:

- `tutorials/README.md` for the full reading order
- `tutorials/00_tutorial_index.ipynb` for onboarding and suggested learning paths

## Curriculum Summary

- Tier 0: orientation, minimal model building, units, frames, and sign conventions
- Tier 1: cavity displacement, qubit driving, and basic observables
- Tier 2: spectroscopy, number splitting, and dressed dispersive signatures
- Tier 3: Rabi, `T1`, Ramsey, and echo calibration-style notebooks
- Tier 4: Kerr, cross-Kerr, storage dynamics, and readout response
- Tier 5: multilevel transmon effects, leakage, and truncation convergence
- Tier 6: pulse/gate composition, parameter sweeps, and result extraction
- Tier 7: effective sideband dynamics, compact end-to-end calibration, and debugging workflows

## Recommended Entry Points

- First-time user: `00`, `01`, `02`, `03`, `06`
- Experimentalist workflow: `00`, `02`, `06`, `09`, `11`, `12`, `25`
- Simulation/theory path: `00`, `01`, `02`, `08`, `14`, `15`, `20`
- Developer/API path: `00`, `01`, `02`, `21`, `22`, `23`, `26`

## Conventions

The tutorials follow the runtime conventions documented in:

- `physics_and_conventions/physics_conventions_report.tex`
- `tutorials/conventions_quick_reference.md`

In particular, frame choice, carrier sign, dispersive `chi`, Kerr terms, and truncation meaning are treated consistently with the current public API.

## Tutorials vs Examples

`tutorials/` is for structured, user-facing, step-by-step learning material.

`examples/` is for standalone scripts, audits, studies, paper reproductions, and specialized workflow helpers that are useful in the repository but are not the primary onboarding curriculum.
