# Developer Guide

This page covers the development workflow, testing, and conventions for contributing to `cqed_sim`.

---

## Development Setup

### Install in editable mode with dev dependencies

```bash
git clone <repo-url>
cd cQED_simulation
pip install -e ".[dev]"
```

This installs `pytest` and all runtime dependencies. The editable install means source changes take effect immediately without reinstalling.

### Optional: JAX backend

```bash
pip install -e ".[jax]"
```

### Build the documentation locally

```bash
pip install mkdocs-material
mkdocs serve
```

Then open `http://127.0.0.1:8000` in a browser. The site live-reloads on file changes.

To build a static site:

```bash
mkdocs build
```

Output goes to `site/`.

---

## Repository Layout

```text
cQED_simulation/
├── cqed_sim/               # Installed Python package
│   ├── core/               # Models, frames, ideal gates, state prep
│   ├── pulses/             # Pulse dataclass, envelopes, builders
│   ├── sequence/           # SequenceCompiler
│   ├── sim/                # Solver, noise, extractors
│   ├── floquet/            # Floquet analysis
│   ├── measurement/        # Qubit measurement, readout chain
│   ├── optimal_control/    # GRAPE, hardware maps, objectives
│   ├── control/            # ControlLine, HardwareContext, CalibrationMap
│   ├── gates/              # Ideal gate operators
│   ├── unitary_synthesis/  # Gate sequence optimization
│   ├── rl_control/         # RL environment and reward layers
│   ├── calibration/        # SQR calibration
│   ├── calibration_targets/# Spectroscopy, Rabi, Ramsey, T1/T2 protocols
│   ├── observables/        # Wigner, Bloch, phase diagnostics
│   ├── plotting/           # Visualization helpers
│   ├── operators/          # Pauli, ladder, embedding operators
│   ├── backends/           # NumPy / JAX dense backends
│   ├── analysis/           # Parameter translation
│   ├── tomo/               # Fock tomography, all-XY
│   ├── io/                 # Gate sequence JSON I/O
│   └── system_id/          # System identification hooks
├── tests/                  # pytest test suite
├── tutorials/              # Jupyter notebook curriculum
├── examples/               # Standalone scripts, studies, audits
├── documentations/         # MkDocs source (this site)
├── docs/                   # Design notes, internal audit docs (not user-facing)
├── physics_and_conventions/# LaTeX conventions reference + compiled PDF
├── pyproject.toml
└── mkdocs.yml
```

---

## Running Tests

```bash
pytest tests/
```

For verbose output with test names:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_10_core_sanity.py -v
```

### Test organization

| File prefix | Coverage area |
|---|---|
| `test_1x_*` | Core models, Hamiltonians, free evolution |
| `test_2x_*` | Cavity drive, Kerr, dispersive, Ramsey |
| `test_3x_*` | Leakage, DRAG, higher-order dispersive |
| `test_4x_*` | Timeline, hardware compilation, gates |
| `test_5x_*` | Optimal control, hardware constraints |
| `test_6x_*` | Hardware context extensions |

### Test conventions

- Tests use `pytest` with assertion-based checking (no `unittest.TestCase`).
- Physics tests compare against analytical predictions to within a tolerance set by the simulation timestep.
- New features must include tests covering the expected physical behavior, not just API surface.

---

## Documentation

Documentation source lives in `documentations/`. The site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/).

### Adding a new page

1. Create a `.md` file in the appropriate subdirectory under `documentations/`.
2. Add the file path to the `nav:` section of `mkdocs.yml`.
3. Run `mkdocs serve` to preview.

### Adding an API reference page

API pages live under `documentations/api/`. Follow the structure of existing pages:

- Start with a purpose section.
- List key classes and functions with signatures and parameter tables.
- Include usage examples with code blocks.
- Cross-link to related pages using relative Markdown links.
- Use `!!! note` / `!!! warning` admonitions for important caveats.

### Math

MathJax is enabled. Use `$...$` for inline math and `$$...$$` for display math.

```markdown
The dispersive shift enters as $+\chi n_c n_q$ in the Hamiltonian.
```

### Code annotations

Use `# (1)` markers in code blocks with `annotate` language tag for inline callouts:

````markdown
```python hl_lines="3" annotate
model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5e9,
    chi=2 * np.pi * (-2.84e6),  # (1)
)
```
1. χ is negative for typical transmon–cavity dispersive coupling.
````

---

## Adding New Features

### Checklist

- [ ] Implementation in the appropriate submodule
- [ ] Export through the subpackage `__init__.py`
- [ ] Export through `cqed_sim/__init__.py` if it belongs in the top-level API
- [ ] Docstring with parameters, return value, and any physics/convention notes
- [ ] Unit or integration test(s) in `tests/`
- [ ] Documentation in `documentations/` (API page and/or user guide entry)
- [ ] If the feature has physics-convention implications, update `documentations/physics_conventions.md`

### Physics conventions

All Hamiltonian terms, operators, and drive conventions must be consistent with those documented in:

- `physics_and_conventions/physics_conventions_report.tex` — canonical LaTeX reference
- `documentations/physics_conventions.md` — rendered web version

If you add a new gate, coupling term, or drive convention, update both.

### Docstring style

Use Google-style docstrings. Include:

- One-line summary
- `Args:` section with types and units
- `Returns:` section
- Any sign conventions, unit requirements, or ordering conventions in the body

```python
def manifold_transition_frequency(self, n: int, frame: FrameSpec) -> float:
    """Qubit transition frequency in the given rotating frame at cavity photon number n.

    Includes first- and higher-order dispersive corrections.

    Args:
        n: Cavity photon number (Fock state index).
        frame: Rotating frame. Use ``FrameSpec()`` for the lab frame.

    Returns:
        Angular frequency omega_ge(n) in rad/s.
    """
```

---

## Physics Conventions Reference

The authoritative conventions document is:

```text
physics_and_conventions/physics_conventions_report.tex
```

A compiled PDF is at `physics_and_conventions/physics_conventions_report.pdf`.

If any discrepancy is found between the code and that document, it should be treated as a **bug** in the implementation.

---

## Code Style

- Python ≥ 3.10
- No hard formatter requirement, but PEP 8 is expected
- Type hints on all public functions and class constructors
- Keep functions focused and testable
- Prefer explicit physical constants over magic numbers

---

## Notebooks

Tutorial notebooks live in `tutorials/`. They are not part of the installed package.

- Name notebooks descriptively: `01_protocol_style_simulation.ipynb`
- Use `%matplotlib inline` or `%matplotlib widget` at the top
- Clear all outputs before committing to keep diffs readable
- Each notebook should be runnable top-to-bottom from a fresh kernel

---

## Known Gaps and Planned Work

The following areas are documented but not yet fully implemented:

- **Open-system GRAPE** — `evaluate_control_with_simulator()` supports open-system replay as a diagnostic, but the GRAPE optimizer itself is currently closed-system only.
- **Floquet-Lindblad** — Floquet-Markov evolution is not yet exposed in the public API.
- **Distributed parameter sweep** — no built-in parallelism for parameter sweeps; use `multiprocessing` or Dask externally.
- **Notebook execution in CI** — tutorials are not currently auto-run in CI; they are manually validated.

---

## Questions

For questions about the physics conventions or simulation design, consult:

- `physics_and_conventions/physics_conventions_report.pdf` for the Hamiltonian reference
- `docs/` for internal design notes and audit reports from feature development
- The `tests/` directory for behavioral specifications
