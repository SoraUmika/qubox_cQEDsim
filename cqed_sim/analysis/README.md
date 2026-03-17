# `cqed_sim.analysis`

The `analysis` module provides parameter translation utilities for converting between different parameterizations of transmon-cavity systems: bare transmon Josephson-junction parameters, measured dispersive shifts, and the runtime `cqed_sim` dispersive convention.

## Relevance in `cqed_sim`

The `cqed_sim` models (`DispersiveTransmonCavityModel`, `UniversalCQEDModel`) are parameterized in terms of the dispersive approximation: `omega_q`, `omega_c`, `alpha`, `chi`, and `kerr`. These parameters are not the same as the bare Josephson energy parameters commonly used in transmon design, nor are they always directly observable from a single spectroscopy measurement. The `analysis` module bridges the gap:

- from bare transmon circuit parameters (`EJ`, `EC`, `g`) to the dispersive convention,
- from measured experimental quantities (qubit frequency, anharmonicity, dispersive shift) to the `cqed_sim` runtime parameters.

## Main Capabilities

- **`from_transmon_params(EJ, EC, g, ...)`**: Translates bare transmon parameters (Josephson energy, charging energy, coupling strength) into the `HamiltonianParams` dispersive convention used by `cqed_sim`.
- **`from_measured(omega_q, alpha, chi, ...)`**: Translates directly measured dispersive-regime parameters into `HamiltonianParams`.
- **`HamiltonianParams`**: Dataclass holding the full set of parameters in the dispersive convention ready for model construction.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `from_transmon_params(...)` | Bare transmon → dispersive parameters |
| `from_measured(...)` | Measured dispersive → `HamiltonianParams` |
| `HamiltonianParams` | Dispersive-convention parameter container |

## Usage Guidance

```python
import numpy as np
from cqed_sim.analysis import from_measured, HamiltonianParams
from cqed_sim.core import DispersiveTransmonCavityModel

params: HamiltonianParams = from_measured(
    omega_q=2*np.pi*6.0e9,
    omega_c=2*np.pi*5.0e9,
    alpha=2*np.pi*(-200.0e6),
    chi=2*np.pi*(-2.84e6),
    kerr=2*np.pi*(-2.0e3),
)

model = DispersiveTransmonCavityModel(
    omega_c=params.omega_c,
    omega_q=params.omega_q,
    alpha=params.alpha,
    chi=params.chi,
    kerr=params.kerr,
    n_cav=8,
    n_tr=2,
)
```

## Important Assumptions / Conventions

- All frequencies are in `rad/s`.
- The dispersive convention used here matches the convention documented in `physics_and_conventions/physics_conventions_report.tex`.
- `chi` is the per-photon qubit-frequency shift: negative `chi` means photons lower the qubit frequency.

## Relationships to Other Modules

- **`cqed_sim.core`**: `HamiltonianParams` fields map directly to arguments of `DispersiveTransmonCavityModel` and `UniversalCQEDModel`.

## Limitations / Non-Goals

- The translation from bare transmon parameters uses standard perturbation-theory results for the dispersive approximation; it is not a full numerical diagonalization.
- Does not currently support multi-mode parameter translation (e.g. three-mode systems).
