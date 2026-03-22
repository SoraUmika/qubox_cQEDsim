# Floquet Analysis

`cqed_sim.floquet` adds periodic-drive Floquet analysis on top of the same model and operator conventions used by the time-domain runtime.

Use it when the Hamiltonian is strictly periodic,

$$H(t + T) = H(t), \qquad \Omega = \frac{2 \pi}{T},$$

and you want quasienergies, dressed-state structure, resonance identification, or sideband analysis rather than a finite-duration trajectory.

---

## When Floquet analysis is the right tool

Typical cQED use cases include:

- strong continuous transmon drive,
- strong cavity or readout dressing,
- periodic parametric modulation,
- sideband or SWAP activation by modulation,
- commensurate multi-tone periodic drives.

If the waveform is not strictly periodic, or if the transient shape matters, use `simulate_sequence(...)` instead.

---

## Basic workflow

```python
import numpy as np

from cqed_sim import FloquetConfig, FloquetProblem, PeriodicDriveTerm
from cqed_sim.core import DispersiveTransmonCavityModel
from cqed_sim.floquet import solve_floquet

model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.0,
    omega_q=2.0 * np.pi * 6.0,
    alpha=2.0 * np.pi * (-0.22),
    chi=2.0 * np.pi * (-0.015),
    n_cav=4,
    n_tr=4,
)

drive = PeriodicDriveTerm(
    target="qubit",
    amplitude=2.0 * np.pi * 0.08,
    frequency=2.0 * np.pi * 6.0,
    waveform="cos",
)

problem = FloquetProblem(
    model=model,
    periodic_terms=[drive],
    period=2.0 * np.pi / drive.frequency,
)

result = solve_floquet(problem, FloquetConfig(n_time_samples=128))

print(result.quasienergies)
print(result.warnings)
```

Target-based periodic terms reuse the model's drive-target semantics. By default, `PeriodicDriveTerm(target="qubit")` builds the in-phase Hermitian quadrature from the model's raising and lowering operators.

---

## Parameter modulation

The Floquet path is not limited to additive drives. It also supports periodic modulation of Hamiltonian parameters by driving explicit operators such as:

- `model.transmon_number()` for qubit-frequency modulation,
- `model.mode_number("storage")` for cavity-frequency modulation,
- `model.mode_number(mode) * model.transmon_number()` for dispersive modulation.

Convenience builders are provided for common cQED cases:

- `build_transmon_frequency_modulation_term(...)`
- `build_mode_frequency_modulation_term(...)`
- `build_dispersive_modulation_term(...)`

---

## Sweeping a driven transmon

```python
import numpy as np

from cqed_sim import FloquetConfig, FloquetProblem, PeriodicDriveTerm
from cqed_sim.core import TransmonModeSpec, UniversalCQEDModel
from cqed_sim.floquet import identify_multiphoton_resonances, run_floquet_sweep

model = UniversalCQEDModel(
    transmon=TransmonModeSpec(
        omega=2.0 * np.pi * 5.2,
        dim=4,
        alpha=2.0 * np.pi * (-0.22),
        label="qubit",
        aliases=("qubit", "transmon"),
        frame_channel="q",
    ),
    bosonic_modes=(),
)

drive_amplitude = 2.0 * np.pi * 0.08
drive_frequencies = 2.0 * np.pi * np.linspace(4.6, 5.8, 61)
problems = [
    FloquetProblem(
        model=model,
        periodic_terms=[
            PeriodicDriveTerm(
                target="qubit",
                amplitude=drive_amplitude,
                frequency=frequency,
                waveform="cos",
            )
        ],
        period=2.0 * np.pi / frequency,
    )
    for frequency in drive_frequencies
]

sweep = run_floquet_sweep(problems, parameter_values=drive_frequencies, config=FloquetConfig(n_time_samples=128))
midpoint_resonances = identify_multiphoton_resonances(sweep.results[len(sweep.results) // 2], max_photon_order=3)
```

`run_floquet_sweep(...)` solves each point and then uses Floquet-mode overlaps to track branches across the sweep.

---

## Sideband and parametric examples

The Floquet module works with the same structured sideband target used by the pulse runtime:

```python
import numpy as np

from cqed_sim import FloquetProblem, SidebandDriveSpec
from cqed_sim.core import DispersiveTransmonCavityModel
from cqed_sim.floquet import build_target_drive_term, solve_floquet

model = DispersiveTransmonCavityModel(
    omega_c=2.0 * np.pi * 5.05,
    omega_q=2.0 * np.pi * 6.25,
    alpha=2.0 * np.pi * (-0.25),
    chi=2.0 * np.pi * (-0.015),
    n_cav=3,
    n_tr=3,
)

sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
frequency = model.sideband_transition_frequency(cavity_level=0)
drive = build_target_drive_term(
    model,
    sideband,
    amplitude=2.0 * np.pi * 0.03,
    frequency=frequency,
    waveform="cos",
)

problem = FloquetProblem(model=model, periodic_terms=[drive], period=2.0 * np.pi / frequency)
result = solve_floquet(problem)
```

For full runnable scripts, see:

- `examples/floquet/driven_transmon_quasienergy_sweep.py`
- `examples/floquet/transmon_cavity_sideband_floquet_scan.py`

---

## Convergence and interpretation notes

- Floquet quasienergies are defined modulo `Omega`.
- Strong-drive problems can require more transmon and cavity levels than static calculations.
- Use `FloquetConfig.sambe_harmonic_cutoff` only when you specifically want harmonic-space analysis.
- Commensurate multi-tone drives are fine, but you must supply a consistent common period.
- The current public API is closed-system. Dissipative Floquet-Markov extensions are a future step.
