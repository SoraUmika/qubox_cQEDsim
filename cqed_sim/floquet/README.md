# `cqed_sim.floquet`

The `cqed_sim.floquet` package adds periodic-drive Floquet analysis for closed cQED Hamiltonians built from the same model and operator conventions used elsewhere in `cqed_sim`.

## Why this module exists

Static diagonalization is not enough when the Hamiltonian is explicitly periodic in time,

$$H(t + T) = H(t).$$

That regime appears in superconducting-circuit workflows such as:

- strong continuous transmon drive,
- strong cavity or readout drive,
- parametric frequency modulation,
- modulated couplers and sideband activation,
- periodic multitone control with a common period.

Floquet analysis is useful when you care about:

- quasienergies and drive-dressed spectra,
- avoided crossings and multiphoton resonances,
- leakage into higher transmon levels,
- sideband engineering and effective resonant couplings,
- drive-induced hybridization beyond a naive weak-drive RWA picture.

## Main entry points

- `PeriodicDriveTerm`
- `PeriodicFourierComponent`
- `FloquetProblem`
- `FloquetConfig`
- `FloquetResult`
- `solve_floquet(...)`
- `build_effective_floquet_hamiltonian(...)`
- `build_sambe_hamiltonian(...)`
- `compute_floquet_transition_strengths(...)`
- `identify_multiphoton_resonances(...)`
- `run_floquet_sweep(...)`
- `track_floquet_branches(...)`

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
    phase=0.0,
    waveform="cos",
)

problem = FloquetProblem(
    model=model,
    periodic_terms=[drive],
    period=2.0 * np.pi / drive.frequency,
)

result = solve_floquet(problem, FloquetConfig(n_time_samples=128))
print(result.quasienergies)
```

Target-based terms reuse the existing model drive-target semantics. For example, `target="qubit"` resolves through `model.drive_coupling_operators()`, and structured `SidebandDriveSpec(...)` targets reuse the same multilevel sideband path as the pulse runtime.

## Frequency and parameter modulation

Periodic modulation is not limited to additive qubit or cavity drives. You can also modulate Hamiltonian parameters by supplying an explicit operator term, for example:

- transmon frequency modulation with `model.transmon_number()`,
- cavity or storage frequency modulation with `model.mode_number("storage")`,
- dispersive modulation with `model.mode_number(mode) * model.transmon_number()`.

Helper builders are provided for those common cases:

- `build_transmon_frequency_modulation_term(...)`
- `build_mode_frequency_modulation_term(...)`
- `build_dispersive_modulation_term(...)`

## Propagator route and Sambe route

The primary closed-system solver wraps QuTiP's `FloquetBasis`, which provides:

- one-period propagators,
- quasienergies,
- Floquet modes and states at arbitrary time.

An optional truncated Sambe-space builder is also included for harmonic-space analysis and sideband interpretation:

$$
\langle m | H_F | n \rangle = H^{(m-n)} + n \Omega \delta_{mn}.
$$

Use the Sambe builder when you want explicit harmonic blocks or harmonic-cutoff comparisons. Use the propagator route as the default numerical answer.

For sweep workflows, `run_floquet_sweep(...)` evaluates Floquet-mode overlaps at
the explicitly supplied `reference_time` when you pass one, or at
`FloquetConfig.overlap_reference_time` otherwise.

## When to use Floquet analysis instead of other tools

Use `cqed_sim.floquet` when the drive is genuinely periodic and you want a spectral or dressed-state description.

Use static diagonalization when the Hamiltonian is time independent.

Use the regular `simulate_sequence(...)` time-domain path when:

- the drive is not strictly periodic,
- you care about finite-duration pulse transients,
- you need open-system Lindblad evolution today.

## Caveats

- Floquet analysis assumes exact periodicity.
- Multi-tone drives are only strictly Floquet when the tones are commensurate with a common period.
- Quasienergies are defined modulo the drive angular frequency `Omega = 2 pi / T`.
- Strong-drive transmon problems often require additional transmon levels.
- The current package surface is closed-system. QuTiP Floquet-Markov primitives exist underneath, but they are not yet wired into the public `cqed_sim.floquet` API.

## Related files

- `examples/floquet/driven_transmon_quasienergy_sweep.py`
- `examples/floquet/transmon_cavity_sideband_floquet_scan.py`
- `documentations/user_guides/floquet_analysis.md`
- `documentations/api/floquet.md`
