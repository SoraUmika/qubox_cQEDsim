# `cqed_sim.measurement`

The `measurement` module provides the reusable readout-facing layer for `cqed_sim`. It covers three related tasks:

- lightweight qubit measurement from a simulated state,
- semi-classical readout-chain modeling for experiment-style I/Q data,
- and stronger readout studies via stochastic continuous-measurement replay plus operational high-power disturbance envelopes.

## Relevance in `cqed_sim`

After `cqed_sim.sim` has produced a state or a compiled replay problem, this module is the standard place to:

- convert a state into qubit probabilities or shot samples,
- model resonator response, amplifier noise, and simple backaction,
- generate continuous homodyne or heterodyne trajectories from monitored loss,
- and build occupancy-activated auxiliary transmon drives for strong-readout studies.

## Main Capabilities

### Qubit measurement

- **`measure_qubit(state, spec)`**: Computes qubit probabilities from `state`, optionally applies a confusion matrix, and optionally samples shot outcomes.
- **`QubitMeasurementSpec`**: Configuration dataclass for shots, random seed, confusion matrix, and optional readout-chain replay.
- **`QubitMeasurementResult`**: Exact probabilities, sampled outcomes, optional I/Q samples, and readout diagnostics.

### Physical readout chain

- **`ReadoutResonator`**: Single-pole dispersive resonator response model.
- **`PurcellFilter`**: Frequency-dependent linewidth suppression model.
- **`AmplifierChain`**: Linear gain plus additive thermal noise.
- **`ReadoutChain`**: Full semi-classical readout model with I/Q clustering, dephasing, and Purcell estimates.
- **`ReadoutTrace`**: Time-domain cavity/output/voltage/I/Q record.
- **`ReadoutChain.simulate_waveform(...)`**: Replay an arbitrary complex readout waveform rather than a fixed steady-state drive.

### Continuous readout replay

- **`ContinuousReadoutSpec`**: SME replay options such as monitored subsystem, number of trajectories, and record storage.
- **`simulate_continuous_readout(...)`**: QuTiP `smesolve(...)` wrapper using `cqed_sim` drive, frame, and noise conventions.
- **`ContinuousReadoutResult`**: Average expectations plus per-trajectory measurement records and states.
- **`integrate_measurement_record(...)`**: Integrates homodyne or heterodyne records along their final time axis.

### Strong-readout disturbance helpers

- **`StrongReadoutMixingSpec`**: Occupancy- and slew-activated phenomenological disturbance model.
- **`build_strong_readout_disturbance(...)`**: Estimates state-averaged readout occupancy from the linear dispersive response and builds auxiliary `g-e` / `e-f` drive envelopes.
- **`strong_readout_drive_targets(...)`**: Matching `TransmonTransitionDriveSpec` mapping for those disturbance channels.
- **`infer_dispersive_coupling(...)`** and **`estimate_dispersive_critical_photon_number(...)`**: Coarse helpers for `g` and `n_crit` when only dispersive parameters are known.
- **Higher-ladder continuation**: `StrongReadoutMixingSpec(higher_ladder_scales=...)` extends the calibrated disturbance onto additional adjacent transmon transitions, with the extra envelopes exposed through `StrongReadoutDisturbance.higher_envelopes`.

## Key Entry Points

| Symbol | Purpose |
|---|---|
| `measure_qubit(state, spec)` | Main qubit-measurement entry point |
| `QubitMeasurementSpec` | Measurement configuration |
| `QubitMeasurementResult` | Measurement result bundle |
| `ReadoutChain` | Semi-classical resonator + amplifier readout model |
| `ReadoutChain.simulate_waveform(...)` | Arbitrary-waveform readout replay |
| `ContinuousReadoutSpec` | Configuration for stochastic continuous readout |
| `simulate_continuous_readout(...)` | Monitored SME replay |
| `integrate_measurement_record(...)` | Integrate homodyne/heterodyne records |
| `StrongReadoutMixingSpec` | Operational high-power disturbance model |
| `build_strong_readout_disturbance(...)` | Occupancy-activated disturbance envelopes |

## Usage Guidance

### Lightweight measurement

```python
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit

result = measure_qubit(
    final_state,
    QubitMeasurementSpec(shots=1024, seed=42),
)
```

### Physical readout chain

```python
import numpy as np

from cqed_sim.measurement import (
    AmplifierChain,
    PurcellFilter,
    QubitMeasurementSpec,
    ReadoutChain,
    ReadoutResonator,
    measure_qubit,
)

chain = ReadoutChain(
    resonator=ReadoutResonator(
        omega_r=2*np.pi*7.0e9,
        kappa=2*np.pi*8.0e6,
        g=2*np.pi*90.0e6,
        epsilon=2*np.pi*0.6e6,
        chi=2*np.pi*1.5e6,
    ),
    purcell_filter=PurcellFilter(bandwidth=2*np.pi*40.0e6),
    amplifier=AmplifierChain(noise_temperature=4.0, gain=12.0),
    integration_time=300.0e-9,
    dt=5.0e-9,
)

result = measure_qubit(
    final_state,
    QubitMeasurementSpec(
        shots=1024,
        seed=42,
        readout_chain=chain,
        readout_duration=300.0e-9,
        classify_from_iq=True,
    ),
)
```

### Continuous readout replay

```python
from cqed_sim.measurement import ContinuousReadoutSpec, simulate_continuous_readout
from cqed_sim.sim import NoiseSpec

replay = simulate_continuous_readout(
    model,
    compiled,
    initial_state,
    {"readout": "readout"},
    noise=NoiseSpec(kappa_readout=2*np.pi*2.4e6),
    spec=ContinuousReadoutSpec(
        frame=frame,
        monitored_subsystem="readout",
        ntraj=16,
        max_step=dt,
    ),
)
record = replay.measurement_records[0]
```

### Strong-readout disturbance envelopes

```python
from cqed_sim.measurement import (
    ReadoutResonator,
    StrongReadoutMixingSpec,
    build_strong_readout_disturbance,
)

resonator = ReadoutResonator(
    omega_r=model.omega_r,
    kappa=kappa_r,
    g=g_ro,
    epsilon=0.0,
    chi=chi_ro,
)
disturbance = build_strong_readout_disturbance(
    resonator,
    compiled.channels["readout"].distorted[:-1],
    dt=dt,
    spec=StrongReadoutMixingSpec(n_crit=20.0),
    drive_frequency=model.omega_r,
)
```

To attach the disturbance to more than the `g-e` and `e-f` channels, pass
`higher_ladder_scales=(...)` in the mixing spec and request the matching drive targets:

```python
targets = strong_readout_drive_targets(
    StrongReadoutMixingSpec(n_crit=20.0, higher_ladder_scales=(0.4, 0.2)),
    max_transmon_level=model.n_tr,
)
# -> {"mix_ge": ..., "mix_ef": ..., "mix_high_2_3": ..., "mix_high_3_4": ...}
```

## Important Assumptions / Conventions

- `measure_qubit(...)` traces out non-qubit subsystems before computing probabilities.
- The confusion matrix convention is `p_observed = M @ p_latent` with `(g, e)` ordering.
- The readout-chain model is semi-classical: it is not a full input-output field simulation.
- `simulate_continuous_readout(...)` promotes one selected bosonic emission channel into the monitored SME path and leaves relaxation, thermal excitation, and dephasing as unmonitored Lindblad terms.
- `integrate_measurement_record(...)` always treats the final axis as time and preserves any leading monitored-operator or heterodyne-quadrature axes.
- `build_strong_readout_disturbance(...)` is deliberately operational rather than microscopic. It is intended for calibrated non-QND threshold studies, not as a literal continuum-ionization model.
- Units remain consistent with the rest of `cqed_sim`: typically `rad/s` and `s`.

## Relationships to Other Modules

- **`cqed_sim.sim`**: provides runtime states, Hamiltonian assembly, and `split_collapse_operators(...)` for stochastic replay.
- **`cqed_sim.core`**: provides the model, rotating frame, and structured drive targets consumed by these helpers.
- **`cqed_sim.tomo`**: tomography routines can still build on `measure_qubit(...)`.

## Limitations / Non-Goals

- The semi-classical readout chain does not insert readout photons into the runtime Hamiltonian by itself.
- Purcell estimates remain analytic first-order approximations.
- The stochastic replay wrapper inherits QuTiP SME solver assumptions and option semantics.
- The strong-readout disturbance layer is a phenomenological approximation whose coefficients should be calibrated against experiment or a more microscopic model.
