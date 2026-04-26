# Microwave Thermal-Noise Cascades

Cryogenic microwave wiring can set the effective bath occupation seen by a
storage mode, readout resonator, or qubit port. `cqed_sim.microwave_noise`
models that wiring as passive thermal attenuators, lossy lines, directional
loss, and optional passive S-matrix components.

The key convention is that the propagated variable is the normally ordered
thermal photon occupation `nbar(f)`, not a symmetrized noise temperature.

---

## Fridge Input Line

```python
from cqed_sim.microwave_noise import NoiseCascade, PassiveLoss

cascade = NoiseCascade([
    PassiveLoss("4K", temp_K=4.0, loss_db=20.0),
    PassiveLoss("still", temp_K=0.7, loss_db=10.0),
    PassiveLoss("MXC", temp_K=0.02, loss_db=40.0),
])

result = cascade.propagate(6.0e9, source_temp_K=300.0)
print(result.n_out)
print(result.effective_temperature)
```

The returned `result.budget` decomposes the final occupation into source and
component-emission contributions. This is useful for deciding which stage
dominates the residual bath occupation.

---

## Feed NoiseSpec

The microwave-noise module is an input model for existing Lindblad simulations.
It does not change solver behavior.

```python
from cqed_sim.sim import NoiseSpec

noise = NoiseSpec(
    kappa_readout=2.0e6,
    nth_readout=float(result.n_out),
)
```

For multiple ports coupled to the same resonator, combine bath occupations by
linewidth:

```python
from cqed_sim.microwave_noise import resonator_thermal_occupation

n_cav = resonator_thermal_occupation(
    kappas=[1.0e3, 4.0e3],
    n_baths=[0.01, 0.001],
)
```

The typed equivalent is useful when each port has a physical role:

```python
from cqed_sim.microwave_noise import BathSpec, ModeBathModel

mode = ModeBathModel(
    "readout",
    omega_rad_s=2.0 * 3.141592653589793 * 7.5e9,
    baths=[
        BathSpec("cold cavity wall", kappa_rad_s=6.0e6, nbar=0.0, kind="cold_internal"),
        BathSpec("input line", kappa_rad_s=1.0e6, nbar=7.0e-4, kind="hot_line"),
    ],
)
noise = NoiseSpec(kappa_readout=mode.total_kappa(), nth_readout=mode.effective_nbar())
```

---

## Qubit And Dephasing Helpers

For a thermal bath occupation at the qubit frequency:

```python
from cqed_sim.microwave_noise import bose_occupation, qubit_thermal_rates

n_q = bose_occupation(6.0e9, 0.06)
gamma_down, gamma_up, gamma_1 = qubit_thermal_rates(1.0 / 80.0e-6, n_q)
```

Thermal photons in a dispersively coupled resonator can be converted into a
photon-shot-noise dephasing estimate:

```python
import numpy as np

from cqed_sim.microwave_noise import thermal_photon_dephasing

kappa = 2.0 * np.pi * 5.0e6
chi = 2.0 * np.pi * 50.0e6
n_cav = 8.7e-3

gamma_phi = thermal_photon_dephasing(kappa, chi, n_cav)
gamma_phi_strong = thermal_photon_dephasing(
    kappa,
    chi,
    n_cav,
    approximation="strong_low_occupation",
)
```

Use `gamma_phi_thermal(...)` for the public residual-photon wrapper. It uses the
same canonical expression as `thermal_photon_dephasing(...)`. The compact
`nbar*kappa*chi**2/(kappa**2+chi**2)` estimate is available as
`gamma_phi_lorentzian_interpolation(...)` for explicit comparisons.

---

## Cavity Attenuators Versus Lossless Filters

A cold dissipative attenuator changes the in-band bath because it mixes the
incoming field with a real cold physical bath:

```python
from cqed_sim.microwave_noise import EffectiveCavityAttenuator

attenuator = EffectiveCavityAttenuator(
    omega_ro_rad_s=2.0 * 3.141592653589793 * 7.573e9,
    kappa_internal_rad_s=6.0,
    kappa_external_rad_s=1.0,
    internal_nbar=0.0,
    external_nbar=7.0e-4,
)
print(attenuator.effective_nbar())  # 1e-4
```

A lossless filter is different. It can reject out-of-band radiation, but an
in-band readout mode sees the same occupation unless the filter introduces
cold dissipation at that frequency.

```python
from cqed_sim.microwave_noise import LosslessFilterStage, MicrowaveNoiseChain

chain = MicrowaveNoiseChain(
    [LosslessFilterStage("reflective", 7.5e9, 20e6, 40.0)],
    input_temperature_K=0.060,
)
```

For Wang-style design sweeps, use `sweep_cavity_attenuator_design(...)` to scan
`kappa_internal/kappa_external`, residual occupation, dephasing rate, and
`T2/(2*T1)`.

---

## Examples

- `examples/microwave_noise_fridge_chain.py`
- `examples/microwave_noise_resonator_ports.py`
- `examples/microwave_noise_qubit_rates.py`
- `examples/microwave_noise_photon_dephasing.py`
- `examples/noise/photon_shot_noise_dephasing.py`
- `examples/noise/multimode_thermal_photons.py`
- `examples/microwave/cavity_attenuator_design.py`
- `examples/noise/noise_induced_dephasing_extraction.py`
