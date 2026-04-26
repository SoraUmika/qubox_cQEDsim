# API Reference - Microwave Thermal Noise (`cqed_sim.microwave_noise`)

`cqed_sim.microwave_noise` propagates normally ordered microwave thermal photon
occupation through passive cryogenic wiring and returns effective bath
occupations suitable for `NoiseSpec`.

The propagated quantity is

$$
n_B(f,T) = \frac{1}{\exp(h f / k_B T)-1},
$$

not a symmetrized noise temperature.

---

## Helper Functions

| Function | Description |
|---|---|
| `bose_occupation(freq_hz, temp_K)` | Vectorized Bose occupation; returns 0 for `temp_K <= 0` |
| `loss_db_to_power_transmission(loss_db)` | Positive insertion loss convention, `eta = 10 ** (-loss_db/10)` |
| `occupation_to_effective_temperature(freq_hz, n)` | Inverts Bose occupation; returns 0 for `n <= 0` |
| `sym_noise_temperature(freq_hz, temp_K)` | Reporting-only symmetrized noise temperature |
| `n_bose(frequency_hz, temperature_K)` | Strict scalar Bose occupation helper; rejects negative frequency or temperature |
| `n_bose_angular(omega_rad_s, temperature_K)` | Strict scalar Bose occupation helper using angular frequency |
| `effective_temperature(frequency_hz, nbar)` | Strict scalar Bose inversion; rejects negative occupation |
| `thermal_lindblad_rates(kappa_rad_s, nbar)` | Strict scalar bosonic downward/upward Lindblad rates |

Frequencies are in Hz. Rates such as `kappa`, `chi`, and `gamma_zero_temp` are
in angular-rate units when used in rate/dephasing helpers.

---

## Scalar Components

### PassiveLoss

```python
PassiveLoss(name: str, temp_K: float, loss_db: float | Callable)
```

Matched passive loss with

$$
n_\mathrm{out} = \eta n_\mathrm{in} + (1-\eta)n_B(f,T),
\qquad
\eta = 10^{-\mathrm{loss\_db}/10}.
$$

`loss_db` may be a scalar or `loss_db(freq_hz)`.

### DistributedLine

```python
DistributedLine(
    name: str,
    length_m: float,
    attenuation_db_per_m: float | Callable,
    temperature_K: float | Callable,
    num_slices: int = 100,
)
```

Implements

$$
\frac{dn}{dz} = \alpha(z,f)\left[n_B(f,T(z))-n(z,f)\right]
$$

by midpoint slicing. Attenuation in dB/m is converted to power Nepers/m via
`alpha = ln(10)/10 * attenuation_db_per_m`.

### DirectionalLoss

```python
DirectionalLoss(name, temp_K, forward_loss_db, reverse_isolation_db)
```

Scalar approximation to an isolator/circulator path. `propagate(...,
direction="forward")` uses `forward_loss_db`; `direction="reverse"` uses
`reverse_isolation_db`.

---

## Passive S-Matrix Components

```python
PassiveSMatrixComponent(name, temp_K, S_matrix)
```

Propagates normally ordered covariance matrices as

$$
C_\mathrm{out} =
S C_\mathrm{in} S^\dagger + n_B(f,T)\left(I-S S^\dagger\right).
$$

The component checks that `I - S S^dagger` is positive semidefinite and raises
`ValueError` for active/non-passive matrices.

---

## NoiseCascade

```python
NoiseCascade(components).propagate(
    freq_hz,
    source_temp_K=None,
    source_n=None,
    direction="forward",
)
```

`source_temp_K` and `source_n` are mutually exclusive. If neither is supplied,
the source occupation defaults to vacuum, `source_n=0`.

Returns `NoiseCascadeResult`:

| Field | Description |
|---|---|
| `n_out` | Final bath occupation |
| `effective_temperature` | Bose-equivalent temperature |
| `trace` | Component-level diagnostic traces |
| `budget` | Exact scalar matched noise budget |

The scalar budget contains `weights`, `occupations`, and `contributions` with

$$
n_\mathrm{out} = w_\mathrm{source} n_\mathrm{source}
 + \sum_i w_i n_B(f,T_i),
\qquad
w_\mathrm{source} + \sum_i w_i = 1.
$$

Distributed lines are budgeted exactly at the slice level.

---

## cQED Helpers

| Function | Description |
|---|---|
| `resonator_thermal_occupation(kappas, n_baths)` | Returns `sum(kappa_j*n_j)/sum(kappa_j)` |
| `resonator_lindblad_rates(kappa, n)` | Returns downward/upward bosonic rates |
| `qubit_thermal_rates(gamma_zero_temp, n)` | Returns `Gamma_down`, `Gamma_up`, and `Gamma_1` |
| `thermal_photon_dephasing(kappa, chi, n_cav, exact=True, approximation=None)` | Exact, weak-dispersive, or strong-dispersive low-occupation photon-shot-noise dephasing |
| `BathSpec`, `ModeBathModel` | Typed multi-bath effective mode occupation and thermal Lindblad rates |
| `gamma_phi_thermal` | Canonical residual-photon dephasing wrapper using angular-rate inputs |
| `gamma_phi_lorentzian_interpolation` | Explicit compact interpolation formula for comparison studies |
| `gamma_phi_strong_dispersive_N` | Strong-dispersive photon-number-conditioned escape/dephasing rate |
| `PassiveLossStage`, `LosslessFilterStage`, `MicrowaveNoiseChain` | Design-facing scalar noise-chain models distinguishing dissipative loss from lossless filtering |
| `EffectiveCavityAttenuator`, `TwoModeCavityAttenuatorModel` | Wang-style effective cold-bath and two-mode hybridization cavity-attenuator models |
| `simulate_ramsey_with_residual_photons` | Deterministic Lindblad Ramsey helper initialized with thermal bosonic states |
| `simulate_noise_induced_dephasing`, `fit_noise_induced_dephasing` | Synthetic added-noise dephasing measurement and offset extraction |

`thermal_photon_dephasing(..., approximation="weak")` returns
`4*chi**2/kappa*n_cav*(n_cav+1)`. `approximation="strong_low_occupation"`
returns `kappa*n_cav`.

The newer `gamma_phi_thermal(...)` wrapper preserves this canonical expression.
The simpler `nbar*kappa*chi**2/(kappa**2+chi**2)` interpolation is available as
`gamma_phi_lorentzian_interpolation(...)` so examples can compare it without
changing the default convention.

For one mode coupled to several ports, `ModeBathModel` uses

$$
\bar n_\mathrm{eff} =
\frac{\sum_j \kappa_j \bar n_j}{\sum_j \kappa_j}.
$$

For a cold dissipative cavity attenuator, `EffectiveCavityAttenuator` uses the
same rule with internal cold coupling and external hot-line coupling. A
lossless filter is modeled separately: it may reject transmitted out-of-band
occupation, but it does not add a cold bath and therefore does not reduce
in-band occupation by itself.

---

## Example

```python
from cqed_sim.microwave_noise import NoiseCascade, PassiveLoss
from cqed_sim.sim import NoiseSpec

cascade = NoiseCascade([
    PassiveLoss("4K", temp_K=4.0, loss_db=20.0),
    PassiveLoss("MXC", temp_K=0.02, loss_db=40.0),
])

result = cascade.propagate(6.0e9, source_temp_K=300.0)
noise = NoiseSpec(kappa_readout=2.0e6, nth_readout=float(result.n_out))
```

Reference scripts:

- `examples/microwave_noise_fridge_chain.py`
- `examples/microwave_noise_resonator_ports.py`
- `examples/microwave_noise_qubit_rates.py`
- `examples/microwave_noise_photon_dephasing.py`
- `examples/noise/photon_shot_noise_dephasing.py`
- `examples/noise/multimode_thermal_photons.py`
- `examples/microwave/cavity_attenuator_design.py`
- `examples/noise/noise_induced_dephasing_extraction.py`

---

## References

[1] S. Krinner et al., "Engineering cryogenic setups for 100-qubit scale superconducting circuit systems," EPJ Quantum Technology 6, 2 (2019). DOI: 10.1140/epjqt/s40507-019-0072-0.

[2] A. A. Clerk, M. H. Devoret, S. M. Girvin, F. Marquardt, and R. J. Schoelkopf, "Introduction to quantum noise, measurement, and amplification," Reviews of Modern Physics 82, 1155 (2010). DOI: 10.1103/RevModPhys.82.1155.

[3] G. Zhang et al., "Suppression of photon shot noise dephasing in a tunable coupling superconducting qubit," npj Quantum Information 3, 1 (2017). DOI: 10.1038/s41534-016-0002-2.

[4] A. P. Sears et al., "Photon shot noise dephasing in the strong-dispersive limit of circuit QED," Physical Review B 86, 180504(R) (2012). DOI: 10.1103/PhysRevB.86.180504.

[5] Z. Wang et al., "Cavity Attenuators for Superconducting Qubits," Physical Review Applied 11, 014031 (2019). DOI: 10.1103/PhysRevApplied.11.014031.
