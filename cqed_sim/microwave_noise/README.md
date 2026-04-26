# `cqed_sim.microwave_noise`

`cqed_sim.microwave_noise` propagates normally ordered thermal photon
occupation through passive cryogenic microwave wiring. It is intended to turn a
room-temperature or calibrated source occupation into effective bath
occupations that can be passed to `NoiseSpec(nth=..., nth_storage=...,
nth_readout=...)`.

The propagated quantity is the Bose occupation

```text
n_B(f, T) = 1 / (exp(h f / (k_B T)) - 1)
```

not a symmetrized noise temperature. Symmetrized temperatures are available only
for reporting calibrated spectra and should not be used as Lindblad bath
occupations.

## Main Entry Points

| Symbol | Purpose |
|---|---|
| `bose_occupation(freq_hz, temp_K)` | Normally ordered Bose occupation |
| `PassiveLoss` | Matched attenuator, lossy filter, or cable segment with scalar loss |
| `DistributedLine` | Sliced lossy line with position-dependent attenuation and temperature |
| `DirectionalLoss` | Scalar isolator/circulator approximation with forward and reverse paths |
| `PassiveSMatrixComponent` | Multiport passive covariance propagation |
| `NoiseCascade` | Scalar matched cascade with exact source/component noise budget |
| `resonator_thermal_occupation` | Linewidth-weighted resonator bath occupation |
| `resonator_lindblad_rates` | Bosonic downward/upward Lindblad rates |
| `qubit_thermal_rates` | Qubit downward/upward thermal rates |
| `thermal_photon_dephasing` | Photon-shot-noise dephasing estimates |
| `BathSpec`, `ModeBathModel` | Typed multi-bath resonator occupation and Lindblad-rate helpers |
| `gamma_phi_thermal` | Canonical residual-photon dephasing wrapper |
| `gamma_phi_lorentzian_interpolation` | Explicit compact interpolation formula |
| `EffectiveCavityAttenuator` | Wang-style cold-bath cavity attenuator model |
| `TwoModeCavityAttenuatorModel` | Readout/attenuator hybridization model |
| `simulate_noise_induced_dephasing` | Synthetic added-noise dephasing extraction workflow |

## Scalar Cascade Usage

```python
from cqed_sim.microwave_noise import NoiseCascade, PassiveLoss

cascade = NoiseCascade(
    [
        PassiveLoss("4K", temp_K=4.0, loss_db=20.0),
        PassiveLoss("MXC", temp_K=0.02, loss_db=40.0),
    ]
)

result = cascade.propagate(6.0e9, source_temp_K=300.0)
print(result.n_out)
print(result.effective_temperature)
print(result.budget.contributions)
```

For each scalar element with power transmission `eta`,

```text
n_out = eta * n_in + (1 - eta) * n_B(f, T)
```

where positive insertion loss in dB maps to `eta = 10 ** (-loss_db / 10)`.

## Distributed Lines

`DistributedLine` implements

```text
dn/dz = alpha(z, f) * (n_B(f, T(z)) - n)
```

by midpoint slicing. If attenuation is specified in dB/m, it is converted to
power Nepers per meter with `alpha = ln(10) / 10 * attenuation_db_per_m`.

## Passive S-Matrix Components

`PassiveSMatrixComponent` propagates normally ordered covariance matrices:

```text
C_out = S C_in S^dagger + n_B(f, T) * (I - S S^dagger)
```

The component validates that `I - S S^dagger` is positive semidefinite, which is
the passivity check for this approximation.

## cQED Integration

Use the cascade output as the bath occupation in existing open-system
simulations:

```python
from cqed_sim.sim import NoiseSpec

noise = NoiseSpec(kappa_readout=2.0e6, nth_readout=float(result.n_out))
```

Rates returned by this module assume angular-rate units when the input rate is
an angular rate. Frequencies used in thermal occupation helpers are always in Hz.

## Residual Photons and Cavity Attenuators

For one resonator coupled to several ports, use `BathSpec` and `ModeBathModel`
to compute the effective occupation

```text
n_eff = sum(kappa_j * n_j) / sum(kappa_j)
```

and convert it directly into bosonic Lindblad rates. Photon-shot-noise
dephasing is exposed through `gamma_phi_thermal(...)`, which preserves the
canonical Zhang/Clerk-Utami expression already implemented by
`thermal_photon_dephasing(...)`. The compact Lorentzian-style formula is
available only as `gamma_phi_lorentzian_interpolation(...)`.

`EffectiveCavityAttenuator` models a readout mode coupled to a cold internal
dissipative bath and a hotter external line. If `kappa_internal/kappa_external`
is six and the internal bath is near vacuum, the model returns the Wang-style
`n_external/7` reduction. `LosslessFilterStage` is intentionally different: it
does not add a cold bath and therefore does not reduce in-band occupation by
itself.

## References

[1] S. Krinner et al., "Engineering cryogenic setups for 100-qubit scale superconducting circuit systems," EPJ Quantum Technology 6, 2 (2019). DOI: 10.1140/epjqt/s40507-019-0072-0.

[2] A. A. Clerk, M. H. Devoret, S. M. Girvin, F. Marquardt, and R. J. Schoelkopf, "Introduction to quantum noise, measurement, and amplification," Reviews of Modern Physics 82, 1155 (2010). DOI: 10.1103/RevModPhys.82.1155.

[3] G. Zhang et al., "Suppression of photon shot noise dephasing in a tunable coupling superconducting qubit," npj Quantum Information 3, 1 (2017). DOI: 10.1038/s41534-016-0002-2.

[4] A. P. Sears et al., "Photon shot noise dephasing in the strong-dispersive limit of circuit QED," Physical Review B 86, 180504(R) (2012). DOI: 10.1103/PhysRevB.86.180504.

[5] Z. Wang et al., "Cavity Attenuators for Superconducting Qubits," Physical Review Applied 11, 014031 (2019). DOI: 10.1103/PhysRevApplied.11.014031.
