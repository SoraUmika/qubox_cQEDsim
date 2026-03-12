# Defining Models

Models define the physical system being simulated. `cqed_sim` provides three model classes for different use cases.

---

## Two-Mode: DispersiveTransmonCavityModel

The most common starting point for qubit + storage/cavity simulations:

```python
import numpy as np
from cqed_sim.core import DispersiveTransmonCavityModel

model = DispersiveTransmonCavityModel(
    omega_c=2 * np.pi * 5.0e9,       # Cavity frequency (rad/s)
    omega_q=2 * np.pi * 6.0e9,       # Qubit frequency (rad/s)
    alpha=2 * np.pi * (-200e6),       # Anharmonicity (rad/s), negative for transmon
    chi=2 * np.pi * (-2.84e6),        # Dispersive shift (rad/s)
    kerr=2 * np.pi * (-2e3),          # Cavity self-Kerr (rad/s)
    n_cav=8,                          # Cavity Hilbert-space dimension
    n_tr=2,                           # Transmon levels (2 for qubit, 3+ for multilevel)
)
```

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `omega_c` | `float` | Cavity/storage frequency (rad/s) |
| `omega_q` | `float` | Qubit frequency (rad/s) |
| `alpha` | `float` | Transmon anharmonicity (rad/s), typically negative |
| `chi` | `float` | First-order dispersive shift (rad/s) |
| `kerr` | `float` | Cavity self-Kerr (rad/s) |
| `n_cav` | `int` | Cavity Hilbert-space truncation dimension |
| `n_tr` | `int` | Transmon Hilbert-space dimension |
| `chi_higher` | `Sequence[float]` | Higher-order dispersive terms: (χ₂, χ₃, ...) |
| `kerr_higher` | `Sequence[float]` | Higher-order Kerr terms: (K₂, ...) |

### Useful Methods

```python
# Static Hamiltonian in rotating frame
H0 = model.static_hamiltonian(frame=frame)

# Basis states
psi_g0 = model.basis_state(0, 0)  # |g, 0⟩
psi_e3 = model.basis_state(1, 3)  # |e, 3⟩

# Operators
ops = model.operators()  # {"a", "adag", "b", "bdag", "n_c", "n_q"}

# Transition frequencies
omega_ge_n0 = model.manifold_transition_frequency(n=0, frame=frame)
omega_ge_n3 = model.manifold_transition_frequency(n=3, frame=frame)
```

---

## Three-Mode: DispersiveReadoutTransmonStorageModel

For systems with a readout resonator:

```python
import numpy as np
from cqed_sim.core import DispersiveReadoutTransmonStorageModel

model = DispersiveReadoutTransmonStorageModel(
    omega_s=2 * np.pi * 5.0e9,       # Storage frequency (rad/s)
    omega_r=2 * np.pi * 7.5e9,       # Readout frequency (rad/s)
    omega_q=2 * np.pi * 6.0e9,       # Qubit frequency (rad/s)
    alpha=2 * np.pi * (-220e6),       # Anharmonicity (rad/s)
    chi_s=2 * np.pi * (-2.8e6),      # Storage–qubit dispersive shift (rad/s)
    chi_r=2 * np.pi * (-1.2e6),      # Readout–qubit dispersive shift (rad/s)
    chi_sr=2 * np.pi * 15e3,         # Storage–readout cross-Kerr (rad/s)
    kerr_s=2 * np.pi * (-2e3),       # Storage self-Kerr (rad/s)
    kerr_r=2 * np.pi * (-30e3),      # Readout self-Kerr (rad/s)
    n_storage=10,                     # Storage dimension
    n_readout=12,                     # Readout dimension
    n_tr=2,                           # Transmon levels
)
```

### Tensor Ordering

Three-mode: qubit ⊗ storage ⊗ readout → `|q, n_s, n_r⟩`

```python
psi = model.basis_state(0, 0, 0)  # |g, 0, 0⟩
```

### Transition Frequencies

```python
# Qubit transition at given storage/readout occupation
omega_q = model.qubit_transition_frequency(ns=0, nr=0, q=0, frame=frame)

# Storage transition at given occupations
omega_s = model.storage_transition_frequency(q=0, ns=0, nr=0, frame=frame)

# Readout transition
omega_r = model.readout_transition_frequency(q=0, ns=0, nr=0, frame=frame)
```

---

## General: UniversalCQEDModel

For maximum flexibility, including multilevel transmon, arbitrary number of bosonic modes, and custom couplings:

```python
import numpy as np
from cqed_sim.core import (
    BosonicModeSpec,
    DispersiveCouplingSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
)

model = UniversalCQEDModel(
    transmon=TransmonModeSpec(
        omega=2 * np.pi * 6.0e9,
        dim=5,                          # 5-level transmon
        alpha=2 * np.pi * (-200e6),
        label="qubit",
        aliases=("qubit", "transmon"),
        frame_channel="q",
    ),
    bosonic_modes=(
        BosonicModeSpec(
            label="storage",
            omega=2 * np.pi * 5.0e9,
            dim=12,
            kerr=2 * np.pi * (-2e3),
            aliases=("storage", "cavity"),
            frame_channel="c",
        ),
        BosonicModeSpec(
            label="readout",
            omega=2 * np.pi * 7.5e9,
            dim=10,
            aliases=("readout",),
            frame_channel="r",
        ),
    ),
    dispersive_couplings=(
        DispersiveCouplingSpec(mode="storage", chi=2 * np.pi * (-2.8e6)),
        DispersiveCouplingSpec(mode="readout", chi=2 * np.pi * (-1.1e6)),
    ),
)
```

### Key Methods

```python
# Hamiltonian
H = model.hamiltonian(frame=frame)

# Basis state
psi = model.basis_state(0, 0, 0)  # transmon level 0, storage 0, readout 0

# Operators
b = model.transmon_lowering()
a_s = model.mode_annihilation("storage")

# Transition frequencies
omega_01 = model.transmon_transition_frequency(
    mode_occupations={"storage": 0, "readout": 0},
    lower_level=0,
    upper_level=1,
    frame=frame,
)
```

### Additional Coupling Terms

Add cross-Kerr, self-Kerr, and exchange couplings:

```python
from cqed_sim.core import CrossKerrSpec, SelfKerrSpec, ExchangeSpec

model = UniversalCQEDModel(
    ...,
    cross_kerr_terms=(
        CrossKerrSpec(left="storage", right="readout", chi=2 * np.pi * 15e3),
    ),
    self_kerr_terms=(
        SelfKerrSpec(mode="readout", kerr=2 * np.pi * (-30e3)),
    ),
    exchange_terms=(
        ExchangeSpec(left="storage", right="readout", coupling=2 * np.pi * 1e3),
    ),
)
```

---

## Wrapper Relationship

`DispersiveTransmonCavityModel` and `DispersiveReadoutTransmonStorageModel` are **convenience wrappers** around `UniversalCQEDModel`. You can always access the underlying universal model:

```python
universal = model.as_universal_model()
```

All three model types expose the same core interface: `operators()`, `static_hamiltonian()`, `basis_state()`, `drive_coupling_operators()`, etc.
