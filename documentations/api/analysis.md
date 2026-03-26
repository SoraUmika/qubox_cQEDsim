# API Reference — Analysis (`cqed_sim.analysis`)

The analysis module translates between bare transmon parameters, measured dispersive parameters, and the runtime convention.

---

## HamiltonianParams

**Module path:** `cqed_sim.analysis.parameter_translation`

```python
@dataclass(frozen=True)
class HamiltonianParams:
    omega_q: float          # Transmon frequency (rad/s)
    omega_r: float          # Resonator frequency (rad/s)
    alpha: float            # Anharmonicity (rad/s), typically negative
    chi: float              # First-order dispersive shift (rad/s)
    chi_2: float            # Second-order dispersive shift (rad/s)
    g: float                # Coupling strength (rad/s)
    delta: float            # Detuning ω_q − ω_r (rad/s)
    ec: float               # Charging energy (rad/s)
    ej: float               # Josephson energy (rad/s)
    synthesis_chi: float    # Same canonical chi for synthesis callers
    synthesis_chi_2: float  # Same canonical chi_2 for synthesis callers
    regime: str = "dispersive"
    metadata: dict = field(default_factory=dict)
```

---

## Translation Functions

### `from_transmon_params`

```python
def from_transmon_params(
    ej: float, ec: float, g: float, omega_r: float,
    *, resonator_dim: int = 5, transmon_dim: int = 6,
) -> HamiltonianParams
```

Translates bare transmon parameters ($E_J$, $E_C$, $g$, $\omega_r$) to dressed runtime coefficients. Uses the large-$E_J$/$E_C$ expansion for the bare frequency and exact numerical diagonalization for dressed chi values.

### `from_measured`

```python
def from_measured(
    omega_01: float, alpha: float, chi: float, g: float,
    *, omega_r: float | None = None, detuning_branch: str = "positive",
    resonator_dim: int = 5, transmon_dim: int = 6,
) -> HamiltonianParams
```

Translates experimentally measured parameters ($\omega_{01}$, $\alpha$, $\chi$, $g$) to `HamiltonianParams`. Infers $E_J$, $E_C$ from the measured qubit frequency and anharmonicity, then optionally refines dressed dispersive coefficients via exact diagonalization.

---

## Usage

```python
from cqed_sim.analysis import from_transmon_params, from_measured

# From bare Josephson junction parameters
params = from_transmon_params(
    ej=15e9 * 2 * 3.14159,   # EJ in rad/s
    ec=250e6 * 2 * 3.14159,  # EC in rad/s
    g=80e6 * 2 * 3.14159,    # coupling in rad/s
    omega_r=8.0e9 * 2 * 3.14159,
)
print(f"chi = {params.chi / (2 * 3.14159) / 1e6:.2f} MHz")

# From measured spectroscopy data
params = from_measured(
    omega_01=6.15e9 * 2 * 3.14159,
    alpha=-256e6 * 2 * 3.14159,
    chi=-2.84e6 * 2 * 3.14159,
    g=80e6 * 2 * 3.14159,
)
```

!!! note "Units"
    All parameters are in **rad/s**. This matches the `cqed_sim` runtime convention. The `DeviceParameters` class in `cqed_sim.tomo` uses Hz and rad/ns internally — use its `.to_model()` method for unit-safe conversion.

### `from_measured`

```python
def from_measured(
    omega_01: float, alpha: float, chi: float, g: float,
    *, omega_r: float | None = None, detuning_branch: str = "positive",
    resonator_dim: int = 5, transmon_dim: int = 6,
) -> HamiltonianParams
```

Inverts measured qubit parameters into approximate circuit parameters. Solves the dispersive equation to find the detuning, then calls `from_transmon_params`.

**Detuning branch options:** `"positive"`, `"negative"`, `"largest-magnitude"`, or auto-select from closest root if `omega_r` is provided.
