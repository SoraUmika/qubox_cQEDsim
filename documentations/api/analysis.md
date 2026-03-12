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

Inverts measured qubit parameters into approximate circuit parameters. Solves the dispersive equation to find the detuning, then calls `from_transmon_params`.

**Detuning branch options:** `"positive"`, `"negative"`, `"largest-magnitude"`, or auto-select from closest root if `omega_r` is provided.
