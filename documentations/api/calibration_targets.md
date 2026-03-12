# API Reference — Calibration Targets (`cqed_sim.calibration_targets`)

Lightweight surrogate-model calibration sweeps that return fitted parameters without running full time-domain simulations.

---

## CalibrationResult

```python
@dataclass
class CalibrationResult:
    fitted_parameters: dict[str, float]
    uncertainties: dict[str, float]
    raw_data: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
```

---

## Calibration Sweep Functions

All functions take a `model` object (with `omega_q` and/or `alpha` attributes) and return a `CalibrationResult`.

### `run_spectroscopy`

```python
def run_spectroscopy(
    model,
    drive_frequencies: np.ndarray,
    *,
    linewidth: float | None = None,       # Defaults to |alpha|/10
    excited_state_fraction: float = 0.0,  # Thermal population fraction
) -> CalibrationResult
```

**Fitted parameters:** `omega_01`, `omega_12`. Synthesizes Lorentzian response and locates peaks via quadratic interpolation.

### `run_rabi`

```python
def run_rabi(
    model,
    amplitudes: np.ndarray,
    *,
    duration: float = 40e-9,
    omega_scale: float = 1.0,
) -> CalibrationResult
```

**Fitted parameters:** `omega_scale`, `duration`. Fits $\sin^2(\Omega A T / 2)$ via `curve_fit`.

### `run_ramsey`

```python
def run_ramsey(
    model,
    delays: np.ndarray,
    *,
    detuning: float,
    t2_star: float = 20e-6,
) -> CalibrationResult
```

**Fitted parameters:** `delta_omega`, `t2_star`. Fits Ramsey fringe $0.5(1 + e^{-t/T_2^*} \cos(\Delta\omega \cdot t))$.

### `run_t1`

```python
def run_t1(
    model,
    delays: np.ndarray,
    *,
    t1: float = 30e-6,
) -> CalibrationResult
```

**Fitted parameters:** `t1`.

### `run_t2_echo`

```python
def run_t2_echo(
    model,
    delays: np.ndarray,
    *,
    t2_echo: float = 40e-6,
) -> CalibrationResult
```

**Fitted parameters:** `t2_echo`.

### `run_drag_tuning`

```python
def run_drag_tuning(
    model,
    drag_values: np.ndarray,
    *,
    optimal_drag: float | None = None,   # Defaults to −1/alpha
    baseline_leakage: float = 1e-3,
    curvature: float = 0.25,
) -> CalibrationResult
```

**Fitted parameters:** `drag_optimal`. Synthesizes quadratic leakage curve and fits vertex.
