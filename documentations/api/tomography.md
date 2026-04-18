# API Reference — Tomography (`cqed_sim.tomo`)

The tomography module provides Fock-resolved tomography protocols, all-XY calibration, leakage calibration, and device parameter management.

---

## DeviceParameters

**Module path:** `cqed_sim.tomo.device.DeviceParameters`

```python
@dataclass(frozen=True)
class DeviceParameters:
    ro_fq: float = 8596222556.078796       # Readout frequency (Hz)
    qb_fq: float = 6150369694.524461       # Qubit frequency (Hz)
    st_fq: float = 5240932800.0            # Storage frequency (Hz)
    ro_kappa: float = 4156000.0            # Readout linewidth (Hz)
    ro_chi: float = -913148.5              # Readout chi (Hz)
    anharmonicity: float = -255669694.5    # Alpha (Hz)
    st_chi: float = -2840421.354           # Storage chi (Hz)
    st_chi2: float = -21912.638            # Storage chi2 (Hz)
    st_chi3: float = -327.379              # Storage chi3 (Hz)
    st_K: float = -28844.0                 # Storage Kerr (Hz)
    st_K2: float = 1406.0                  # Storage Kerr2 (Hz)
    ro_therm_clks: float = 1000.0
    qb_therm_clks: float = 19625.0
    st_therm_clks: float = 200000.0
    qb_t1_relax_ns: float = 9812.87
    qb_t2_ramsey_ns: float = 6324.73
    qb_t2_echo_ns: float = 8381.0
```

| Method | Description |
|---|---|
| `hz_to_rad_per_ns(f_hz)` | Convert Hz to rad/ns: $2\pi f \cdot 10^{-9}$ |
| `to_model(n_cav=12, n_tr=3)` | Build `DispersiveTransmonCavityModel` with all params in rad/ns |

!!! warning
    `DeviceParameters.to_model()` uses a helper-specific **rad/ns** conversion path for tomography workflows that are parameterized in nanoseconds. The core `cqed_sim` model layer remains unit-coherent, so this does not imply a global rad/ns-only simulator convention.

---

## Tomography Protocol

**Module path:** `cqed_sim.tomo.protocol`

### QubitPulseCal

```python
@dataclass
class QubitPulseCal:
    amp90: float
    y_phase: float = np.pi / 2
    drag: float = 0.0
    detuning: float = 0.0
    duration_ns: float = 16.0

    @staticmethod
    def nominal() -> QubitPulseCal
    def amp(self, label: str) -> float    # "x90", "y90", "x180", "y180", "i"
    def phase(self, label: str) -> float
```

### All-XY Calibration

```python
ALL_XY_21: list[tuple[str, str]]  # 21 standard gate pairs

def run_all_xy(
    model, cal: QubitPulseCal, dt_ns=0.2, frame=None, noise=None,
) -> dict[str, np.ndarray]  # {"measured_z", "expected_z", "rms_error"}

def autocalibrate_all_xy(
    model, initial_cal, dt_ns=0.2, max_iter=12, target_rms=0.08,
) -> tuple[QubitPulseCal, dict]
```

### Fock-Resolved Tomography

```python
def run_fock_resolved_tomo(
    model, state_prep: Callable, n_max: int, cal: QubitPulseCal,
    tag_duration_ns=1000.0, tag_amp=0.0015, dt_ns=1.0,
    noise=None, ideal_tag=False, pre_rotation_mode="pulse",
    leakage_cal=None, unmix_lambda=1e-2,
) -> FockTomographyResult
```

| Field | Type | Description |
|---|---|---|
| `v_hat` | `dict[str, ndarray]` | `"x"`, `"y"`, `"z"` → raw Bloch components per Fock level |
| `p_n` | `ndarray` | Fock populations |
| `conditioned_bloch` | `dict[int, ndarray]` | n → [x, y, z] per Fock level |
| `v_rec` | `dict[str, ndarray] \| None` | Leakage-corrected vectors |

### Helper Functions

| Function | Description |
|---|---|
| `selective_qubit_drive_frequency(model, n)` | Positive physical qubit drive frequency for the Fock-selective manifold-$n$ tag tone |
| `selective_pi_pulse(n, t0_ns, duration_ns, amp, model, drag=0.0)` | Gaussian π-pulse targeting Fock manifold n. Internally converts the positive drive frequency into the raw `Pulse.carrier` expected by the runtime |
| `true_fock_resolved_vectors(state, n_max)` | Exact Bloch vectors by projecting onto each Fock manifold |
| `calibrate_leakage_matrix(model, n_max, alphas, bloch_states, cal, ...)` | Returns (W matrix, bias dict, condition number) |
