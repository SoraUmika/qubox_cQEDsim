# API Reference — Unitary Synthesis (`cqed_sim.unitary_synthesis`)

Gradient-free optimization of pulse/gate sequences to implement target unitaries within a qubit–cavity subspace.

!!! note "Convention"
    The synthesis drift-phase layer matches the runtime dispersive/Kerr convention. The remaining sign distinction is the waveform convention `Pulse.carrier = -omega_transition(frame)`.

---

## Subspace

**Module path:** `cqed_sim.unitary_synthesis.subspace.Subspace`

```python
@dataclass(frozen=True)
class Subspace:
    full_dim: int                      # Full Hilbert-space dimension
    indices: tuple[int, ...]           # Subspace state indices in full space
    labels: tuple[str, ...]            # Human-readable basis labels
    kind: str = "custom"
    metadata: dict | None = None
```

| Factory Method | Description |
|---|---|
| `Subspace.qubit_cavity_block(n_match, n_cav=None)` | Qubit–cavity block: \|g,0⟩, \|e,0⟩, \|g,1⟩, \|e,1⟩, ... |
| `Subspace.cavity_only(n_match, qubit="g", n_cav=None)` | Cavity subspace for fixed qubit state |
| `Subspace.custom(full_dim, indices, labels=None)` | Arbitrary index selection |

| Method | Description |
|---|---|
| `projector()` | Full-space projection operator |
| `embed(vec_sub)` | Subspace → full-space embedding |
| `extract(vec_full)` | Full-space → subspace extraction |
| `restrict_operator(op_full)` | Full-space operator → subspace restriction |
| `per_fock_blocks()` | Block slices for qubit_cavity_block kind |

---

## Target Construction

**Module path:** `cqed_sim.unitary_synthesis.targets`

```python
def make_target(
    name: str,               # "easy", "ghz", "cluster"
    n_match: int,
    variant: str = "analytic",
    **kwargs,
) -> np.ndarray
```

---

## Gate Sequence Primitives

**Module path:** `cqed_sim.unitary_synthesis.sequence`

### Gate Types

| Gate Class | Parameters | Description |
|---|---|---|
| `QubitRotation` | `theta`, `phi` | Qubit rotation ⊗ I_cav |
| `SQR` | `theta_n[]`, `phi_n[]` | Selective qubit rotation per Fock |
| `SNAP` | `phases[]` | Diagonal cavity phase gate |
| `Displacement` | `alpha = re + i·im` | D(α) ⊗ I_qubit |
| `ConditionalPhaseSQR` | `phases_n[]` | Conditional phase per n |
| `FreeEvolveCondPhase` | (uses drift_model) | Idle evolution under drift |

All gates inherit from `GateBase` with: `parameter_names(n_cav)`, `get_parameters(n_cav)`, `set_parameters(params, n_cav)`, `ideal_unitary(n_cav)`, `pulse_unitary(n_cav)`, `duration`, `optimize_time`, `time_bounds`.

### DriftPhaseModel

```python
@dataclass(frozen=True)
class DriftPhaseModel:
    chi: float = 0.0; chi2: float = 0.0
    kerr: float = 0.0; kerr2: float = 0.0
    delta_c: float = 0.0; delta_q: float = 0.0
    frame: str = "rotating_omega_c_omega_q"
```

**Drift energies (shared runtime/synthesis convention):**

$$E_{g,n} = \Delta_c n + K(n)$$
$$E_{e,n} = \Delta_c n + \Delta_q + (\chi n + \chi_2 n(n-1)) + K(n)$$

| Function | Description |
|---|---|
| `drift_phase_table(n_cav, duration, model)` | Precompute drift phases |
| `drift_phase_unitary(n_cav, duration, model)` | $\exp(-iH_0 t)$ as Qobj |
| `drift_hamiltonian_qobj(n_cav, model)` | Diagonal $H_0$ Hamiltonian |

### GateSequence

```python
@dataclass
class GateSequence:
    gates: list[GatePrimitive]
    n_cav: int
```

| Method | Description |
|---|---|
| `unitary(backend="ideal")` | Full sequence unitary |
| `total_duration()` | Sum of gate durations |
| `get_parameter_vector()` / `set_parameter_vector(v)` | Gate-angle parameters |
| `get_time_vector()` / `set_time_vector(t)` | Duration parameters |
| `serialize()` | List of gate records |

---

## UnitarySynthesizer

**Module path:** `cqed_sim.unitary_synthesis.optim.UnitarySynthesizer`

```python
class UnitarySynthesizer:
    def __init__(
        self,
        subspace: Subspace,
        backend: str = "ideal",
        gateset: list[str] | None = None,
        optimize_times: bool = True,
        time_bounds: dict | None = None,
        time_policy: dict | None = None,
        time_mode: str = "per-instance",
        time_groups: dict | None = None,
        leakage_weight: float = 0.0,
        time_reg_weight: float = 0.0,
        time_smooth_weight: float = 0.0,
        gauge: str = "global",
        drift_config: Mapping | None = None,
        include_conditional_phase_in_sqr: bool = False,
        hardware_limits: Mapping | None = None,
        time_grid: Mapping | None = None,
        constraints: Mapping | None = None,
        parallel: Mapping | None = None,
        progress: Mapping | None = None,
        seed: int = 0,
    )
```

```python
def fit(
    self,
    target: np.ndarray,
    init_guess: str = "heuristic",
    multistart: int = 1,
    maxiter: int = 300,
) -> SynthesisResult
```

**Optimization objective:**

$$L = (1 - F_{\text{subspace}}) + \lambda_L \cdot \text{leakage}_{\text{worst}} + \lambda_t \cdot \text{time\_reg} + \text{constraint\_penalties}$$

### SynthesisResult

```python
@dataclass
class SynthesisResult:
    success: bool
    objective: float
    sequence: GateSequence
    simulation: SimulationResult
    report: dict[str, Any]
    history: list[dict]
    history_by_run: dict[str, list[dict]]
```

---

## Metrics

**Module path:** `cqed_sim.unitary_synthesis.metrics`

| Function | Signature | Description |
|---|---|---|
| `subspace_unitary_fidelity(actual, target, gauge, block_slices)` | `-> float` | Phase-invariant fidelity in [0, 1] |
| `leakage_metrics(full_operator, subspace, probes, n_jobs)` | `-> LeakageMetrics` | average, worst, per_probe leakage |
| `objective_breakdown(actual_sub, target_sub, full_op, subspace, ...)` | `-> dict` | Full objective decomposition |
| `unitarity_error(op)` | `-> float` | $\|U^\dagger U - I\|_F$ |

---

## Constraints

**Module path:** `cqed_sim.unitary_synthesis.constraints`

| Function | Returns | Description |
|---|---|---|
| `snap_times_to_grid(times, dt, mode)` | `TimeGridResult` | Quantize to nearest dt |
| `piecewise_constant_samples(amplitudes, durations, dt)` | `ndarray` | Sample piecewise-constant waveform |
| `enforce_slew_limit(samples, dt, s_max, mode)` | `SlewConstraintResult` | Slew-rate violation check |
| `evaluate_tone_spacing(freqs, domega_min, ...)` | `ToneSpacingResult` | Tone-spacing constraint |
| `project_tone_frequencies(freqs, domega_min, forbidden_bands)` | `ndarray` | Projected frequencies |

---

## Progress Reporting

**Module path:** `cqed_sim.unitary_synthesis.progress`

| Class | Description |
|---|---|
| `ProgressReporter` | Base class with `on_start`, `on_event`, `on_end` hooks |
| `NullReporter` | No-op |
| `HistoryReporter` | Accumulate events in memory. Has `to_dataframe()`. |
| `JupyterLiveReporter` | Live Jupyter progress plots |

| Function | Description |
|---|---|
| `history_to_dataframe(history)` | Convert events to pandas DataFrame |
| `plot_history(history, what="objective_total", ...)` | Plot optimization history |

---

## Reporting

```python
def make_run_report(base_report, subspace_operator) -> dict
```

Adds per-Fock-block breakdown to the synthesis result report.
