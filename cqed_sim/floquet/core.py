from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.drive_targets import DriveTarget
from cqed_sim.core.frame import FrameSpec
from cqed_sim.sim.noise import NoiseSpec, collapse_operators as runtime_collapse_operators
from cqed_sim.solvers.options import build_qutip_solver_options

from .utils import angular_frequency_from_period, bare_state_overlap_matrix, boundary_populations, fold_quasienergies, wrap_phase


WaveformSpec = str | Callable[[np.ndarray], np.ndarray]
SpectrumCallback = Callable[[np.ndarray], np.ndarray]


def flat_markov_spectrum(scale: float = 1.0) -> SpectrumCallback:
    level = float(scale)

    def _callback(frequency: np.ndarray) -> np.ndarray:
        values = np.asarray(frequency, dtype=float)
        return np.full(values.shape, level, dtype=float)

    return _callback


def _wrap_spectrum_callback(callback: SpectrumCallback | None) -> SpectrumCallback:
    if callback is None:
        return flat_markov_spectrum()

    def _wrapped(frequency: np.ndarray) -> np.ndarray:
        values = np.asarray(frequency, dtype=float)
        result = callback(values)
        array = np.asarray(result, dtype=float)
        if array.shape == ():
            return np.full(values.shape, float(array), dtype=float)
        if array.shape != values.shape:
            raise ValueError(
                f"Floquet-Markov spectrum callback returned shape {array.shape}, expected {values.shape}."
            )
        return array

    return _wrapped


@dataclass(frozen=True)
class PeriodicFourierComponent:
    harmonic: int
    amplitude: complex

    def __post_init__(self) -> None:
        object.__setattr__(self, "harmonic", int(self.harmonic))
        object.__setattr__(self, "amplitude", complex(self.amplitude))


@dataclass(frozen=True)
class PeriodicDriveTerm:
    operator: qt.Qobj | None = None
    target: DriveTarget | None = None
    quadrature: str = "x"
    amplitude: complex = 1.0
    frequency: float = 0.0
    phase: float = 0.0
    waveform: WaveformSpec = "cos"
    fourier_components: Sequence[PeriodicFourierComponent] = field(default_factory=tuple)
    label: str | None = None

    def __post_init__(self) -> None:
        if (self.operator is None) == (self.target is None):
            raise ValueError("Exactly one of operator or target must be supplied for a PeriodicDriveTerm.")
        if self.operator is not None and not isinstance(self.operator, qt.Qobj):
            raise TypeError("PeriodicDriveTerm.operator must be a QuTiP Qobj when provided.")
        if self.target is not None and isinstance(self.target, str) and not self.target.strip():
            raise ValueError("PeriodicDriveTerm.target cannot be an empty string.")
        object.__setattr__(self, "quadrature", str(self.quadrature).strip().lower())
        object.__setattr__(self, "amplitude", complex(self.amplitude))
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "phase", float(self.phase))
        object.__setattr__(
            self,
            "fourier_components",
            tuple(
                component if isinstance(component, PeriodicFourierComponent) else PeriodicFourierComponent(**component)
                for component in self.fourier_components
            ),
        )
        if self.fourier_components and not callable(self.waveform) and str(self.waveform).strip().lower() == "constant":
            raise ValueError("Do not combine fourier_components with waveform='constant'; encode the DC harmonic explicitly instead.")

    def _waveform_values(self, theta: np.ndarray) -> np.ndarray:
        if self.fourier_components:
            values = np.zeros_like(theta, dtype=np.complex128)
            for component in self.fourier_components:
                values = values + component.amplitude * np.exp(1j * float(component.harmonic) * theta)
            return values
        if callable(self.waveform):
            return np.asarray(self.waveform(theta), dtype=np.complex128)

        waveform = str(self.waveform).strip().lower()
        if waveform in {"cos", "cosine"}:
            return np.cos(theta).astype(np.complex128)
        if waveform in {"sin", "sine"}:
            return np.sin(theta).astype(np.complex128)
        if waveform in {"exp", "complex_exp", "complex", "cw"}:
            return np.exp(1j * theta)
        if waveform in {"constant", "dc"}:
            return np.ones_like(theta, dtype=np.complex128)
        if waveform == "square":
            return np.where(np.cos(theta) >= 0.0, 1.0, -1.0).astype(np.complex128)
        raise ValueError(f"Unsupported Floquet waveform '{self.waveform}'.")

    def coefficient(self, t: float | np.ndarray) -> complex | np.ndarray:
        values = np.asarray(t, dtype=float)
        theta = float(self.frequency) * values + float(self.phase)
        coeff = complex(self.amplitude) * self._waveform_values(theta)
        if np.ndim(t) == 0:
            return complex(np.asarray(coeff).reshape(-1)[0])
        return np.asarray(coeff, dtype=np.complex128)

    def exact_fourier_components(self, period: float, *, rtol: float = 1.0e-8, atol: float = 1.0e-10) -> dict[int, complex] | None:
        omega = angular_frequency_from_period(period)
        if self.fourier_components:
            return {
                int(component.harmonic): complex(self.amplitude) * component.amplitude * np.exp(1j * float(component.harmonic) * self.phase)
                for component in self.fourier_components
            }

        waveform = None if callable(self.waveform) else str(self.waveform).strip().lower()
        if waveform is None:
            return None
        if abs(self.frequency) <= atol:
            return {0: complex(self.coefficient(0.0))}

        harmonic = self.frequency / omega
        nearest_harmonic = int(np.rint(harmonic))
        if not np.isclose(harmonic, nearest_harmonic, rtol=rtol, atol=atol):
            return None

        amplitude = complex(self.amplitude)
        phase = float(self.phase)
        if waveform in {"cos", "cosine"}:
            return {
                nearest_harmonic: 0.5 * amplitude * np.exp(1j * phase),
                -nearest_harmonic: 0.5 * amplitude * np.exp(-1j * phase),
            }
        if waveform in {"sin", "sine"}:
            return {
                nearest_harmonic: -0.5j * amplitude * np.exp(1j * phase),
                -nearest_harmonic: 0.5j * amplitude * np.exp(-1j * phase),
            }
        if waveform in {"exp", "complex_exp", "complex", "cw"}:
            return {nearest_harmonic: amplitude * np.exp(1j * phase)}
        if waveform in {"constant", "dc"}:
            return {0: amplitude}
        return None


@dataclass(frozen=True)
class FloquetProblem:
    period: float
    periodic_terms: Sequence[PeriodicDriveTerm] = field(default_factory=tuple)
    static_hamiltonian: qt.Qobj | None = None
    model: Any | None = None
    frame: FrameSpec = FrameSpec()
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.period) <= 0.0:
            raise ValueError("FloquetProblem.period must be positive.")
        terms = tuple(
            term if isinstance(term, PeriodicDriveTerm) else PeriodicDriveTerm(**term)
            for term in self.periodic_terms
        )
        object.__setattr__(self, "periodic_terms", terms)
        if self.static_hamiltonian is None:
            if self.model is None:
                raise ValueError("Provide either static_hamiltonian or model when constructing a FloquetProblem.")
            if not hasattr(self.model, "static_hamiltonian"):
                raise ValueError("Model must provide static_hamiltonian(frame=...) to build a FloquetProblem without an explicit static Hamiltonian.")
            object.__setattr__(self, "static_hamiltonian", self.model.static_hamiltonian(frame=self.frame))
        elif not isinstance(self.static_hamiltonian, qt.Qobj):
            raise TypeError("FloquetProblem.static_hamiltonian must be a QuTiP Qobj.")
        object.__setattr__(self, "period", float(self.period))


@dataclass(frozen=True)
class FloquetConfig:
    n_time_samples: int = 401
    atol: float = 1.0e-8
    rtol: float = 1.0e-7
    max_step: float | None = None
    nsteps: int | None = None
    solver_options: dict[str, Any] = field(default_factory=dict)
    sort: bool = True
    sparse: bool = False
    zone_center: float = 0.0
    precompute_times: Sequence[float] | None = None
    overlap_reference_time: float = 0.0
    sambe_harmonic_cutoff: int | None = None
    sambe_n_time_samples: int | None = None
    commensurability_rtol: float = 1.0e-8
    commensurability_atol: float = 1.0e-10
    boundary_population_warning_threshold: float = 1.0e-3

    def __post_init__(self) -> None:
        if int(self.n_time_samples) < 8:
            raise ValueError("FloquetConfig.n_time_samples must be at least 8.")
        if self.max_step is not None and float(self.max_step) <= 0.0:
            raise ValueError("FloquetConfig.max_step must be positive when provided.")
        if self.nsteps is not None and int(self.nsteps) <= 0:
            raise ValueError("FloquetConfig.nsteps must be positive when provided.")
        if self.sambe_harmonic_cutoff is not None and int(self.sambe_harmonic_cutoff) < 0:
            raise ValueError("FloquetConfig.sambe_harmonic_cutoff must be non-negative when provided.")
        if self.precompute_times is not None:
            object.__setattr__(self, "precompute_times", tuple(float(value) for value in self.precompute_times))
        object.__setattr__(self, "solver_options", dict(self.solver_options))


@dataclass(frozen=True)
class FloquetMarkovBath:
    operator: qt.Qobj
    spectrum: SpectrumCallback | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.operator, qt.Qobj):
            raise TypeError("FloquetMarkovBath.operator must be a QuTiP Qobj.")
        if self.spectrum is not None and not callable(self.spectrum):
            raise TypeError("FloquetMarkovBath.spectrum must be callable when provided.")

    def resolved_spectrum(self) -> SpectrumCallback:
        return _wrap_spectrum_callback(self.spectrum)


@dataclass(frozen=True)
class FloquetMarkovConfig:
    floquet: FloquetConfig = field(default_factory=FloquetConfig)
    kmax: int = 5
    nT: int | None = None
    w_th: float = 0.0
    store_states: bool | None = None
    store_final_state: bool = True
    store_floquet_states: bool = False
    normalize_output: bool = True
    progress_bar: str = ""
    method: str | None = None
    atol: float | None = None
    rtol: float | None = None
    max_step: float | None = None
    solver_options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.kmax) < 0:
            raise ValueError("FloquetMarkovConfig.kmax must be non-negative.")
        if self.nT is not None and int(self.nT) <= 0:
            raise ValueError("FloquetMarkovConfig.nT must be positive when provided.")
        if self.atol is not None and float(self.atol) <= 0.0:
            raise ValueError("FloquetMarkovConfig.atol must be positive when provided.")
        if self.rtol is not None and float(self.rtol) <= 0.0:
            raise ValueError("FloquetMarkovConfig.rtol must be positive when provided.")
        if self.max_step is not None and float(self.max_step) <= 0.0:
            raise ValueError("FloquetMarkovConfig.max_step must be positive when provided.")


@dataclass
class FloquetMarkovResult:
    floquet_result: "FloquetResult"
    config: FloquetMarkovConfig
    baths: tuple[FloquetMarkovBath, ...]
    tlist: np.ndarray
    solver_result: Any
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def states(self):
        return self.solver_result.states

    @property
    def expect(self):
        return self.solver_result.expect

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self.solver_result.times, dtype=float)

    @property
    def final_state(self):
        return getattr(self.solver_result, "final_state", None)

    @property
    def floquet_states(self):
        return getattr(self.solver_result, "floquet_states", None)


@dataclass
class FloquetResult:
    problem: FloquetProblem
    config: FloquetConfig
    qutip_hamiltonian: qt.Qobj | qt.QobjEvo
    floquet_basis: qt.FloquetBasis
    period_propagator: qt.Qobj
    eigenphases: np.ndarray
    quasienergies: np.ndarray
    mode_order: np.ndarray
    floquet_modes_0: tuple[qt.Qobj, ...]
    bare_hamiltonian_eigenenergies: np.ndarray
    bare_hamiltonian_eigenstates: tuple[qt.Qobj, ...]
    bare_state_overlaps: np.ndarray
    dominant_bare_state_indices: np.ndarray
    harmonic_component_norms: dict[int, float] | None = None
    effective_hamiltonian: qt.Qobj | None = None
    sambe_hamiltonian: qt.Qobj | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def modes(self, t: float = 0.0) -> tuple[qt.Qobj, ...]:
        modes = self.floquet_basis.mode(float(t))
        return tuple(modes[int(index)] for index in self.mode_order)

    def states(self, t: float = 0.0) -> tuple[qt.Qobj, ...]:
        states = self.floquet_basis.state(float(t))
        return tuple(states[int(index)] for index in self.mode_order)

    def mode(self, index: int, t: float = 0.0) -> qt.Qobj:
        return self.modes(t)[int(index)]

    def state(self, index: int, t: float = 0.0) -> qt.Qobj:
        return self.states(t)[int(index)]


def _validate_periodicity(problem: FloquetProblem, config: FloquetConfig) -> None:
    omega = angular_frequency_from_period(problem.period)
    for term in problem.periodic_terms:
        exact = term.exact_fourier_components(
            problem.period,
            rtol=config.commensurability_rtol,
            atol=config.commensurability_atol,
        )
        if exact is not None:
            continue
        if abs(term.frequency) <= config.commensurability_atol:
            continue
        ratio = float(term.frequency) / omega
        nearest = np.rint(ratio)
        if not np.isclose(ratio, nearest, rtol=config.commensurability_rtol, atol=config.commensurability_atol):
            raise ValueError(
                f"Periodic drive term '{term.label or term.target or 'operator'}' is not commensurate with the Floquet period. "
                f"Expected frequency / Omega to be an integer, got {ratio}."
            )


def solve_floquet(problem: FloquetProblem, config: FloquetConfig | None = None) -> FloquetResult:
    from .builders import build_floquet_hamiltonian, harmonic_component_norms
    from .effective_models import _effective_hamiltonian_from_modes, build_sambe_hamiltonian

    cfg = config or FloquetConfig()
    _validate_periodicity(problem, cfg)
    qutip_hamiltonian = build_floquet_hamiltonian(problem)

    options = build_qutip_solver_options(
        atol=cfg.atol,
        rtol=cfg.rtol,
        max_step=cfg.max_step,
        nsteps=cfg.nsteps,
        solver_options=cfg.solver_options,
    )

    floquet_kwargs: dict[str, Any] = {
        "options": options,
        "sparse": cfg.sparse,
        "sort": cfg.sort,
    }
    if cfg.precompute_times is not None:
        floquet_kwargs["precompute"] = np.asarray(cfg.precompute_times, dtype=float)

    floquet_basis = qt.FloquetBasis(qutip_hamiltonian, problem.period, **floquet_kwargs)
    raw_quasienergies = np.asarray(floquet_basis.e_quasi, dtype=float)
    omega = angular_frequency_from_period(problem.period)
    folded_quasienergies = fold_quasienergies(raw_quasienergies, omega, zone_center=cfg.zone_center)
    order = np.argsort(folded_quasienergies)
    quasienergies = folded_quasienergies[order]
    floquet_modes_0 = tuple(floquet_basis.mode(0.0)[int(index)] for index in order)
    period_propagator = floquet_basis.U(problem.period)
    eigenphases = wrap_phase(-quasienergies * float(problem.period))

    bare_eigenenergies, bare_eigenstates = problem.static_hamiltonian.eigenstates()
    bare_state_overlaps = bare_state_overlap_matrix(floquet_modes_0, tuple(bare_eigenstates))
    dominant_bare_state_indices = np.argmax(bare_state_overlaps, axis=1)
    effective_hamiltonian = _effective_hamiltonian_from_modes(quasienergies, floquet_modes_0)

    warnings: list[str] = []
    edge_population = boundary_populations(floquet_modes_0, problem.static_hamiltonian.dims[0])
    if edge_population.size and float(np.max(edge_population)) > float(cfg.boundary_population_warning_threshold):
        warnings.append(
            "Floquet modes place noticeable weight on the truncation boundary; increase the Hilbert-space dimension for quantitative strong-drive analysis."
        )

    harmonic_norms = None
    sambe_hamiltonian = None
    if cfg.sambe_harmonic_cutoff is not None:
        harmonic_norms = harmonic_component_norms(
            problem,
            cfg.sambe_harmonic_cutoff,
            n_time_samples=max(int(cfg.sambe_n_time_samples or cfg.n_time_samples), 8),
        )
        sambe_hamiltonian = build_sambe_hamiltonian(
            problem,
            harmonic_cutoff=cfg.sambe_harmonic_cutoff,
            n_time_samples=max(int(cfg.sambe_n_time_samples or cfg.n_time_samples), 8),
        )

    metadata = {
        "fundamental_angular_frequency": omega,
        "zone_center": float(cfg.zone_center),
        "boundary_populations": edge_population,
    }

    return FloquetResult(
        problem=problem,
        config=cfg,
        qutip_hamiltonian=qutip_hamiltonian,
        floquet_basis=floquet_basis,
        period_propagator=period_propagator,
        eigenphases=np.asarray(eigenphases, dtype=float),
        quasienergies=np.asarray(quasienergies, dtype=float),
        mode_order=np.asarray(order, dtype=int),
        floquet_modes_0=floquet_modes_0,
        bare_hamiltonian_eigenenergies=np.asarray(bare_eigenenergies, dtype=float),
        bare_hamiltonian_eigenstates=tuple(bare_eigenstates),
        bare_state_overlaps=np.asarray(bare_state_overlaps, dtype=float),
        dominant_bare_state_indices=np.asarray(dominant_bare_state_indices, dtype=int),
        harmonic_component_norms=harmonic_norms,
        effective_hamiltonian=effective_hamiltonian,
        sambe_hamiltonian=sambe_hamiltonian,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def build_floquet_markov_baths(
    problem: FloquetProblem,
    noise: NoiseSpec,
    *,
    spectrum: SpectrumCallback | None = None,
) -> tuple[FloquetMarkovBath, ...]:
    if problem.model is None:
        raise ValueError("Floquet-Markov noise bridging requires FloquetProblem(model=...) so collapse operators can be resolved.")
    operators = runtime_collapse_operators(problem.model, noise)
    wrapped_spectrum = _wrap_spectrum_callback(spectrum)
    return tuple(
        FloquetMarkovBath(
            operator=operator if operator.isherm else operator.dag(),
            spectrum=wrapped_spectrum,
            label=f"bath_{index}",
        )
        for index, operator in enumerate(operators)
    )


def solve_floquet_markov(
    problem: FloquetProblem,
    initial_state: qt.Qobj,
    tlist: Sequence[float],
    *,
    baths: Sequence[FloquetMarkovBath] | None = None,
    noise: NoiseSpec | None = None,
    e_ops: Any = None,
    args: dict[str, Any] | None = None,
    config: FloquetMarkovConfig | None = None,
) -> FloquetMarkovResult:
    cfg = config or FloquetMarkovConfig()
    floquet_result = solve_floquet(problem, cfg.floquet)

    resolved_baths: tuple[FloquetMarkovBath, ...]
    if baths is not None:
        resolved_baths = tuple(
            bath if isinstance(bath, FloquetMarkovBath) else FloquetMarkovBath(**bath)
            for bath in baths
        )
    elif noise is not None:
        resolved_baths = build_floquet_markov_baths(problem, noise)
    else:
        resolved_baths = ()

    if not resolved_baths:
        raise ValueError("solve_floquet_markov requires at least one Floquet-Markov bath or a non-zero NoiseSpec.")

    for bath in resolved_baths:
        if bath.operator.dims != problem.static_hamiltonian.dims:
            raise ValueError(
                f"Floquet-Markov bath '{bath.label or 'operator'}' has dims {bath.operator.dims}, but the Floquet problem uses dims {problem.static_hamiltonian.dims}."
            )

    extra_options: dict[str, Any] = {
        "store_floquet_states": bool(cfg.store_floquet_states),
        "normalize_output": bool(cfg.normalize_output),
    }
    if cfg.store_states is not None:
        extra_options["store_states"] = bool(cfg.store_states)
    active_max_step = cfg.floquet.max_step if cfg.max_step is None else cfg.max_step
    if cfg.method is not None:
        extra_options["method"] = str(cfg.method)
    options = build_qutip_solver_options(
        atol=cfg.floquet.atol if cfg.atol is None else cfg.atol,
        rtol=cfg.floquet.rtol if cfg.rtol is None else cfg.rtol,
        max_step=active_max_step,
        nsteps=cfg.floquet.nsteps,
        store_final_state=bool(cfg.store_final_state),
        progress_bar=str(cfg.progress_bar),
        extra_options=extra_options,
        solver_options=cfg.solver_options,
    )

    solver = qt.FMESolver(
        floquet_result.floquet_basis,
        [(bath.operator, bath.resolved_spectrum()) for bath in resolved_baths],
        float(cfg.w_th),
        kmax=int(cfg.kmax),
        nT=None if cfg.nT is None else int(cfg.nT),
        options=options,
    )
    qutip_result = solver.run(
        initial_state,
        np.asarray(tlist, dtype=float),
        args=args,
        e_ops=e_ops,
    )

    metadata = {
        "bath_count": len(resolved_baths),
        "used_noise_bridge": bool(noise is not None and baths is None),
    }
    return FloquetMarkovResult(
        floquet_result=floquet_result,
        config=cfg,
        baths=resolved_baths,
        tlist=np.asarray(tlist, dtype=float),
        solver_result=qutip_result,
        warnings=tuple(floquet_result.warnings),
        metadata=metadata,
    )


__all__ = [
    "FloquetConfig",
    "FloquetMarkovBath",
    "FloquetMarkovConfig",
    "FloquetMarkovResult",
    "FloquetProblem",
    "FloquetResult",
    "PeriodicDriveTerm",
    "PeriodicFourierComponent",
    "build_floquet_markov_baths",
    "flat_markov_spectrum",
    "solve_floquet",
    "solve_floquet_markov",
]
