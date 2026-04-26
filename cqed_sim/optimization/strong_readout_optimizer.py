from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from cqed_sim.metrics import ReadoutMetricSet, compute_readout_metrics
from cqed_sim.pulses.clear import SampledReadoutPulse, clear_readout_seed, square_readout_seed
from cqed_sim.readout.input_output import linear_pointer_response


MetricScorer = Callable[[SampledReadoutPulse], ReadoutMetricSet]
MistScorer = Callable[[SampledReadoutPulse], float]


@dataclass(frozen=True)
class StrongReadoutObjectiveWeights:
    wA: float = 1.0
    wQ: float = 1.0
    wL: float = 1.0
    wR: float = 1.0
    wE: float = 0.0
    wS: float = 0.0
    wM: float = 0.0


@dataclass(frozen=True)
class PulseConstraints:
    max_amplitude: float | None = None
    max_slew_rate: float | None = None
    bandwidth: float | None = None
    fixed_total_duration: float | None = None
    drive_frequency: float | None = None
    optimize_drive_frequency: bool = False


@dataclass(frozen=True)
class LinearPointerSeedModel:
    kappa: float
    chi: float
    noise_sigma: float = 0.25
    residual_weight_time: float = 0.0
    resonator_frequency: float | None = None
    measurement_efficiency: float = 1.0


@dataclass(frozen=True)
class StrongReadoutOptimizerConfig:
    method: str = "Powell"
    maxiter: int = 40
    n_candidates: int = 8
    random_seed: int | None = None
    parameter_scale: float = 1.0
    parameter_scale_mode: str = "relative"
    include_clear_seed: bool = True


@dataclass
class StrongReadoutCandidate:
    pulse: SampledReadoutPulse
    metrics: ReadoutMetricSet
    objective: float
    mist_penalty: float = 0.0
    stage_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class StrongReadoutOptimizationResult:
    candidates: list[StrongReadoutCandidate]
    pareto_set: list[StrongReadoutCandidate]
    history: list[StrongReadoutCandidate]
    config: StrongReadoutOptimizerConfig

    @property
    def best(self) -> StrongReadoutCandidate:
        if not self.candidates:
            raise ValueError("No candidates are available.")
        return self.candidates[0]


def _apply_bandwidth(samples: np.ndarray, *, dt: float, bandwidth: float | None) -> np.ndarray:
    if bandwidth is None or bandwidth <= 0.0 or samples.size <= 1:
        return np.asarray(samples, dtype=np.complex128)
    freqs = np.abs(2.0 * np.pi * np.fft.fftfreq(samples.size, d=float(dt)))
    spectrum = np.fft.fft(samples)
    spectrum[freqs > float(bandwidth)] = 0.0
    return np.fft.ifft(spectrum)


def enforce_pulse_constraints(
    pulse: SampledReadoutPulse,
    constraints: PulseConstraints,
) -> SampledReadoutPulse:
    samples = np.asarray(pulse.samples, dtype=np.complex128).copy()
    if constraints.fixed_total_duration is not None:
        target_n = max(1, int(np.round(float(constraints.fixed_total_duration) / pulse.dt)))
        if target_n != samples.size:
            old_t = np.linspace(0.0, 1.0, samples.size, endpoint=False)
            new_t = np.linspace(0.0, 1.0, target_n, endpoint=False)
            samples = np.interp(new_t, old_t, samples.real) + 1j * np.interp(new_t, old_t, samples.imag)
    samples = _apply_bandwidth(samples, dt=pulse.dt, bandwidth=constraints.bandwidth)
    if constraints.max_amplitude is not None:
        mag = np.abs(samples)
        scale = np.ones_like(mag)
        mask = mag > float(constraints.max_amplitude)
        scale[mask] = float(constraints.max_amplitude) / np.maximum(mag[mask], 1.0e-30)
        samples = samples * scale
    if constraints.max_slew_rate is not None and samples.size > 1:
        max_step = float(constraints.max_slew_rate) * pulse.dt
        out = samples.copy()
        for idx in range(1, out.size):
            delta = out[idx] - out[idx - 1]
            mag = abs(delta)
            if mag > max_step > 0.0:
                out[idx] = out[idx - 1] + delta * (max_step / mag)
        samples = out
    drive_frequency = pulse.drive_frequency if constraints.drive_frequency is None else float(constraints.drive_frequency)
    return SampledReadoutPulse(
        samples=samples,
        dt=pulse.dt,
        drive_frequency=drive_frequency,
        phase=pulse.phase,
        label=pulse.label,
    )


def _two_state_confusion_from_separation(separation: float, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        p_correct = 1.0 if separation > 0.0 else 0.5
    else:
        from math import erf, sqrt

        p_correct = 0.5 * (1.0 + erf(float(separation) / (2.0 * sqrt(2.0) * float(sigma))))
    p_correct = float(np.clip(p_correct, 0.0, 1.0))
    return np.array([[p_correct, 1.0 - p_correct], [1.0 - p_correct, p_correct]], dtype=float)


def linear_pointer_metrics(
    pulse: SampledReadoutPulse,
    model: LinearPointerSeedModel,
    *,
    max_slew: float | None = None,
) -> ReadoutMetricSet:
    dt = float(pulse.dt)
    if model.resonator_frequency is None:
        det_g = -0.5 * float(model.chi)
        det_e = 0.5 * float(model.chi)
    else:
        det_g = float(model.resonator_frequency) - 0.5 * float(model.chi) - float(pulse.drive_frequency)
        det_e = float(model.resonator_frequency) + 0.5 * float(model.chi) - float(pulse.drive_frequency)
    t, alpha_g = linear_pointer_response(pulse.samples, dt=dt, kappa=model.kappa, detuning=det_g)
    _t, alpha_e = linear_pointer_response(pulse.samples, dt=dt, kappa=model.kappa, detuning=det_e)
    output_separation = np.sqrt(max(float(model.kappa), 0.0)) * (alpha_e - alpha_g)
    snr2 = float(
        max(float(model.measurement_efficiency), 0.0)
        * np.trapezoid(np.abs(output_separation) ** 2, x=t)
    )
    from math import erf, sqrt

    f_assign = 0.5 * (1.0 + erf(sqrt(max(snr2, 0.0)) / 2.0))
    confusion = np.array([[f_assign, 1.0 - f_assign], [1.0 - f_assign, f_assign]], dtype=float)
    transition = np.eye(2, dtype=float)
    residual = max(float(abs(alpha_g[-1]) ** 2), float(abs(alpha_e[-1]) ** 2))
    return compute_readout_metrics(
        confusion=confusion,
        transition_matrix=transition,
        pulse_samples=pulse.samples,
        dt=dt,
        residual_resonator=residual,
        max_slew=max_slew,
        extra={
            "snr2": snr2,
            "detuning_g": det_g,
            "detuning_e": det_e,
            "legacy_noise_sigma": float(model.noise_sigma),
        },
    )


def _pulse_from_parameters(
    params: Sequence[float],
    *,
    template: SampledReadoutPulse,
    constraints: PulseConstraints,
) -> SampledReadoutPulse:
    data = np.asarray(params, dtype=float)
    n = template.samples.size
    if data.size not in {2 * n, 2 * n + 1}:
        raise ValueError("Parameter vector must contain I/Q samples and optional drive-frequency offset.")
    samples = data[:n] + 1j * data[n : 2 * n]
    drive_frequency = template.drive_frequency
    if data.size == 2 * n + 1 and constraints.optimize_drive_frequency:
        drive_frequency = float(template.drive_frequency + data[-1])
    return enforce_pulse_constraints(
        SampledReadoutPulse(samples=samples, dt=template.dt, drive_frequency=drive_frequency, label="optimized_readout"),
        constraints,
    )


def _parameters_from_pulse(pulse: SampledReadoutPulse, constraints: PulseConstraints) -> np.ndarray:
    base = np.concatenate([pulse.samples.real, pulse.samples.imag]).astype(float)
    if constraints.optimize_drive_frequency:
        base = np.concatenate([base, np.zeros(1, dtype=float)])
    return base


def _is_dominated(candidate: StrongReadoutCandidate, other: StrongReadoutCandidate) -> bool:
    c = candidate.metrics
    o = other.metrics
    other_no_worse = (
        o.assignment_fidelity >= c.assignment_fidelity
        and o.physical_qnd_fidelity >= c.physical_qnd_fidelity
        and o.leakage_probability <= c.leakage_probability
        and (o.residual_resonator_photons + o.residual_filter_photons)
        <= (c.residual_resonator_photons + c.residual_filter_photons)
    )
    other_better = (
        o.assignment_fidelity > c.assignment_fidelity
        or o.physical_qnd_fidelity > c.physical_qnd_fidelity
        or o.leakage_probability < c.leakage_probability
        or (o.residual_resonator_photons + o.residual_filter_photons)
        < (c.residual_resonator_photons + c.residual_filter_photons)
    )
    return bool(other_no_worse and other_better)


def _jitter_scale_for_parameters(
    x0: np.ndarray,
    *,
    constraints: PulseConstraints,
    parameter_scale: float,
    mode: str,
) -> np.ndarray:
    scale = np.full_like(x0, float(parameter_scale), dtype=float)
    if str(mode).lower() == "absolute":
        return scale
    control_count = x0.size - (1 if constraints.optimize_drive_frequency else 0)
    control_reference = float(constraints.max_amplitude or 0.0)
    if control_reference <= 0.0 and control_count > 0:
        control_reference = float(np.max(np.abs(x0[:control_count])))
    control_reference = max(control_reference, 1.0)
    scale[:control_count] = float(parameter_scale) * control_reference
    if constraints.optimize_drive_frequency and x0.size:
        freq_reference = float(constraints.bandwidth or control_reference)
        scale[-1] = float(parameter_scale) * max(freq_reference, 1.0)
    return scale


def pareto_front(candidates: Iterable[StrongReadoutCandidate]) -> list[StrongReadoutCandidate]:
    pool = list(candidates)
    front = [candidate for candidate in pool if not any(_is_dominated(candidate, other) for other in pool if other is not candidate)]
    return sorted(front, key=lambda item: item.objective)


@dataclass
class StrongReadoutOptimizer:
    linear_model: LinearPointerSeedModel
    weights: StrongReadoutObjectiveWeights = field(default_factory=StrongReadoutObjectiveWeights)
    constraints: PulseConstraints = field(default_factory=PulseConstraints)
    config: StrongReadoutOptimizerConfig = field(default_factory=StrongReadoutOptimizerConfig)
    stage_b_scorer: MetricScorer | None = None
    stage_c_scorer: MetricScorer | None = None
    mist_scorer: MistScorer | None = None

    def _score(self, pulse: SampledReadoutPulse) -> StrongReadoutCandidate:
        constrained = enforce_pulse_constraints(pulse, self.constraints)
        stage_a = linear_pointer_metrics(
            constrained,
            self.linear_model,
            max_slew=self.constraints.max_slew_rate,
        )
        metrics = stage_a
        stage_scores = {"A_linear_seed": 0.0}
        if self.stage_b_scorer is not None:
            metrics = self.stage_b_scorer(constrained)
            stage_scores["B_master_equation"] = 0.0
        if self.stage_c_scorer is not None:
            metrics = self.stage_c_scorer(constrained)
            stage_scores["C_trajectories"] = 0.0
        mist = 0.0 if self.mist_scorer is None else float(self.mist_scorer(constrained))
        objective = metrics.objective(
            wA=self.weights.wA,
            wQ=self.weights.wQ,
            wL=self.weights.wL,
            wR=self.weights.wR,
            wE=self.weights.wE,
            wS=self.weights.wS,
            wM=self.weights.wM,
            mist_penalty=mist,
        )
        stage_scores["D_MIST"] = mist
        return StrongReadoutCandidate(
            pulse=constrained,
            metrics=metrics,
            objective=float(objective),
            mist_penalty=mist,
            stage_scores=stage_scores,
        )

    def _default_seeds(self, *, amplitude: complex, duration: float, dt: float, drive_frequency: float) -> list[SampledReadoutPulse]:
        seeds = [
            square_readout_seed(
                amplitude=amplitude,
                duration=duration,
                dt=dt,
                drive_frequency=drive_frequency,
            )
        ]
        if self.config.include_clear_seed:
            seeds.append(
                clear_readout_seed(
                    amplitude=amplitude,
                    duration=duration,
                    dt=dt,
                    drive_frequency=drive_frequency,
                )
            )
        return [enforce_pulse_constraints(seed, self.constraints) for seed in seeds]

    def optimize(
        self,
        seeds: Sequence[SampledReadoutPulse] | None = None,
        *,
        amplitude: complex = 1.0,
        duration: float = 1.0,
        dt: float = 0.02,
        drive_frequency: float = 0.0,
    ) -> StrongReadoutOptimizationResult:
        rng = np.random.default_rng(self.config.random_seed)
        seed_pulses = list(seeds) if seeds is not None else self._default_seeds(
            amplitude=amplitude,
            duration=duration,
            dt=dt,
            drive_frequency=drive_frequency,
        )
        if not seed_pulses:
            raise ValueError("At least one seed pulse is required.")
        history: list[StrongReadoutCandidate] = []
        candidates: list[StrongReadoutCandidate] = []
        for seed in seed_pulses:
            seed = enforce_pulse_constraints(seed, self.constraints)
            x0 = _parameters_from_pulse(seed, self.constraints)

            def objective(x: np.ndarray) -> float:
                pulse = _pulse_from_parameters(x, template=seed, constraints=self.constraints)
                scored = self._score(pulse)
                history.append(scored)
                return float(scored.objective)

            candidates.append(self._score(seed))
            if self.config.maxiter > 0:
                result = minimize(
                    objective,
                    x0,
                    method=self.config.method,
                    options={"maxiter": int(self.config.maxiter), "disp": False},
                )
                candidates.append(self._score(_pulse_from_parameters(result.x, template=seed, constraints=self.constraints)))

            for _ in range(max(0, int(self.config.n_candidates) - 1)):
                jitter_scale = _jitter_scale_for_parameters(
                    x0,
                    constraints=self.constraints,
                    parameter_scale=float(self.config.parameter_scale),
                    mode=self.config.parameter_scale_mode,
                )
                jitter = rng.normal(scale=jitter_scale, size=x0.shape)
                candidates.append(self._score(_pulse_from_parameters(x0 + jitter, template=seed, constraints=self.constraints)))

        ranked = sorted(candidates, key=lambda candidate: candidate.objective)
        return StrongReadoutOptimizationResult(
            candidates=ranked,
            pareto_set=pareto_front(ranked),
            history=history,
            config=self.config,
        )


__all__ = [
    "LinearPointerSeedModel",
    "MetricScorer",
    "MistScorer",
    "PulseConstraints",
    "StrongReadoutCandidate",
    "StrongReadoutObjectiveWeights",
    "StrongReadoutOptimizationResult",
    "StrongReadoutOptimizer",
    "StrongReadoutOptimizerConfig",
    "enforce_pulse_constraints",
    "linear_pointer_metrics",
    "pareto_front",
]
