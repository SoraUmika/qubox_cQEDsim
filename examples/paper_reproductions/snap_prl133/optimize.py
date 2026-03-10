from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from examples.studies.snap_opt.experiments import SnapRunConfig
from examples.studies.snap_opt.pulses import SnapToneParameters

from .errors import CoherentMetricResult, ManifoldErrors, compute_mean_squared_overlap


@dataclass
class SnapOptimizationResult:
    params: SnapToneParameters
    history_fidelity: list[float]
    history_epsilon_coh: list[float]
    history_error_vector_norm: list[float]
    history_max_component: list[float]
    history_params: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    history_error: list[float]
    converged: bool
    threshold_hit: bool
    iters_to_threshold: int | None
    final_metric: CoherentMetricResult
    final_errors: ManifoldErrors


def _apply_update(
    params: SnapToneParameters,
    metric: CoherentMetricResult,
    *,
    eta: float,
    duration: float,
    base_amp: float,
) -> SnapToneParameters:
    # Paper mapping:
    #   Delta lambda_n = -eps_L/(2T)
    #   Delta alpha_n = -Delta theta_n
    #   Delta omega_n = +pi*eps_T/(2T)
    delta_lambda = -metric.eps_l / (2.0 * max(duration, 1e-12))
    amp_shift = delta_lambda / max(base_amp, 1e-12)
    new_a = params.amplitudes + eta * amp_shift
    new_d = params.detunings + eta * (np.pi * metric.eps_t / (2.0 * max(duration, 1e-12)))
    new_p = params.phases - eta * metric.dtheta
    new_a = np.clip(new_a, 0.05, 5.0)
    return SnapToneParameters(amplitudes=new_a, detunings=new_d, phases=new_p)


def _to_manifold_errors(metric: CoherentMetricResult) -> ManifoldErrors:
    return ManifoldErrors(
        dtheta=metric.dtheta.copy(),
        dlambda=metric.eps_l.copy(),
        dalpha=metric.eps_t.copy(),
        mean_overlap_error=float(metric.error_vector_norm),
    )


def optimize_snap_prl133(
    model: DispersiveTransmonCavityModel,
    target_phases,
    cfg: SnapRunConfig,
    *,
    initial_params: SnapToneParameters | None = None,
    frame: FrameSpec | None = None,
    max_iter: int = 60,
    learning_rate: float = 0.25,
    threshold: float = 1e-5,
    near_monotonic_window: int = 5,
    jitter_tol: float = 1e-8,
    local_refine_maxiter: int = 0,
) -> SnapOptimizationResult:
    frame = frame or FrameSpec(omega_q_frame=model.omega_q)
    target_phases = np.asarray(target_phases, dtype=float)
    params = initial_params or SnapToneParameters.vanilla(target_phases)
    history_fidelity: list[float] = []
    history_epsilon: list[float] = []
    history_norm: list[float] = []
    history_maxc: list[float] = []
    history_params: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    metric = compute_mean_squared_overlap(
        model=model,
        target_phases=target_phases,
        cfg=cfg,
        params=params,
        frame=frame,
        context="optimize:init",
    )
    history_fidelity.append(metric.fidelity)
    history_epsilon.append(metric.epsilon_coh)
    history_norm.append(metric.error_vector_norm)
    history_maxc.append(metric.max_component_error)
    history_params.append((params.amplitudes.copy(), params.detunings.copy(), params.phases.copy()))
    threshold_hit = metric.epsilon_coh <= threshold
    iters_to_threshold = 0 if threshold_hit else None

    for iteration in range(1, max_iter + 1):
        if threshold_hit:
            break
        eta = learning_rate
        improved = False
        best_candidate = params
        best_metric = metric
        for _ in range(10):
            cand = _apply_update(
                params,
                metric,
                eta=eta,
                duration=cfg.duration,
                base_amp=cfg.base_amp,
            )
            cand_metric = compute_mean_squared_overlap(
                model=model,
                target_phases=target_phases,
                cfg=cfg,
                params=cand,
                frame=frame,
                context=f"optimize:iter{iteration}",
            )
            if cand_metric.epsilon_coh <= metric.epsilon_coh + jitter_tol:
                improved = True
                best_candidate = cand
                best_metric = cand_metric
                break
            eta *= 0.5
        if not improved:
            break

        params = best_candidate
        metric = best_metric
        history_fidelity.append(metric.fidelity)
        history_epsilon.append(metric.epsilon_coh)
        history_norm.append(metric.error_vector_norm)
        history_maxc.append(metric.max_component_error)
        history_params.append((params.amplitudes.copy(), params.detunings.copy(), params.phases.copy()))
        if metric.epsilon_coh <= threshold:
            threshold_hit = True
            iters_to_threshold = iteration
            break

        if len(history_epsilon) >= near_monotonic_window:
            tail = np.asarray(history_epsilon[-near_monotonic_window:], dtype=float)
            if np.any(np.diff(tail) > jitter_tol):
                # Stop if near-convergence monotonicity is violated.
                break

    if threshold_hit and iters_to_threshold is None:
        iters_to_threshold = max(0, len(history_epsilon) - 1)

    if (not threshold_hit) and local_refine_maxiter > 0:
        try:
            from scipy.optimize import minimize
        except Exception:
            minimize = None
        if minimize is not None:
            l = target_phases.size

            def obj(x: np.ndarray) -> float:
                p = SnapToneParameters(
                    amplitudes=np.clip(x[:l], 0.05, 5.0),
                    detunings=x[l : 2 * l],
                    phases=x[2 * l :],
                )
                m = compute_mean_squared_overlap(
                    model=model,
                    target_phases=target_phases,
                    cfg=cfg,
                    params=p,
                    frame=frame,
                    context="optimize:local_refine",
                )
                return m.epsilon_coh

            x0 = np.concatenate([params.amplitudes, params.detunings, params.phases])
            res = minimize(
                obj,
                x0,
                method="Powell",
                options={"maxiter": int(local_refine_maxiter), "xtol": 1e-4, "ftol": 1e-7, "disp": False},
            )
            x = np.asarray(res.x, dtype=float)
            params = SnapToneParameters(
                amplitudes=np.clip(x[:l], 0.05, 5.0),
                detunings=x[l : 2 * l],
                phases=x[2 * l :],
            )
            metric = compute_mean_squared_overlap(
                model=model,
                target_phases=target_phases,
                cfg=cfg,
                params=params,
                frame=frame,
                context="optimize:local_refine_final",
            )
            history_fidelity.append(metric.fidelity)
            history_epsilon.append(metric.epsilon_coh)
            history_norm.append(metric.error_vector_norm)
            history_maxc.append(metric.max_component_error)
            history_params.append((params.amplitudes.copy(), params.detunings.copy(), params.phases.copy()))
            if metric.epsilon_coh <= threshold:
                threshold_hit = True
                iters_to_threshold = max(0, len(history_epsilon) - 1)

    final_errors = _to_manifold_errors(metric)
    return SnapOptimizationResult(
        params=params,
        history_fidelity=history_fidelity,
        history_epsilon_coh=history_epsilon,
        history_error_vector_norm=history_norm,
        history_max_component=history_maxc,
        history_params=history_params,
        history_error=history_epsilon.copy(),
        converged=threshold_hit,
        threshold_hit=threshold_hit,
        iters_to_threshold=iters_to_threshold,
        final_metric=metric,
        final_errors=final_errors,
    )


__all__ = ["SnapOptimizationResult", "optimize_snap_prl133"]
