from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.snap_opt.experiments import SnapRunConfig, run_snap_stage
from cqed_sim.snap_opt.pulses import SnapToneParameters


@dataclass
class ManifoldErrors:
    dtheta: np.ndarray
    dlambda: np.ndarray
    dalpha: np.ndarray
    mean_overlap_error: float


@dataclass
class CoherentMetricResult:
    fidelity: float
    epsilon_coh: float
    per_manifold_overlap: np.ndarray
    dtheta: np.ndarray
    eps_l: np.ndarray
    eps_t: np.ndarray
    error_vector_norm: float
    max_component_error: float
    excited_amplitudes: np.ndarray
    ground_amplitudes: np.ndarray


def _validate_metric_bounds(metric: CoherentMetricResult, context: str, tol: float = 1e-9) -> None:
    if metric.fidelity < -tol or metric.fidelity > 1.0 + tol:
        raise ValueError(f"[{context}] fidelity out of bounds: {metric.fidelity}")
    if metric.epsilon_coh < -tol or metric.epsilon_coh > 1.0 + tol:
        raise ValueError(f"[{context}] epsilon_coh out of bounds: {metric.epsilon_coh}")
    lo = float(np.min(metric.per_manifold_overlap))
    hi = float(np.max(metric.per_manifold_overlap))
    if lo < -tol or hi > 1.0 + tol:
        raise ValueError(f"[{context}] per-manifold overlap out of bounds: min={lo}, max={hi}")


def compute_mean_squared_overlap(
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
    cfg: SnapRunConfig,
    params: SnapToneParameters,
    frame: FrameSpec | None = None,
    *,
    validate_bounds: bool = True,
    context: str = "snap_prl133",
) -> CoherentMetricResult:
    """Compute paper-style coherent metric for ge-protocol slow stage.

    Supplement fidelity definition:
      F = avg_{||c||=1} <psi_target(c)|rho_out(c)|psi_target(c)>
    evaluated analytically from manifold amplitudes under number-conserving dynamics.
    """
    frame = frame or FrameSpec(omega_q_frame=model.omega_q)
    target_phases = np.asarray(target_phases, dtype=float)
    n_max = target_phases.size - 1
    a = np.zeros(n_max + 1, dtype=np.complex128)
    b = np.zeros(n_max + 1, dtype=np.complex128)
    dtheta_raw = np.zeros(n_max + 1, dtype=float)
    dtheta = np.zeros(n_max + 1, dtype=float)
    eps_l = np.zeros(n_max + 1, dtype=float)
    eps_t = np.zeros(n_max + 1, dtype=float)
    per_overlap = np.zeros(n_max + 1, dtype=float)

    for n in range(n_max + 1):
        psi0 = model.basis_state( 0,n)
        out, _, _ = run_snap_stage(model, target_phases, params, cfg, psi0, frame=frame)
        a_n = model.basis_state( 1,n).overlap(out)
        b_n = model.basis_state( 0,n).overlap(out)
        a[n] = a_n
        b[n] = b_n
        dtheta_raw[n] = float(np.angle(a_n * np.exp(-1j * target_phases[n])))
        per_overlap[n] = float(np.abs(a_n) ** 2)

    # Remove global phase offset (not physically relevant for SNAP action).
    phase_ref = dtheta_raw[0] if dtheta_raw.size > 0 else 0.0
    dtheta = np.angle(np.exp(1j * (dtheta_raw - phase_ref)))
    for n in range(n_max + 1):
        eps_complex_rot = -2.0 * b[n] * np.exp(-1j * dtheta[n])
        eps_l[n] = float(np.real(eps_complex_rot))
        eps_t[n] = float(np.imag(eps_complex_rot))

    s = a * np.exp(-1j * target_phases)
    l = float(n_max + 1)
    weights = (np.ones((n_max + 1, n_max + 1), dtype=float) + np.eye(n_max + 1, dtype=float)) / (l * (l + 1.0))
    fidelity = float(np.real(np.sum(weights * np.outer(s, np.conj(s)))))
    fidelity = float(np.clip(fidelity, 0.0, 1.0))
    metric = CoherentMetricResult(
        fidelity=fidelity,
        epsilon_coh=float(1.0 - fidelity),
        per_manifold_overlap=np.clip(per_overlap, 0.0, 1.0),
        dtheta=dtheta,
        eps_l=eps_l,
        eps_t=eps_t,
        error_vector_norm=float(np.sqrt(np.mean(dtheta**2 + eps_l**2 + eps_t**2))),
        max_component_error=float(np.max(np.stack([np.abs(dtheta), np.abs(eps_l), np.abs(eps_t)]))),
        excited_amplitudes=a,
        ground_amplitudes=b,
    )
    if validate_bounds:
        _validate_metric_bounds(metric, context=context)
    return metric


def evaluate_manifold_errors(
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
    cfg: SnapRunConfig,
    params: SnapToneParameters,
    frame: FrameSpec | None = None,
) -> ManifoldErrors:
    metric = compute_mean_squared_overlap(
        model=model,
        target_phases=target_phases,
        cfg=cfg,
        params=params,
        frame=frame,
    )
    return ManifoldErrors(
        dtheta=metric.dtheta.copy(),
        dlambda=metric.eps_l.copy(),
        dalpha=metric.eps_t.copy(),
        mean_overlap_error=float(metric.error_vector_norm),
    )


__all__ = ["ManifoldErrors", "CoherentMetricResult", "compute_mean_squared_overlap", "evaluate_manifold_errors"]
