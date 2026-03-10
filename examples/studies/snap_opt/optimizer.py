from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from .experiments import SnapRunConfig, run_snap_stage
from .metrics import ManifoldErrors
from .pulses import SnapToneParameters


@dataclass
class SnapOptimizationResult:
    params: SnapToneParameters
    history_error: list[float]
    history_params: list[tuple[np.ndarray, np.ndarray, np.ndarray]]
    converged: bool
    final_errors: ManifoldErrors


def _apply_update(params: SnapToneParameters, err: ManifoldErrors, eta: float) -> SnapToneParameters:
    # Paper-inspired first-order corrections:
    # phase -> phi, longitudinal -> amplitude, transversal -> detuning.
    new_a = params.amplitudes - eta * err.dlambda
    new_d = params.detunings - eta * err.dalpha * np.sign(err.dtheta + 1e-12)
    new_p = params.phases - eta * err.dtheta
    new_a = np.clip(new_a, 0.2, 3.0)
    return SnapToneParameters(amplitudes=new_a, detunings=new_d, phases=new_p)


def _evaluate_errors(
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
    cfg: SnapRunConfig,
    params: SnapToneParameters,
    frame: FrameSpec,
) -> ManifoldErrors:
    n_max = target_phases.size - 1
    dtheta = np.zeros(n_max + 1, dtype=float)
    dlambda = np.zeros(n_max + 1, dtype=float)
    dalpha = np.zeros(n_max + 1, dtype=float)
    for n in range(n_max + 1):
        psi0 = model.basis_state( 0,n)
        out, _, _ = run_snap_stage(model, target_phases, params, cfg, psi0, frame=frame)
        cg = model.basis_state( 0,n).overlap(out)
        ce = model.basis_state( 1,n).overlap(out)
        dtheta[n] = np.angle(cg * np.exp(-1j * target_phases[n]))
        dlambda[n] = np.abs(cg) - 1.0
        dalpha[n] = np.abs(ce)
    mean_err = float(np.sqrt(np.mean(dtheta**2 + dlambda**2 + dalpha**2)))
    return ManifoldErrors(dtheta=dtheta, dlambda=dlambda, dalpha=dalpha, mean_overlap_error=mean_err)


def optimize_snap_parameters(
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
    cfg: SnapRunConfig,
    initial_params: SnapToneParameters | None = None,
    frame: FrameSpec | None = None,
    max_iter: int = 60,
    learning_rate: float = 0.25,
    threshold: float = 1e-4,
) -> SnapOptimizationResult:
    frame = frame or FrameSpec(omega_q_frame=model.omega_q)
    params = initial_params or SnapToneParameters.vanilla(target_phases)
    history_error: list[float] = []
    history_params: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    errs = _evaluate_errors(model, target_phases, cfg, params, frame)
    history_error.append(errs.mean_overlap_error)
    history_params.append((params.amplitudes.copy(), params.detunings.copy(), params.phases.copy()))

    converged = errs.mean_overlap_error <= threshold
    for _ in range(max_iter):
        if converged:
            break
        eta = learning_rate
        improved = False
        best_candidate = params
        best_err = errs
        for _try in range(8):
            cand = _apply_update(params, errs, eta=eta)
            err_c = _evaluate_errors(model, target_phases, cfg, cand, frame)
            if err_c.mean_overlap_error <= errs.mean_overlap_error:
                improved = True
                best_candidate = cand
                best_err = err_c
                break
            eta *= 0.5
        if not improved:
            break
        params = best_candidate
        errs = best_err
        history_error.append(errs.mean_overlap_error)
        history_params.append((params.amplitudes.copy(), params.detunings.copy(), params.phases.copy()))
        if errs.mean_overlap_error <= threshold:
            converged = True
            break

    return SnapOptimizationResult(
        params=params,
        history_error=history_error,
        history_params=history_params,
        converged=converged,
        final_errors=errs,
    )
