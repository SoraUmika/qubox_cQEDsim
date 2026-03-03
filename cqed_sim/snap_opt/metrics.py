from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.snap_opt.model import manifold_basis


@dataclass
class ManifoldErrors:
    dtheta: np.ndarray
    dlambda: np.ndarray
    dalpha: np.ndarray
    mean_overlap_error: float


def extract_manifold_amplitudes(state: qt.Qobj, model: DispersiveTransmonCavityModel, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    if state.isoper:
        raise ValueError("Expected pure state for coherent-error extraction.")
    c_g = np.zeros(n_max + 1, dtype=np.complex128)
    c_e = np.zeros(n_max + 1, dtype=np.complex128)
    for n in range(n_max + 1):
        g, e = manifold_basis(model, n)
        c_g[n] = g.overlap(state)
        c_e[n] = e.overlap(state)
    return c_g, c_e


def coherent_errors_from_state(
    state: qt.Qobj,
    model: DispersiveTransmonCavityModel,
    target_phases: np.ndarray,
) -> ManifoldErrors:
    n_max = target_phases.size - 1
    c_g, c_e = extract_manifold_amplitudes(state, model, n_max=n_max)
    target = np.exp(1j * target_phases)
    # Phase mismatch on the intended ground component.
    dtheta = np.angle(c_g * np.conj(target))
    # Longitudinal mismatch in target component amplitude.
    dlambda = np.abs(c_g) - 1.0
    # Transversal residual population in excited branch.
    dalpha = np.abs(c_e)
    mean_err = float(np.sqrt(np.mean(dtheta**2 + dlambda**2 + dalpha**2)))
    return ManifoldErrors(dtheta=dtheta, dlambda=dlambda, dalpha=dalpha, mean_overlap_error=mean_err)


def gate_infidelity_like(state: qt.Qobj, model: DispersiveTransmonCavityModel, target_phases: np.ndarray) -> float:
    errs = coherent_errors_from_state(state, model, target_phases)
    return errs.mean_overlap_error

