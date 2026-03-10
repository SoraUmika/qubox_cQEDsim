from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from cqed_sim.core.frequencies import falling_factorial_scalar, manifold_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel


@dataclass(frozen=True)
class SnapModelConfig:
    """SNAP optimization model wrapper.

    References:
    - J. Landgraf et al., arXiv:2310.10498 / PRL 133, 260802 (2024).
    """

    n_cav: int = 8
    n_tr: int = 2
    chi: float = 2 * np.pi * 0.02
    alpha: float = 0.0
    kerr: float = 0.0
    chi2: float = 0.0
    chi3: float = 0.0
    omega_q: float = 0.0
    omega_c: float = 0.0

    def build_model(self) -> DispersiveTransmonCavityModel:
        higher = tuple(x for x in (self.chi2, self.chi3) if x != 0.0)
        return DispersiveTransmonCavityModel(
            omega_c=self.omega_c,
            omega_q=self.omega_q,
            alpha=self.alpha,
            chi=self.chi,
            chi_higher=higher,
            kerr=self.kerr,
            n_cav=self.n_cav,
            n_tr=self.n_tr,
        )


def manifold_basis(model: DispersiveTransmonCavityModel, n: int) -> tuple[qt.Qobj, qt.Qobj]:
    return model.basis_state( 0,n), model.basis_state( 1,n)

