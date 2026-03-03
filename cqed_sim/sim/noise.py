from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel


@dataclass(frozen=True)
class NoiseSpec:
    """Lindblad noise parameters in SI-style units.

    Times are in seconds and rates are in 1/seconds (rad/s compatible for
    our dimensionless Hamiltonian convention used across tests/examples).
    """

    t1: float | None = None
    tphi: float | None = None
    kappa: float | None = None
    nth: float = 0.0

    @property
    def gamma1(self) -> float:
        return 0.0 if self.t1 is None else 1.0 / self.t1

    @property
    def gamma_phi(self) -> float:
        # Convention requested in spec: gamma_phi = 1 / (2*Tphi) for collapse op scaling.
        return 0.0 if self.tphi is None else 1.0 / (2.0 * self.tphi)


def collapse_operators(model: DispersiveTransmonCavityModel, noise: NoiseSpec | None) -> list[qt.Qobj]:
    if noise is None:
        return []
    ops = model.operators()
    c_ops: list[qt.Qobj] = []

    if noise.gamma1 > 0.0:
        # Multi-level transmon relaxation via lowering operator.
        c_ops.append(np.sqrt(noise.gamma1) * ops["b"])

    if noise.gamma_phi > 0.0:
        if model.n_tr == 2:
            i_tot = qt.tensor(qt.qeye(model.n_cav), qt.qeye(model.n_tr))
            sigma_z = i_tot - 2.0 * ops["n_q"]  # |g><g| - |e><e|
            c_ops.append(np.sqrt(noise.gamma_phi) * sigma_z)
        else:
            # Ladder dephasing approximation for multi-level transmon.
            c_ops.append(np.sqrt(noise.gamma_phi) * ops["n_q"])

    if noise.kappa is not None and noise.kappa > 0.0:
        nth = max(0.0, float(noise.nth))
        c_ops.append(np.sqrt(noise.kappa * (nth + 1.0)) * ops["a"])
        if nth > 0.0:
            c_ops.append(np.sqrt(noise.kappa * nth) * ops["adag"])

    return c_ops

