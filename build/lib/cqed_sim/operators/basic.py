from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core.ideal_gates import embed_cavity_op as _embed_cavity_op
from cqed_sim.core.ideal_gates import embed_qubit_op as _embed_qubit_op


def sigma_x() -> qt.Qobj:
    return qt.sigmax()


def sigma_y() -> qt.Qobj:
    return qt.sigmay()


def sigma_z() -> qt.Qobj:
    return qt.sigmaz()


def tensor_qubit_cavity(op_q: qt.Qobj, op_c: qt.Qobj) -> qt.Qobj:
    return qt.tensor(op_q, op_c)


def embed_qubit_op(op_q: qt.Qobj, n_cav: int) -> qt.Qobj:
    return _embed_qubit_op(op_q, n_cav)


def embed_cavity_op(op_c: qt.Qobj, n_tr: int = 2) -> qt.Qobj:
    return _embed_cavity_op(op_c, n_tr=n_tr)


def as_dm(state: qt.Qobj) -> qt.Qobj:
    return state if state.isoper else state.proj()


def purity(state: qt.Qobj) -> float:
    rho = as_dm(state)
    return float(np.real((rho * rho).tr()))


def build_qubit_state(label: str) -> qt.Qobj:
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    lookup = {
        "g": g,
        "e": e,
        "+x": (g + e).unit(),
        "-x": (g - e).unit(),
        "+y": (g + 1j * e).unit(),
        "-y": (g - 1j * e).unit(),
    }
    if label not in lookup:
        raise ValueError(f"Unsupported qubit label '{label}'.")
    return lookup[label]


def joint_basis_state(n_cav_dim: int, qubit_label: str, n: int) -> qt.Qobj:
    qubit_index = 0 if qubit_label == "g" else 1
    return qt.tensor(qt.basis(2, qubit_index), qt.basis(n_cav_dim, n))
