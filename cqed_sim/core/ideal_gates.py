from __future__ import annotations

import numpy as np
import qutip as qt


def qubit_rotation_xy(theta: float, phi: float) -> qt.Qobj:
    sx = qt.sigmax()
    sy = qt.sigmay()
    identity = qt.qeye(2)
    axis = np.cos(phi) * sx + np.sin(phi) * sy
    c = float(np.cos(theta / 2.0))
    s = float(np.sin(theta / 2.0))
    return c * identity - 1j * s * axis


def qubit_rotation_axis(theta: float, axis: str) -> qt.Qobj:
    if axis == "x":
        return qubit_rotation_xy(theta, 0.0)
    if axis == "y":
        return qubit_rotation_xy(theta, np.pi / 2.0)
    if axis == "z":
        identity = qt.qeye(2)
        c = float(np.cos(theta / 2.0))
        s = float(np.sin(theta / 2.0))
        return c * identity - 1j * s * qt.sigmaz()
    raise ValueError(f"Unsupported axis '{axis}'.")


def displacement_op(n_cav: int, alpha: complex) -> qt.Qobj:
    return qt.displace(n_cav, alpha)


def snap_op(phases: np.ndarray | list[float]) -> qt.Qobj:
    phases = np.asarray(phases, dtype=float)
    diag = np.exp(1j * phases)
    return qt.Qobj(np.diag(diag), dims=[[phases.size], [phases.size]])


def sqr_op(thetas: np.ndarray | list[float], phis: np.ndarray | list[float]) -> qt.Qobj:
    thetas = np.asarray(thetas, dtype=float)
    phis = np.asarray(phis, dtype=float)
    if thetas.shape != phis.shape:
        raise ValueError("thetas and phis must have the same shape.")
    n_cav = thetas.size
    out = 0 * qt.tensor(qt.qeye(n_cav), qt.qeye(2))
    for n in range(n_cav):
        pn = qt.basis(n_cav, n) * qt.basis(n_cav, n).dag()
        out += qt.tensor(pn, qubit_rotation_xy(float(thetas[n]), float(phis[n])))
    return out


def embed_qubit_op(op_q: qt.Qobj, n_cav: int) -> qt.Qobj:
    return qt.tensor(qt.qeye(n_cav), op_q)


def embed_cavity_op(op_c: qt.Qobj, n_tr: int = 2) -> qt.Qobj:
    return qt.tensor(op_c, qt.qeye(n_tr))


def beamsplitter_unitary(n_a: int, n_b: int, theta: float) -> qt.Qobj:
    """Ideal two-mode beam-splitter unitary exp[-i theta (a b^† + a^† b)]."""
    a = qt.tensor(qt.destroy(n_a), qt.qeye(n_b))
    b = qt.tensor(qt.qeye(n_a), qt.destroy(n_b))
    h = a * b.dag() + a.dag() * b
    return (-1j * theta * h).expm()
