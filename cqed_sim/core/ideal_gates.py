from __future__ import annotations

import numpy as np
import qutip as qt


def _resolve_block_phase_levels(
    phases: np.ndarray | list[float],
    *,
    fock_levels: np.ndarray | list[int] | tuple[int, ...] | None = None,
    cavity_dim: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...], int]:
    phase_array = np.asarray(phases, dtype=float).reshape(-1)
    if fock_levels is None:
        dim = int(phase_array.size) if cavity_dim is None else int(cavity_dim)
        if dim <= 0:
            raise ValueError("cavity_dim must be positive.")
        if phase_array.size > dim:
            raise ValueError("Number of phases cannot exceed cavity_dim.")
        levels = tuple(range(int(phase_array.size)))
        return phase_array, levels, dim

    levels = tuple(int(level) for level in fock_levels)
    if len(levels) != int(phase_array.size):
        raise ValueError("phases and fock_levels must have the same length.")
    if len(set(levels)) != len(levels):
        raise ValueError("fock_levels must not contain duplicates.")
    if cavity_dim is None:
        dim = 0 if not levels else int(max(levels) + 1)
    else:
        dim = int(cavity_dim)
    if dim <= 0:
        raise ValueError("cavity_dim must be positive.")
    for level in levels:
        if level < 0 or level >= dim:
            raise ValueError(f"Fock level {level} is outside the truncated cavity dimension {dim}.")
    return phase_array, levels, dim


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


def cavity_block_phase_op(
    phases: np.ndarray | list[float],
    *,
    fock_levels: np.ndarray | list[int] | tuple[int, ...] | None = None,
    cavity_dim: int | None = None,
) -> qt.Qobj:
    phase_array, levels, dim = _resolve_block_phase_levels(phases, fock_levels=fock_levels, cavity_dim=cavity_dim)
    diag = np.ones(dim, dtype=np.complex128)
    for phase, level in zip(phase_array, levels, strict=True):
        diag[int(level)] = np.exp(1j * float(phase))
    return qt.Qobj(np.diag(diag), dims=[[dim], [dim]])


def logical_block_phase_op(
    phases: np.ndarray | list[float],
    *,
    fock_levels: np.ndarray | list[int] | tuple[int, ...] | None = None,
    cavity_dim: int | None = None,
    qubit_dim: int = 2,
) -> qt.Qobj:
    if int(qubit_dim) <= 0:
        raise ValueError("qubit_dim must be positive.")
    return qt.tensor(
        qt.qeye(int(qubit_dim)),
        cavity_block_phase_op(phases, fock_levels=fock_levels, cavity_dim=cavity_dim),
    )


def snap_op(phases: np.ndarray | list[float]) -> qt.Qobj:
    return cavity_block_phase_op(phases)


def sqr_op(thetas: np.ndarray | list[float], phis: np.ndarray | list[float]) -> qt.Qobj:
    thetas = np.asarray(thetas, dtype=float)
    phis = np.asarray(phis, dtype=float)
    if thetas.shape != phis.shape:
        raise ValueError("thetas and phis must have the same shape.")
    n_cav = thetas.size
    out = 0 * qt.tensor(qt.qeye(2), qt.qeye(n_cav))
    for n in range(n_cav):
        pn = qt.basis(n_cav, n) * qt.basis(n_cav, n).dag()
        out += qt.tensor(qubit_rotation_xy(float(thetas[n]), float(phis[n])), pn)
    return out


def embed_qubit_op(op_q: qt.Qobj, n_cav: int) -> qt.Qobj:
    return qt.tensor(op_q, qt.qeye(n_cav))


def embed_cavity_op(op_c: qt.Qobj, n_tr: int = 2) -> qt.Qobj:
    return qt.tensor(qt.qeye(n_tr), op_c)


def beamsplitter_unitary(n_a: int, n_b: int, theta: float) -> qt.Qobj:
    """Ideal two-mode beam-splitter unitary exp[-i theta (a b^† + a^† b)]."""
    a = qt.tensor(qt.destroy(n_a), qt.qeye(n_b))
    b = qt.tensor(qt.qeye(n_a), qt.destroy(n_b))
    h = a * b.dag() + a.dag() * b
    return (-1j * theta * h).expm()
