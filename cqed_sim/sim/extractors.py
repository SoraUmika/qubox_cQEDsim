from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.operators.cavity import fock_projector


def _as_dm(state: qt.Qobj) -> qt.Qobj:
    return state if state.isoper else state.proj()


def _joint_dims(rho: qt.Qobj) -> tuple[int, int]:
    if len(rho.dims[0]) != 2 or len(rho.dims[1]) != 2:
        raise ValueError(f"Expected a qubit-cavity bipartite state, got dims={rho.dims}.")
    n_qubit = int(rho.dims[0][0])
    n_cav = int(rho.dims[0][1])
    return n_qubit, n_cav


def reduced_qubit_state(state: qt.Qobj) -> qt.Qobj:
    return qt.ptrace(_as_dm(state), 0)


def reduced_cavity_state(state: qt.Qobj) -> qt.Qobj:
    return qt.ptrace(_as_dm(state), 1)


def bloch_xyz_from_qubit_state(rho_q: qt.Qobj) -> tuple[float, float, float]:
    """Return Bloch coordinates in the qubox convention.

    The Y component follows qubox storage-tomography outputs and is defined as
    2 * Im(rho_ge), which is the negative of Tr(rho sigma_y) in the standard
    physics convention.
    """
    return (
        float(np.real((rho_q * qt.sigmax()).tr())),
        2.0 * float(np.imag(rho_q[0, 1])),
        float(np.real((rho_q * qt.sigmaz()).tr())),
    )


def qubit_density_from_bloch_xyz(x: float, y: float, z: float) -> qt.Qobj:
    """Build a qubit density matrix from Bloch coordinates in the qubox convention."""
    return 0.5 * (qt.qeye(2) + float(x) * qt.sigmax() - float(y) * qt.sigmay() + float(z) * qt.sigmaz())


def bloch_xyz_from_joint(state: qt.Qobj) -> tuple[float, float, float]:
    rho_q = reduced_qubit_state(state)
    return bloch_xyz_from_qubit_state(rho_q)


def conditioned_population(state: qt.Qobj, n: int) -> float:
    rho = _as_dm(state)
    n_qubit, n_cav = _joint_dims(rho)
    if n < 0 or n >= n_cav:
        raise IndexError("Conditioning index n out of range.")
    proj_n = qt.tensor(qt.qeye(n_qubit), fock_projector(n_cav, n))
    return float(np.real((proj_n * rho).tr()))


def conditioned_qubit_state(state: qt.Qobj, n: int, fallback: str = "nan") -> tuple[qt.Qobj, float, bool]:
    rho = _as_dm(state)
    n_qubit, n_cav = _joint_dims(rho)
    if n < 0 or n >= n_cav:
        raise IndexError("Conditioning index n out of range.")
    proj_n = qt.tensor(qt.qeye(n_qubit), fock_projector(n_cav, n))
    block = proj_n * rho * proj_n
    rho_q_tilde = qt.ptrace(block, 0)
    p_n = float(np.real(rho_q_tilde.tr()))
    if p_n > 1.0e-15:
        return rho_q_tilde / p_n, p_n, True
    if fallback == "zero":
        return 0 * qt.qeye(n_qubit), 0.0, False
    if fallback == "nan":
        nan_dm = qt.Qobj(np.full((n_qubit, n_qubit), np.nan, dtype=np.complex128), dims=[[n_qubit], [n_qubit]])
        return nan_dm, 0.0, False
    raise ValueError(f"Unsupported fallback '{fallback}'.")


def conditioned_bloch_xyz(state: qt.Qobj, n: int, fallback: str = "nan") -> tuple[float, float, float, float, bool]:
    rho_q, p_n, valid = conditioned_qubit_state(state, n=n, fallback=fallback)
    if not valid and fallback == "nan":
        return (np.nan, np.nan, np.nan, p_n, valid)
    x, y, z = bloch_xyz_from_qubit_state(rho_q)
    return (x, y, z, p_n, valid)


def cavity_moments(state: qt.Qobj, n_cav: int | None = None) -> dict[str, complex]:
    rho_c = reduced_cavity_state(state)
    n_c = n_cav if n_cav is not None else rho_c.dims[0][0]
    a = qt.destroy(n_c)
    adag = a.dag()
    return {
        "a": complex((rho_c * a).tr()),
        "adag_a": complex((rho_c * adag * a).tr()),
        "n": float(np.real((rho_c * adag * a).tr())),
    }


def cavity_wigner(
    rho_c: qt.Qobj,
    xvec: np.ndarray | None = None,
    yvec: np.ndarray | None = None,
    n_points: int = 41,
    extent: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xvec = np.linspace(-extent, extent, n_points) if xvec is None else np.asarray(xvec, dtype=float)
    yvec = np.linspace(-extent, extent, n_points) if yvec is None else np.asarray(yvec, dtype=float)
    w = qt.wigner(rho_c, xvec, yvec)
    return xvec, yvec, np.asarray(w, dtype=float)
