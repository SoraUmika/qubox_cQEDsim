from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.operators.cavity import fock_projector


def _as_dm(state: qt.Qobj) -> qt.Qobj:
    return state if state.isoper else state.proj()


def _joint_dims(rho: qt.Qobj) -> tuple[int, int]:
    if len(rho.dims[0]) != 2 or len(rho.dims[1]) != 2:
        raise ValueError(f"Expected a cavity-qubit bipartite state, got dims={rho.dims}.")
    n_cav = int(rho.dims[0][0])
    n_qubit = int(rho.dims[0][1])
    return n_cav, n_qubit


def reduced_qubit_state(state: qt.Qobj) -> qt.Qobj:
    return qt.ptrace(_as_dm(state), 1)


def reduced_cavity_state(state: qt.Qobj) -> qt.Qobj:
    return qt.ptrace(_as_dm(state), 0)


def bloch_xyz_from_joint(state: qt.Qobj) -> tuple[float, float, float]:
    rho_q = reduced_qubit_state(state)
    return (
        float(np.real((rho_q * qt.sigmax()).tr())),
        float(np.real((rho_q * qt.sigmay()).tr())),
        float(np.real((rho_q * qt.sigmaz()).tr())),
    )


def conditioned_population(state: qt.Qobj, n: int) -> float:
    rho = _as_dm(state)
    n_cav, n_qubit = _joint_dims(rho)
    if n < 0 or n >= n_cav:
        raise IndexError("Conditioning index n out of range.")
    proj_n = qt.tensor(fock_projector(n_cav, n), qt.qeye(n_qubit))
    return float(np.real((proj_n * rho).tr()))


def conditioned_qubit_state(state: qt.Qobj, n: int, fallback: str = "nan") -> tuple[qt.Qobj, float, bool]:
    rho = _as_dm(state)
    n_cav, n_qubit = _joint_dims(rho)
    if n < 0 or n >= n_cav:
        raise IndexError("Conditioning index n out of range.")
    proj_n = qt.tensor(fock_projector(n_cav, n), qt.qeye(n_qubit))
    block = proj_n * rho * proj_n
    rho_q_tilde = qt.ptrace(block, 1)
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
    return (
        float(np.real((rho_q * qt.sigmax()).tr())),
        float(np.real((rho_q * qt.sigmay()).tr())),
        float(np.real((rho_q * qt.sigmaz()).tr())),
        p_n,
        valid,
    )


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

