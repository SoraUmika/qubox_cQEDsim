from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.operators.cavity import fock_projector


def _as_dm(state: qt.Qobj) -> qt.Qobj:
    return state if state.isoper else state.proj()


def _subsystem_dims(rho: qt.Qobj) -> tuple[int, ...]:
    left_dims = tuple(int(dim) for dim in rho.dims[0])
    right_dims = tuple(int(dim) for dim in rho.dims[1])
    if left_dims != right_dims:
        raise ValueError(f"Expected a square state operator, got dims={rho.dims}.")
    if len(left_dims) < 2:
        raise ValueError(f"Expected a multipartite state, got dims={rho.dims}.")
    return left_dims


def _resolve_subsystem_index(rho: qt.Qobj, subsystem: int | str) -> int:
    dims = _subsystem_dims(rho)
    if isinstance(subsystem, int):
        if subsystem < 0 or subsystem >= len(dims):
            raise IndexError(f"Subsystem index {subsystem} out of range for dims={dims}.")
        return int(subsystem)

    aliases = {
        "qubit": 0,
        "transmon": 0,
        "cavity": 1,
        "storage": 1,
        "readout": 2,
    }
    if subsystem not in aliases:
        raise ValueError(f"Unsupported subsystem label '{subsystem}'.")
    idx = aliases[subsystem]
    if idx >= len(dims):
        raise ValueError(f"Subsystem label '{subsystem}' is not valid for dims={dims}.")
    return idx


def reduced_subsystem_state(state: qt.Qobj, subsystem: int | str) -> qt.Qobj:
    rho = _as_dm(state)
    return qt.ptrace(rho, _resolve_subsystem_index(rho, subsystem))


def reduced_qubit_state(state: qt.Qobj) -> qt.Qobj:
    return reduced_subsystem_state(state, 0)


def reduced_transmon_state(state: qt.Qobj) -> qt.Qobj:
    return reduced_subsystem_state(state, 0)


def reduced_cavity_state(state: qt.Qobj) -> qt.Qobj:
    rho = _as_dm(state)
    dims = _subsystem_dims(rho)
    if len(dims) != 2:
        raise ValueError("reduced_cavity_state is only defined for the two-mode qubit-storage path.")
    return qt.ptrace(rho, 1)


def reduced_storage_state(state: qt.Qobj) -> qt.Qobj:
    return reduced_subsystem_state(state, "storage")


def reduced_readout_state(state: qt.Qobj) -> qt.Qobj:
    return reduced_subsystem_state(state, "readout")


def bloch_xyz_from_qubit_state(rho_q: qt.Qobj) -> tuple[float, float, float]:
    return (
        float(np.real((rho_q * qt.sigmax()).tr())),
        float(np.real((rho_q * qt.sigmay()).tr())),
        float(np.real((rho_q * qt.sigmaz()).tr())),
    )


def qubit_density_from_bloch_xyz(x: float, y: float, z: float) -> qt.Qobj:
    return 0.5 * (qt.qeye(2) + float(x) * qt.sigmax() + float(y) * qt.sigmay() + float(z) * qt.sigmaz())


def bloch_xyz_from_joint(state: qt.Qobj) -> tuple[float, float, float]:
    return bloch_xyz_from_qubit_state(reduced_qubit_state(state))


def conditioned_population(state: qt.Qobj, n: int) -> float:
    rho = _as_dm(state)
    dims = _subsystem_dims(rho)
    if len(dims) != 2:
        raise ValueError("conditioned_population is only defined for the two-mode qubit-storage path.")
    n_qubit, n_cav = dims
    if n < 0 or n >= n_cav:
        raise IndexError("Conditioning index n out of range.")
    proj_n = qt.tensor(qt.qeye(n_qubit), fock_projector(n_cav, n))
    return float(np.real((proj_n * rho).tr()))


def conditioned_qubit_state(state: qt.Qobj, n: int, fallback: str = "nan") -> tuple[qt.Qobj, float, bool]:
    rho = _as_dm(state)
    dims = _subsystem_dims(rho)
    if len(dims) != 2:
        raise ValueError("conditioned_qubit_state is only defined for the two-mode qubit-storage path.")
    n_qubit, n_cav = dims
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


def mode_moments(state: qt.Qobj, subsystem: int | str, dim: int | None = None) -> dict[str, complex]:
    rho_mode = reduced_subsystem_state(state, subsystem)
    n_mode = int(rho_mode.dims[0][0] if dim is None else dim)
    a = qt.destroy(n_mode)
    adag = a.dag()
    return {
        "a": complex((rho_mode * a).tr()),
        "adag_a": complex((rho_mode * adag * a).tr()),
        "n": float(np.real((rho_mode * adag * a).tr())),
    }


def cavity_moments(state: qt.Qobj, n_cav: int | None = None) -> dict[str, complex]:
    return mode_moments(state, "storage", dim=n_cav)


def storage_moments(state: qt.Qobj, n_storage: int | None = None) -> dict[str, complex]:
    return mode_moments(state, "storage", dim=n_storage)


def readout_moments(state: qt.Qobj, n_readout: int | None = None) -> dict[str, complex]:
    return mode_moments(state, "readout", dim=n_readout)


def storage_photon_number(state: qt.Qobj) -> float:
    return float(np.real(storage_moments(state)["n"]))


def readout_photon_number(state: qt.Qobj) -> float:
    return float(np.real(readout_moments(state)["n"]))


def joint_expectation(state: qt.Qobj, operator: qt.Qobj) -> complex:
    rho = _as_dm(state)
    return complex((rho * operator).tr())


def qubit_conditioned_subsystem_state(
    state: qt.Qobj,
    subsystem: int | str,
    qubit_level: int,
    fallback: str = "nan",
) -> tuple[qt.Qobj, float, bool]:
    rho = _as_dm(state)
    dims = _subsystem_dims(rho)
    qubit_level = int(qubit_level)
    if qubit_level < 0 or qubit_level >= dims[0]:
        raise IndexError("qubit_level out of range.")

    factors = [qt.basis(dims[0], qubit_level) * qt.basis(dims[0], qubit_level).dag()]
    factors.extend(qt.qeye(dim) for dim in dims[1:])
    proj_q = qt.tensor(*factors)
    block = proj_q * rho * proj_q
    subsystem_index = _resolve_subsystem_index(rho, subsystem)
    rho_sub_tilde = qt.ptrace(block, subsystem_index)
    p_q = float(np.real(rho_sub_tilde.tr()))
    if p_q > 1.0e-15:
        return rho_sub_tilde / p_q, p_q, True
    dim_sub = int(dims[subsystem_index])
    if fallback == "zero":
        return 0 * qt.qeye(dim_sub), 0.0, False
    if fallback == "nan":
        nan_dm = qt.Qobj(np.full((dim_sub, dim_sub), np.nan, dtype=np.complex128), dims=[[dim_sub], [dim_sub]])
        return nan_dm, 0.0, False
    raise ValueError(f"Unsupported fallback '{fallback}'.")


def qubit_conditioned_mode_moments(
    state: qt.Qobj,
    subsystem: int | str,
    qubit_level: int,
) -> dict[str, complex | float | bool]:
    rho_sub, prob, valid = qubit_conditioned_subsystem_state(state, subsystem, qubit_level, fallback="zero")
    n_mode = int(rho_sub.dims[0][0])
    a = qt.destroy(n_mode)
    adag = a.dag()
    return {
        "probability": prob,
        "valid": valid,
        "a": complex((rho_sub * a).tr()),
        "adag_a": complex((rho_sub * adag * a).tr()),
        "n": float(np.real((rho_sub * adag * a).tr())),
    }


def readout_response_by_qubit_state(state: qt.Qobj) -> dict[int, dict[str, complex | float | bool]]:
    return {
        0: qubit_conditioned_mode_moments(state, "readout", 0),
        1: qubit_conditioned_mode_moments(state, "readout", 1),
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
