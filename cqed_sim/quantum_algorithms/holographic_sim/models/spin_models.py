from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm


PAULI_I = np.eye(2, dtype=np.complex128)
PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def _kron2(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.kron(np.asarray(left, dtype=np.complex128), np.asarray(right, dtype=np.complex128))


@dataclass(frozen=True)
class IsingTransferSpec:
    """Minimal spin-inspired transfer model for a physical qubit and bond qubit."""

    zz_coupling: float
    hx_physical: float = 0.0
    hx_bond: float = 0.0
    hz_physical: float = 0.0
    hz_bond: float = 0.0
    time: float = 1.0

    def hamiltonian(self) -> np.ndarray:
        return (
            float(self.zz_coupling) * _kron2(PAULI_Z, PAULI_Z)
            + float(self.hx_physical) * _kron2(PAULI_X, PAULI_I)
            + float(self.hx_bond) * _kron2(PAULI_I, PAULI_X)
            + float(self.hz_physical) * _kron2(PAULI_Z, PAULI_I)
            + float(self.hz_bond) * _kron2(PAULI_I, PAULI_Z)
        )

    def unitary(self) -> np.ndarray:
        return expm(-1j * float(self.time) * self.hamiltonian())


def transverse_field_ising_transfer_unitary(
    *,
    zz_coupling: float,
    hx_physical: float = 0.0,
    hx_bond: float = 0.0,
    hz_physical: float = 0.0,
    hz_bond: float = 0.0,
    time: float = 1.0,
) -> np.ndarray:
    """Dense transfer unitary for a qubit physical register and qubit bond register."""

    return IsingTransferSpec(
        zz_coupling=zz_coupling,
        hx_physical=hx_physical,
        hx_bond=hx_bond,
        hz_physical=hz_physical,
        hz_bond=hz_bond,
        time=time,
    ).unitary()
