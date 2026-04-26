from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import qutip as qt
from scipy import sparse


TransmonBackend = Literal["cosine", "duffing"]


@dataclass(frozen=True)
class TransmonCosineSpec:
    """Charge-basis transmon specification.

    All energies are angular frequencies, usually rad/s.  The offset charge
    ``ng`` is dimensionless and the charge basis is ordered from ``-n_cut`` to
    ``+n_cut``.
    """

    EJ: float
    EC: float
    ng: float = 0.0
    n_cut: int = 25
    levels: int = 4

    def __post_init__(self) -> None:
        if float(self.EJ) <= 0.0:
            raise ValueError("EJ must be positive.")
        if float(self.EC) <= 0.0:
            raise ValueError("EC must be positive.")
        if int(self.n_cut) < 1:
            raise ValueError("n_cut must be at least 1.")
        dim = 2 * int(self.n_cut) + 1
        if int(self.levels) < 1 or int(self.levels) > dim:
            raise ValueError("levels must be between 1 and the charge-basis dimension.")
        object.__setattr__(self, "EJ", float(self.EJ))
        object.__setattr__(self, "EC", float(self.EC))
        object.__setattr__(self, "ng", float(self.ng))
        object.__setattr__(self, "n_cut", int(self.n_cut))
        object.__setattr__(self, "levels", int(self.levels))

    @property
    def charge_numbers(self) -> np.ndarray:
        return np.arange(-self.n_cut, self.n_cut + 1, dtype=float)


@dataclass(frozen=True)
class DuffingTransmonSpec:
    """Lightweight multilevel transmon approximation.

    ``omega_01`` and ``alpha`` are angular frequencies.  The diagonal energy is
    ``E_m = m*omega_01 + alpha*m*(m-1)/2`` with ``E_0 = 0``.
    """

    omega_01: float
    alpha: float
    levels: int = 4

    def __post_init__(self) -> None:
        if int(self.levels) < 2:
            raise ValueError("levels must be at least 2.")
        object.__setattr__(self, "omega_01", float(self.omega_01))
        object.__setattr__(self, "alpha", float(self.alpha))
        object.__setattr__(self, "levels", int(self.levels))


@dataclass
class TransmonSpectrum:
    """Lowest transmon eigenstates and charge matrix elements."""

    energies: np.ndarray
    eigenstates: list[qt.Qobj]
    charge_numbers: np.ndarray
    charge_hamiltonian: qt.Qobj
    charge_operator: qt.Qobj
    n_matrix: np.ndarray
    backend: TransmonBackend = "cosine"

    @property
    def levels(self) -> int:
        return int(self.energies.size)

    @property
    def shifted_energies(self) -> np.ndarray:
        return np.asarray(self.energies - self.energies[0], dtype=float)

    def hamiltonian_eigenbasis(self, *, shift_ground: bool = True) -> qt.Qobj:
        energies = self.shifted_energies if shift_ground else np.asarray(self.energies, dtype=float)
        return qt.Qobj(np.diag(energies), dims=[[self.levels], [self.levels]])

    def charge_operator_eigenbasis(self) -> qt.Qobj:
        return qt.Qobj(np.asarray(self.n_matrix, dtype=np.complex128), dims=[[self.levels], [self.levels]])


def charge_basis_operators(n_cut: int) -> tuple[np.ndarray, qt.Qobj, qt.Qobj, qt.Qobj]:
    """Return charge numbers plus ``n``, ``cos(phi)``, and ``sin(phi)`` operators."""

    n_cut = int(n_cut)
    if n_cut < 1:
        raise ValueError("n_cut must be at least 1.")
    dim = 2 * n_cut + 1
    charges = np.arange(-n_cut, n_cut + 1, dtype=float)
    n_op = qt.Qobj(sparse.diags(charges, format="csr"), dims=[[dim], [dim]])
    shift_plus = sparse.diags(np.ones(dim - 1, dtype=np.complex128), offsets=1, format="csr")
    shift_minus = shift_plus.conjugate().transpose()
    cos_phi = qt.Qobj(0.5 * (shift_plus + shift_minus), dims=[[dim], [dim]])
    sin_phi = qt.Qobj((shift_plus - shift_minus) / (2.0j), dims=[[dim], [dim]])
    return charges, n_op, cos_phi, sin_phi


def transmon_charge_hamiltonian(spec: TransmonCosineSpec) -> qt.Qobj:
    """Build ``H = 4 EC (n-ng)^2 - EJ cos(phi)`` in the charge basis."""

    charges, _n_op, cos_phi, _sin_phi = charge_basis_operators(spec.n_cut)
    charging = 4.0 * spec.EC * (charges - spec.ng) ** 2
    dim = charges.size
    return qt.Qobj(
        sparse.diags(charging, format="csr") - spec.EJ * cos_phi.data.as_scipy(),
        dims=[[dim], [dim]],
    )


def _phase_fix_state(state: qt.Qobj) -> qt.Qobj:
    data = np.asarray(state.full()).reshape(-1)
    pivot = int(np.argmax(np.abs(data)))
    if abs(data[pivot]) <= 0.0:
        return state
    phase = np.exp(-1j * np.angle(data[pivot]))
    return phase * state


def diagonalize_transmon(spec: TransmonCosineSpec) -> TransmonSpectrum:
    """Diagonalize the cosine transmon and return the lowest requested levels.

    The returned energies are in the same angular-frequency units as ``EJ`` and
    ``EC``.  ``n_matrix`` is the physical Cooper-pair number operator ``n`` in
    the transmon eigenbasis, not ``n-ng``.
    """

    h = transmon_charge_hamiltonian(spec)
    charges, n_op, _cos_phi, _sin_phi = charge_basis_operators(spec.n_cut)
    energies_all, states_all = h.eigenstates()
    order = np.argsort(np.asarray(energies_all, dtype=float))[: spec.levels]
    energies = np.asarray(energies_all, dtype=float)[order]
    states = [_phase_fix_state(states_all[int(idx)]) for idx in order]
    n_matrix = np.empty((spec.levels, spec.levels), dtype=np.complex128)
    for row, left in enumerate(states):
        for col, right in enumerate(states):
            n_matrix[row, col] = complex(left.dag() * n_op * right)
    return TransmonSpectrum(
        energies=energies,
        eigenstates=states,
        charge_numbers=charges,
        charge_hamiltonian=h,
        charge_operator=n_op,
        n_matrix=n_matrix,
        backend="cosine",
    )


def duffing_transmon_spectrum(spec: DuffingTransmonSpec) -> TransmonSpectrum:
    """Return a Duffing-oscillator transmon approximation."""

    levels = int(spec.levels)
    indices = np.arange(levels, dtype=float)
    energies = spec.omega_01 * indices + 0.5 * spec.alpha * indices * (indices - 1.0)
    lowering = qt.destroy(levels)
    n_matrix = np.asarray(((lowering + lowering.dag()) / np.sqrt(2.0)).full(), dtype=np.complex128)
    h = qt.Qobj(np.diag(energies), dims=[[levels], [levels]])
    eigenstates = [qt.basis(levels, level) for level in range(levels)]
    return TransmonSpectrum(
        energies=np.asarray(energies, dtype=float),
        eigenstates=eigenstates,
        charge_numbers=np.arange(levels, dtype=float),
        charge_hamiltonian=h,
        charge_operator=qt.Qobj(n_matrix, dims=[[levels], [levels]]),
        n_matrix=n_matrix,
        backend="duffing",
    )


@dataclass(frozen=True)
class TransmonModel:
    """Convenience wrapper selecting either the cosine or Duffing backend."""

    backend: TransmonBackend
    cosine: TransmonCosineSpec | None = None
    duffing: DuffingTransmonSpec | None = None

    @classmethod
    def from_cosine(
        cls,
        *,
        EJ: float,
        EC: float,
        ng: float = 0.0,
        n_cut: int = 25,
        levels: int = 4,
    ) -> "TransmonModel":
        return cls("cosine", cosine=TransmonCosineSpec(EJ=EJ, EC=EC, ng=ng, n_cut=n_cut, levels=levels))

    @classmethod
    def from_duffing(
        cls,
        *,
        omega_01: float,
        alpha: float,
        levels: int = 4,
    ) -> "TransmonModel":
        return cls("duffing", duffing=DuffingTransmonSpec(omega_01=omega_01, alpha=alpha, levels=levels))

    def spectrum(self) -> TransmonSpectrum:
        if self.backend == "cosine":
            if self.cosine is None:
                raise ValueError("cosine backend requires a TransmonCosineSpec.")
            return diagonalize_transmon(self.cosine)
        if self.backend == "duffing":
            if self.duffing is None:
                raise ValueError("duffing backend requires a DuffingTransmonSpec.")
            return duffing_transmon_spectrum(self.duffing)
        raise ValueError(f"Unsupported transmon backend {self.backend!r}.")


def transmon_convergence_sweep(
    specs: Sequence[TransmonCosineSpec],
) -> list[TransmonSpectrum]:
    """Diagonalize a sequence of truncations for charge-basis convergence checks."""

    return [diagonalize_transmon(spec) for spec in specs]


__all__ = [
    "DuffingTransmonSpec",
    "TransmonBackend",
    "TransmonCosineSpec",
    "TransmonModel",
    "TransmonSpectrum",
    "charge_basis_operators",
    "diagonalize_transmon",
    "duffing_transmon_spectrum",
    "transmon_charge_hamiltonian",
    "transmon_convergence_sweep",
]
