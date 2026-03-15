from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.linalg import null_space, svd

from .channel import HolographicChannel
from .utils import as_complex_array


def check_state_normalized(state: Any, *, atol: float = 1.0e-12) -> np.ndarray:
    arr = np.asarray(as_complex_array(state), dtype=np.complex128)
    norm_sq = float(np.sum(np.abs(arr) ** 2))
    if not np.isclose(norm_sq, 1.0, atol=atol):
        raise ValueError(f"State is not normalized: norm^2 = {norm_sq:.12f}")
    return arr


def contract_mps(mps_tensors: Sequence[np.ndarray]) -> np.ndarray:
    if not mps_tensors:
        raise ValueError("contract_mps requires at least one tensor.")
    chi = int(mps_tensors[0].shape[0])
    left = np.zeros(chi, dtype=np.complex128)
    right = np.zeros(chi, dtype=np.complex128)
    left[0] = 1.0
    right[0] = 1.0
    psi = np.tensordot(left, mps_tensors[0], axes=(0, 0))
    for tensor in mps_tensors[1:]:
        psi = np.tensordot(psi, tensor, axes=([-1], [0]))
    return np.tensordot(psi, right, axes=([-1], [0]))


def complete_right_isometry(tensor: Any, chi: int) -> np.ndarray:
    arr = np.asarray(as_complex_array(tensor), dtype=np.complex128)
    if arr.ndim != 3:
        raise ValueError("complete_right_isometry expects a rank-3 MPS tensor.")
    chi_left, physical_dim, chi_right = arr.shape
    if chi < max(chi_left, chi_right):
        raise ValueError("chi must be at least as large as the tensor bond dimensions.")
    isometry_check = np.zeros((chi_left, chi_left), dtype=np.complex128)
    for idx in range(physical_dim):
        block = arr[:, idx, :]
        isometry_check += block @ block.conj().T
    if not np.allclose(isometry_check, np.eye(chi_left), atol=1.0e-12):
        raise ValueError("Tensor does not satisfy the right-isometry condition.")

    in_embed = np.vstack((np.eye(chi_right), np.zeros((chi - chi_right, chi_right), dtype=np.complex128)))
    out_embed = np.vstack((np.eye(chi_left), np.zeros((chi - chi_left, chi_left), dtype=np.complex128)))
    padded = np.zeros((chi, physical_dim, chi), dtype=np.complex128)
    for idx in range(physical_dim):
        padded[:, idx, :] = out_embed @ arr[:, idx, :] @ in_embed.conj().T

    isometry = padded.reshape(chi, chi * physical_dim).conj().T
    rank = np.linalg.matrix_rank(isometry)
    null_basis = null_space(isometry.conj().T)
    if null_basis.shape[1] < chi - rank:
        raise ValueError("Could not complete the right-isometry to the requested bond dimension.")
    isometry[:, rank:chi] = null_basis[:, : chi - rank]
    if not np.allclose(isometry.conj().T @ isometry, np.eye(chi), atol=1.0e-12):
        raise ValueError("Completed tensor failed the right-isometry check.")
    return isometry.conj().T.reshape((chi, physical_dim, chi))


@dataclass
class MatrixProductState:
    state: np.ndarray
    tensors: list[np.ndarray] | None = None
    uniform_tensors: list[np.ndarray] | None = None
    bond_dimension: int | None = None

    def __post_init__(self) -> None:
        self.state = check_state_normalized(self.state)

    @property
    def num_sites(self) -> int:
        return int(self.state.ndim)

    @property
    def physical_dim(self) -> int:
        return int(self.state.shape[0])

    @property
    def psi(self) -> np.ndarray:
        return self.state

    @property
    def mps(self) -> list[np.ndarray] | None:
        return self.tensors

    @property
    def mps_uniform(self) -> list[np.ndarray] | None:
        return self.uniform_tensors

    @property
    def chi(self) -> int | None:
        return self.bond_dimension

    def _update_bond_dimension(self) -> None:
        if not self.tensors:
            raise ValueError("No MPS tensors are available.")
        bond_dims: list[int] = []
        for idx, tensor in enumerate(self.tensors):
            if tensor.ndim != 3:
                raise ValueError(f"Tensor at site {idx} is not rank-3.")
            bond_dims.extend([int(tensor.shape[0]), int(tensor.shape[2])])
        self.bond_dimension = max(bond_dims)

    def make_right_canonical(self, *, cast_complete: bool = False, chi_max: int | None = None) -> None:
        psi = self.state
        num_sites = self.num_sites
        physical_dim = self.physical_dim
        psi_2d = psi.reshape(physical_dim**num_sites, 1)
        tensors: list[np.ndarray] = [None] * num_sites  # type: ignore[list-item]
        bond_dim_right = 1
        for site in reversed(range(1, num_sites)):
            psi_2d = psi_2d.reshape(-1, physical_dim * bond_dim_right)
            u, s, vh = svd(psi_2d, full_matrices=False)
            chi = len(s) if chi_max is None else min(len(s), int(chi_max))
            u = u[:, :chi]
            s = s[:chi]
            vh = vh[:chi, :]
            tensors[site] = vh.reshape(chi, physical_dim, bond_dim_right)
            psi_2d = u @ np.diag(s)
            bond_dim_right = chi
        first = psi_2d.reshape(1, physical_dim, bond_dim_right)
        first /= np.linalg.norm(first)
        tensors[0] = first
        self.tensors = list(tensors)
        self._update_bond_dimension()
        if cast_complete:
            assert self.bond_dimension is not None
            self.uniform_tensors = [complete_right_isometry(tensor, self.bond_dimension) for tensor in self.tensors]

    def make_rcf(self, cast_complete: bool = False, chi_max: int | None = None) -> None:
        self.make_right_canonical(cast_complete=cast_complete, chi_max=chi_max)

    def make_left_canonical(self, *, chi_max: int | None = None) -> None:
        psi = self.state
        num_sites = self.num_sites
        physical_dim = self.physical_dim
        psi_2d = psi.reshape(1, physical_dim**num_sites)
        tensors: list[np.ndarray] = []
        bond_dim_left = 1
        for site in range(num_sites - 1):
            psi_2d = psi_2d.reshape(bond_dim_left * physical_dim, -1)
            u, s, vh = svd(psi_2d, full_matrices=False)
            chi = len(s) if chi_max is None else min(len(s), int(chi_max))
            u = u[:, :chi]
            s = s[:chi]
            vh = vh[:chi, :]
            tensors.append(u.reshape(bond_dim_left, physical_dim, chi))
            bond_dim_left = chi
            psi_2d = np.diag(s) @ vh
        tensors.append(psi_2d.reshape(bond_dim_left, physical_dim, 1))
        self.tensors = tensors
        self.uniform_tensors = None
        self._update_bond_dimension()

    def make_lcf(self, chi_max: int | None = None) -> None:
        self.make_left_canonical(chi_max=chi_max)

    def expect_operator_product(self, operators: Iterable[tuple[int, Any]]) -> complex:
        if self.tensors is None:
            raise ValueError("MPS tensors have not been constructed yet.")
        operators_by_site: dict[int, np.ndarray] = {}
        for site, operator in operators:
            if not (0 <= int(site) < self.num_sites):
                raise ValueError(f"Site index {site} is outside [0, {self.num_sites - 1}].")
            op = np.asarray(as_complex_array(operator), dtype=np.complex128)
            if op.shape != (self.physical_dim, self.physical_dim):
                raise ValueError(
                    f"Operator at site {site} must have shape {(self.physical_dim, self.physical_dim)}, got {op.shape}."
                )
            operators_by_site[int(site)] = operators_by_site.get(int(site), np.eye(self.physical_dim, dtype=np.complex128)) @ op

        env = np.eye(self.tensors[0].shape[0], dtype=np.complex128)
        identity = np.eye(self.physical_dim, dtype=np.complex128)
        for site, tensor in enumerate(self.tensors):
            op = operators_by_site.get(site, identity)
            env = np.einsum("ab,asx,bty,st->xy", env, tensor, tensor.conjugate(), op)
        return complex(env.item())

    def expect_O(self, operator: Any) -> np.ndarray:
        return np.asarray(
            [self.expect_operator_product([(site, operator)]) for site in range(self.num_sites)],
            dtype=np.complex128,
        )

    def expect_OO(self, operator: Any) -> np.ndarray:
        corr = np.zeros((self.num_sites, self.num_sites), dtype=np.complex128)
        for i in range(self.num_sites):
            for j in range(self.num_sites):
                corr[i, j] = self.expect_operator_product([(i, operator), (j, operator)])
        return corr

    def site_tensor(self, site: int, *, complete: bool = False) -> np.ndarray:
        if complete:
            if self.uniform_tensors is None:
                self.make_right_canonical(cast_complete=True)
            assert self.uniform_tensors is not None
            return self.uniform_tensors[int(site)]
        if self.tensors is None:
            self.make_right_canonical()
        assert self.tensors is not None
        return self.tensors[int(site)]

    def to_holographic_channel(
        self,
        *,
        site: int = 0,
        complete: bool = True,
        label: str | None = None,
    ) -> HolographicChannel:
        tensor = self.site_tensor(int(site), complete=complete)
        if tensor.shape[0] != tensor.shape[2]:
            chi = max(int(tensor.shape[0]), int(tensor.shape[2]))
            tensor = complete_right_isometry(tensor, chi)
        return HolographicChannel.from_right_canonical_mps(
            tensor,
            label=label if label is not None else f"mps_site_{int(site)}",
            metadata={"source": "MatrixProductState", "site": int(site)},
        )

    def fidelity(self, other: "MatrixProductState") -> float:
        other_state = np.asarray(as_complex_array(other.state), dtype=np.complex128)
        overlap = np.vdot(self.state.reshape(-1), other_state.reshape(-1))
        return float(abs(overlap) ** 2)
