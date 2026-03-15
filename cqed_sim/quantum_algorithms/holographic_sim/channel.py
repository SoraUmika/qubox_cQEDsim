from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import (
    as_complex_array,
    basis_vector,
    coerce_density_matrix,
    ensure_positive_int,
    ensure_square_matrix,
    json_ready,
    validate_unitary,
)


def _kraus_completeness_error(kraus_ops: Sequence[np.ndarray]) -> float:
    if not kraus_ops:
        return float("inf")
    bond_dim = kraus_ops[0].shape[0]
    completeness = np.zeros((bond_dim, bond_dim), dtype=np.complex128)
    for op in kraus_ops:
        completeness += op.conj().T @ op
    ident = np.eye(bond_dim, dtype=np.complex128)
    return float(np.linalg.norm(completeness - ident, ord="fro"))


@dataclass
class HolographicChannel:
    """A single holographic transfer step on the bond Hilbert space.

    The stored Kraus operators follow the standard quantum-channel convention
    `rho -> sum_k K_k rho K_k^dagger`.

    When constructed from right-canonical MPS matrices `V_sigma`, the mapping is
    `K_sigma = V_sigma^dagger`, matching the paper's transfer-channel
    orientation while keeping the implementation in standard quantum-information
    channel form.
    """

    physical_dim: int
    bond_dim: int
    kraus_ops: Sequence[np.ndarray]
    label: str | None = None
    joint_unitary: np.ndarray | None = None
    reference_state: np.ndarray | None = None
    mps_matrices: Sequence[np.ndarray] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.physical_dim = ensure_positive_int(self.physical_dim, name="physical_dim")
        self.bond_dim = ensure_positive_int(self.bond_dim, name="bond_dim")
        normalized = tuple(ensure_square_matrix(op, name="kraus operator") for op in self.kraus_ops)
        if len(normalized) == 0:
            raise ValueError("HolographicChannel requires at least one Kraus operator.")
        for op in normalized:
            if op.shape != (self.bond_dim, self.bond_dim):
                raise ValueError(
                    f"Kraus operators must all have shape {(self.bond_dim, self.bond_dim)}, got {op.shape}."
                )
        if len(normalized) > self.physical_dim:
            raise ValueError("The number of Kraus operators cannot exceed the physical dimension.")
        if len(normalized) < self.physical_dim:
            zeros = tuple(np.zeros((self.bond_dim, self.bond_dim), dtype=np.complex128) for _ in range(self.physical_dim - len(normalized)))
            normalized = normalized + zeros
        self.kraus_ops = normalized
        if self.reference_state is None:
            self.reference_state = basis_vector(self.physical_dim, 0)
        else:
            ref = np.asarray(as_complex_array(self.reference_state), dtype=np.complex128).reshape(-1)
            if ref.size != self.physical_dim:
                raise ValueError(f"reference_state has dimension {ref.size}, expected {self.physical_dim}.")
            norm = float(np.linalg.norm(ref))
            if norm <= 1.0e-12:
                raise ValueError("reference_state norm is zero.")
            self.reference_state = ref / norm
        if self.joint_unitary is not None:
            self.joint_unitary = validate_unitary(self.joint_unitary)
            dim = self.physical_dim * self.bond_dim
            if self.joint_unitary.shape != (dim, dim):
                raise ValueError(f"joint_unitary must have shape {(dim, dim)}, got {self.joint_unitary.shape}.")
        if self.mps_matrices is not None:
            matrices = tuple(ensure_square_matrix(op, name="mps matrix") for op in self.mps_matrices)
            if len(matrices) != self.physical_dim:
                raise ValueError("mps_matrices must have one square matrix per physical basis outcome.")
            for op in matrices:
                if op.shape != (self.bond_dim, self.bond_dim):
                    raise ValueError("All mps_matrices must have shape (bond_dim, bond_dim).")
            self.mps_matrices = matrices
        else:
            self.mps_matrices = tuple(op.conj().T for op in self.kraus_ops)

    @classmethod
    def from_unitary(
        cls,
        unitary: Any,
        *,
        physical_dim: int,
        bond_dim: int | None = None,
        reference_state: Any | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "HolographicChannel":
        physical_dim = ensure_positive_int(physical_dim, name="physical_dim")
        unitary_arr = validate_unitary(unitary)
        full_dim = int(unitary_arr.shape[0])
        if bond_dim is None:
            if full_dim % physical_dim != 0:
                raise ValueError(f"Unitary dimension {full_dim} is not divisible by physical_dim={physical_dim}.")
            bond_dim = full_dim // physical_dim
        bond_dim = ensure_positive_int(bond_dim, name="bond_dim")
        if full_dim != physical_dim * bond_dim:
            raise ValueError(
                f"Unitary dimension {full_dim} does not match physical_dim * bond_dim = {physical_dim * bond_dim}."
            )
        ref = basis_vector(physical_dim, 0) if reference_state is None else np.asarray(as_complex_array(reference_state), dtype=np.complex128).reshape(-1)
        if ref.size != physical_dim:
            raise ValueError(f"reference_state has dimension {ref.size}, expected {physical_dim}.")
        ref = ref / np.linalg.norm(ref)
        blocks = unitary_arr.reshape(physical_dim, bond_dim, physical_dim, bond_dim)
        kraus = [np.tensordot(blocks[outcome], ref, axes=([1], [0])) for outcome in range(physical_dim)]
        return cls(
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            kraus_ops=tuple(kraus),
            label=label,
            joint_unitary=unitary_arr,
            reference_state=ref,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_kraus(
        cls,
        kraus_ops: Sequence[Any],
        *,
        physical_dim: int | None = None,
        bond_dim: int | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "HolographicChannel":
        if not kraus_ops:
            raise ValueError("from_kraus requires at least one Kraus operator.")
        normalized = tuple(ensure_square_matrix(op, name="kraus operator") for op in kraus_ops)
        inferred_bond_dim = int(normalized[0].shape[0])
        for op in normalized:
            if op.shape != (inferred_bond_dim, inferred_bond_dim):
                raise ValueError("All Kraus operators must have the same square shape.")
        resolved_bond_dim = inferred_bond_dim if bond_dim is None else ensure_positive_int(bond_dim, name="bond_dim")
        if resolved_bond_dim != inferred_bond_dim:
            raise ValueError("bond_dim does not match the Kraus operator dimensions.")
        resolved_physical_dim = len(normalized) if physical_dim is None else ensure_positive_int(physical_dim, name="physical_dim")
        return cls(
            physical_dim=resolved_physical_dim,
            bond_dim=resolved_bond_dim,
            kraus_ops=normalized,
            label=label,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_right_canonical_mps(
        cls,
        tensor: Any,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "HolographicChannel":
        arr = as_complex_array(tensor)
        if arr.ndim == 3:
            bond_left, physical_dim, bond_right = arr.shape
            if bond_left != bond_right:
                raise ValueError("Right-canonical MPS tensor must be square on the bond legs for direct channel construction.")
            matrices = tuple(arr[:, idx, :] for idx in range(physical_dim))
        elif arr.ndim == 2:
            raise ValueError("from_right_canonical_mps expects a rank-3 tensor or a sequence of matrices, not one matrix.")
        else:
            matrices = tuple(ensure_square_matrix(op, name="mps matrix") for op in tensor)
            if not matrices:
                raise ValueError("from_right_canonical_mps requires at least one MPS matrix.")
            physical_dim = len(matrices)
            bond_left = matrices[0].shape[0]
            bond_right = matrices[0].shape[1]
            if bond_left != bond_right:
                raise ValueError("Right-canonical MPS matrices must be square on the bond Hilbert space.")
        kraus = tuple(matrix.conj().T for matrix in matrices)
        return cls(
            physical_dim=int(physical_dim),
            bond_dim=int(bond_left),
            kraus_ops=kraus,
            label=label,
            mps_matrices=matrices,
            metadata=dict(metadata or {}),
        )

    @property
    def num_physical_qubits(self) -> int:
        return int(np.ceil(np.log2(self.physical_dim)))

    @property
    def num_bond_qubits(self) -> int:
        return int(np.ceil(np.log2(self.bond_dim)))

    def apply(self, bond_state: Any) -> np.ndarray:
        rho = coerce_density_matrix(bond_state, dim=self.bond_dim)
        out = np.zeros((self.bond_dim, self.bond_dim), dtype=np.complex128)
        for op in self.kraus_ops:
            out += op @ rho @ op.conj().T
        return 0.5 * (out + out.conj().T)

    def joint_output_state(self, bond_state: Any) -> np.ndarray:
        rho = coerce_density_matrix(bond_state, dim=self.bond_dim)
        if self.joint_unitary is not None:
            ref_dm = np.outer(self.reference_state, self.reference_state.conj())
            rho_in = np.kron(ref_dm, rho)
            out = self.joint_unitary @ rho_in @ self.joint_unitary.conj().T
            return 0.5 * (out + out.conj().T)

        dim = self.physical_dim * self.bond_dim
        joint = np.zeros((dim, dim), dtype=np.complex128)
        for i, op_i in enumerate(self.kraus_ops):
            for j, op_j in enumerate(self.kraus_ops):
                block = op_i @ rho @ op_j.conj().T
                row = slice(i * self.bond_dim, (i + 1) * self.bond_dim)
                col = slice(j * self.bond_dim, (j + 1) * self.bond_dim)
                joint[row, col] = block
        return 0.5 * (joint + joint.conj().T)

    def kraus_completeness_error(self) -> float:
        return _kraus_completeness_error(self.kraus_ops)

    def right_canonical_error(self) -> float:
        matrices = self.mps_matrices if self.mps_matrices is not None else tuple(op.conj().T for op in self.kraus_ops)
        total = np.zeros((self.bond_dim, self.bond_dim), dtype=np.complex128)
        for matrix in matrices:
            total += matrix @ matrix.conj().T
        ident = np.eye(self.bond_dim, dtype=np.complex128)
        return float(np.linalg.norm(total - ident, ord="fro"))

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "label": self.label,
                "physical_dim": int(self.physical_dim),
                "bond_dim": int(self.bond_dim),
                "num_physical_qubits": int(self.num_physical_qubits),
                "num_bond_qubits": int(self.num_bond_qubits),
                "kraus_ops": tuple(np.asarray(op, dtype=np.complex128) for op in self.kraus_ops),
                "reference_state": np.asarray(self.reference_state, dtype=np.complex128),
                "has_joint_unitary": self.joint_unitary is not None,
                "metadata": dict(self.metadata),
            }
        )
