from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from .utils import coerce_density_matrix, ensure_positive_int, ensure_square_matrix, json_ready


def _kraus_completeness_error(kraus_ops: Sequence[np.ndarray]) -> float:
    if not kraus_ops:
        return float("inf")
    bond_dim = kraus_ops[0].shape[0]
    completeness = np.zeros((bond_dim, bond_dim), dtype=np.complex128)
    for op in kraus_ops:
        completeness += op.conj().T @ op
    ident = np.eye(bond_dim, dtype=np.complex128)
    return float(np.linalg.norm(completeness - ident, ord="fro"))


def _import_qutip() -> Any:
    try:
        import qutip as qt
    except Exception as exc:  # pragma: no cover - exercised only when QuTiP is unavailable.
        raise ImportError("BondNoiseChannel QuTiP conversions require the qutip package.") from exc
    return qt


def _weyl_unitaries(dim: int) -> tuple[np.ndarray, ...]:
    shift = np.zeros((dim, dim), dtype=np.complex128)
    for basis_index in range(dim):
        shift[(basis_index + 1) % dim, basis_index] = 1.0
    phase = np.diag(np.exp(2.0j * np.pi * np.arange(dim) / float(dim)).astype(np.complex128))
    shift_powers = tuple(np.linalg.matrix_power(shift, power) for power in range(dim))
    phase_powers = tuple(np.linalg.matrix_power(phase, power) for power in range(dim))
    return tuple(shift_power @ phase_power for shift_power in shift_powers for phase_power in phase_powers)


@dataclass
class BondNoiseChannel:
    """A CPTP bond-space noise map applied between holographic steps."""

    bond_dim: int
    kraus_ops: Sequence[np.ndarray]
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.bond_dim = ensure_positive_int(self.bond_dim, name="bond_dim")
        normalized = tuple(ensure_square_matrix(op, name="bond-noise Kraus operator") for op in self.kraus_ops)
        if not normalized:
            raise ValueError("BondNoiseChannel requires at least one Kraus operator.")
        for op in normalized:
            if op.shape != (self.bond_dim, self.bond_dim):
                raise ValueError(
                    f"Bond-noise Kraus operators must all have shape {(self.bond_dim, self.bond_dim)}, got {op.shape}."
                )
        self.kraus_ops = normalized

    @classmethod
    def from_kraus(
        cls,
        kraus_ops: Sequence[Any],
        *,
        bond_dim: int | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BondNoiseChannel":
        if not kraus_ops:
            raise ValueError("from_kraus requires at least one Kraus operator.")
        normalized = tuple(ensure_square_matrix(op, name="bond-noise Kraus operator") for op in kraus_ops)
        inferred_bond_dim = int(normalized[0].shape[0])
        for op in normalized:
            if op.shape != (inferred_bond_dim, inferred_bond_dim):
                raise ValueError("All bond-noise Kraus operators must have the same square shape.")
        resolved_bond_dim = inferred_bond_dim if bond_dim is None else ensure_positive_int(bond_dim, name="bond_dim")
        if resolved_bond_dim != inferred_bond_dim:
            raise ValueError("bond_dim does not match the bond-noise Kraus operator dimensions.")
        return cls(
            bond_dim=resolved_bond_dim,
            kraus_ops=normalized,
            label=label,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_qutip_super(
        cls,
        superoperator: Any,
        *,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BondNoiseChannel":
        qt = _import_qutip()
        kraus_ops = tuple(qt.to_kraus(superoperator))
        resolved_metadata = {"source": "qutip_superoperator"}
        if metadata is not None:
            resolved_metadata.update(dict(metadata))
        return cls.from_kraus(kraus_ops, label=label, metadata=resolved_metadata)

    @classmethod
    def dephasing(
        cls,
        *,
        bond_dim: int,
        probability: float | None = None,
        coherence_scale: float | None = None,
        duration: float | None = None,
        tphi: float | None = None,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BondNoiseChannel":
        resolved_bond_dim = ensure_positive_int(bond_dim, name="bond_dim")
        specification_count = int(probability is not None) + int(coherence_scale is not None) + int(duration is not None or tphi is not None)
        if specification_count != 1:
            raise ValueError(
                "Specify exactly one dephasing parameterization: probability, coherence_scale, or duration together with tphi."
            )
        if duration is not None or tphi is not None:
            if duration is None or tphi is None:
                raise ValueError("duration and tphi must be provided together.")
            resolved_duration = float(duration)
            resolved_tphi = float(tphi)
            if resolved_duration < 0.0:
                raise ValueError("duration must be non-negative.")
            if resolved_tphi <= 0.0:
                raise ValueError("tphi must be positive.")
            coherence_scale = float(np.exp(-resolved_duration / resolved_tphi))
        if coherence_scale is not None:
            resolved_coherence_scale = float(coherence_scale)
            if not 0.0 <= resolved_coherence_scale <= 1.0:
                raise ValueError("coherence_scale must lie in [0, 1].")
            resolved_probability = 1.0 - resolved_coherence_scale
        else:
            assert probability is not None
            resolved_probability = float(probability)
            if not 0.0 <= resolved_probability <= 1.0:
                raise ValueError("probability must lie in [0, 1].")
            resolved_coherence_scale = 1.0 - resolved_probability

        identity = np.eye(resolved_bond_dim, dtype=np.complex128)
        kraus_ops = [np.sqrt(max(0.0, 1.0 - resolved_probability)) * identity]
        if resolved_probability > 0.0:
            projector_weight = np.sqrt(resolved_probability)
            for basis_index in range(resolved_bond_dim):
                projector = np.zeros((resolved_bond_dim, resolved_bond_dim), dtype=np.complex128)
                projector[basis_index, basis_index] = projector_weight
                kraus_ops.append(projector)

        resolved_metadata = {
            "noise_model": "bond_dephasing",
            "basis": "bond_computational",
            "probability": float(resolved_probability),
            "coherence_scale": float(resolved_coherence_scale),
        }
        if duration is not None and tphi is not None:
            resolved_metadata["duration"] = float(duration)
            resolved_metadata["tphi"] = float(tphi)
        if metadata is not None:
            resolved_metadata.update(dict(metadata))
        return cls(
            bond_dim=resolved_bond_dim,
            kraus_ops=tuple(kraus_ops),
            label=label if label is not None else "bond_dephasing",
            metadata=resolved_metadata,
        )

    @classmethod
    def amplitude_damping(
        cls,
        *,
        bond_dim: int,
        probability: float | None = None,
        duration: float | None = None,
        t1: float | None = None,
        target_index: int = 0,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BondNoiseChannel":
        resolved_bond_dim = ensure_positive_int(bond_dim, name="bond_dim")
        if not 0 <= int(target_index) < resolved_bond_dim:
            raise ValueError(f"target_index must lie in [0, {resolved_bond_dim - 1}].")

        specification_count = int(probability is not None) + int(duration is not None or t1 is not None)
        if specification_count != 1:
            raise ValueError("Specify exactly one amplitude-damping parameterization: probability or duration together with t1.")
        if duration is not None or t1 is not None:
            if duration is None or t1 is None:
                raise ValueError("duration and t1 must be provided together.")
            resolved_duration = float(duration)
            resolved_t1 = float(t1)
            if resolved_duration < 0.0:
                raise ValueError("duration must be non-negative.")
            if resolved_t1 <= 0.0:
                raise ValueError("t1 must be positive.")
            resolved_probability = float(1.0 - np.exp(-resolved_duration / resolved_t1))
        else:
            assert probability is not None
            resolved_probability = float(probability)
            if not 0.0 <= resolved_probability <= 1.0:
                raise ValueError("probability must lie in [0, 1].")

        keep_scale = np.sqrt(max(0.0, 1.0 - resolved_probability))
        diagonal = np.full(resolved_bond_dim, keep_scale, dtype=np.complex128)
        diagonal[int(target_index)] = 1.0
        kraus_ops = [np.diag(diagonal)]
        if resolved_probability > 0.0:
            jump_scale = np.sqrt(resolved_probability)
            for source_index in range(resolved_bond_dim):
                if source_index == int(target_index):
                    continue
                jump = np.zeros((resolved_bond_dim, resolved_bond_dim), dtype=np.complex128)
                jump[int(target_index), source_index] = jump_scale
                kraus_ops.append(jump)

        resolved_metadata = {
            "noise_model": "bond_amplitude_damping",
            "basis": "bond_computational",
            "probability": float(resolved_probability),
            "target_index": int(target_index),
            "generalization": "direct_relaxation_to_target_basis_state",
        }
        if duration is not None and t1 is not None:
            resolved_metadata["duration"] = float(duration)
            resolved_metadata["t1"] = float(t1)
        if metadata is not None:
            resolved_metadata.update(dict(metadata))
        return cls(
            bond_dim=resolved_bond_dim,
            kraus_ops=tuple(kraus_ops),
            label=label if label is not None else "bond_amplitude_damping",
            metadata=resolved_metadata,
        )

    @classmethod
    def depolarizing(
        cls,
        *,
        bond_dim: int,
        probability: float,
        label: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BondNoiseChannel":
        resolved_bond_dim = ensure_positive_int(bond_dim, name="bond_dim")
        resolved_probability = float(probability)
        if not 0.0 <= resolved_probability <= 1.0:
            raise ValueError("probability must lie in [0, 1].")

        kraus_ops: list[np.ndarray] = []
        if resolved_probability < 1.0:
            kraus_ops.append(np.sqrt(1.0 - resolved_probability) * np.eye(resolved_bond_dim, dtype=np.complex128))
        if resolved_probability > 0.0:
            twirl_scale = np.sqrt(resolved_probability) / float(resolved_bond_dim)
            kraus_ops.extend(twirl_scale * unitary for unitary in _weyl_unitaries(resolved_bond_dim))

        resolved_metadata = {
            "noise_model": "bond_depolarizing",
            "probability": float(resolved_probability),
            "fixed_point": "maximally_mixed",
            "operator_basis": "weyl",
        }
        if metadata is not None:
            resolved_metadata.update(dict(metadata))
        return cls(
            bond_dim=resolved_bond_dim,
            kraus_ops=tuple(kraus_ops),
            label=label if label is not None else "bond_depolarizing",
            metadata=resolved_metadata,
        )

    def apply(self, bond_state: Any) -> np.ndarray:
        rho = coerce_density_matrix(bond_state, dim=self.bond_dim)
        out = np.zeros((self.bond_dim, self.bond_dim), dtype=np.complex128)
        for op in self.kraus_ops:
            out += op @ rho @ op.conj().T
        return 0.5 * (out + out.conj().T)

    def kraus_completeness_error(self) -> float:
        return _kraus_completeness_error(self.kraus_ops)

    def to_qutip_super(self) -> Any:
        qt = _import_qutip()
        kraus_list = [qt.Qobj(op, dims=[[self.bond_dim], [self.bond_dim]]) for op in self.kraus_ops]
        return qt.kraus_to_super(kraus_list)

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "label": self.label,
                "bond_dim": int(self.bond_dim),
                "kraus_ops": tuple(np.asarray(op, dtype=np.complex128) for op in self.kraus_ops),
                "metadata": dict(self.metadata),
            }
        )