from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .utils import ensure_positive_int, json_ready, validate_unitary


_ALLOWED_UNITARY_SPACES = {"joint", "physical", "bond"}


def embed_step_unitary(
    unitary: Any,
    *,
    acts_on: str,
    physical_dim: int,
    bond_dim: int,
) -> np.ndarray:
    resolved_acts_on = str(acts_on).strip().lower()
    if resolved_acts_on not in _ALLOWED_UNITARY_SPACES:
        allowed = ", ".join(sorted(_ALLOWED_UNITARY_SPACES))
        raise ValueError(f"acts_on must be one of {{{allowed}}}, got {acts_on!r}.")

    resolved_physical_dim = ensure_positive_int(physical_dim, name="physical_dim")
    resolved_bond_dim = ensure_positive_int(bond_dim, name="bond_dim")
    unitary_arr = validate_unitary(unitary)
    if resolved_acts_on == "joint":
        expected = resolved_physical_dim * resolved_bond_dim
        if unitary_arr.shape != (expected, expected):
            raise ValueError(
                f"Joint step unitary must have shape {(expected, expected)}, got {unitary_arr.shape}."
            )
        return unitary_arr

    if resolved_acts_on == "physical":
        if unitary_arr.shape != (resolved_physical_dim, resolved_physical_dim):
            raise ValueError(
                "Physical-only step unitary must have shape "
                f"{(resolved_physical_dim, resolved_physical_dim)}, got {unitary_arr.shape}."
            )
        return np.kron(unitary_arr, np.eye(resolved_bond_dim, dtype=np.complex128))

    if unitary_arr.shape != (resolved_bond_dim, resolved_bond_dim):
        raise ValueError(
            f"Bond-only step unitary must have shape {(resolved_bond_dim, resolved_bond_dim)}, got {unitary_arr.shape}."
        )
    return np.kron(np.eye(resolved_physical_dim, dtype=np.complex128), unitary_arr)


@dataclass(frozen=True)
class StepUnitarySpec:
    """One holographic step unitary with explicit subsystem semantics.

    The supported embedding convention is always `physical ⊗ bond`, meaning:

    - `acts_on="joint"`: the provided unitary already acts on the full
      `physical ⊗ bond` Hilbert space.
    - `acts_on="physical"`: the provided unitary is embedded as
      `U_physical ⊗ I_bond`.
    - `acts_on="bond"`: the provided unitary is embedded as
      `I_physical ⊗ U_bond`.
    """

    unitary: Any
    acts_on: str = "joint"
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unitary", validate_unitary(self.unitary))
        resolved_acts_on = str(self.acts_on).strip().lower()
        if resolved_acts_on not in _ALLOWED_UNITARY_SPACES:
            allowed = ", ".join(sorted(_ALLOWED_UNITARY_SPACES))
            raise ValueError(f"acts_on must be one of {{{allowed}}}, got {self.acts_on!r}.")
        object.__setattr__(self, "acts_on", resolved_acts_on)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def resolve_joint_unitary(self, *, physical_dim: int, bond_dim: int) -> np.ndarray:
        return embed_step_unitary(
            self.unitary,
            acts_on=self.acts_on,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
        )

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "label": self.label,
                "acts_on": self.acts_on,
                "unitary_shape": list(np.asarray(self.unitary).shape),
                "metadata": dict(self.metadata),
            }
        )