from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import qutip as qt

from .frame import FrameSpec
from .universal_model import UniversalCQEDModel

_TRANSMON_LEVEL_LABELS = ("g", "e", "f", "h", "i", "j", "k", "l", "m", "n")


@dataclass(frozen=True)
class EnergyLevel:
    index: int
    energy: float
    raw_energy: float
    eigenstate: qt.Qobj
    dominant_basis_levels: tuple[int, ...]
    dominant_basis_label: str
    dominant_basis_overlap: float


@dataclass(frozen=True)
class EnergySpectrum:
    hamiltonian: qt.Qobj
    frame: FrameSpec
    levels: tuple[EnergyLevel, ...]
    vacuum_energy: float
    vacuum_basis_levels: tuple[int, ...]
    vacuum_basis_label: str
    vacuum_level_index: int | None
    vacuum_level_overlap: float
    vacuum_residual_norm: float
    subsystem_labels: tuple[str, ...]
    subsystem_dims: tuple[int, ...]
    basis_levels: tuple[tuple[int, ...], ...]
    basis_labels: tuple[str, ...]

    @property
    def energies(self) -> np.ndarray:
        return np.asarray([level.energy for level in self.levels], dtype=float)

    @property
    def raw_energies(self) -> np.ndarray:
        return np.asarray([level.raw_energy for level in self.levels], dtype=float)

    @property
    def eigenstates(self) -> tuple[qt.Qobj, ...]:
        return tuple(level.eigenstate for level in self.levels)

    def find_level(self, dominant_basis_label: str) -> EnergyLevel:
        matches = [level for level in self.levels if level.dominant_basis_label == str(dominant_basis_label)]
        if not matches:
            raise KeyError(f"No level is labeled by dominant basis state '{dominant_basis_label}'.")
        if len(matches) > 1:
            raise KeyError(
                f"Multiple levels share the dominant basis label '{dominant_basis_label}'. "
                "Inspect EnergySpectrum.levels directly to disambiguate."
            )
        return matches[0]

    def level_rows(self, max_levels: int | None = None) -> list[dict[str, Any]]:
        if max_levels is None:
            selected = self.levels
        else:
            selected = self.levels[: max(0, int(max_levels))]
        return [
            {
                "index": int(level.index),
                "energy": float(level.energy),
                "raw_energy": float(level.raw_energy),
                "dominant_basis_label": level.dominant_basis_label,
                "dominant_basis_levels": tuple(int(value) for value in level.dominant_basis_levels),
                "dominant_basis_overlap": float(level.dominant_basis_overlap),
            }
            for level in selected
        ]


def _as_spectrum_model(model: Any) -> UniversalCQEDModel | Any:
    if isinstance(model, UniversalCQEDModel):
        return model
    if hasattr(model, "as_universal_model"):
        return model.as_universal_model()
    return model


def _transmon_level_label(level: int) -> str:
    if 0 <= int(level) < len(_TRANSMON_LEVEL_LABELS):
        return _TRANSMON_LEVEL_LABELS[int(level)]
    return f"t{int(level)}"


def _basis_label(model: UniversalCQEDModel | Any, levels: tuple[int, ...]) -> str:
    text_parts: list[str] = []
    values = list(int(level) for level in levels)
    has_transmon = getattr(model, "transmon", None) is not None
    bosonic_modes = tuple(getattr(model, "bosonic_modes", ()))

    if has_transmon:
        text_parts.append(_transmon_level_label(values.pop(0)))

    multiple_bosonic_modes = len(bosonic_modes) > 1
    for mode, level in zip(bosonic_modes, values):
        if multiple_bosonic_modes:
            text_parts.append(f"{int(level)}_{mode.label}")
        else:
            text_parts.append(str(int(level)))

    if not text_parts:
        return "|>"
    return f"|{','.join(text_parts)}>"


def compute_energy_spectrum(
    model: Any,
    *,
    frame: FrameSpec | None = None,
    levels: int | None = None,
    sort: str = "low",
) -> EnergySpectrum:
    """Diagonalize a model's static Hamiltonian in the selected frame.

    The returned energies are shifted so the bare vacuum basis state sits at
    zero energy. For the current number-conserving Hamiltonians in ``cqed_sim``,
    this vacuum state is also an eigenstate of the static Hamiltonian.
    """

    resolved_frame = FrameSpec() if frame is None else frame
    hamiltonian = model.hamiltonian(frame=resolved_frame) if hasattr(model, "hamiltonian") else model.static_hamiltonian(frame=resolved_frame)
    eigvals = 0 if levels is None else int(levels)
    if eigvals < 0:
        raise ValueError("levels must be non-negative or None.")

    raw_energies, eigenstates_out = hamiltonian.eigenstates(sort=str(sort), eigvals=eigvals)
    raw_energies = np.asarray(np.real_if_close(raw_energies), dtype=float)
    eigenstates = tuple(eigenstates_out.tolist() if isinstance(eigenstates_out, np.ndarray) else tuple(eigenstates_out))

    spectrum_model = _as_spectrum_model(model)
    subsystem_dims = tuple(int(dim) for dim in spectrum_model.subsystem_dims)
    subsystem_labels = tuple(str(label) for label in spectrum_model.subsystem_labels)
    basis_levels = tuple(tuple(int(level) for level in combo) for combo in product(*(range(dim) for dim in subsystem_dims)))
    basis_labels = tuple(_basis_label(spectrum_model, combo) for combo in basis_levels)
    basis_states = tuple(spectrum_model.basis_state(*combo) for combo in basis_levels)

    vacuum_basis_levels = tuple(0 for _ in subsystem_dims)
    vacuum_basis_label = _basis_label(spectrum_model, vacuum_basis_levels)
    vacuum_state = spectrum_model.basis_state(*vacuum_basis_levels)
    vacuum_energy = float(np.real_if_close(qt.expect(hamiltonian, vacuum_state)))
    vacuum_residual_norm = float((hamiltonian * vacuum_state - vacuum_energy * vacuum_state).norm())

    vacuum_overlaps = np.asarray(
        [abs(complex(vacuum_state.overlap(state))) ** 2 for state in eigenstates],
        dtype=float,
    )
    vacuum_level_index: int | None = None
    vacuum_level_overlap = 0.0
    if vacuum_overlaps.size:
        best_index = int(np.argmax(vacuum_overlaps))
        vacuum_level_overlap = float(vacuum_overlaps[best_index])
        if vacuum_level_overlap > 1.0e-12:
            vacuum_level_index = best_index

    resolved_levels: list[EnergyLevel] = []
    for index, (raw_energy, eigenstate) in enumerate(zip(raw_energies, eigenstates)):
        overlaps = np.asarray(
            [abs(complex(basis_state.overlap(eigenstate))) ** 2 for basis_state in basis_states],
            dtype=float,
        )
        dominant_index = int(np.argmax(overlaps))
        resolved_levels.append(
            EnergyLevel(
                index=int(index),
                energy=float(raw_energy - vacuum_energy),
                raw_energy=float(raw_energy),
                eigenstate=eigenstate,
                dominant_basis_levels=basis_levels[dominant_index],
                dominant_basis_label=basis_labels[dominant_index],
                dominant_basis_overlap=float(overlaps[dominant_index]),
            )
        )

    return EnergySpectrum(
        hamiltonian=hamiltonian,
        frame=resolved_frame,
        levels=tuple(resolved_levels),
        vacuum_energy=float(vacuum_energy),
        vacuum_basis_levels=vacuum_basis_levels,
        vacuum_basis_label=vacuum_basis_label,
        vacuum_level_index=vacuum_level_index,
        vacuum_level_overlap=float(vacuum_level_overlap),
        vacuum_residual_norm=float(vacuum_residual_norm),
        subsystem_labels=subsystem_labels,
        subsystem_dims=subsystem_dims,
        basis_levels=basis_levels,
        basis_labels=basis_labels,
    )


__all__ = ["EnergyLevel", "EnergySpectrum", "compute_energy_spectrum"]
