from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import qutip as qt


BareLabel = tuple[int, ...]


@dataclass
class TransitionMatrixResult:
    matrix: np.ndarray
    rows: tuple[int, ...]
    columns: tuple[int, ...]
    missing_weight: np.ndarray


@dataclass
class DressedBasis:
    energies: np.ndarray
    states: list[qt.Qobj]
    labels: tuple[BareLabel, ...]
    subsystem_dims: tuple[int, ...]
    projectors: dict[int, qt.Qobj]

    @property
    def retained_subspace_projector(self) -> qt.Qobj:
        if not self.states:
            raise ValueError("No dressed states were retained.")
        total = 0 * (self.states[0] * self.states[0].dag())
        for state in self.states:
            total = total + state * state.dag()
        return total

    def qubit_projector(self, level: int) -> qt.Qobj:
        return self.projectors[int(level)]

    def labeled_projector(
        self,
        level: int,
        *,
        allowed_occupations: Mapping[int, Sequence[int]] | None = None,
    ) -> qt.Qobj:
        """Project onto dressed states assigned to a qubit level and optional bare occupations.

        ``allowed_occupations`` maps subsystem index to allowed bare labels.  For
        example, ``{1: (0,)}`` selects the cavity-empty part of a qubit branch in
        a two-subsystem transmon-resonator model.
        """

        if not self.states:
            raise ValueError("No dressed states were retained.")
        allowed = {
            int(index): {int(value) for value in values}
            for index, values in dict(allowed_occupations or {}).items()
        }
        total = 0 * (self.states[0] * self.states[0].dag())
        for label, state in zip(self.labels, self.states):
            if int(label[0]) != int(level):
                continue
            if any(int(index) >= len(label) or int(label[int(index)]) not in values for index, values in allowed.items()):
                continue
            total = total + state * state.dag()
        return total

    def cavity_empty_projector(self, level: int) -> qt.Qobj:
        allowed = {index: (0,) for index in range(1, len(self.subsystem_dims))}
        return self.labeled_projector(level, allowed_occupations=allowed)

    def computational_projector(self, levels: Sequence[int] = (0, 1)) -> qt.Qobj:
        if not self.states:
            raise ValueError("No dressed states were retained.")
        total = 0 * (self.states[0] * self.states[0].dag())
        for level in levels:
            total = total + self.projectors.get(int(level), 0 * total)
        return total

    def leakage_probability(self, rho: qt.Qobj, computational_levels: Sequence[int] = (0, 1)) -> float:
        density = rho if rho.isoper else rho.proj()
        inside = complex((self.computational_projector(computational_levels) * density).tr()).real
        return float(max(0.0, 1.0 - inside))

    def transition_matrix(
        self,
        rho_final_by_initial: Mapping[int, qt.Qobj] | Sequence[qt.Qobj],
        *,
        prepared_levels: Sequence[int] | None = None,
        measured_levels: Sequence[int] | None = None,
        projector_mode: str = "branch",
    ) -> TransitionMatrixResult:
        if isinstance(rho_final_by_initial, Mapping):
            columns = tuple(int(key) for key in rho_final_by_initial.keys()) if prepared_levels is None else tuple(int(v) for v in prepared_levels)
            rhos = [rho_final_by_initial[level] for level in columns]
        else:
            rhos = list(rho_final_by_initial)
            columns = tuple(range(len(rhos))) if prepared_levels is None else tuple(int(v) for v in prepared_levels)
        rows = tuple(sorted(self.projectors)) if measured_levels is None else tuple(int(v) for v in measured_levels)
        matrix = np.zeros((len(rows), len(columns)), dtype=float)
        missing = np.zeros(len(columns), dtype=float)
        for col, rho in enumerate(rhos):
            density = rho if rho.isoper else rho.proj()
            for row, level in enumerate(rows):
                if str(projector_mode) == "cavity_empty":
                    projector = self.cavity_empty_projector(int(level))
                elif str(projector_mode) == "branch":
                    projector = self.projectors.get(int(level))
                else:
                    raise ValueError("projector_mode must be 'branch' or 'cavity_empty'.")
                if projector is not None:
                    matrix[row, col] = float(np.real((projector * density).tr()))
            missing[col] = float(max(0.0, 1.0 - np.sum(matrix[:, col])))
        return TransitionMatrixResult(matrix=matrix, rows=rows, columns=columns, missing_weight=missing)


def _bare_basis_states(dims: Sequence[int]) -> tuple[list[BareLabel], list[qt.Qobj]]:
    dims = tuple(int(dim) for dim in dims)
    labels: list[BareLabel] = []
    states: list[qt.Qobj] = []

    def _walk(prefix: list[int], index: int) -> None:
        if index == len(dims):
            label = tuple(prefix)
            labels.append(label)
            states.append(qt.tensor(*(qt.basis(dim, value) for dim, value in zip(dims, label))))
            return
        for value in range(dims[index]):
            prefix.append(value)
            _walk(prefix, index + 1)
            prefix.pop()

    _walk([], 0)
    return labels, states


def diagonalize_dressed_hamiltonian(
    hamiltonian: qt.Qobj,
    *,
    subsystem_dims: Sequence[int] | None = None,
    levels: int | None = None,
) -> DressedBasis:
    """Diagonalize an undriven coupled Hamiltonian and label states by overlap."""

    dims = tuple(int(dim) for dim in (hamiltonian.dims[0] if subsystem_dims is None else subsystem_dims))
    if np.prod(dims, dtype=int) != hamiltonian.shape[0]:
        raise ValueError("subsystem_dims do not match the Hamiltonian dimension.")
    energies_all, states_all = hamiltonian.eigenstates()
    order = np.argsort(np.asarray(energies_all, dtype=float))
    if levels is not None:
        order = order[: int(levels)]
    bare_labels, bare_states = _bare_basis_states(dims)
    assigned: set[int] = set()
    labels: list[BareLabel] = []
    states: list[qt.Qobj] = []
    energies: list[float] = []
    for idx in order:
        state = states_all[int(idx)]
        overlaps = np.asarray([abs(complex(bare.dag() * state)) ** 2 for bare in bare_states], dtype=float)
        for candidate in np.argsort(overlaps)[::-1]:
            if int(candidate) not in assigned:
                assigned.add(int(candidate))
                labels.append(bare_labels[int(candidate)])
                break
        else:
            labels.append(bare_labels[int(np.argmax(overlaps))])
        states.append(state)
        energies.append(float(energies_all[int(idx)]))

    zero = 0 * (states[0] * states[0].dag()) if states else 0 * hamiltonian
    projectors: dict[int, qt.Qobj] = {}
    for label, state in zip(labels, states):
        q_level = int(label[0])
        projectors.setdefault(q_level, zero.copy())
        projectors[q_level] = projectors[q_level] + state * state.dag()
    return DressedBasis(
        energies=np.asarray(energies, dtype=float),
        states=states,
        labels=tuple(labels),
        subsystem_dims=dims,
        projectors=projectors,
    )


def transition_matrix(
    rho_final_by_initial: Mapping[int, qt.Qobj] | Sequence[qt.Qobj],
    dressed_basis: DressedBasis,
    *,
    prepared_levels: Sequence[int] | None = None,
    measured_levels: Sequence[int] | None = None,
    projector_mode: str = "branch",
) -> TransitionMatrixResult:
    return dressed_basis.transition_matrix(
        rho_final_by_initial,
        prepared_levels=prepared_levels,
        measured_levels=measured_levels,
        projector_mode=projector_mode,
    )


__all__ = [
    "BareLabel",
    "DressedBasis",
    "TransitionMatrixResult",
    "diagonalize_dressed_hamiltonian",
    "transition_matrix",
]
