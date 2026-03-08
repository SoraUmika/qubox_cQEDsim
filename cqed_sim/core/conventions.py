from __future__ import annotations


def qubit_cavity_dims(n_qubit: int, n_cav: int) -> list[list[int]]:
    """Return QuTiP dims for the repository convention qubit ⊗ cavity."""
    return [[int(n_qubit), int(n_cav)], [int(n_qubit), int(n_cav)]]


def qubit_cavity_index(n_cav: int, qubit_level: int, cavity_level: int) -> int:
    """Return the flat basis index for |q> ⊗ |n> in qubit-major ordering."""
    return int(qubit_level) * int(n_cav) + int(cavity_level)


def qubit_cavity_block_indices(n_cav: int, cavity_level: int) -> tuple[int, int]:
    """Return the two flat indices spanning {|g,n>, |e,n>} in qubit ⊗ cavity order."""
    n_cav = int(n_cav)
    cavity_level = int(cavity_level)
    return (
        qubit_cavity_index(n_cav, 0, cavity_level),
        qubit_cavity_index(n_cav, 1, cavity_level),
    )
