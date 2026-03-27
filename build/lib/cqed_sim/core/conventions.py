from __future__ import annotations


def qubit_cavity_dims(n_qubit: int, n_cav: int) -> list[list[int]]:
    """Return QuTiP dims for the repository convention qubit tensor cavity."""
    return [[int(n_qubit), int(n_cav)], [int(n_qubit), int(n_cav)]]


def qubit_cavity_index(n_cav: int, qubit_level: int, cavity_level: int) -> int:
    """Return the flat basis index for |q> tensor |n> in qubit-major ordering."""
    return int(qubit_level) * int(n_cav) + int(cavity_level)


def qubit_cavity_block_indices(n_cav: int, cavity_level: int) -> tuple[int, int]:
    """Return the flat indices spanning {|g,n>, |e,n>} in qubit tensor cavity order."""
    n_cav = int(n_cav)
    cavity_level = int(cavity_level)
    return (
        qubit_cavity_index(n_cav, 0, cavity_level),
        qubit_cavity_index(n_cav, 1, cavity_level),
    )


def qubit_storage_readout_dims(n_qubit: int, n_storage: int, n_readout: int) -> list[list[int]]:
    """Return QuTiP dims for qubit tensor storage tensor readout ordering."""
    dims = [int(n_qubit), int(n_storage), int(n_readout)]
    return [dims, dims]


def qubit_storage_readout_index(
    n_storage: int,
    n_readout: int,
    qubit_level: int,
    storage_level: int,
    readout_level: int,
) -> int:
    """Return the flat basis index for |q,n_s,n_r> in qubit-major ordering."""
    n_storage = int(n_storage)
    n_readout = int(n_readout)
    return ((int(qubit_level) * n_storage) + int(storage_level)) * n_readout + int(readout_level)


def qubit_storage_readout_block_indices(
    n_storage: int,
    n_readout: int,
    storage_level: int,
    readout_level: int,
) -> tuple[int, int]:
    """Return the flat indices spanning {|g,n_s,n_r>, |e,n_s,n_r>} in qubit-major order."""
    return (
        qubit_storage_readout_index(n_storage, n_readout, 0, storage_level, readout_level),
        qubit_storage_readout_index(n_storage, n_readout, 1, storage_level, readout_level),
    )
