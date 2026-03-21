from __future__ import annotations

from functools import reduce
from typing import Iterable

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

from .base_backend import BaseBackend

# Dimension threshold: use sparse Kronecker products for the Liouvillian
# when the Hilbert-space dimension exceeds this value.
_SPARSE_KRON_DIM_THRESHOLD = 20


class NumPyBackend(BaseBackend):
    """Dense NumPy/SciPy backend for small-system time evolution.

    Uses ``numpy`` for array operations and ``scipy.linalg.expm`` for the matrix
    exponential.  Suitable for closed-system propagation (via ``expm``) or
    open-system evolution (via the Liouvillian superoperator).

    Limitations
    -----------
    - Dense storage: memory scales as O(dim^2) for states and O(dim^4) for
      Liouvillians.  For dim > ~50 the JAX or QuTiP backends should be preferred.
    - ``expm`` calls ``scipy.linalg.expm`` which is CPU-only and single-threaded.
    """

    name = "numpy"

    def asarray(self, value):
        return np.asarray(value, dtype=np.complex128)

    def to_numpy(self, value):
        return np.asarray(value, dtype=np.complex128)

    def eye(self, dim: int):
        return np.eye(int(dim), dtype=np.complex128)

    def zeros(self, shape: tuple[int, ...]):
        return np.zeros(shape, dtype=np.complex128)

    def reshape(self, value, shape: tuple[int, ...]):
        return np.reshape(value, shape)

    def dagger(self, value):
        return np.asarray(value, dtype=np.complex128).conj().T

    def matmul(self, left, right):
        return np.asarray(left, dtype=np.complex128) @ np.asarray(right, dtype=np.complex128)

    def kron(self, values: Iterable):
        return reduce(np.kron, [np.asarray(value, dtype=np.complex128) for value in values])

    def expm(self, value):
        return expm(np.asarray(value, dtype=np.complex128))

    def trace(self, value):
        return np.trace(np.asarray(value, dtype=np.complex128))

    def expectation(self, operator, state):
        operator = np.asarray(operator, dtype=np.complex128)
        state = np.asarray(state, dtype=np.complex128)
        if state.ndim == 1:
            return np.vdot(state, operator @ state)
        return np.trace(state @ operator)

    def lindbladian(self, hamiltonian, collapse_ops):
        hamiltonian = np.asarray(hamiltonian, dtype=np.complex128)
        dim = hamiltonian.shape[0]
        if dim >= _SPARSE_KRON_DIM_THRESHOLD:
            return self._lindbladian_sparse(hamiltonian, collapse_ops)
        identity = self.eye(dim)
        liouvillian = -1j * (np.kron(identity, hamiltonian) - np.kron(hamiltonian.T, identity))
        for collapse in collapse_ops:
            c = np.asarray(collapse, dtype=np.complex128)
            cd_c = c.conj().T @ c
            liouvillian += np.kron(c.conj(), c)
            liouvillian -= 0.5 * np.kron(identity, cd_c)
            liouvillian -= 0.5 * np.kron(cd_c.T, identity)
        return liouvillian

    def _lindbladian_sparse(self, hamiltonian, collapse_ops):
        """Build Liouvillian using sparse Kronecker products (large dim)."""
        dim = hamiltonian.shape[0]
        h_sp = sp.csr_matrix(hamiltonian)
        identity = sp.eye(dim, dtype=np.complex128, format="csr")
        liouvillian = -1j * (sp.kron(identity, h_sp, format="csr") - sp.kron(h_sp.T, identity, format="csr"))
        for collapse in collapse_ops:
            c = sp.csr_matrix(np.asarray(collapse, dtype=np.complex128))
            cd_c = c.conj().T @ c
            liouvillian = liouvillian + sp.kron(c.conj(), c, format="csr")
            liouvillian = liouvillian - 0.5 * sp.kron(identity, cd_c, format="csr")
            liouvillian = liouvillian - 0.5 * sp.kron(cd_c.T, identity, format="csr")
        return liouvillian.toarray()


__all__ = ["NumPyBackend"]
