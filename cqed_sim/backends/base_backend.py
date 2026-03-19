from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class BaseBackend(ABC):
    """Dense linear-algebra backend used by the optional solver path.

    These backends operate on dense NumPy/JAX arrays and are intended for small
    systems where the Hilbert-space dimension is modest (typically up to ~100).
    For large Hilbert spaces the default QuTiP backend (which uses sparse matrices
    internally) will be more efficient.

    Limitations
    -----------
    - All matrices are stored and manipulated as dense arrays; memory usage scales
      as dim^2 (states) or dim^4 (superoperators/Liouvillians).
    - The Lindbladian constructed by :meth:`lindbladian` is a dense dim^2 x dim^2
      superoperator matrix; this can be prohibitively large for systems with
      dim > ~50.
    - The :class:`JaxBackend` is optional and requires JAX to be installed.  When
      JAX is not available ``cqed_sim.backends.JaxBackend`` will be ``None``.
    - Neither ``NumPyBackend`` nor ``JaxBackend`` are drop-in replacements for
      QuTiP's adaptive ODE solver.  They use piecewise-constant matrix
      exponentials, which is accurate for small step sizes but significantly
      slower than QuTiP for large or stiff systems.
    - GPU acceleration via JAX for GRAPE gradient computation is not yet
      implemented.  The current ``JaxBackend`` runs on CPU only.  GPU/JAX
      GRAPE integration is deferred (see ``docs/performance_design.md``).
    """

    name: str = "base"

    @abstractmethod
    def asarray(self, value): ...

    @abstractmethod
    def to_numpy(self, value): ...

    @abstractmethod
    def eye(self, dim: int): ...

    @abstractmethod
    def zeros(self, shape: tuple[int, ...]): ...

    @abstractmethod
    def reshape(self, value, shape: tuple[int, ...]): ...

    @abstractmethod
    def dagger(self, value): ...

    @abstractmethod
    def matmul(self, left, right): ...

    @abstractmethod
    def kron(self, values: Iterable): ...

    @abstractmethod
    def expm(self, value): ...

    @abstractmethod
    def trace(self, value): ...

    @abstractmethod
    def expectation(self, operator, state): ...

    @abstractmethod
    def lindbladian(self, hamiltonian, collapse_ops): ...
