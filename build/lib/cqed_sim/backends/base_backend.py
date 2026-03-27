from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable


class BaseBackend(ABC):
    """Dense linear-algebra backend used by the optional solver path."""

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
