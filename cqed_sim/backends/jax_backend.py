from __future__ import annotations

from functools import reduce
from typing import Iterable

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np

from .base_backend import BaseBackend

jax.config.update("jax_enable_x64", True)


class JaxBackend(BaseBackend):
    """Dense JAX backend for small-system time evolution with optional GPU support.

    Uses ``jax.numpy`` for array operations and ``jax.scipy.linalg.expm`` for
    the matrix exponential.  When JAX is configured with a GPU device, computations
    can be offloaded to the GPU by passing ``device="gpu"`` to the constructor.

    Prerequisites
    -------------
    JAX must be installed (``pip install jax``).  64-bit precision is enabled
    automatically via ``jax.config.update("jax_enable_x64", True)``.

    Limitations
    -----------
    - Dense storage: same O(dim^2) / O(dim^4) scaling as :class:`NumPyBackend`.
    - JIT compilation adds overhead for first-call latency; for very short
      simulations the NumPy backend may be faster.
    - JAX operations are asynchronous by default; call ``.block_until_ready()``
      on results if you need synchronization with wall-clock timings.
    """

    name = "jax"

    def __init__(self, device: str | None = None):
        self.device = device

    def _put(self, value):
        array = jnp.asarray(value, dtype=jnp.complex128)
        if self.device is None:
            return array
        matching = [dev for dev in jax.devices() if dev.platform == self.device]
        return array if not matching else jax.device_put(array, matching[0])

    def asarray(self, value):
        return self._put(value)

    def to_numpy(self, value):
        return np.asarray(value, dtype=np.complex128)

    def eye(self, dim: int):
        return self._put(jnp.eye(int(dim), dtype=jnp.complex128))

    def zeros(self, shape: tuple[int, ...]):
        return self._put(jnp.zeros(shape, dtype=jnp.complex128))

    def reshape(self, value, shape: tuple[int, ...]):
        return jnp.reshape(value, shape)

    def dagger(self, value):
        return jnp.conjugate(jnp.swapaxes(value, -1, -2))

    def matmul(self, left, right):
        return jnp.asarray(left, dtype=jnp.complex128) @ jnp.asarray(right, dtype=jnp.complex128)

    def kron(self, values: Iterable):
        arrays = [jnp.asarray(value, dtype=jnp.complex128) for value in values]
        return reduce(jnp.kron, arrays)

    def expm(self, value):
        return jsp_linalg.expm(jnp.asarray(value, dtype=jnp.complex128))

    def trace(self, value):
        return jnp.trace(jnp.asarray(value, dtype=jnp.complex128))

    def expectation(self, operator, state):
        operator = jnp.asarray(operator, dtype=jnp.complex128)
        state = jnp.asarray(state, dtype=jnp.complex128)
        if state.ndim == 1:
            return jnp.vdot(state, operator @ state)
        return jnp.trace(state @ operator)

    def lindbladian(self, hamiltonian, collapse_ops):
        hamiltonian = jnp.asarray(hamiltonian, dtype=jnp.complex128)
        dim = int(hamiltonian.shape[0])
        identity = self.eye(dim)
        liouvillian = -1j * (jnp.kron(identity, hamiltonian) - jnp.kron(hamiltonian.T, identity))
        for collapse in collapse_ops:
            c = jnp.asarray(collapse, dtype=jnp.complex128)
            cd_c = jnp.conjugate(c.T) @ c
            liouvillian += jnp.kron(jnp.conjugate(c), c)
            liouvillian -= 0.5 * jnp.kron(identity, cd_c)
            liouvillian -= 0.5 * jnp.kron(cd_c.T, identity)
        return liouvillian


__all__ = ["JaxBackend"]
