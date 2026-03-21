from __future__ import annotations

from functools import lru_cache

import qutip as qt


@lru_cache(maxsize=16)
def destroy_cavity(n_cav_dim: int) -> qt.Qobj:
    return qt.destroy(n_cav_dim)


@lru_cache(maxsize=16)
def create_cavity(n_cav_dim: int) -> qt.Qobj:
    return destroy_cavity(n_cav_dim).dag()


@lru_cache(maxsize=16)
def number_operator(n_cav_dim: int) -> qt.Qobj:
    a = destroy_cavity(n_cav_dim)
    return a.dag() * a


@lru_cache(maxsize=64)
def fock_projector(n_cav_dim: int, n: int) -> qt.Qobj:
    ket = qt.basis(n_cav_dim, n)
    return ket * ket.dag()
