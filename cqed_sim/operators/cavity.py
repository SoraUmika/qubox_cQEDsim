from __future__ import annotations

import qutip as qt


def destroy_cavity(n_cav_dim: int) -> qt.Qobj:
    return qt.destroy(n_cav_dim)


def create_cavity(n_cav_dim: int) -> qt.Qobj:
    return destroy_cavity(n_cav_dim).dag()


def number_operator(n_cav_dim: int) -> qt.Qobj:
    a = destroy_cavity(n_cav_dim)
    return a.dag() * a


def fock_projector(n_cav_dim: int, n: int) -> qt.Qobj:
    ket = qt.basis(n_cav_dim, n)
    return ket * ket.dag()
