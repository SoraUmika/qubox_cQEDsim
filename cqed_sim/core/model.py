from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import qutip as qt

from .frame import FrameSpec
from .frequencies import manifold_transition_frequency


def _falling_factorial_number_op(n_op: qt.Qobj, order: int) -> qt.Qobj:
    """Return n(n-1)...(n-order+1) for number operator n."""
    if order <= 0:
        return 0 * n_op + qt.qeye(n_op.dims[0])
    out = 0 * n_op + qt.qeye(n_op.dims[0])
    for k in range(order):
        out = out * (n_op - k * qt.qeye(n_op.dims[0]))
    return out


@dataclass
class DispersiveTransmonCavityModel:
    omega_c: float
    omega_q: float
    alpha: float
    chi: float = 0.0
    chi_higher: Sequence[float] = field(default_factory=tuple)
    kerr: float = 0.0
    kerr_higher: Sequence[float] = field(default_factory=tuple)
    n_cav: int = 12
    n_tr: int = 3

    def operators(self) -> dict[str, qt.Qobj]:
        a = qt.tensor(qt.qeye(self.n_tr), qt.destroy(self.n_cav))
        b = qt.tensor(qt.destroy(self.n_tr), qt.qeye(self.n_cav))
        adag = a.dag()
        bdag = b.dag()
        n_c = adag * a
        n_q = bdag * b
        return {"a": a, "adag": adag, "b": b, "bdag": bdag, "n_c": n_c, "n_q": n_q}

    def static_hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        frame = frame or FrameSpec()
        ops = self.operators()
        n_c, n_q = ops["n_c"], ops["n_q"]
        b, bdag = ops["b"], ops["bdag"]

        delta_c = self.omega_c - frame.omega_c_frame
        delta_q = self.omega_q - frame.omega_q_frame
        h = delta_c * n_c + delta_q * n_q

        # Duffing transmon anharmonicity.
        h += 0.5 * self.alpha * (bdag * bdag * b * b)

        # Cavity Kerr hierarchy: K1 * n(n-1)/2 + K2 * n(n-1)(n-2)/6 + ...
        if self.kerr != 0.0:
            h += self.kerr * _falling_factorial_number_op(n_c, 2) / math.factorial(2)
        for i, coeff in enumerate(self.kerr_higher, start=2):
            order = i + 1
            h += coeff * _falling_factorial_number_op(n_c, order) / math.factorial(order)

        # qubox convention: omega_ge(n) = omega_ge(0) - chi*n - chi2*n(n-1) - chi3*n(n-1)(n-2) - ...
        if self.chi != 0.0:
            h += -self.chi * n_c * n_q
        for i, coeff in enumerate(self.chi_higher, start=2):
            h += -coeff * _falling_factorial_number_op(n_c, i) * n_q
        return h

    def basis_state(self, q_level: int, cavity_level: int) -> qt.Qobj:
        """Return the joint basis ket |q> tensor |n> with qubit first, cavity second."""
        return qt.tensor(qt.basis(self.n_tr, q_level), qt.basis(self.n_cav, cavity_level))

    def coherent_qubit_superposition(self, n_cav: int = 0) -> qt.Qobj:
        g = self.basis_state(0, n_cav)
        e = self.basis_state(1, n_cav)
        return (g + e).unit()

    def manifold_transition_frequency(self, n: int, frame: FrameSpec | None = None) -> float:
        return manifold_transition_frequency(self, n=n, frame=frame)
