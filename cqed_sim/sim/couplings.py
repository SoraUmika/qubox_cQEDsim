from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import qutip as qt


def cross_kerr(a: qt.Qobj, b: qt.Qobj, chi: float) -> qt.Qobj:
    """Return a cross-Kerr interaction ``chi * a^† a * b^† b``."""
    return float(chi) * (a.dag() * a) * (b.dag() * b)


def self_kerr(a: qt.Qobj, kerr: float) -> qt.Qobj:
    """Return a self-Kerr interaction ``-(K / 2) * a^† a^† a a``."""
    return -0.5 * float(kerr) * (a.dag() * a.dag() * a * a)


def exchange(a: qt.Qobj, b: qt.Qobj, coupling: float | complex) -> qt.Qobj:
    """Return an exchange interaction ``J * (a^† b + a b^†)``."""
    coupling = complex(coupling)
    return coupling * (a.dag() * b + a * b.dag())


@dataclass(frozen=True)
class TunableCoupler:
    """Flux-tunable exchange coupling with a cosine transfer function."""

    j_max: float
    flux_period: float = 1.0
    phase_offset: float = 0.0
    dc_offset: float = 0.0

    def exchange_rate(self, flux: float) -> float:
        angle = math.tau * float(flux) / float(self.flux_period) + float(self.phase_offset)
        return float(self.dc_offset + self.j_max * np.cos(angle))

    def operator(self, a: qt.Qobj, b: qt.Qobj, flux: float) -> qt.Qobj:
        return exchange(a, b, self.exchange_rate(flux))


__all__ = ["cross_kerr", "self_kerr", "exchange", "TunableCoupler"]
