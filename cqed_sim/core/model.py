from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import qutip as qt

from .hamiltonian import CrossKerrSpec, ExchangeSpec, SelfKerrSpec, assemble_static_hamiltonian, coupling_term_key
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
    cross_kerr_terms: Sequence[CrossKerrSpec] = field(default_factory=tuple)
    self_kerr_terms: Sequence[SelfKerrSpec] = field(default_factory=tuple)
    exchange_terms: Sequence[ExchangeSpec] = field(default_factory=tuple)
    n_cav: int = 12
    n_tr: int = 3

    subsystem_labels: tuple[str, ...] = ("qubit", "storage")
    _operators_cache: dict[str, qt.Qobj] | None = field(default=None, init=False, repr=False, compare=False)
    _static_h_cache: dict[tuple[float, ...], qt.Qobj] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def subsystem_dims(self) -> tuple[int, ...]:
        return (int(self.n_tr), int(self.n_cav))

    def operators(self) -> dict[str, qt.Qobj]:
        if self._operators_cache is None:
            a = qt.tensor(qt.qeye(self.n_tr), qt.destroy(self.n_cav))
            b = qt.tensor(qt.destroy(self.n_tr), qt.qeye(self.n_cav))
            adag = a.dag()
            bdag = b.dag()
            n_c = adag * a
            n_q = bdag * b
            self._operators_cache = {"a": a, "adag": adag, "b": b, "bdag": bdag, "n_c": n_c, "n_q": n_q}
        return self._operators_cache

    def drive_coupling_operators(self) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
        ops = self.operators()
        return {
            "cavity": (ops["adag"], ops["a"]),
            "storage": (ops["adag"], ops["a"]),
            "qubit": (ops["bdag"], ops["b"]),
            "sideband": (ops["adag"] * ops["b"], ops["a"] * ops["bdag"]),
        }

    def transmon_level_projector(self, level: int) -> qt.Qobj:
        level = int(level)
        projector = qt.basis(self.n_tr, level) * qt.basis(self.n_tr, level).dag()
        return qt.tensor(projector, qt.qeye(self.n_cav))

    def transmon_transition_operators(self, lower_level: int, upper_level: int) -> tuple[qt.Qobj, qt.Qobj]:
        lower_level = int(lower_level)
        upper_level = int(upper_level)
        transition_up = qt.basis(self.n_tr, upper_level) * qt.basis(self.n_tr, lower_level).dag()
        transition_down = transition_up.dag()
        return (
            qt.tensor(transition_up, qt.qeye(self.n_cav)),
            qt.tensor(transition_down, qt.qeye(self.n_cav)),
        )

    def mode_operators(self, mode: str = "storage") -> tuple[qt.Qobj, qt.Qobj]:
        ops = self.operators()
        mode_key = str(mode).strip().lower()
        if mode_key not in {"storage", "cavity"}:
            raise ValueError(f"Unsupported bosonic mode '{mode}'.")
        return ops["a"], ops["adag"]

    def sideband_drive_operators(
        self,
        *,
        mode: str = "storage",
        lower_level: int = 0,
        upper_level: int = 1,
        sideband: str = "red",
    ) -> tuple[qt.Qobj, qt.Qobj]:
        mode_lowering, mode_raising = self.mode_operators(mode)
        transmon_up, transmon_down = self.transmon_transition_operators(lower_level, upper_level)
        sideband_key = str(sideband).strip().lower()
        if sideband_key == "red":
            return 1j * transmon_up * mode_lowering, -1j * transmon_down * mode_raising
        if sideband_key == "blue":
            return 1j * transmon_up * mode_raising, -1j * transmon_down * mode_lowering
        raise ValueError(f"Unsupported sideband '{sideband}'.")

    def static_hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        frame = frame or FrameSpec()
        key = (
            float(self.omega_c),
            float(self.omega_q),
            float(self.alpha),
            float(self.chi),
            tuple(float(value) for value in self.chi_higher),
            float(self.kerr),
            tuple(float(value) for value in self.kerr_higher),
            coupling_term_key(self.cross_kerr_terms, self.self_kerr_terms, self.exchange_terms),
            float(frame.omega_c_frame),
            float(frame.omega_q_frame),
        )
        cached = self._static_h_cache.get(key)
        if cached is not None:
            return cached
        ops = self.operators()
        n_c, n_q = ops["n_c"], ops["n_q"]
        b, bdag = ops["b"], ops["bdag"]

        delta_c = self.omega_c - frame.omega_c_frame
        delta_q = self.omega_q - frame.omega_q_frame
        h = delta_c * n_c + delta_q * n_q

        h += 0.5 * self.alpha * (bdag * bdag * b * b)

        if self.kerr != 0.0:
            h += self.kerr * _falling_factorial_number_op(n_c, 2) / math.factorial(2)
        for i, coeff in enumerate(self.kerr_higher, start=2):
            order = i + 1
            h += coeff * _falling_factorial_number_op(n_c, order) / math.factorial(order)

        if self.chi != 0.0:
            h += self.chi * n_c * n_q
        for i, coeff in enumerate(self.chi_higher, start=2):
            h += coeff * _falling_factorial_number_op(n_c, i) * n_q
        h = assemble_static_hamiltonian(
            h,
            ops,
            cross_kerr_terms=self.cross_kerr_terms,
            self_kerr_terms=self.self_kerr_terms,
            exchange_terms=self.exchange_terms,
        )
        self._static_h_cache[key] = h
        return h

    def basis_energy(self, q_level: int, cavity_level: int, frame: FrameSpec | None = None) -> float:
        frame = frame or FrameSpec()
        q_level = int(q_level)
        cavity_level = int(cavity_level)

        delta_c = float(self.omega_c - frame.omega_c_frame)
        delta_q = float(self.omega_q - frame.omega_q_frame)
        energy = delta_c * cavity_level + delta_q * q_level
        energy += 0.5 * float(self.alpha) * q_level * (q_level - 1)
        energy += 0.5 * float(self.kerr) * cavity_level * (cavity_level - 1)
        for i, coeff in enumerate(self.kerr_higher, start=2):
            order = i + 1
            factor = 1
            for k in range(order):
                factor *= cavity_level - k
            energy += float(coeff) * factor / math.factorial(order)
        energy += float(self.chi) * cavity_level * q_level
        for i, coeff in enumerate(self.chi_higher, start=2):
            factor = 1
            for k in range(i):
                factor *= cavity_level - k
            energy += float(coeff) * factor * q_level
        return float(energy)

    def basis_state(self, q_level: int, cavity_level: int) -> qt.Qobj:
        """Return the joint basis ket |q> tensor |n> with qubit first, cavity second."""
        return qt.tensor(qt.basis(self.n_tr, q_level), qt.basis(self.n_cav, cavity_level))

    def coherent_qubit_superposition(self, n_cav: int = 0) -> qt.Qobj:
        g = self.basis_state(0, n_cav)
        e = self.basis_state(1, n_cav)
        return (g + e).unit()

    def manifold_transition_frequency(self, n: int, frame: FrameSpec | None = None) -> float:
        return manifold_transition_frequency(self, n=n, frame=frame)

    def transmon_transition_frequency(
        self,
        cavity_level: int = 0,
        *,
        lower_level: int = 0,
        upper_level: int = 1,
        frame: FrameSpec | None = None,
    ) -> float:
        return float(
            self.basis_energy(int(upper_level), int(cavity_level), frame=frame)
            - self.basis_energy(int(lower_level), int(cavity_level), frame=frame)
        )

    def sideband_transition_frequency(
        self,
        cavity_level: int = 0,
        *,
        lower_level: int = 0,
        upper_level: int = 1,
        sideband: str = "red",
        frame: FrameSpec | None = None,
    ) -> float:
        cavity_level = int(cavity_level)
        sideband_key = str(sideband).strip().lower()
        if sideband_key == "red":
            if cavity_level + 1 >= self.n_cav:
                raise IndexError("red sideband requires cavity_level + 1 to be within the cavity dimension.")
            return float(
                self.basis_energy(int(upper_level), cavity_level, frame=frame)
                - self.basis_energy(int(lower_level), cavity_level + 1, frame=frame)
            )
        if sideband_key == "blue":
            if cavity_level + 1 >= self.n_cav:
                raise IndexError("blue sideband requires cavity_level + 1 to be within the cavity dimension.")
            return float(
                self.basis_energy(int(upper_level), cavity_level + 1, frame=frame)
                - self.basis_energy(int(lower_level), cavity_level, frame=frame)
            )
        raise ValueError(f"Unsupported sideband '{sideband}'.")
