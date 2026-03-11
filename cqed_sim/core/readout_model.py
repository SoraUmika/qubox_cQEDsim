from __future__ import annotations

from dataclasses import dataclass, field

import qutip as qt

from .hamiltonian import CrossKerrSpec, ExchangeSpec, SelfKerrSpec, assemble_static_hamiltonian, coupling_term_key
from .frame import FrameSpec


@dataclass
class DispersiveReadoutTransmonStorageModel:
    omega_s: float
    omega_r: float
    omega_q: float
    alpha: float
    chi_s: float = 0.0
    chi_r: float = 0.0
    chi_sr: float = 0.0
    kerr_s: float = 0.0
    kerr_r: float = 0.0
    cross_kerr_terms: tuple[CrossKerrSpec, ...] = field(default_factory=tuple)
    self_kerr_terms: tuple[SelfKerrSpec, ...] = field(default_factory=tuple)
    exchange_terms: tuple[ExchangeSpec, ...] = field(default_factory=tuple)
    n_storage: int = 12
    n_readout: int = 8
    n_tr: int = 3

    subsystem_labels: tuple[str, ...] = ("qubit", "storage", "readout")
    _operators_cache: dict[str, qt.Qobj] | None = field(default=None, init=False, repr=False, compare=False)
    _static_h_cache: dict[tuple[float, ...], qt.Qobj] = field(default_factory=dict, init=False, repr=False, compare=False)

    @property
    def subsystem_dims(self) -> tuple[int, ...]:
        return (int(self.n_tr), int(self.n_storage), int(self.n_readout))

    def operators(self) -> dict[str, qt.Qobj]:
        if self._operators_cache is None:
            i_q = qt.qeye(self.n_tr)
            i_s = qt.qeye(self.n_storage)
            i_r = qt.qeye(self.n_readout)

            b = qt.tensor(qt.destroy(self.n_tr), i_s, i_r)
            a_s = qt.tensor(i_q, qt.destroy(self.n_storage), i_r)
            a_r = qt.tensor(i_q, i_s, qt.destroy(self.n_readout))

            bdag = b.dag()
            adag_s = a_s.dag()
            adag_r = a_r.dag()
            n_q = bdag * b
            n_s = adag_s * a_s
            n_r = adag_r * a_r
            self._operators_cache = {
                "b": b,
                "bdag": bdag,
                "a_s": a_s,
                "adag_s": adag_s,
                "a_r": a_r,
                "adag_r": adag_r,
                "n_q": n_q,
                "n_s": n_s,
                "n_r": n_r,
            }
        return self._operators_cache

    def drive_coupling_operators(self) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
        ops = self.operators()
        return {
            "storage": (ops["adag_s"], ops["a_s"]),
            "cavity": (ops["adag_s"], ops["a_s"]),
            "qubit": (ops["bdag"], ops["b"]),
            "transmon": (ops["bdag"], ops["b"]),
            "readout": (ops["adag_r"], ops["a_r"]),
        }

    def static_hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        frame = frame or FrameSpec()
        key = (
            float(self.omega_s),
            float(self.omega_r),
            float(self.omega_q),
            float(self.alpha),
            float(self.chi_s),
            float(self.chi_r),
            float(self.chi_sr),
            float(self.kerr_s),
            float(self.kerr_r),
            coupling_term_key(self.cross_kerr_terms, self.self_kerr_terms, self.exchange_terms),
            float(frame.omega_c_frame),
            float(frame.omega_q_frame),
            float(frame.omega_r_frame),
        )
        cached = self._static_h_cache.get(key)
        if cached is not None:
            return cached
        ops = self.operators()
        n_q = ops["n_q"]
        n_s = ops["n_s"]
        n_r = ops["n_r"]
        b = ops["b"]
        bdag = ops["bdag"]

        delta_s = self.omega_s - frame.omega_s_frame
        delta_r = self.omega_r - frame.omega_r_frame
        delta_q = self.omega_q - frame.omega_q_frame

        h = delta_s * n_s + delta_r * n_r + delta_q * n_q
        h += 0.5 * self.alpha * (bdag * bdag * b * b)
        if self.chi_s != 0.0:
            h += -self.chi_s * n_s * n_q
        if self.chi_r != 0.0:
            h += -self.chi_r * n_r * n_q
        if self.chi_sr != 0.0:
            h += self.chi_sr * n_s * n_r
        if self.kerr_s != 0.0:
            h += 0.5 * self.kerr_s * n_s * (n_s - qt.qeye(n_s.dims[0]))
        if self.kerr_r != 0.0:
            h += 0.5 * self.kerr_r * n_r * (n_r - qt.qeye(n_r.dims[0]))
        h = assemble_static_hamiltonian(
            h,
            ops,
            cross_kerr_terms=self.cross_kerr_terms,
            self_kerr_terms=self.self_kerr_terms,
            exchange_terms=self.exchange_terms,
        )
        self._static_h_cache[key] = h
        return h

    def basis_energy(
        self,
        q_level: int,
        storage_level: int,
        readout_level: int,
        frame: FrameSpec | None = None,
    ) -> float:
        frame = frame or FrameSpec()
        q_level = int(q_level)
        storage_level = int(storage_level)
        readout_level = int(readout_level)

        delta_s = float(self.omega_s - frame.omega_s_frame)
        delta_r = float(self.omega_r - frame.omega_r_frame)
        delta_q = float(self.omega_q - frame.omega_q_frame)
        energy = delta_s * storage_level + delta_r * readout_level + delta_q * q_level
        energy += 0.5 * float(self.alpha) * q_level * (q_level - 1)
        energy += -float(self.chi_s) * storage_level * q_level
        energy += -float(self.chi_r) * readout_level * q_level
        energy += float(self.chi_sr) * storage_level * readout_level
        energy += 0.5 * float(self.kerr_s) * storage_level * (storage_level - 1)
        energy += 0.5 * float(self.kerr_r) * readout_level * (readout_level - 1)
        return float(energy)

    def basis_state(self, q_level: int, storage_level: int, readout_level: int) -> qt.Qobj:
        """Return |q,n_s,n_r> = |q> tensor |n_s> tensor |n_r>."""
        return qt.tensor(
            qt.basis(self.n_tr, q_level),
            qt.basis(self.n_storage, storage_level),
            qt.basis(self.n_readout, readout_level),
        )

    def coherent_qubit_superposition(self, storage_level: int = 0, readout_level: int = 0) -> qt.Qobj:
        g = self.basis_state(0, storage_level, readout_level)
        e = self.basis_state(1, storage_level, readout_level)
        return (g + e).unit()

    def qubit_transition_frequency(
        self,
        storage_level: int = 0,
        readout_level: int = 0,
        qubit_level: int = 0,
        frame: FrameSpec | None = None,
    ) -> float:
        frame = frame or FrameSpec()
        delta_q = float(self.omega_q - frame.omega_q_frame)
        return float(
            delta_q
            + float(self.alpha) * int(qubit_level)
            - float(self.chi_s) * int(storage_level)
            - float(self.chi_r) * int(readout_level)
        )

    def storage_transition_frequency(
        self,
        qubit_level: int = 0,
        storage_level: int = 0,
        readout_level: int = 0,
        frame: FrameSpec | None = None,
    ) -> float:
        frame = frame or FrameSpec()
        delta_s = float(self.omega_s - frame.omega_s_frame)
        return float(
            delta_s
            - float(self.chi_s) * int(qubit_level)
            + float(self.chi_sr) * int(readout_level)
            + float(self.kerr_s) * int(storage_level)
        )

    def readout_transition_frequency(
        self,
        qubit_level: int = 0,
        storage_level: int = 0,
        readout_level: int = 0,
        frame: FrameSpec | None = None,
    ) -> float:
        frame = frame or FrameSpec()
        delta_r = float(self.omega_r - frame.omega_r_frame)
        return float(
            delta_r
            - float(self.chi_r) * int(qubit_level)
            + float(self.chi_sr) * int(storage_level)
            + float(self.kerr_r) * int(readout_level)
        )
