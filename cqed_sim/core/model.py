from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import qutip as qt

from .hamiltonian import CrossKerrSpec, ExchangeSpec, SelfKerrSpec, coupling_term_key
from .frame import FrameSpec
from .frequencies import manifold_transition_frequency
from .universal_model import BosonicModeSpec, DispersiveCouplingSpec, TransmonModeSpec, UniversalCQEDModel


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
    _delegate_cache: UniversalCQEDModel | None = field(default=None, init=False, repr=False, compare=False)
    _delegate_signature: tuple | None = field(default=None, init=False, repr=False, compare=False)

    @property
    def subsystem_dims(self) -> tuple[int, ...]:
        return self.as_universal_model().subsystem_dims

    def operators(self) -> dict[str, qt.Qobj]:
        return self.as_universal_model().operators()

    def drive_coupling_operators(self) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
        return self.as_universal_model().drive_coupling_operators()

    def transmon_level_projector(self, level: int) -> qt.Qobj:
        return self.as_universal_model().transmon_level_projector(level)

    def transmon_transition_operators(self, lower_level: int, upper_level: int) -> tuple[qt.Qobj, qt.Qobj]:
        return self.as_universal_model().transmon_transition_operators(lower_level, upper_level)

    def mode_operators(self, mode: str = "storage") -> tuple[qt.Qobj, qt.Qobj]:
        return self.as_universal_model().mode_operators(mode)

    def sideband_drive_operators(
        self,
        *,
        mode: str = "storage",
        lower_level: int = 0,
        upper_level: int = 1,
        sideband: str = "red",
    ) -> tuple[qt.Qobj, qt.Qobj]:
        return self.as_universal_model().sideband_drive_operators(
            mode=mode,
            lower_level=lower_level,
            upper_level=upper_level,
            sideband=sideband,
        )

    def static_hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        return self.as_universal_model().static_hamiltonian(frame=frame)

    def hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        return self.static_hamiltonian(frame=frame)

    def energy_spectrum(self, *, frame: FrameSpec | None = None, levels: int | None = None):
        return self.as_universal_model().energy_spectrum(frame=frame, levels=levels)

    def basis_energy(self, q_level: int, cavity_level: int, frame: FrameSpec | None = None) -> float:
        return self.as_universal_model().basis_energy(int(q_level), int(cavity_level), frame=frame)

    def basis_state(self, q_level: int, cavity_level: int) -> qt.Qobj:
        return self.as_universal_model().basis_state(int(q_level), int(cavity_level))

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
        return self.as_universal_model().transmon_transition_frequency(
            mode_levels={"storage": int(cavity_level)},
            lower_level=lower_level,
            upper_level=upper_level,
            frame=frame,
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
        return self.as_universal_model().sideband_transition_frequency(
            mode="storage",
            mode_levels={"storage": int(cavity_level)},
            lower_level=lower_level,
            upper_level=upper_level,
            sideband=sideband,
            frame=frame,
        )

    def transmon_lowering(self) -> qt.Qobj:
        return self.as_universal_model().transmon_lowering()

    def transmon_raising(self) -> qt.Qobj:
        return self.as_universal_model().transmon_raising()

    def transmon_number(self) -> qt.Qobj:
        return self.as_universal_model().transmon_number()

    def cavity_annihilation(self) -> qt.Qobj:
        return self.as_universal_model().cavity_annihilation()

    def cavity_creation(self) -> qt.Qobj:
        return self.as_universal_model().cavity_creation()

    def cavity_number(self) -> qt.Qobj:
        return self.as_universal_model().cavity_number()

    def as_universal_model(self) -> UniversalCQEDModel:
        signature = (
            float(self.omega_c),
            float(self.omega_q),
            float(self.alpha),
            float(self.chi),
            tuple(float(value) for value in self.chi_higher),
            float(self.kerr),
            tuple(float(value) for value in self.kerr_higher),
            int(self.n_cav),
            int(self.n_tr),
            coupling_term_key(self.cross_kerr_terms, self.self_kerr_terms, self.exchange_terms),
        )
        if self._delegate_signature != signature or self._delegate_cache is None:
            self._delegate_signature = signature
            self._delegate_cache = UniversalCQEDModel(
                transmon=TransmonModeSpec(
                    omega=float(self.omega_q),
                    dim=int(self.n_tr),
                    alpha=float(self.alpha),
                    label="qubit",
                    aliases=("qubit", "transmon"),
                    frame_channel="q",
                ),
                bosonic_modes=(
                    BosonicModeSpec(
                        label="storage",
                        omega=float(self.omega_c),
                        dim=int(self.n_cav),
                        kerr=float(self.kerr),
                        kerr_higher=tuple(float(value) for value in self.kerr_higher),
                        aliases=("storage", "cavity"),
                        frame_channel="c",
                    ),
                ),
                dispersive_couplings=(
                    DispersiveCouplingSpec(
                        mode="storage",
                        chi=float(self.chi),
                        chi_higher=tuple(float(value) for value in self.chi_higher),
                        transmon="qubit",
                    ),
                ),
                cross_kerr_terms=self.cross_kerr_terms,
                self_kerr_terms=self.self_kerr_terms,
                exchange_terms=self.exchange_terms,
            )
        return self._delegate_cache
