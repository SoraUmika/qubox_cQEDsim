from __future__ import annotations

from dataclasses import dataclass, field

import qutip as qt

from .frame import FrameSpec
from .hamiltonian import CrossKerrSpec, ExchangeSpec, SelfKerrSpec, coupling_term_key
from .universal_model import BosonicModeSpec, DispersiveCouplingSpec, TransmonModeSpec, UniversalCQEDModel


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

    def basis_energy(
        self,
        q_level: int,
        storage_level: int,
        readout_level: int,
        frame: FrameSpec | None = None,
    ) -> float:
        return self.as_universal_model().basis_energy(int(q_level), int(storage_level), int(readout_level), frame=frame)

    def basis_state(self, q_level: int, storage_level: int, readout_level: int) -> qt.Qobj:
        return self.as_universal_model().basis_state(int(q_level), int(storage_level), int(readout_level))

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
        return self.transmon_transition_frequency(
            storage_level=storage_level,
            readout_level=readout_level,
            lower_level=qubit_level,
            upper_level=int(qubit_level) + 1,
            frame=frame,
        )

    def transmon_transition_frequency(
        self,
        *,
        storage_level: int = 0,
        readout_level: int = 0,
        lower_level: int = 0,
        upper_level: int = 1,
        frame: FrameSpec | None = None,
    ) -> float:
        return self.as_universal_model().transmon_transition_frequency(
            mode_levels={"storage": int(storage_level), "readout": int(readout_level)},
            lower_level=lower_level,
            upper_level=upper_level,
            frame=frame,
        )

    def sideband_transition_frequency(
        self,
        *,
        mode: str = "storage",
        storage_level: int = 0,
        readout_level: int = 0,
        lower_level: int = 0,
        upper_level: int = 1,
        sideband: str = "red",
        frame: FrameSpec | None = None,
    ) -> float:
        return self.as_universal_model().sideband_transition_frequency(
            mode=mode,
            mode_levels={"storage": int(storage_level), "readout": int(readout_level)},
            lower_level=lower_level,
            upper_level=upper_level,
            sideband=sideband,
            frame=frame,
        )

    def storage_transition_frequency(
        self,
        qubit_level: int = 0,
        storage_level: int = 0,
        readout_level: int = 0,
        frame: FrameSpec | None = None,
    ) -> float:
        return self.as_universal_model().mode_transition_frequency(
            "storage",
            mode_levels={"storage": int(storage_level), "readout": int(readout_level)},
            transmon_level=qubit_level,
            frame=frame,
        )

    def readout_transition_frequency(
        self,
        qubit_level: int = 0,
        storage_level: int = 0,
        readout_level: int = 0,
        frame: FrameSpec | None = None,
    ) -> float:
        return self.as_universal_model().mode_transition_frequency(
            "readout",
            mode_levels={"storage": int(storage_level), "readout": int(readout_level)},
            transmon_level=qubit_level,
            frame=frame,
        )

    def transmon_lowering(self) -> qt.Qobj:
        return self.as_universal_model().transmon_lowering()

    def transmon_raising(self) -> qt.Qobj:
        return self.as_universal_model().transmon_raising()

    def transmon_number(self) -> qt.Qobj:
        return self.as_universal_model().transmon_number()

    def storage_annihilation(self) -> qt.Qobj:
        return self.as_universal_model().storage_annihilation()

    def storage_creation(self) -> qt.Qobj:
        return self.as_universal_model().storage_creation()

    def storage_number(self) -> qt.Qobj:
        return self.as_universal_model().storage_number()

    def readout_annihilation(self) -> qt.Qobj:
        return self.as_universal_model().readout_annihilation()

    def readout_creation(self) -> qt.Qobj:
        return self.as_universal_model().readout_creation()

    def readout_number(self) -> qt.Qobj:
        return self.as_universal_model().readout_number()

    def as_universal_model(self) -> UniversalCQEDModel:
        signature = (
            float(self.omega_s),
            float(self.omega_r),
            float(self.omega_q),
            float(self.alpha),
            float(self.chi_s),
            float(self.chi_r),
            float(self.chi_sr),
            float(self.kerr_s),
            float(self.kerr_r),
            int(self.n_storage),
            int(self.n_readout),
            int(self.n_tr),
            coupling_term_key(self.cross_kerr_terms, self.self_kerr_terms, self.exchange_terms),
        )
        if self._delegate_signature != signature or self._delegate_cache is None:
            self._delegate_signature = signature
            base_cross_kerr_terms = ()
            if float(self.chi_sr) != 0.0:
                base_cross_kerr_terms = (CrossKerrSpec("storage", "readout", float(self.chi_sr)),)
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
                        omega=float(self.omega_s),
                        dim=int(self.n_storage),
                        kerr=float(self.kerr_s),
                        aliases=("storage", "cavity"),
                        frame_channel="c",
                    ),
                    BosonicModeSpec(
                        label="readout",
                        omega=float(self.omega_r),
                        dim=int(self.n_readout),
                        kerr=float(self.kerr_r),
                        aliases=("readout",),
                        frame_channel="r",
                    ),
                ),
                dispersive_couplings=(
                    DispersiveCouplingSpec(mode="storage", chi=float(self.chi_s), transmon="qubit"),
                    DispersiveCouplingSpec(mode="readout", chi=float(self.chi_r), transmon="qubit"),
                ),
                cross_kerr_terms=base_cross_kerr_terms + tuple(self.cross_kerr_terms),
                self_kerr_terms=self.self_kerr_terms,
                exchange_terms=self.exchange_terms,
            )
        return self._delegate_cache
