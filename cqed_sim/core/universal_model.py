from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import numpy as np
import qutip as qt

from .frame import FrameSpec
from .hamiltonian import CrossKerrSpec, ExchangeSpec, SelfKerrSpec, assemble_static_hamiltonian, coupling_term_key


def _dedupe_labels(*labels: str) -> tuple[str, ...]:
    out: list[str] = []
    for label in labels:
        text = str(label).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _falling_factorial_number_op(n_op: qt.Qobj, order: int) -> qt.Qobj:
    if int(order) <= 0:
        return 0 * n_op + qt.qeye(n_op.dims[0])
    out = 0 * n_op + qt.qeye(n_op.dims[0])
    for k in range(int(order)):
        out = out * (n_op - float(k) * qt.qeye(n_op.dims[0]))
    return out


def _falling_factorial_scalar(n: int, order: int) -> float:
    out = 1.0
    for k in range(int(order)):
        out *= float(int(n) - k)
    return out


def _bosonic_annihilation_operator(dim: int) -> qt.Qobj:
    dimension = int(dim)
    if dimension < 1:
        raise ValueError("Bosonic mode dimension must be positive.")
    if dimension == 1:
        return qt.Qobj(np.zeros((1, 1), dtype=np.complex128), dims=[[1], [1]])
    return qt.destroy(dimension)


def _frame_frequency(frame: FrameSpec, channel: str | None) -> float:
    channel_key = "" if channel is None else str(channel).strip().lower()
    if channel_key in {"", "lab", "none"}:
        return 0.0
    if channel_key in {"c", "storage", "cavity", "omega_c_frame", "omega_s_frame"}:
        return float(frame.omega_c_frame)
    if channel_key in {"q", "qubit", "transmon", "omega_q_frame"}:
        return float(frame.omega_q_frame)
    if channel_key in {"r", "readout", "omega_r_frame"}:
        return float(frame.omega_r_frame)
    raise ValueError(f"Unsupported frame channel '{channel}'.")


def _set_operator_alias(operators: dict[str, qt.Qobj], key: str, value: qt.Qobj) -> None:
    existing = operators.get(key)
    if existing is not None:
        if existing is value:
            return
        if existing.dims == value.dims and (existing - value).norm() < 1.0e-12:
            return
        raise ValueError(f"Operator alias '{key}' resolves to multiple incompatible operators.")
    operators[key] = value


@dataclass(frozen=True)
class TransmonModeSpec:
    omega: float
    dim: int = 3
    alpha: float = 0.0
    label: str = "qubit"
    aliases: Sequence[str] = field(default_factory=lambda: ("qubit", "transmon"))
    frame_channel: str = "q"

    def __post_init__(self) -> None:
        if int(self.dim) < 2:
            raise ValueError("TransmonModeSpec.dim must be at least 2.")
        object.__setattr__(self, "aliases", _dedupe_labels(self.label, *self.aliases, "qubit", "transmon"))


@dataclass(frozen=True)
class BosonicModeSpec:
    label: str
    omega: float
    dim: int
    kerr: float = 0.0
    kerr_higher: Sequence[float] = field(default_factory=tuple)
    aliases: Sequence[str] = field(default_factory=tuple)
    frame_channel: str = "c"

    def __post_init__(self) -> None:
        if int(self.dim) < 1:
            raise ValueError("BosonicModeSpec.dim must be positive.")
        object.__setattr__(self, "kerr_higher", tuple(float(value) for value in self.kerr_higher))
        object.__setattr__(self, "aliases", _dedupe_labels(self.label, *self.aliases))


@dataclass(frozen=True)
class DispersiveCouplingSpec:
    mode: str
    chi: float = 0.0
    chi_higher: Sequence[float] = field(default_factory=tuple)
    transmon: str = "qubit"

    def __post_init__(self) -> None:
        object.__setattr__(self, "chi_higher", tuple(float(value) for value in self.chi_higher))


@dataclass
class UniversalCQEDModel:
    transmon: TransmonModeSpec | None = None
    bosonic_modes: Sequence[BosonicModeSpec] = field(default_factory=tuple)
    dispersive_couplings: Sequence[DispersiveCouplingSpec] = field(default_factory=tuple)
    cross_kerr_terms: Sequence[CrossKerrSpec] = field(default_factory=tuple)
    self_kerr_terms: Sequence[SelfKerrSpec] = field(default_factory=tuple)
    exchange_terms: Sequence[ExchangeSpec] = field(default_factory=tuple)

    subsystem_labels: tuple[str, ...] = field(init=False)
    _operators_cache: dict[str, qt.Qobj] | None = field(default=None, init=False, repr=False, compare=False)
    _structure_signature_cache: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _static_h_cache: dict[tuple[tuple, float, float, float], qt.Qobj] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.bosonic_modes = tuple(self.bosonic_modes)
        self.dispersive_couplings = tuple(self.dispersive_couplings)
        self.cross_kerr_terms = tuple(self.cross_kerr_terms)
        self.self_kerr_terms = tuple(self.self_kerr_terms)
        self.exchange_terms = tuple(self.exchange_terms)

        labels: list[str] = []
        aliases: dict[str, str] = {}

        if self.transmon is not None:
            labels.append(self.transmon.label)
            for alias in self.transmon.aliases:
                previous = aliases.setdefault(alias, self.transmon.label)
                if previous != self.transmon.label:
                    raise ValueError(f"Alias '{alias}' collides between subsystems '{previous}' and '{self.transmon.label}'.")

        for mode in self.bosonic_modes:
            labels.append(mode.label)
            for alias in mode.aliases:
                previous = aliases.setdefault(alias, mode.label)
                if previous != mode.label:
                    raise ValueError(f"Alias '{alias}' collides between subsystems '{previous}' and '{mode.label}'.")

        if not labels:
            raise ValueError("UniversalCQEDModel requires at least one subsystem.")

        self.subsystem_labels = tuple(labels)
        self._validate_coupling_labels()

    @property
    def subsystem_dims(self) -> tuple[int, ...]:
        dims: list[int] = []
        if self.transmon is not None:
            dims.append(int(self.transmon.dim))
        dims.extend(int(mode.dim) for mode in self.bosonic_modes)
        return tuple(dims)

    @property
    def has_transmon(self) -> bool:
        return self.transmon is not None

    def _structure_signature(self) -> tuple:
        return (
            self.transmon,
            tuple(self.bosonic_modes),
            tuple(self.dispersive_couplings),
            coupling_term_key(self.cross_kerr_terms, self.self_kerr_terms, self.exchange_terms),
        )

    def _invalidate_caches_if_needed(self) -> tuple:
        signature = self._structure_signature()
        if self._structure_signature_cache != signature:
            self._structure_signature_cache = signature
            self._operators_cache = None
            self._static_h_cache.clear()
        return signature

    def _validate_coupling_labels(self) -> None:
        for coupling in self.dispersive_couplings:
            self._resolve_bosonic_mode(coupling.mode)
            self._resolve_transmon_label(coupling.transmon)

    def _resolve_transmon_label(self, label: str) -> str:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        text = str(label).strip()
        if text in self.transmon.aliases:
            return self.transmon.label
        raise ValueError(f"Unknown transmon label '{label}'.")

    def _resolve_bosonic_mode(self, label: str) -> BosonicModeSpec:
        text = str(label).strip()
        for mode in self.bosonic_modes:
            if text in mode.aliases:
                return mode
        raise ValueError(f"Unknown bosonic mode '{label}'.")

    def _transmon_operator_aliases(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        assert self.transmon is not None
        lowering = _dedupe_labels("b", *self.transmon.aliases)
        raising = _dedupe_labels("bdag", *(f"{alias}_dag" for alias in self.transmon.aliases))
        number = _dedupe_labels("n_q", *(f"n_{alias}" for alias in self.transmon.aliases))
        return lowering, raising, number

    def _bosonic_operator_aliases(
        self,
        mode: BosonicModeSpec,
        *,
        mode_index: int,
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        lowering = list(_dedupe_labels(mode.label, *mode.aliases))
        raising = list(_dedupe_labels(f"{mode.label}_dag", *(f"{alias}_dag" for alias in mode.aliases)))
        number = list(_dedupe_labels(f"n_{mode.label}", *(f"n_{alias}" for alias in mode.aliases)))

        alias_set = set(mode.aliases)
        if "storage" in alias_set and len(self.bosonic_modes) > 1:
            lowering.append("a_s")
            raising.append("adag_s")
            number.append("n_s")
        if "readout" in alias_set:
            lowering.append("a_r")
            raising.append("adag_r")
            number.append("n_r")
        if mode_index == 0 and len(self.bosonic_modes) == 1:
            lowering.append("a")
            raising.append("adag")
            number.append("n_c")
        return _dedupe_labels(*lowering), _dedupe_labels(*raising), _dedupe_labels(*number)

    def _embed_local_operator(self, local_op: qt.Qobj, subsystem_index: int) -> qt.Qobj:
        factors = [qt.qeye(dim) for dim in self.subsystem_dims]
        factors[int(subsystem_index)] = local_op
        return qt.tensor(*factors)

    def operators(self) -> dict[str, qt.Qobj]:
        self._invalidate_caches_if_needed()
        if self._operators_cache is not None:
            return self._operators_cache

        operators: dict[str, qt.Qobj] = {}
        next_index = 0

        if self.transmon is not None:
            lowering = self._embed_local_operator(qt.destroy(int(self.transmon.dim)), next_index)
            raising = lowering.dag()
            number = raising * lowering
            lowering_aliases, raising_aliases, number_aliases = self._transmon_operator_aliases()
            for key in lowering_aliases:
                _set_operator_alias(operators, key, lowering)
            for key in raising_aliases:
                _set_operator_alias(operators, key, raising)
            for key in number_aliases:
                _set_operator_alias(operators, key, number)
            next_index += 1

        for mode_index, mode in enumerate(self.bosonic_modes):
            lowering = self._embed_local_operator(_bosonic_annihilation_operator(int(mode.dim)), next_index + mode_index)
            raising = lowering.dag()
            number = raising * lowering
            lowering_aliases, raising_aliases, number_aliases = self._bosonic_operator_aliases(mode, mode_index=mode_index)
            for key in lowering_aliases:
                _set_operator_alias(operators, key, lowering)
            for key in raising_aliases:
                _set_operator_alias(operators, key, raising)
            for key in number_aliases:
                _set_operator_alias(operators, key, number)

        self._operators_cache = operators
        return operators

    def transmon_level_projector(self, level: int) -> qt.Qobj:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        projector = qt.basis(self.transmon.dim, int(level)) * qt.basis(self.transmon.dim, int(level)).dag()
        return self._embed_local_operator(projector, 0)

    def transmon_transition_operators(self, lower_level: int, upper_level: int) -> tuple[qt.Qobj, qt.Qobj]:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        transition_up = qt.basis(self.transmon.dim, int(upper_level)) * qt.basis(self.transmon.dim, int(lower_level)).dag()
        transition_down = transition_up.dag()
        return (
            self._embed_local_operator(transition_up, 0),
            self._embed_local_operator(transition_down, 0),
        )

    def mode_operators(self, mode: str) -> tuple[qt.Qobj, qt.Qobj]:
        resolved = self._resolve_bosonic_mode(mode)
        ops = self.operators()
        return ops[resolved.label], ops[f"{resolved.label}_dag"]

    def transmon_lowering(self) -> qt.Qobj:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        return self.operators()["b"]

    def transmon_raising(self) -> qt.Qobj:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        return self.operators()["bdag"]

    def transmon_number(self) -> qt.Qobj:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        return self.operators()["n_q"]

    def mode_annihilation(self, mode: str) -> qt.Qobj:
        return self.mode_operators(mode)[0]

    def mode_creation(self, mode: str) -> qt.Qobj:
        return self.mode_operators(mode)[1]

    def mode_number(self, mode: str) -> qt.Qobj:
        resolved = self._resolve_bosonic_mode(mode)
        return self.operators()[f"n_{resolved.label}"]

    def cavity_annihilation(self) -> qt.Qobj:
        return self.mode_annihilation("cavity")

    def cavity_creation(self) -> qt.Qobj:
        return self.mode_creation("cavity")

    def cavity_number(self) -> qt.Qobj:
        return self.mode_number("cavity")

    def storage_annihilation(self) -> qt.Qobj:
        return self.mode_annihilation("storage")

    def storage_creation(self) -> qt.Qobj:
        return self.mode_creation("storage")

    def storage_number(self) -> qt.Qobj:
        return self.mode_number("storage")

    def readout_annihilation(self) -> qt.Qobj:
        return self.mode_annihilation("readout")

    def readout_creation(self) -> qt.Qobj:
        return self.mode_creation("readout")

    def readout_number(self) -> qt.Qobj:
        return self.mode_number("readout")

    def drive_coupling_operators(self) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
        ops = self.operators()
        couplings: dict[str, tuple[qt.Qobj, qt.Qobj]] = {}
        if self.transmon is not None:
            transmon_pair = (ops["bdag"], ops["b"])
            for alias in self.transmon.aliases:
                couplings[alias] = transmon_pair
            couplings["qubit"] = transmon_pair
            couplings["transmon"] = transmon_pair

        for mode in self.bosonic_modes:
            pair = (ops[f"{mode.label}_dag"], ops[mode.label])
            for alias in mode.aliases:
                couplings[alias] = pair
            couplings[mode.label] = pair

        if self.transmon is not None and self.bosonic_modes:
            try:
                couplings["sideband"] = self.sideband_drive_operators(mode=self.bosonic_modes[0].label)
            except Exception:
                pass
        return couplings

    def sideband_drive_operators(
        self,
        *,
        mode: str,
        lower_level: int = 0,
        upper_level: int = 1,
        sideband: str = "red",
    ) -> tuple[qt.Qobj, qt.Qobj]:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        mode_lowering, mode_raising = self.mode_operators(mode)
        transmon_up, transmon_down = self.transmon_transition_operators(lower_level, upper_level)
        sideband_key = str(sideband).strip().lower()
        if sideband_key == "red":
            return 1j * transmon_up * mode_lowering, -1j * transmon_down * mode_raising
        if sideband_key == "blue":
            return 1j * transmon_up * mode_raising, -1j * transmon_down * mode_lowering
        raise ValueError(f"Unsupported sideband '{sideband}'.")

    def _occupation_for_operator_label(self, label: str, levels: tuple[int, ...]) -> int | None:
        text = str(label).strip()
        if self.transmon is not None and text in {"b", *self.transmon.aliases}:
            return int(levels[0])
        boson_offset = 1 if self.transmon is not None else 0
        for idx, mode in enumerate(self.bosonic_modes):
            if text in {mode.label, *mode.aliases, f"a_{idx}", f"mode_{idx}"}:
                return int(levels[boson_offset + idx])
            if text == "a" and idx == 0 and (len(self.bosonic_modes) == 1 or {"storage", "cavity"} & set(mode.aliases)):
                return int(levels[boson_offset + idx])
            if text == "a_s" and "storage" in mode.aliases:
                return int(levels[boson_offset + idx])
            if text == "a_r" and "readout" in mode.aliases:
                return int(levels[boson_offset + idx])
        return None

    def static_hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        frame = frame or FrameSpec()
        signature = self._invalidate_caches_if_needed()
        key = (signature, float(frame.omega_c_frame), float(frame.omega_q_frame), float(frame.omega_r_frame))
        cached = self._static_h_cache.get(key)
        if cached is not None:
            return cached

        operators = self.operators()
        h = None
        boson_offset = 0

        if self.transmon is not None:
            n_q = operators["n_q"]
            b = operators["b"]
            bdag = operators["bdag"]
            delta_q = float(self.transmon.omega) - _frame_frequency(frame, self.transmon.frame_channel)
            h = delta_q * n_q
            h += 0.5 * float(self.transmon.alpha) * (bdag * bdag * b * b)
            boson_offset = 1

        for idx, mode in enumerate(self.bosonic_modes):
            number = self.mode_number(mode.label)
            delta = float(mode.omega) - _frame_frequency(frame, mode.frame_channel)
            term = delta * number
            if float(mode.kerr) != 0.0:
                term += float(mode.kerr) * _falling_factorial_number_op(number, 2) / math.factorial(2)
            for order_index, coeff in enumerate(mode.kerr_higher, start=2):
                order = order_index + 1
                term += float(coeff) * _falling_factorial_number_op(number, order) / math.factorial(order)
            h = term if h is None else h + term

        if h is None:
            h = 0 * qt.qeye(1)

        if self.transmon is not None:
            n_q = operators["n_q"]
            for coupling in self.dispersive_couplings:
                number = self.mode_number(coupling.mode)
                if float(coupling.chi) != 0.0:
                    h += float(coupling.chi) * number * n_q
                for order, coeff in enumerate(coupling.chi_higher, start=2):
                    h += float(coeff) * _falling_factorial_number_op(number, order) * n_q

        h = assemble_static_hamiltonian(
            h,
            operators,
            cross_kerr_terms=self.cross_kerr_terms,
            self_kerr_terms=self.self_kerr_terms,
            exchange_terms=self.exchange_terms,
        )
        self._static_h_cache[key] = h
        return h

    def hamiltonian(self, frame: FrameSpec | None = None) -> qt.Qobj:
        return self.static_hamiltonian(frame=frame)

    def energy_spectrum(self, *, frame: FrameSpec | None = None, levels: int | None = None):
        from .spectrum import compute_energy_spectrum

        return compute_energy_spectrum(self, frame=frame, levels=levels)

    def _normalize_basis_levels(self, levels: Sequence[int] | tuple[int, ...]) -> tuple[int, ...]:
        if len(levels) == 1 and isinstance(levels[0], (tuple, list)):  # type: ignore[index]
            levels = tuple(levels[0])  # type: ignore[assignment]
        dims = self.subsystem_dims
        if len(levels) != len(dims):
            raise ValueError(f"Expected {len(dims)} basis indices, got {len(levels)}.")
        normalized = tuple(int(level) for level in levels)
        for level, dim in zip(normalized, dims):
            if level < 0 or level >= int(dim):
                raise IndexError(f"Basis level {level} out of range for subsystem dimension {dim}.")
        return normalized

    def basis_state(self, *levels: int | Sequence[int]) -> qt.Qobj:
        normalized = self._normalize_basis_levels(levels)
        return qt.tensor(*(qt.basis(dim, level) for dim, level in zip(self.subsystem_dims, normalized)))

    def basis_energy(self, *levels: int | Sequence[int], frame: FrameSpec | None = None) -> float:
        frame = frame or FrameSpec()
        normalized = self._normalize_basis_levels(levels)
        energy = 0.0
        boson_levels: dict[str, int] = {}
        index = 0

        if self.transmon is not None:
            q_level = int(normalized[0])
            energy += (float(self.transmon.omega) - _frame_frequency(frame, self.transmon.frame_channel)) * q_level
            energy += 0.5 * float(self.transmon.alpha) * q_level * (q_level - 1)
            index = 1

        for mode_index, mode in enumerate(self.bosonic_modes):
            n_level = int(normalized[index + mode_index])
            boson_levels[mode.label] = n_level
            energy += (float(mode.omega) - _frame_frequency(frame, mode.frame_channel)) * n_level
            energy += 0.5 * float(mode.kerr) * n_level * (n_level - 1)
            for order_index, coeff in enumerate(mode.kerr_higher, start=2):
                order = order_index + 1
                energy += float(coeff) * _falling_factorial_scalar(n_level, order) / math.factorial(order)

        if self.transmon is not None:
            q_level = int(normalized[0])
            for coupling in self.dispersive_couplings:
                n_level = boson_levels[self._resolve_bosonic_mode(coupling.mode).label]
                energy += float(coupling.chi) * n_level * q_level
                for order, coeff in enumerate(coupling.chi_higher, start=2):
                    energy += float(coeff) * _falling_factorial_scalar(n_level, order) * q_level

        for spec in self.cross_kerr_terms:
            left_occ = self._occupation_for_operator_label(spec.left, normalized)
            right_occ = self._occupation_for_operator_label(spec.right, normalized)
            if left_occ is not None and right_occ is not None:
                energy += float(spec.chi) * float(left_occ) * float(right_occ)

        for spec in self.self_kerr_terms:
            occ = self._occupation_for_operator_label(spec.mode, normalized)
            if occ is not None:
                energy += 0.5 * float(spec.kerr) * float(occ) * float(occ - 1)

        return float(energy)

    def _normalize_mode_levels(self, mode_levels: Mapping[str, int] | Sequence[int] | None) -> tuple[int, ...]:
        if mode_levels is None:
            return tuple(0 for _ in self.bosonic_modes)
        if isinstance(mode_levels, Mapping):
            resolved: list[int] = []
            for mode in self.bosonic_modes:
                value = None
                for alias in mode.aliases:
                    if alias in mode_levels:
                        value = mode_levels[alias]
                        break
                resolved.append(0 if value is None else int(value))
            return tuple(resolved)
        if len(mode_levels) != len(self.bosonic_modes):
            raise ValueError(f"Expected {len(self.bosonic_modes)} bosonic mode levels, got {len(mode_levels)}.")
        return tuple(int(value) for value in mode_levels)

    def transmon_transition_frequency(
        self,
        *,
        mode_levels: Mapping[str, int] | Sequence[int] | None = None,
        lower_level: int = 0,
        upper_level: int = 1,
        frame: FrameSpec | None = None,
    ) -> float:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        boson_levels = self._normalize_mode_levels(mode_levels)
        return float(
            self.basis_energy(int(upper_level), *boson_levels, frame=frame)
            - self.basis_energy(int(lower_level), *boson_levels, frame=frame)
        )

    def mode_transition_frequency(
        self,
        mode: str,
        *,
        mode_levels: Mapping[str, int] | Sequence[int] | None = None,
        transmon_level: int = 0,
        frame: FrameSpec | None = None,
    ) -> float:
        target = self._resolve_bosonic_mode(mode)
        boson_levels = list(self._normalize_mode_levels(mode_levels))
        target_index = next(index for index, spec in enumerate(self.bosonic_modes) if spec.label == target.label)
        before = list(boson_levels)
        after = list(boson_levels)
        after[target_index] += 1
        if after[target_index] >= int(target.dim):
            raise IndexError(f"Selected transition exceeds the truncation of mode '{target.label}'.")
        if self.transmon is not None:
            return float(
                self.basis_energy(int(transmon_level), *after, frame=frame)
                - self.basis_energy(int(transmon_level), *before, frame=frame)
            )
        return float(self.basis_energy(*after, frame=frame) - self.basis_energy(*before, frame=frame))

    def sideband_transition_frequency(
        self,
        *,
        mode: str,
        mode_levels: Mapping[str, int] | Sequence[int] | None = None,
        lower_level: int = 0,
        upper_level: int = 1,
        sideband: str = "red",
        frame: FrameSpec | None = None,
    ) -> float:
        if self.transmon is None:
            raise ValueError("This model does not define a transmon subsystem.")
        target = self._resolve_bosonic_mode(mode)
        boson_levels = list(self._normalize_mode_levels(mode_levels))
        target_index = next(index for index, spec in enumerate(self.bosonic_modes) if spec.label == target.label)
        sideband_key = str(sideband).strip().lower()

        if sideband_key == "red":
            excited = list(boson_levels)
            ground = list(boson_levels)
            ground[target_index] += 1
            if ground[target_index] >= int(target.dim):
                raise IndexError(f"Selected red sideband exceeds the truncation of mode '{target.label}'.")
            return float(
                self.basis_energy(int(upper_level), *excited, frame=frame)
                - self.basis_energy(int(lower_level), *ground, frame=frame)
            )
        if sideband_key == "blue":
            excited = list(boson_levels)
            ground = list(boson_levels)
            excited[target_index] += 1
            if excited[target_index] >= int(target.dim):
                raise IndexError(f"Selected blue sideband exceeds the truncation of mode '{target.label}'.")
            return float(
                self.basis_energy(int(upper_level), *excited, frame=frame)
                - self.basis_energy(int(lower_level), *ground, frame=frame)
            )
        raise ValueError(f"Unsupported sideband '{sideband}'.")


__all__ = [
    "BosonicModeSpec",
    "DispersiveCouplingSpec",
    "TransmonModeSpec",
    "UniversalCQEDModel",
]
