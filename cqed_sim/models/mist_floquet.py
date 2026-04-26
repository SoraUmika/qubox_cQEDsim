from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from cqed_sim.models.transmon import TransmonCosineSpec, diagonalize_transmon


@dataclass(frozen=True)
class MISTScanConfig:
    EJ: float
    EC: float
    n_cut: int
    levels: int
    drive_amplitudes: Sequence[float]
    drive_frequencies: Sequence[float]
    offset_charges: Sequence[float] = (0.0,)
    max_multiphoton_order: int = 6
    computational_levels: tuple[int, int] = (0, 1)
    detuning_width: float | None = None
    coupling_floor: float = 1.0e-30
    include_computational_transitions: bool = True
    resonance_storage_threshold: float = 1.0e-4


@dataclass
class MISTResonance:
    ng: float
    amplitude: float
    frequency: float
    initial_level: int
    target_level: int
    photon_order: int
    detuning: float
    strength: float


@dataclass
class MISTScanResult:
    config: MISTScanConfig
    penalty_grid: np.ndarray
    resonances: list[MISTResonance]
    drive_amplitudes: np.ndarray
    drive_frequencies: np.ndarray
    offset_charges: np.ndarray

    def penalty(self, amplitude: float, frequency: float, ng: float | None = None) -> float:
        ng_values = self.offset_charges
        ng_index = int(np.argmin(np.abs(ng_values - float(ng_values[0] if ng is None else ng))))
        amp_index = int(np.argmin(np.abs(self.drive_amplitudes - float(amplitude))))
        freq_index = int(np.argmin(np.abs(self.drive_frequencies - float(frequency))))
        return float(self.penalty_grid[ng_index, amp_index, freq_index])


def _default_width(frequencies: np.ndarray) -> float:
    if frequencies.size <= 1:
        return max(abs(float(frequencies[0])) * 1.0e-3, 1.0)
    diffs = np.diff(np.sort(frequencies))
    positive = diffs[diffs > 0.0]
    return float(max(np.min(positive), 1.0e-12))


def scan_mist(config: MISTScanConfig) -> MISTScanResult:
    """Semiclassical MIST scanner using a charge-driven transmon model.

    The scanner estimates high-risk drive regions from multiphoton resonance
    proximity in the cosine transmon spectrum.  It is intentionally a penalty
    map for optimization, not a replacement for a full driven-dissipative
    validation simulation.
    """

    amplitudes = np.asarray(config.drive_amplitudes, dtype=float)
    frequencies = np.asarray(config.drive_frequencies, dtype=float)
    ng_values = np.asarray(config.offset_charges, dtype=float)
    if amplitudes.ndim != 1 or frequencies.ndim != 1 or ng_values.ndim != 1:
        raise ValueError("MIST scan axes must be one-dimensional.")
    width = _default_width(frequencies) if config.detuning_width is None else float(config.detuning_width)
    width = max(width, 1.0e-30)
    grid = np.zeros((ng_values.size, amplitudes.size, frequencies.size), dtype=float)
    resonances: list[MISTResonance] = []

    for ng_index, ng in enumerate(ng_values):
        spectrum = diagonalize_transmon(
            TransmonCosineSpec(
                EJ=float(config.EJ),
                EC=float(config.EC),
                ng=float(ng),
                n_cut=int(config.n_cut),
                levels=int(config.levels),
            )
        )
        energies = spectrum.shifted_energies
        n_matrix = np.asarray(spectrum.n_matrix, dtype=np.complex128)
        for amp_index, amplitude in enumerate(amplitudes):
            for freq_index, frequency in enumerate(frequencies):
                local_penalty = 0.0
                for initial in config.computational_levels:
                    for target in range(config.levels):
                        if target == int(initial):
                            continue
                        if target in config.computational_levels and not config.include_computational_transitions:
                            continue
                        transition = abs(float(energies[target] - energies[int(initial)]))
                        matrix_element = abs(n_matrix[target, int(initial)])
                        if matrix_element <= config.coupling_floor:
                            continue
                        for order in range(1, int(config.max_multiphoton_order) + 1):
                            detuning = transition - order * float(frequency)
                            resonance_weight = 1.0 / (1.0 + (detuning / width) ** 2)
                            drive_ratio = abs(float(amplitude)) * matrix_element / max(abs(transition), width, 1.0)
                            strength = (drive_ratio ** (2 * order)) / float(math.factorial(order) ** 2)
                            contribution = float(resonance_weight * strength / max(order, 1))
                            local_penalty += contribution
                            if contribution > float(config.resonance_storage_threshold):
                                resonances.append(
                                    MISTResonance(
                                        ng=float(ng),
                                        amplitude=float(amplitude),
                                        frequency=float(frequency),
                                        initial_level=int(initial),
                                        target_level=int(target),
                                        photon_order=int(order),
                                        detuning=float(detuning),
                                        strength=float(contribution),
                                    )
                                )
                grid[ng_index, amp_index, freq_index] = local_penalty
    return MISTScanResult(
        config=config,
        penalty_grid=grid,
        resonances=resonances,
        drive_amplitudes=amplitudes,
        drive_frequencies=frequencies,
        offset_charges=ng_values,
    )


def mist_penalty(
    amplitude: float,
    frequency: float,
    *,
    scan: MISTScanResult,
    ng: float | None = None,
) -> float:
    return scan.penalty(amplitude, frequency, ng=ng)


__all__ = [
    "MISTResonance",
    "MISTScanConfig",
    "MISTScanResult",
    "mist_penalty",
    "scan_mist",
]
