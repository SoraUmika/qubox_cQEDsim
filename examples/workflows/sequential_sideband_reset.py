from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
import qutip as qt

from physics_and_conventions.conventions import to_internal_units

from cqed_sim.core import (
    DispersiveReadoutTransmonStorageModel,
    FrameSpec,
    SidebandDriveSpec,
    TransmonTransitionDriveSpec,
    carrier_for_transition_frequency,
)
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import CompiledSequence, SequenceCompiler
from cqed_sim.sim.extractors import readout_photon_number, storage_photon_number, transmon_level_populations
from cqed_sim.sim.noise import NoiseSpec, pure_dephasing_time_from_t1_t2
from cqed_sim.sim.runner import SimulationConfig, SimulationResult, simulate_sequence


@dataclass(frozen=True)
class SequentialSidebandResetDevice:
    readout_frequency_hz: float
    qubit_frequency_hz: float
    storage_frequency_hz: float
    readout_kappa_hz: float
    qubit_anharmonicity_hz: float
    chi_storage_hz: float
    chi_readout_hz: float
    storage_gf_sideband_frequency_hz: float | None = None
    storage_t1_s: float | None = None
    storage_t2_ramsey_s: float | None = None
    chi_storage_readout_hz: float = 0.0
    storage_kerr_hz: float = 0.0
    readout_kerr_hz: float = 0.0


@dataclass(frozen=True)
class SequentialSidebandResetCalibration:
    storage_sideband_rate_hz: float
    readout_sideband_rate_hz: float
    ef_rate_hz: float
    ringdown_multiple: float = 5.0


@dataclass(frozen=True)
class ResetStageRecord:
    cycle_index: int
    stage: str
    target_storage_level: int | None
    duration_s: float
    frequency_rad_s: float | None
    amplitude_rad_s: float | None
    compiled: CompiledSequence
    simulation: SimulationResult


@dataclass
class SequentialResetResult:
    initial_state: qt.Qobj
    final_state: qt.Qobj
    stage_records: list[ResetStageRecord]
    cycle_final_storage_photon_number: np.ndarray
    cycle_final_readout_photon_number: np.ndarray
    cycle_final_transmon_populations: list[dict[int, float]]


def build_sideband_reset_model(
    device: SequentialSidebandResetDevice,
    *,
    n_storage: int,
    n_readout: int,
    n_transmon: int = 3,
) -> DispersiveReadoutTransmonStorageModel:
    return DispersiveReadoutTransmonStorageModel(
        omega_s=to_internal_units(float(device.storage_frequency_hz)),
        omega_r=to_internal_units(float(device.readout_frequency_hz)),
        omega_q=to_internal_units(float(device.qubit_frequency_hz)),
        alpha=to_internal_units(float(device.qubit_anharmonicity_hz)),
        chi_s=to_internal_units(float(device.chi_storage_hz)),
        chi_r=to_internal_units(float(device.chi_readout_hz)),
        chi_sr=to_internal_units(float(device.chi_storage_readout_hz)),
        kerr_s=to_internal_units(float(device.storage_kerr_hz)),
        kerr_r=to_internal_units(float(device.readout_kerr_hz)),
        n_storage=int(n_storage),
        n_readout=int(n_readout),
        n_tr=int(n_transmon),
    )


def build_sideband_reset_frame(model: DispersiveReadoutTransmonStorageModel) -> FrameSpec:
    return FrameSpec(
        omega_c_frame=model.omega_s,
        omega_q_frame=model.omega_q,
        omega_r_frame=model.omega_r,
    )


def build_sideband_reset_noise(device: SequentialSidebandResetDevice) -> NoiseSpec | None:
    storage_tphi_s = pure_dephasing_time_from_t1_t2(t1_s=device.storage_t1_s, t2_s=device.storage_t2_ramsey_s)
    kappa_storage = None if device.storage_t1_s is None or device.storage_t1_s <= 0.0 else 1.0 / float(device.storage_t1_s)
    kappa_readout = None if device.readout_kappa_hz <= 0.0 else float(device.readout_kappa_hz)
    if kappa_storage is None and kappa_readout is None and storage_tphi_s is None:
        return None
    return NoiseSpec(
        kappa_storage=kappa_storage,
        kappa_readout=kappa_readout,
        tphi_storage=storage_tphi_s,
    )


def _square_envelope(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def _compile_single_pulse(pulse: Pulse | None, *, duration_s: float, dt_s: float) -> CompiledSequence:
    pulses = [] if pulse is None else [pulse]
    t_end = max(float(duration_s), float(dt_s))
    return SequenceCompiler(dt=float(dt_s)).compile(pulses, t_end=t_end)


def _simulate_transition_pulse(
    model: DispersiveReadoutTransmonStorageModel,
    initial_state: qt.Qobj,
    *,
    drive_target: TransmonTransitionDriveSpec | SidebandDriveSpec,
    frequency_rad_s: float,
    amplitude_rad_s: float,
    duration_s: float,
    dt_s: float,
    channel: str,
    frame: FrameSpec,
    noise: NoiseSpec | None,
) -> tuple[CompiledSequence, SimulationResult]:
    pulse = Pulse(
        channel,
        0.0,
        float(duration_s),
        _square_envelope,
        amp=float(amplitude_rad_s),
        carrier=carrier_for_transition_frequency(float(frequency_rad_s)),
    )
    compiled = _compile_single_pulse(pulse, duration_s=duration_s, dt_s=dt_s)
    result = simulate_sequence(
        model,
        compiled,
        initial_state,
        {channel: drive_target},
        SimulationConfig(frame=frame, store_states=True, max_step=dt_s),
        noise=noise,
    )
    return compiled, result


def _simulate_idle(
    model: DispersiveReadoutTransmonStorageModel,
    initial_state: qt.Qobj,
    *,
    duration_s: float,
    dt_s: float,
    frame: FrameSpec,
    noise: NoiseSpec | None,
) -> tuple[CompiledSequence, SimulationResult]:
    compiled = _compile_single_pulse(None, duration_s=duration_s, dt_s=dt_s)
    result = simulate_sequence(
        model,
        compiled,
        initial_state,
        {},
        SimulationConfig(frame=frame, store_states=True, max_step=dt_s),
        noise=noise,
    )
    return compiled, result


def _sideband_pi_time(*, amplitude_rad_s: float, initial_mode_level: int) -> float:
    if amplitude_rad_s <= 0.0:
        raise ValueError("amplitude_rad_s must be positive.")
    if initial_mode_level < 0:
        raise ValueError("initial_mode_level must be non-negative.")
    matrix_element = math.sqrt(float(initial_mode_level + 1))
    return math.pi / (2.0 * float(amplitude_rad_s) * matrix_element)


def run_sequential_sideband_reset(
    model: DispersiveReadoutTransmonStorageModel,
    initial_state: qt.Qobj,
    *,
    calibration: SequentialSidebandResetCalibration,
    initial_storage_level: int,
    include_initial_ef_prep: bool = False,
    pulse_dt_s: float = 0.25e-9,
    ringdown_dt_s: float = 4.0e-9,
    frame: FrameSpec | None = None,
    noise: NoiseSpec | None = None,
    max_cycles: int | None = None,
) -> SequentialResetResult:
    frame = build_sideband_reset_frame(model) if frame is None else frame
    state = initial_state
    stage_records: list[ResetStageRecord] = []
    cycle_storage: list[float] = []
    cycle_readout: list[float] = []
    cycle_transmon: list[dict[int, float]] = []
    storage_rate = to_internal_units(float(calibration.storage_sideband_rate_hz))
    readout_rate = to_internal_units(float(calibration.readout_sideband_rate_hz))
    total_cycles = int(initial_storage_level) if max_cycles is None else min(int(initial_storage_level), int(max_cycles))
    ringdown_time_s = float(calibration.ringdown_multiple) / float(noise.kappa_readout) if noise is not None and noise.kappa_readout else 0.0

    if include_initial_ef_prep:
        ef_frequency = model.transmon_transition_frequency(
            storage_level=int(initial_storage_level),
            readout_level=0,
            lower_level=1,
            upper_level=2,
            frame=frame,
        )
        ef_duration = math.pi / (2.0 * to_internal_units(float(calibration.ef_rate_hz)))
        compiled, result = _simulate_transition_pulse(
            model,
            state,
            drive_target=TransmonTransitionDriveSpec(lower_level=1, upper_level=2),
            frequency_rad_s=ef_frequency,
            amplitude_rad_s=to_internal_units(float(calibration.ef_rate_hz)),
            duration_s=ef_duration,
            dt_s=pulse_dt_s,
            channel="ef",
            frame=frame,
            noise=noise,
        )
        stage_records.append(
            ResetStageRecord(
                cycle_index=0,
                stage="ef_prepare",
                target_storage_level=int(initial_storage_level),
                duration_s=ef_duration,
                frequency_rad_s=ef_frequency,
                amplitude_rad_s=to_internal_units(float(calibration.ef_rate_hz)),
                compiled=compiled,
                simulation=result,
            )
        )
        state = result.final_state

        dump_frequency = model.sideband_transition_frequency(
            mode="readout",
            storage_level=int(initial_storage_level),
            readout_level=0,
            lower_level=0,
            upper_level=2,
            frame=frame,
        )
        dump_duration = _sideband_pi_time(amplitude_rad_s=readout_rate, initial_mode_level=0)
        compiled_dump, result_dump = _simulate_transition_pulse(
            model,
            state,
            drive_target=SidebandDriveSpec(mode="readout", lower_level=0, upper_level=2, sideband="red"),
            frequency_rad_s=dump_frequency,
            amplitude_rad_s=readout_rate,
            duration_s=dump_duration,
            dt_s=pulse_dt_s,
            channel="sb_readout_dump",
            frame=frame,
            noise=noise,
        )
        stage_records.append(
            ResetStageRecord(
                cycle_index=0,
                stage="ef_dump_readout_sideband",
                target_storage_level=int(initial_storage_level),
                duration_s=dump_duration,
                frequency_rad_s=dump_frequency,
                amplitude_rad_s=readout_rate,
                compiled=compiled_dump,
                simulation=result_dump,
            )
        )
        state = result_dump.final_state

        if ringdown_time_s > 0.0:
            compiled_dump_ringdown, result_dump_ringdown = _simulate_idle(
                model,
                state,
                duration_s=ringdown_time_s,
                dt_s=ringdown_dt_s,
                frame=frame,
                noise=noise,
            )
            stage_records.append(
                ResetStageRecord(
                    cycle_index=0,
                    stage="ef_dump_ringdown",
                    target_storage_level=int(initial_storage_level),
                    duration_s=ringdown_time_s,
                    frequency_rad_s=None,
                    amplitude_rad_s=None,
                    compiled=compiled_dump_ringdown,
                    simulation=result_dump_ringdown,
                )
            )
            state = result_dump_ringdown.final_state

    for cycle in range(total_cycles):
        target_storage_level = int(initial_storage_level) - cycle - 1
        current_storage_occupancy = target_storage_level

        storage_frequency = model.sideband_transition_frequency(
            mode="storage",
            storage_level=target_storage_level,
            readout_level=0,
            lower_level=0,
            upper_level=2,
            frame=frame,
        )
        storage_duration = _sideband_pi_time(amplitude_rad_s=storage_rate, initial_mode_level=target_storage_level)
        compiled_storage, result_storage = _simulate_transition_pulse(
            model,
            state,
            drive_target=SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2, sideband="red"),
            frequency_rad_s=storage_frequency,
            amplitude_rad_s=storage_rate,
            duration_s=storage_duration,
            dt_s=pulse_dt_s,
            channel="sb_storage",
            frame=frame,
            noise=noise,
        )
        stage_records.append(
            ResetStageRecord(
                cycle_index=cycle + 1,
                stage="storage_sideband",
                target_storage_level=target_storage_level,
                duration_s=storage_duration,
                frequency_rad_s=storage_frequency,
                amplitude_rad_s=storage_rate,
                compiled=compiled_storage,
                simulation=result_storage,
            )
        )
        state = result_storage.final_state

        readout_frequency = model.sideband_transition_frequency(
            mode="readout",
            storage_level=target_storage_level,
            readout_level=0,
            lower_level=0,
            upper_level=2,
            frame=frame,
        )
        readout_duration = _sideband_pi_time(amplitude_rad_s=readout_rate, initial_mode_level=0)
        compiled_readout, result_readout = _simulate_transition_pulse(
            model,
            state,
            drive_target=SidebandDriveSpec(mode="readout", lower_level=0, upper_level=2, sideband="red"),
            frequency_rad_s=readout_frequency,
            amplitude_rad_s=readout_rate,
            duration_s=readout_duration,
            dt_s=pulse_dt_s,
            channel="sb_readout",
            frame=frame,
            noise=noise,
        )
        stage_records.append(
            ResetStageRecord(
                cycle_index=cycle + 1,
                stage="readout_sideband",
                target_storage_level=target_storage_level,
                duration_s=readout_duration,
                frequency_rad_s=readout_frequency,
                amplitude_rad_s=readout_rate,
                compiled=compiled_readout,
                simulation=result_readout,
            )
        )
        state = result_readout.final_state

        if ringdown_time_s > 0.0:
            compiled_ringdown, result_ringdown = _simulate_idle(
                model,
                state,
                duration_s=ringdown_time_s,
                dt_s=ringdown_dt_s,
                frame=frame,
                noise=noise,
            )
            stage_records.append(
                ResetStageRecord(
                    cycle_index=cycle + 1,
                    stage="ringdown",
                    target_storage_level=target_storage_level,
                    duration_s=ringdown_time_s,
                    frequency_rad_s=None,
                    amplitude_rad_s=None,
                    compiled=compiled_ringdown,
                    simulation=result_ringdown,
                )
            )
            state = result_ringdown.final_state

        cycle_storage.append(storage_photon_number(state))
        cycle_readout.append(readout_photon_number(state))
        cycle_transmon.append(transmon_level_populations(state))

    return SequentialResetResult(
        initial_state=initial_state,
        final_state=state,
        stage_records=stage_records,
        cycle_final_storage_photon_number=np.asarray(cycle_storage, dtype=float),
        cycle_final_readout_photon_number=np.asarray(cycle_readout, dtype=float),
        cycle_final_transmon_populations=cycle_transmon,
    )


__all__ = [
    "SequentialSidebandResetCalibration",
    "SequentialSidebandResetDevice",
    "SequentialResetResult",
    "ResetStageRecord",
    "build_sideband_reset_model",
    "build_sideband_reset_frame",
    "build_sideband_reset_noise",
    "run_sequential_sideband_reset",
]
