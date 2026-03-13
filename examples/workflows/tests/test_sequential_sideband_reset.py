from __future__ import annotations

import math

import numpy as np
import qutip as qt

from cqed_sim import (
    Pulse,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    TransmonTransitionDriveSpec,
    carrier_for_transition_frequency,
    simulate_sequence,
    storage_photon_number,
    subsystem_level_population,
    transmon_level_populations,
)
from cqed_sim.sim import pure_dephasing_time_from_t1_t2
from examples.workflows.sequential_sideband_reset import (
    SequentialSidebandResetCalibration,
    SequentialSidebandResetDevice,
    build_sideband_reset_frame,
    build_sideband_reset_model,
    build_sideband_reset_noise,
    run_sequential_sideband_reset,
)


def _device(*, readout_kappa_hz: float = 4.156e6) -> SequentialSidebandResetDevice:
    return SequentialSidebandResetDevice(
        readout_frequency_hz=8596222556.078796,
        qubit_frequency_hz=6150358764.4830475,
        storage_frequency_hz=5240932800.0,
        readout_kappa_hz=readout_kappa_hz,
        qubit_anharmonicity_hz=-255669694.5244608,
        chi_storage_hz=-2840421.0,
        chi_readout_hz=-3.0e6,
        storage_gf_sideband_frequency_hz=6803533628.0,
        storage_t1_s=250.0e-6,
        storage_t2_ramsey_s=150.0e-6,
    )


def _calibration(*, ringdown_multiple: float = 4.0) -> SequentialSidebandResetCalibration:
    return SequentialSidebandResetCalibration(
        storage_sideband_rate_hz=8.0e6,
        readout_sideband_rate_hz=10.0e6,
        ef_rate_hz=12.0e6,
        ringdown_multiple=ringdown_multiple,
    )


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def _simulate_storage_sideband_sweep(model, frame, frequencies, *, duration_s: float, amplitude_hz: float) -> np.ndarray:
    responses = []
    initial = model.basis_state(0, 1, 0)
    target = model.basis_state(2, 0, 0)
    for frequency in frequencies:
        pulse = Pulse(
            "sb",
            0.0,
            duration_s,
            _square,
            amp=2.0 * np.pi * amplitude_hz,
            carrier=carrier_for_transition_frequency(float(frequency)),
        )
        compiled = SequenceCompiler(dt=0.25e-9).compile([pulse], t_end=duration_s)
        result = simulate_sequence(
            model,
            compiled,
            initial,
            {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)},
            SimulationConfig(frame=frame),
        )
        responses.append(abs(target.overlap(result.final_state)) ** 2)
    return np.asarray(responses, dtype=float)


def test_reset_model_basis_order_and_mode_embedding_are_correct():
    model = build_sideband_reset_model(_device(), n_storage=4, n_readout=3)

    storage_lowered = model.storage_annihilation() * model.basis_state(0, 1, 0)
    readout_lowered = model.readout_annihilation() * model.basis_state(0, 0, 1)

    assert abs(storage_lowered.overlap(model.basis_state(0, 0, 0))) > 0.999999
    assert abs(readout_lowered.overlap(model.basis_state(0, 0, 0))) > 0.999999


def test_storage_and_readout_gf_sidebands_connect_expected_states():
    model = build_sideband_reset_model(_device(), n_storage=4, n_readout=3)

    _, storage_lowering = model.sideband_drive_operators(mode="storage", lower_level=0, upper_level=2, sideband="red")
    _, readout_lowering = model.sideband_drive_operators(mode="readout", lower_level=0, upper_level=2, sideband="red")

    assert abs((storage_lowering * model.basis_state(2, 0, 0)).overlap(model.basis_state(0, 1, 0))) > 0.999999
    assert abs((readout_lowering * model.basis_state(2, 0, 0)).overlap(model.basis_state(0, 0, 1))) > 0.999999


def test_storage_sideband_spectroscopy_peak_matches_dressed_transition():
    model = build_sideband_reset_model(_device(), n_storage=4, n_readout=3)
    frame = build_sideband_reset_frame(model)
    predicted = model.sideband_transition_frequency(
        mode="storage",
        storage_level=0,
        readout_level=0,
        lower_level=0,
        upper_level=2,
        frame=frame,
    )
    scan = predicted + np.linspace(-2.0, 2.0, 21) * 2.0 * np.pi * 1.0e6
    response = _simulate_storage_sideband_sweep(model, frame, scan, duration_s=40.0e-9, amplitude_hz=1.5e6)
    peak = float(scan[int(np.argmax(response))])

    assert abs(peak - predicted) < 2.0 * np.pi * 0.35e6


def test_storage_sideband_power_rabi_peak_tracks_pi_condition():
    model = build_sideband_reset_model(_device(), n_storage=4, n_readout=3)
    frame = build_sideband_reset_frame(model)
    frequency = model.sideband_transition_frequency(
        mode="storage",
        storage_level=0,
        readout_level=0,
        lower_level=0,
        upper_level=2,
        frame=frame,
    )
    duration_s = 62.5e-9
    amplitudes_hz = np.linspace(2.0e6, 12.0e6, 21)
    responses = []
    for amplitude_hz in amplitudes_hz:
        pulse = Pulse(
            "sb",
            0.0,
            duration_s,
            _square,
            amp=2.0 * np.pi * float(amplitude_hz),
            carrier=carrier_for_transition_frequency(frequency),
        )
        compiled = SequenceCompiler(dt=0.25e-9).compile([pulse], t_end=duration_s)
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 1, 0),
            {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)},
            SimulationConfig(frame=frame),
        )
        responses.append(abs(model.basis_state(2, 0, 0).overlap(result.final_state)) ** 2)

    best = float(amplitudes_hz[int(np.argmax(responses))])
    expected = 1.0 / (4.0 * duration_s)
    assert abs(best - expected) < 0.8e6


def test_single_step_reset_reduces_storage_excitation_from_g01():
    device = _device()
    model = build_sideband_reset_model(device, n_storage=4, n_readout=3)
    result = run_sequential_sideband_reset(
        model,
        model.basis_state(0, 1, 0),
        calibration=_calibration(),
        initial_storage_level=1,
        frame=build_sideband_reset_frame(model),
        noise=build_sideband_reset_noise(device),
        pulse_dt_s=0.25e-9,
        ringdown_dt_s=4.0e-9,
    )

    assert len(result.cycle_final_storage_photon_number) == 1
    assert result.cycle_final_storage_photon_number[0] < 0.35
    assert result.cycle_final_readout_photon_number[0] < 0.2


def test_multi_step_reset_monotonically_reduces_mean_storage_photon_number():
    device = _device()
    model = build_sideband_reset_model(device, n_storage=6, n_readout=3)
    initial = model.basis_state(0, 3, 0)
    result = run_sequential_sideband_reset(
        model,
        initial,
        calibration=_calibration(ringdown_multiple=3.0),
        initial_storage_level=3,
        frame=build_sideband_reset_frame(model),
        noise=build_sideband_reset_noise(device),
        pulse_dt_s=0.25e-9,
        ringdown_dt_s=4.0e-9,
    )

    assert np.all(np.diff(result.cycle_final_storage_photon_number) <= 1.0e-6)
    assert result.cycle_final_storage_photon_number[-1] < 0.5
    assert storage_photon_number(result.final_state) < 0.5


def test_e_manifold_preparation_then_reset_removes_storage_excitation():
    device = _device()
    model = build_sideband_reset_model(device, n_storage=5, n_readout=3)
    initial = model.basis_state(1, 2, 0)
    result = run_sequential_sideband_reset(
        model,
        initial,
        calibration=_calibration(ringdown_multiple=3.0),
        initial_storage_level=2,
        include_initial_ef_prep=True,
        frame=build_sideband_reset_frame(model),
        noise=build_sideband_reset_noise(device),
        pulse_dt_s=0.25e-9,
        ringdown_dt_s=4.0e-9,
    )

    populations = transmon_level_populations(result.final_state)
    assert result.cycle_final_storage_photon_number[-1] < 0.1
    assert populations.get(1, 0.0) < 1.0e-3


def test_faster_readout_decay_improves_reset_performance():
    slow_device = _device(readout_kappa_hz=1.0e6)
    fast_device = _device(readout_kappa_hz=8.0e6)
    slow_model = build_sideband_reset_model(slow_device, n_storage=5, n_readout=3)
    fast_model = build_sideband_reset_model(fast_device, n_storage=5, n_readout=3)
    initial_slow = slow_model.basis_state(0, 2, 0)
    initial_fast = fast_model.basis_state(0, 2, 0)
    fixed_ringdown_time_s = 0.5e-6

    slow = run_sequential_sideband_reset(
        slow_model,
        initial_slow,
        calibration=_calibration(ringdown_multiple=fixed_ringdown_time_s * slow_device.readout_kappa_hz),
        initial_storage_level=2,
        frame=build_sideband_reset_frame(slow_model),
        noise=build_sideband_reset_noise(slow_device),
        pulse_dt_s=0.25e-9,
        ringdown_dt_s=4.0e-9,
    )
    fast = run_sequential_sideband_reset(
        fast_model,
        initial_fast,
        calibration=_calibration(ringdown_multiple=fixed_ringdown_time_s * fast_device.readout_kappa_hz),
        initial_storage_level=2,
        frame=build_sideband_reset_frame(fast_model),
        noise=build_sideband_reset_noise(fast_device),
        pulse_dt_s=0.25e-9,
        ringdown_dt_s=4.0e-9,
    )

    assert fast.cycle_final_storage_photon_number[-1] < slow.cycle_final_storage_photon_number[-1]
    assert fast.cycle_final_readout_photon_number[-1] < slow.cycle_final_readout_photon_number[-1]


def test_storage_pure_dephasing_from_t1_t2_matches_repository_convention():
    tphi = pure_dephasing_time_from_t1_t2(t1_s=250.0e-6, t2_s=150.0e-6)
    assert math.isclose(tphi, 214.28571428571428e-6, rel_tol=1.0e-12)

    noise = build_sideband_reset_noise(_device())
    assert noise is not None
    assert math.isclose(float(noise.tphi_storage), tphi, rel_tol=1.0e-12)
    assert math.isclose(float(noise.kappa_storage), 1.0 / 250.0e-6, rel_tol=1.0e-12)
    assert math.isclose(float(noise.kappa_readout), _device().readout_kappa_hz, rel_tol=1.0e-12)
