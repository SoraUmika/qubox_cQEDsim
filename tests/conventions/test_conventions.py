from __future__ import annotations

import numpy as np

from cqed_sim.core import (
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    FrameSpec,
    carrier_for_transition_frequency,
    drive_frequency_for_transition_frequency,
    drive_frequency_from_internal_carrier,
    internal_carrier_from_drive_frequency,
    transition_frequency_from_drive_frequency,
)
from physics_and_conventions.conventions import DetuningSign, from_internal_units, to_internal_units, validate_detuning


def test_frequency_unit_helpers_round_trip_between_hz_and_rad_per_second():
    frequency_hz = 5.2e9
    assert np.isclose(from_internal_units(to_internal_units(frequency_hz)), frequency_hz, rtol=0.0, atol=1.0e-9)


def test_validate_detuning_respects_declared_sign_convention():
    delta = 2.0 * np.pi * 1.5e6
    assert np.isclose(validate_detuning(delta, DetuningSign.SYSTEM_MINUS_DRIVE), delta, atol=1.0e-12)
    assert np.isclose(validate_detuning(delta, DetuningSign.DRIVE_MINUS_SYSTEM), -delta, atol=1.0e-12)


def test_positive_drive_frequency_helpers_round_trip_through_internal_carrier():
    frame_frequency = 2.0 * np.pi * 6.0e9
    transition_frequency = -2.0 * np.pi * 2.5e6

    drive_frequency = drive_frequency_for_transition_frequency(transition_frequency, frame_frequency)
    carrier = internal_carrier_from_drive_frequency(drive_frequency, frame_frequency)

    assert np.isclose(drive_frequency, frame_frequency + transition_frequency, atol=1.0e-12)
    assert np.isclose(carrier, carrier_for_transition_frequency(transition_frequency), atol=1.0e-12)
    assert np.isclose(
        transition_frequency_from_drive_frequency(drive_frequency, frame_frequency),
        transition_frequency,
        atol=1.0e-12,
    )
    assert np.isclose(
        drive_frequency_from_internal_carrier(carrier, frame_frequency),
        drive_frequency,
        atol=1.0e-12,
    )


def test_hamiltonian_builders_preserve_positive_single_excitation_energies():
    two_mode = DispersiveTransmonCavityModel(
        omega_c=2.0,
        omega_q=3.0,
        alpha=0.0,
        chi=0.1,
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )
    three_mode = DispersiveReadoutTransmonStorageModel(
        omega_s=2.0,
        omega_r=4.0,
        omega_q=6.0,
        alpha=0.0,
        chi_s=0.1,
        chi_r=0.05,
        chi_sr=0.01,
        n_storage=3,
        n_readout=3,
        n_tr=2,
    )

    assert two_mode.basis_energy(0, 1) > two_mode.basis_energy(0, 0)
    assert two_mode.basis_energy(1, 0) > two_mode.basis_energy(0, 0)
    assert three_mode.basis_energy(0, 1, 0) > three_mode.basis_energy(0, 0, 0)
    assert three_mode.basis_energy(0, 0, 1) > three_mode.basis_energy(0, 0, 0)
    assert three_mode.basis_energy(1, 0, 0) > three_mode.basis_energy(0, 0, 0)


def test_frame_transform_round_trip_recovers_lab_hamiltonian():
    model = DispersiveTransmonCavityModel(
        omega_c=2.5,
        omega_q=4.5,
        alpha=0.0,
        chi=0.2,
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    ops = model.operators()
    h_lab = model.static_hamiltonian(FrameSpec())
    h_rot = model.static_hamiltonian(frame)
    reconstructed = h_rot + frame.omega_c_frame * ops["n_c"] + frame.omega_q_frame * ops["n_q"]
    assert (h_lab - reconstructed).norm() < 1.0e-12
