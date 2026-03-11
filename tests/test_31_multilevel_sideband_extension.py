from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim import (
    DispersiveTransmonCavityModel,
    NoiseSpec,
    Pulse,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    build_sideband_pulse,
    carrier_for_transition_frequency,
    collapse_operators,
    compute_shelving_leakage,
    effective_sideband_rabi_frequency,
    hamiltonian_time_slices,
    sideband_transition_frequency,
    simulate_sequence,
    subsystem_level_population,
)


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def test_structured_sideband_target_produces_hermitian_hamiltonian_and_correct_dims():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    target = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)
    pulses, drive_ops, _meta = build_sideband_pulse(
        target,
        duration_s=4.0,
        amplitude_rad_s=0.2,
        sigma_fraction=0.18,
        channel="sb",
    )
    compiled = SequenceCompiler(dt=0.02).compile(pulses, t_end=4.0)
    terms = hamiltonian_time_slices(model, compiled, drive_ops, frame=SimulationConfig().frame)
    raising, lowering = model.sideband_drive_operators(mode="storage", lower_level=0, upper_level=2)

    assert raising.dims == model.static_hamiltonian().dims
    assert (raising.dag() - lowering).norm() < 1.0e-12

    coeff = compiled.channels["sb"].distorted[len(compiled.tlist) // 2]
    instantaneous = terms[0] + coeff * raising + np.conj(coeff) * lowering
    assert (instantaneous - instantaneous.dag()).norm() < 1.0e-10


def test_multilevel_transmon_decay_channels_resolve_ge_and_ef_separately():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=2,
        n_tr=3,
    )
    c_ops = collapse_operators(model, NoiseSpec(transmon_t1=(5.0, 2.0)))
    assert len(c_ops) == 2

    ge_target = model.basis_state(0, 0)
    ef_target = model.basis_state(1, 0)
    ge_source = model.basis_state(1, 0)
    ef_source = model.basis_state(2, 0)

    assert abs((c_ops[0] * ge_source).overlap(ge_target)) > 0.4
    assert abs((c_ops[1] * ef_source).overlap(ef_target)) > 0.6
    assert (c_ops[0] * ef_source).norm() < 1.0e-12


def test_gf_red_sideband_swap_reaches_target_faster_than_inverse_chi():
    g_sb = 0.35
    chi = -0.03
    t_pi = np.pi / (2.0 * g_sb)
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=chi,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    pulse = Pulse("sb", 0.0, t_pi, _square, amp=g_sb)
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=t_pi)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(2, 0),
        {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)},
        SimulationConfig(),
    )
    p_g1 = abs(model.basis_state(0, 1).overlap(result.final_state)) ** 2
    assert p_g1 > 0.98
    assert t_pi < 0.25 / abs(chi)


def test_zero_sideband_amplitude_gives_no_transfer():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    pulse = Pulse("sb", 0.0, 4.0, _square, amp=0.0)
    compiled = SequenceCompiler(dt=0.02).compile([pulse], t_end=4.0)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(2, 0),
        {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)},
        SimulationConfig(),
    )
    assert abs(model.basis_state(2, 0).overlap(result.final_state)) ** 2 > 0.999999


def test_detuned_sideband_matches_effective_rabi_frequency_model():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    g_sb = 0.22
    detuning = 0.31
    duration = 8.0
    pulse = Pulse(
        "sb",
        0.0,
        duration,
        _square,
        amp=g_sb,
        carrier=carrier_for_transition_frequency(detuning),
    )
    compiled = SequenceCompiler(dt=0.02).compile([pulse], t_end=duration)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(2, 0),
        {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)},
        SimulationConfig(store_states=True),
    )
    p_g1 = np.array([abs(model.basis_state(0, 1).overlap(state)) ** 2 for state in result.states], dtype=float)
    omega_eff = effective_sideband_rabi_frequency(g_sb, detuning)
    predicted = (2.0 * g_sb / omega_eff) ** 2 * np.sin(0.5 * omega_eff * compiled.tlist) ** 2
    assert np.max(np.abs(p_g1 - predicted)) < 1.0e-2


def test_shelved_e_population_stays_fixed_under_gf_sideband():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    duration = 6.0
    amplitude = np.pi / (2.0 * duration)
    pulses, drive_ops, _meta = build_sideband_pulse(
        SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2),
        duration_s=duration,
        amplitude_rad_s=amplitude,
        sigma_fraction=0.18,
        channel="sb",
    )
    compiled = SequenceCompiler(dt=0.01).compile(pulses, t_end=duration)
    initial = (np.sqrt(0.4) * model.basis_state(1, 0) + np.sqrt(0.6) * model.basis_state(2, 0)).unit()
    result = simulate_sequence(model, compiled, initial, drive_ops, SimulationConfig())

    leakage = compute_shelving_leakage(initial, result.final_state, shelved_level=1)
    assert leakage < 1.0e-5
    assert np.isclose(subsystem_level_population(result.final_state, "transmon", 1), 0.4, atol=1.0e-5)
    assert abs(model.basis_state(0, 1).overlap(result.final_state)) ** 2 > 0.59


def test_open_system_sideband_fidelity_degrades_relative_to_unitary_case():
    g_sb = 0.3
    duration = np.pi / (2.0 * g_sb)
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    target = model.basis_state(0, 1)
    pulse = Pulse("sb", 0.0, duration, _square, amp=g_sb)
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=duration)
    drive_ops = {"sb": SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)}

    closed = simulate_sequence(model, compiled, model.basis_state(2, 0), drive_ops, SimulationConfig())
    noisy = simulate_sequence(
        model,
        compiled,
        model.basis_state(2, 0),
        drive_ops,
        SimulationConfig(),
        noise=NoiseSpec(transmon_t1=(120.0, 35.0), tphi=90.0, kappa=0.02),
    )

    closed_fidelity = abs(target.overlap(closed.final_state)) ** 2
    noisy_fidelity = qt.fidelity(noisy.final_state, target.proj())
    assert closed_fidelity > 0.999
    assert noisy_fidelity < closed_fidelity - 0.04


def test_detuned_sync_choice_improves_branch_selectivity_for_two_photon_branches():
    g_sb = 0.35
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=-0.2,
        kerr=0.0,
        n_cav=5,
        n_tr=3,
    )
    target_spec = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)
    initial = (model.basis_state(2, 0) + model.basis_state(2, 1)).unit()
    target = (model.basis_state(0, 1) + model.basis_state(2, 1)).unit()
    base_frequency = sideband_transition_frequency(model, cavity_level=0, lower_level=0, upper_level=2)

    naive_duration = np.pi / (2.0 * g_sb)
    naive_pulse = Pulse(
        "sb",
        0.0,
        naive_duration,
        _square,
        amp=g_sb,
        carrier=carrier_for_transition_frequency(base_frequency),
    )
    naive_compiled = SequenceCompiler(dt=0.02).compile([naive_pulse], t_end=naive_duration)
    naive_result = simulate_sequence(model, naive_compiled, initial, {"sb": target_spec}, SimulationConfig())
    naive_fidelity = abs(target.overlap(naive_result.final_state)) ** 2

    optimized_detuning = 0.16
    optimized_duration = 5.11
    optimized_pulse = Pulse(
        "sb",
        0.0,
        optimized_duration,
        _square,
        amp=g_sb,
        carrier=carrier_for_transition_frequency(base_frequency + optimized_detuning),
    )
    optimized_compiled = SequenceCompiler(dt=0.02).compile([optimized_pulse], t_end=optimized_duration)
    optimized_result = simulate_sequence(model, optimized_compiled, initial, {"sb": target_spec}, SimulationConfig())
    optimized_fidelity = abs(target.overlap(optimized_result.final_state)) ** 2

    assert naive_fidelity < 0.8
    assert optimized_fidelity > naive_fidelity + 0.15
