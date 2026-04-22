from __future__ import annotations

import numpy as np

from cqed_sim import (
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    FrameSpec,
    NoiseSpec,
    Pulse,
    RotationGate,
    SequenceCompiler,
    SimulationConfig,
    build_rotation_pulse,
    simulate_sequence,
)
from cqed_sim.pulses import square_envelope
from tutorials.tutorial_support import (
    GHz,
    MHz,
    cross_kerr_conditional_phase,
    final_expectation,
    fit_rabi_vs_amplitude,
    gaussian_quasistatic_echo_excited_population,
    gaussian_quasistatic_ramsey_excited_population,
    ns,
    ramsey_population,
    resonant_drive_excited_population,
    t1_relaxation_population,
    us,
)


def test_tutorial_04_pi_pulse_matches_repo_rabi_normalization() -> None:
    omega_rabi = 2.0 * np.pi * 8.0e6
    pulse_duration = np.pi / (2.0 * omega_rabi)
    model = DispersiveTransmonCavityModel(
        omega_c=GHz(5.0),
        omega_q=GHz(6.1),
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_q_frame=model.omega_q)
    pulse = Pulse("q", 0.0, pulse_duration, square_envelope, amp=omega_rabi, carrier=0.0, label="pi")
    dt = pulse_duration / 200.0
    compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=pulse_duration + dt)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0),
        {"q": "qubit"},
        config=SimulationConfig(frame=frame, store_states=True, max_step=dt),
    )

    times = np.asarray(compiled.tlist, dtype=float)
    simulated = np.asarray(result.expectations["P_e"], dtype=float)
    theory = resonant_drive_excited_population(omega_rabi, times)

    assert np.max(np.abs(simulated - theory)) < 3.0e-3
    assert simulated[-1] > 0.99


def test_tutorial_09_power_rabi_fit_recovers_drive_scaling() -> None:
    omega_scale_true = 2.0 * np.pi * 18.0e6
    duration_s = 25.0 * ns
    amplitude_scales = np.linspace(0.0, 2.0, 17)

    model = DispersiveTransmonCavityModel(
        omega_c=GHz(5.0),
        omega_q=GHz(6.1),
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_q_frame=model.omega_q)
    dt = duration_s / 150.0

    excited_population = []
    for scale in amplitude_scales:
        pulse = Pulse(
            "q",
            0.0,
            duration_s,
            square_envelope,
            amp=0.5 * omega_scale_true * float(scale),
            carrier=0.0,
            label="power_rabi",
        )
        compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=duration_s + dt)
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 0),
            {"q": "qubit"},
            config=SimulationConfig(frame=frame, max_step=dt),
        )
        excited_population.append(final_expectation(result, "P_e"))

    fit = fit_rabi_vs_amplitude(
        amplitude_scales,
        np.asarray(excited_population, dtype=float),
        duration=duration_s,
        p0=(omega_scale_true, 0.0),
    )

    assert np.isclose(fit.parameters["omega_scale"], omega_scale_true, rtol=2.0e-2)
    assert abs(fit.parameters["offset"]) < 5.0e-3
    assert np.sqrt(np.mean((fit.model_y - np.asarray(excited_population, dtype=float)) ** 2)) < 5.0e-3


def test_tutorial_11_and_12_helpers_match_pulse_level_t1_and_ramsey() -> None:
    model = DispersiveTransmonCavityModel(
        omega_c=GHz(5.0),
        omega_q=GHz(6.2),
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_q_frame=model.omega_q)

    t1_true = 18.0 * us
    delays_t1 = np.linspace(0.0, 30.0, 7) * us
    t1_simulated = []
    for delay_s in delays_t1:
        dt = 20.0 * ns
        compiled = SequenceCompiler(dt=dt).compile([], t_end=float(max(delay_s, dt)))
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(1, 0),
            {},
            config=SimulationConfig(frame=frame, max_step=dt),
            noise=NoiseSpec(t1=t1_true),
        )
        t1_simulated.append(final_expectation(result, "P_e"))
    t1_simulated = np.asarray(t1_simulated, dtype=float)
    t1_theory = t1_relaxation_population(delays_t1, t1_true)
    assert np.max(np.abs(t1_simulated - t1_theory)) < 2.0e-3

    rotation_duration = 30.0 * ns
    pulse_config = {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": 0.18}
    x90 = build_rotation_pulse(
        RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
        pulse_config,
    )[0][0]
    detuning = 2.0 * np.pi * 0.6e6
    t2_star_true = 8.0 * us
    tphi_true = 1.0 / (1.0 / t2_star_true - 1.0 / (2.0 * 30.0 * us))
    delays_ramsey = np.linspace(0.0, 12.0, 7) * us
    ramsey_simulated = []
    for delay_s in delays_ramsey:
        first = Pulse(
            channel=x90.channel,
            t0=0.0,
            duration=x90.duration,
            envelope=x90.envelope,
            amp=x90.amp,
            carrier=x90.carrier,
            phase=0.0,
            label="ramsey_a",
        )
        second = Pulse(
            channel=x90.channel,
            t0=rotation_duration + delay_s,
            duration=x90.duration,
            envelope=x90.envelope,
            amp=x90.amp,
            carrier=x90.carrier,
            phase=float(detuning * delay_s),
            label="ramsey_b",
        )
        dt = 4.0 * ns
        compiled = SequenceCompiler(dt=dt).compile([first, second], t_end=2.0 * rotation_duration + delay_s + dt)
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 0),
            {"qubit": "qubit"},
            config=SimulationConfig(frame=frame, max_step=dt),
            noise=NoiseSpec(t1=30.0 * us, tphi=tphi_true),
        )
        ramsey_simulated.append(final_expectation(result, "P_e"))
    ramsey_simulated = np.asarray(ramsey_simulated, dtype=float)
    ramsey_theory = ramsey_population(delays_ramsey, detuning, t2_star_true)
    assert np.max(np.abs(ramsey_simulated - ramsey_theory)) < 6.0e-3


def test_tutorial_13_echo_refocuses_quasistatic_detuning() -> None:
    sigma_detuning = 2.0 * np.pi * 0.22e6
    detuning_offsets = np.linspace(-4.0, 4.0, 21) * sigma_detuning
    weights = np.exp(-0.5 * (detuning_offsets / sigma_detuning) ** 2)
    weights = weights / np.sum(weights)
    delays_s = np.linspace(0.0, 6.0, 5) * us

    model = DispersiveTransmonCavityModel(
        omega_c=GHz(5.0),
        omega_q=GHz(6.2),
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    pulse_config = {"duration_rotation_s": 30.0 * ns, "rotation_sigma_fraction": 0.18}
    x90 = build_rotation_pulse(
        RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
        pulse_config,
    )[0][0]
    x180 = build_rotation_pulse(
        RotationGate(index=1, name="x180", theta=np.pi, phi=0.0),
        pulse_config,
    )[0][0]
    minus_x90 = build_rotation_pulse(
        RotationGate(index=2, name="minus_x90", theta=np.pi / 2.0, phi=np.pi),
        pulse_config,
    )[0][0]

    ramsey_mean = np.zeros_like(delays_s, dtype=float)
    echo_mean = np.zeros_like(delays_s, dtype=float)
    dt = 4.0 * ns
    for delta_omega, weight in zip(detuning_offsets, weights, strict=True):
        frame = FrameSpec(omega_q_frame=model.omega_q + float(delta_omega))
        for index, delay_s in enumerate(delays_s):
            ramsey_pulses = [
                Pulse(x90.channel, 0.0, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=x90.phase, label="ramsey_a"),
                Pulse(x90.channel, x90.duration + delay_s, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=x90.phase, label="ramsey_b"),
            ]
            ramsey_compiled = SequenceCompiler(dt=dt).compile(ramsey_pulses, t_end=2.0 * x90.duration + delay_s + dt)
            ramsey_result = simulate_sequence(
                model,
                ramsey_compiled,
                model.basis_state(0, 0),
                {"qubit": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
            )
            ramsey_mean[index] += float(weight) * final_expectation(ramsey_result, "P_e")

            echo_pulses = [
                Pulse(x90.channel, 0.0, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=x90.phase, label="echo_a"),
                Pulse(x180.channel, x90.duration + 0.5 * delay_s, x180.duration, x180.envelope, amp=x180.amp, carrier=x180.carrier, phase=x180.phase, label="echo_pi"),
                Pulse(minus_x90.channel, x90.duration + delay_s + x180.duration, minus_x90.duration, minus_x90.envelope, amp=minus_x90.amp, carrier=minus_x90.carrier, phase=minus_x90.phase, label="echo_b"),
            ]
            echo_compiled = SequenceCompiler(dt=dt).compile(echo_pulses, t_end=2.0 * x90.duration + x180.duration + delay_s + dt)
            echo_result = simulate_sequence(
                model,
                echo_compiled,
                model.basis_state(0, 0),
                {"qubit": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
            )
            echo_mean[index] += float(weight) * final_expectation(echo_result, "P_e")

    ramsey_theory = gaussian_quasistatic_ramsey_excited_population(delays_s, sigma_detuning)
    echo_theory = gaussian_quasistatic_echo_excited_population(delays_s)

    assert np.sqrt(np.mean((ramsey_mean - ramsey_theory) ** 2)) < 1.0e-2
    assert np.max(np.abs(echo_mean - echo_theory)) < 2.0e-3


def test_tutorial_16_storage_decay_matches_exponential_and_is_monotone() -> None:
    kappa = 1.0 / (40.0 * us)
    delays_s = np.linspace(0.0, 120.0, 9) * us

    model = DispersiveTransmonCavityModel(
        omega_c=GHz(5.0),
        omega_q=GHz(6.0),
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=6,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

    simulated = []
    for delay_s in delays_s:
        dt = 20.0 * ns
        compiled = SequenceCompiler(dt=dt).compile([], t_end=float(max(delay_s, dt)))
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 1),
            {},
            config=SimulationConfig(frame=frame, max_step=dt),
            noise=NoiseSpec(kappa=kappa),
        )
        simulated.append(final_expectation(result, "n_c"))

    simulated = np.asarray(simulated, dtype=float)
    theory = np.exp(-kappa * delays_s)

    assert np.all(np.diff(simulated) <= 1.0e-8)
    assert np.max(np.abs(simulated - theory)) < 3.0e-3
    assert simulated[-1] < simulated[1]


def test_tutorial_15_cross_kerr_phase_uses_negative_dynamical_sign() -> None:
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=GHz(5.1),
        omega_r=GHz(7.3),
        omega_q=GHz(6.2),
        alpha=MHz(-220.0),
        chi_s=0.0,
        chi_r=0.0,
        chi_sr=MHz(0.030),
        kerr_s=0.0,
        kerr_r=0.0,
        n_storage=4,
        n_readout=4,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_s, omega_q_frame=model.omega_q, omega_r_frame=model.omega_r)
    compiled = SequenceCompiler(dt=0.5 * us).compile([], t_end=8.0 * us)
    initial_r0 = (model.basis_state(0, 0, 0) + model.basis_state(0, 1, 0)).unit()
    initial_r1 = (model.basis_state(0, 0, 1) + model.basis_state(0, 1, 1)).unit()
    result_r0 = simulate_sequence(model, compiled, initial_r0, {}, config=SimulationConfig(frame=frame, store_states=True, max_step=0.5 * us))
    result_r1 = simulate_sequence(model, compiled, initial_r1, {}, config=SimulationConfig(frame=frame, store_states=True, max_step=0.5 * us))

    def relative_phase(states, readout_level: int) -> np.ndarray:
        phases = []
        reference = model.basis_state(0, 0, readout_level)
        shifted = model.basis_state(0, 1, readout_level)
        for state in states:
            amp_reference = complex(reference.overlap(state))
            amp_shifted = complex(shifted.overlap(state))
            phases.append(np.angle(amp_shifted / amp_reference))
        return np.unwrap(np.asarray(phases, dtype=float))

    conditional_phase = relative_phase(result_r1.states, 1) - relative_phase(result_r0.states, 0)
    theory = cross_kerr_conditional_phase(np.asarray(compiled.tlist, dtype=float), model.chi_sr)

    assert conditional_phase[-1] < 0.0
    assert np.max(np.abs(conditional_phase - theory)) < 1.0e-4
