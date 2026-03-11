from __future__ import annotations

import numpy as np

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.pulses.hardware import apply_first_order_lowpass
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, prepare_simulation, simulate_batch, simulate_sequence


def _square(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=np.complex128)


def _assert_state_close(a, b, atol: float = 1e-10) -> None:
    assert np.allclose(np.asarray(a.full()), np.asarray(b.full()), atol=atol)


def _assert_result_close(a, b, atol: float = 1e-10) -> None:
    _assert_state_close(a.final_state, b.final_state, atol=atol)
    assert a.expectations.keys() == b.expectations.keys()
    for key in a.expectations:
        assert np.allclose(a.expectations[key], b.expectations[key], atol=atol)


def test_two_mode_static_hamiltonian_cache_reuses_operator_object() -> None:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0,
        omega_q=3.0,
        alpha=-0.4,
        chi=0.2,
        kerr=-0.03,
        chi_higher=(0.01,),
        kerr_higher=(0.004,),
        n_cav=5,
        n_tr=3,
    )
    frame = FrameSpec(omega_c_frame=1.2, omega_q_frame=2.5)
    h_a = model.static_hamiltonian(frame)
    h_b = model.static_hamiltonian(frame)
    assert h_a is h_b


def test_three_mode_static_hamiltonian_cache_reuses_operator_object() -> None:
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=2.0,
        omega_r=2.5,
        omega_q=3.5,
        alpha=-0.4,
        chi_s=0.1,
        chi_r=0.07,
        chi_sr=0.02,
        kerr_s=-0.005,
        kerr_r=-0.004,
        n_storage=4,
        n_readout=3,
        n_tr=3,
    )
    frame = FrameSpec(omega_c_frame=1.8, omega_q_frame=3.0, omega_r_frame=2.1)
    h_a = model.static_hamiltonian(frame)
    h_b = model.static_hamiltonian(frame)
    assert h_a is h_b


def test_support_slice_compilation_matches_full_grid_sampling() -> None:
    pulses = [
        Pulse("q", 0.10, 0.35, _square, amp=0.3, phase=0.2),
        Pulse("q", 0.25, 0.20, _square, amp=-0.05, carrier=2.0),
        Pulse("c", 0.20, 0.40, _square, amp=0.12, phase=-0.3),
    ]
    compiler = SequenceCompiler(dt=0.05)
    compiled = compiler.compile(pulses, t_end=0.8)

    manual: dict[str, np.ndarray] = {}
    for pulse in pulses:
        manual.setdefault(pulse.channel, np.zeros_like(compiled.tlist, dtype=np.complex128))
        manual[pulse.channel] += pulse.sample(compiled.tlist)

    assert np.allclose(compiled.channels["q"].baseband, manual["q"])
    assert np.allclose(compiled.channels["c"].baseband, manual["c"])


def test_lowpass_fast_path_matches_reference_recurrence() -> None:
    rng = np.random.default_rng(1234)
    x = rng.normal(size=64) + 1j * rng.normal(size=64)
    dt = 0.01
    bw = 8.0

    actual = apply_first_order_lowpass(x, dt=dt, bw=bw)

    tau = 1.0 / (2.0 * np.pi * bw)
    alpha = dt / (tau + dt)
    expected = np.empty_like(actual)
    expected[0] = x[0]
    for idx in range(1, x.size):
        expected[idx] = expected[idx - 1] + alpha * (x[idx] - expected[idx - 1])

    assert np.allclose(actual, expected)


def test_prepare_simulation_matches_simulate_sequence() -> None:
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    compiled = SequenceCompiler(dt=0.01).compile([Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)], t_end=1.1)
    initial = model.basis_state(0, 0)
    config = SimulationConfig(max_step=0.01)

    reference = simulate_sequence(model, compiled, initial, {"q": "qubit"}, config=config)
    prepared = prepare_simulation(model, compiled, {"q": "qubit"}, config=config)
    result = prepared.run(initial)

    _assert_result_close(reference, result)


def test_empty_observables_disable_expectation_collection_without_changing_final_state() -> None:
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    compiled = SequenceCompiler(dt=0.01).compile([Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)], t_end=1.1)
    initial = model.basis_state(0, 0)

    reference = simulate_sequence(model, compiled, initial, {"q": "qubit"}, config=SimulationConfig())
    fast = simulate_sequence(model, compiled, initial, {"q": "qubit"}, config=SimulationConfig(), e_ops={})

    assert fast.expectations == {}
    _assert_state_close(reference.final_state, fast.final_state)


def test_simulate_batch_matches_serial_runs() -> None:
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    compiled = SequenceCompiler(dt=0.01).compile([Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)], t_end=1.1)
    states = [model.basis_state(0, 0), model.basis_state(1, 0)]
    session = prepare_simulation(model, compiled, {"q": "qubit"}, config=SimulationConfig())

    serial = [session.run(state) for state in states]
    batched = simulate_batch(session, states, max_workers=1)

    for lhs, rhs in zip(serial, batched):
        _assert_result_close(lhs, rhs)


def test_simulate_batch_parallel_matches_serial_runs() -> None:
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    compiled = SequenceCompiler(dt=0.01).compile([Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)], t_end=1.1)
    states = [model.basis_state(0, 0), model.basis_state(1, 0)]
    session = prepare_simulation(model, compiled, {"q": "qubit"}, config=SimulationConfig(), e_ops={})

    serial = simulate_batch(session, states, max_workers=1)
    parallel = simulate_batch(session, states, max_workers=2, mp_context="spawn")

    for lhs, rhs in zip(serial, parallel):
        assert rhs.expectations == {}
        _assert_state_close(lhs.final_state, rhs.final_state)
