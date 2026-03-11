from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim import (
    NoiseSpec,
    SimulationConfig,
    qubit_conditioned_mode_moments,
    readout_response_by_qubit_state,
    reduced_qubit_state,
    simulate_sequence,
)


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def _compile(pulses: list[Pulse], t_end: float, dt: float = 0.02):
    return SequenceCompiler(dt=dt).compile(pulses, t_end=t_end)


def test_three_mode_drive_channel_isolation_in_decoupled_limit():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi_s=0.0,
        chi_r=0.0,
        chi_sr=0.0,
        kerr_s=0.0,
        kerr_r=0.0,
        n_storage=6,
        n_readout=6,
        n_tr=2,
    )
    cfg = SimulationConfig(frame=FrameSpec())

    storage = simulate_sequence(
        model,
        _compile([Pulse("s", 0.0, 1.0, _square, amp=0.4)], t_end=1.1),
        model.basis_state(0, 0, 0),
        {"s": "storage"},
        cfg,
    )
    assert storage.expectations["n_s"][-1] > 0.1
    assert storage.expectations["n_r"][-1] < 1.0e-8
    assert storage.expectations["P_e"][-1] < 1.0e-8

    readout = simulate_sequence(
        model,
        _compile([Pulse("r", 0.0, 1.0, _square, amp=0.35)], t_end=1.1),
        model.basis_state(0, 0, 0),
        {"r": "readout"},
        cfg,
    )
    assert readout.expectations["n_r"][-1] > 0.08
    assert readout.expectations["n_s"][-1] < 1.0e-8
    assert readout.expectations["P_e"][-1] < 1.0e-8

    qubit = simulate_sequence(
        model,
        _compile([Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)], t_end=1.1),
        model.basis_state(0, 0, 0),
        {"q": "qubit"},
        cfg,
    )
    assert np.isclose(qubit.expectations["P_e"][-1], 0.5, atol=3.0e-2)
    assert qubit.expectations["n_s"][-1] < 1.0e-8
    assert qubit.expectations["n_r"][-1] < 1.0e-8


def test_three_mode_storage_and_readout_ringdown_follow_distinct_kappas():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        n_storage=3,
        n_readout=3,
        n_tr=2,
    )
    compiled = _compile([], t_end=4.0, dt=0.05)

    storage = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 1, 0),
        {},
        SimulationConfig(),
        noise=NoiseSpec(kappa_storage=0.4),
    )
    readout = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0, 1),
        {},
        SimulationConfig(),
        noise=NoiseSpec(kappa_readout=0.8),
    )

    assert np.isclose(storage.expectations["n_s"][-1], np.exp(-0.4 * compiled.tlist[-1]), atol=3.0e-2)
    assert np.isclose(readout.expectations["n_r"][-1], np.exp(-0.8 * compiled.tlist[-1]), atol=3.0e-2)
    assert readout.expectations["n_r"][-1] < storage.expectations["n_s"][-1]


def test_three_mode_transmon_relaxation_and_dephasing_behave_as_expected():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        n_storage=2,
        n_readout=2,
        n_tr=2,
    )
    compiled = _compile([], t_end=4.0, dt=0.05)

    t1_result = simulate_sequence(
        model,
        compiled,
        model.basis_state(1, 0, 0),
        {},
        SimulationConfig(),
        noise=NoiseSpec(t1=5.0),
    )
    assert np.isclose(t1_result.expectations["P_e"][-1], np.exp(-compiled.tlist[-1] / 5.0), atol=3.0e-2)

    psi_plus = model.coherent_qubit_superposition(storage_level=0, readout_level=0)
    dephasing = simulate_sequence(
        model,
        compiled,
        psi_plus,
        {},
        SimulationConfig(),
        noise=NoiseSpec(tphi=2.0),
    )
    rho_q = reduced_qubit_state(dephasing.final_state)
    assert abs(rho_q[0, 1]) < 0.15


def test_three_mode_noise_path_preserves_trace_and_positivity():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.1,
        omega_r=-0.05,
        omega_q=0.0,
        alpha=-0.2,
        chi_s=0.03,
        chi_r=0.04,
        chi_sr=0.01,
        n_storage=3,
        n_readout=3,
        n_tr=2,
    )
    compiled = _compile([], t_end=2.0, dt=0.05)
    psi = (model.basis_state(0, 1, 0) + 1j * model.basis_state(1, 0, 1)).unit()
    result = simulate_sequence(
        model,
        compiled,
        psi,
        {},
        SimulationConfig(store_states=True),
        noise=NoiseSpec(t1=6.0, tphi=5.0, kappa_storage=0.15, kappa_readout=0.25),
    )

    for state in result.states[::5]:
        assert abs(state.tr() - 1.0) < 1.0e-8
        eigvals = np.linalg.eigvalsh(state.full())
        assert np.min(eigvals) > -1.0e-8


def test_three_mode_readout_response_depends_on_qubit_state_and_conditioned_extractors():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=5.0,
        omega_q=0.0,
        alpha=0.0,
        chi_s=0.0,
        chi_r=0.4,
        chi_sr=0.0,
        kerr_s=0.0,
        kerr_r=0.0,
        n_storage=2,
        n_readout=10,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_s, omega_q_frame=model.omega_q, omega_r_frame=model.omega_r)
    compiled = _compile([Pulse("r", 0.0, 4.0, _square, amp=0.15)], t_end=4.2, dt=0.02)

    g_response = simulate_sequence(model, compiled, model.basis_state(0, 0, 0), {"r": "readout"}, SimulationConfig(frame=frame))
    e_response = simulate_sequence(model, compiled, model.basis_state(1, 0, 0), {"r": "readout"}, SimulationConfig(frame=frame))

    assert g_response.expectations["n_r"][-1] > e_response.expectations["n_r"][-1]

    superposition = simulate_sequence(
        model,
        compiled,
        model.coherent_qubit_superposition(storage_level=0, readout_level=0),
        {"r": "readout"},
        SimulationConfig(frame=frame),
    )
    conditioned = readout_response_by_qubit_state(superposition.final_state)
    assert conditioned[0]["valid"]
    assert conditioned[1]["valid"]
    assert abs(conditioned[0]["a"] - conditioned[1]["a"]) > 5.0e-2

    direct = qubit_conditioned_mode_moments(superposition.final_state, "readout", 1)
    assert np.isclose(direct["n"], conditioned[1]["n"], atol=1.0e-12)
