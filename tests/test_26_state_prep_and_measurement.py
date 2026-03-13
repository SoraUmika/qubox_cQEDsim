from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.core import StatePreparationSpec, density_matrix_state, fock_state, prepare_state, qubit_state
from cqed_sim.measurement import QubitMeasurementSpec, measure_qubit
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, reduced_qubit_state, simulate_sequence


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def test_prepare_state_respects_three_mode_tensor_ordering():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        n_storage=4,
        n_readout=3,
        n_tr=2,
    )
    prepared = prepare_state(
        model,
        StatePreparationSpec(
            qubit=qubit_state("e"),
            storage=fock_state(2),
            readout=fock_state(1),
        ),
    )
    manual = model.basis_state(1, 2, 1)
    assert (prepared - manual).norm() < 1.0e-12


def test_prepare_state_supports_mixed_density_matrices():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )
    rho_q = 0.7 * qt.basis(2, 0).proj() + 0.3 * qt.basis(2, 1).proj()
    rho = prepare_state(
        model,
        StatePreparationSpec(
            qubit=density_matrix_state(rho_q),
            storage=fock_state(1),
        ),
    )
    assert rho.isoper
    assert (reduced_qubit_state(rho) - rho_q).norm() < 1.0e-12


def test_measure_qubit_sampled_mode_tracks_observed_probabilities():
    psi = (np.sqrt(0.8) * qt.basis(2, 0) + np.sqrt(0.2) * qt.basis(2, 1)).unit()
    state = qt.tensor(psi, qt.basis(2, 0))

    spec = QubitMeasurementSpec(
        shots=4000,
        confusion_matrix=np.array([[0.9, 0.2], [0.1, 0.8]], dtype=float),
        iq_sigma=0.2,
        seed=123,
    )
    result = measure_qubit(state, spec)

    assert np.isclose(result.probabilities["g"], 0.8, atol=1.0e-12)
    assert np.isclose(result.probabilities["e"], 0.2, atol=1.0e-12)
    assert np.isclose(result.observed_probabilities["g"], 0.76, atol=1.0e-12)
    assert np.isclose(result.observed_probabilities["e"], 0.24, atol=1.0e-12)
    assert result.counts is not None
    assert result.samples is not None
    assert result.iq_samples is not None
    assert result.iq_samples.shape == (4000, 2)
    freq_e = result.counts["e"] / spec.shots
    assert np.isclose(freq_e, 0.24, atol=0.03)


def test_direct_prepare_compile_simulate_measure_workflow():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=3,
        n_tr=2,
    )
    pulse = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    initial = prepare_state(
        model,
        StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(0)),
    )
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=1.1)
    simulation = simulate_sequence(
        model,
        compiled,
        initial,
        {"q": "qubit"},
        config=SimulationConfig(frame=FrameSpec()),
    )
    measurement = measure_qubit(simulation.final_state, QubitMeasurementSpec(shots=2000, seed=5))

    assert compiled.tlist.size > 50
    assert np.isclose(simulation.expectations["P_e"][-1], 0.5, atol=3.0e-2)
    assert measurement.counts is not None
    assert np.isclose(measurement.counts["e"] / 2000.0, 0.5, atol=0.05)
