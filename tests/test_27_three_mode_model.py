from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core import (
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    FrameSpec,
    qubit_storage_readout_index,
)
from cqed_sim.sim import reduced_qubit_state, reduced_readout_state, reduced_storage_state


def test_three_mode_basis_state_flat_indices_follow_qubit_storage_readout_order():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        n_storage=3,
        n_readout=4,
        n_tr=2,
    )

    for q_level, storage_level, readout_level in [(0, 0, 0), (0, 1, 2), (1, 0, 1), (1, 2, 3)]:
        psi = model.basis_state(q_level, storage_level, readout_level)
        flat_index = qubit_storage_readout_index(model.n_storage, model.n_readout, q_level, storage_level, readout_level)
        assert int(np.argmax(np.abs(np.asarray(psi.full()).ravel()))) == flat_index


def test_three_mode_operator_embeddings_act_only_on_target_subsystems():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        n_storage=4,
        n_readout=3,
        n_tr=3,
    )
    ops = model.operators()
    state = model.basis_state(1, 2, 1)

    assert (ops["a_s"] * state - np.sqrt(2.0) * model.basis_state(1, 1, 1)).norm() < 1.0e-12
    assert (ops["a_r"] * state - model.basis_state(1, 2, 0)).norm() < 1.0e-12
    assert (ops["b"] * state - model.basis_state(0, 2, 1)).norm() < 1.0e-12


def test_three_mode_partial_traces_return_expected_subsystem_states():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=0.0,
        omega_q=0.0,
        alpha=0.0,
        n_storage=5,
        n_readout=4,
        n_tr=2,
    )
    state = model.basis_state(1, 3, 2)

    rho_q = reduced_qubit_state(state)
    rho_s = reduced_storage_state(state)
    rho_r = reduced_readout_state(state)

    assert np.isclose(float(np.real((rho_q * qt.basis(2, 1).proj()).tr())), 1.0, atol=1.0e-12)
    assert np.isclose(float(np.real((rho_s * qt.basis(5, 3).proj()).tr())), 1.0, atol=1.0e-12)
    assert np.isclose(float(np.real((rho_r * qt.basis(4, 2).proj()).tr())), 1.0, atol=1.0e-12)


def test_three_mode_static_hamiltonian_diagonal_matches_basis_energy_formula():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=1.7,
        omega_r=2.3,
        omega_q=5.2,
        alpha=-0.4,
        chi_s=0.12,
        chi_r=0.08,
        chi_sr=0.03,
        kerr_s=-0.02,
        kerr_r=0.05,
        n_storage=4,
        n_readout=3,
        n_tr=3,
    )
    h = model.static_hamiltonian()
    diag = np.real(np.diag(h.full()))

    for q_level in range(model.n_tr):
        for storage_level in range(model.n_storage):
            for readout_level in range(model.n_readout):
                idx = qubit_storage_readout_index(
                    model.n_storage,
                    model.n_readout,
                    q_level,
                    storage_level,
                    readout_level,
                )
                expected = model.basis_energy(q_level, storage_level, readout_level)
                assert np.isclose(diag[idx], expected, atol=1.0e-12)


def test_three_mode_reduces_to_two_mode_storage_transmon_when_readout_is_idle():
    model_two = DispersiveTransmonCavityModel(
        omega_c=1.4,
        omega_q=4.6,
        alpha=-0.3,
        chi=0.11,
        kerr=-0.02,
        n_cav=4,
        n_tr=3,
    )
    model_three = DispersiveReadoutTransmonStorageModel(
        omega_s=model_two.omega_c,
        omega_r=0.7,
        omega_q=model_two.omega_q,
        alpha=model_two.alpha,
        chi_s=model_two.chi,
        chi_r=0.0,
        chi_sr=0.0,
        kerr_s=model_two.kerr,
        kerr_r=0.0,
        n_storage=model_two.n_cav,
        n_readout=2,
        n_tr=model_two.n_tr,
    )

    for q_level in range(model_two.n_tr):
        for storage_level in range(model_two.n_cav):
            assert np.isclose(
                model_three.basis_energy(q_level, storage_level, 0),
                model_two.basis_energy(q_level, storage_level),
                atol=1.0e-12,
            )


def test_three_mode_transition_frequencies_match_configured_dispersive_shifts():
    frame = FrameSpec(omega_c_frame=4.0, omega_q_frame=6.0, omega_r_frame=7.0)
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=4.0,
        omega_r=7.0,
        omega_q=6.0,
        alpha=-0.25,
        chi_s=0.13,
        chi_r=0.07,
        chi_sr=0.02,
        kerr_s=-0.01,
        kerr_r=0.03,
        n_storage=4,
        n_readout=4,
        n_tr=3,
    )

    assert np.isclose(
        model.qubit_transition_frequency(storage_level=2, readout_level=3, frame=frame),
        -(2.0 * model.chi_s + 3.0 * model.chi_r),
        atol=1.0e-12,
    )
    assert np.isclose(
        model.storage_transition_frequency(qubit_level=1, readout_level=2, storage_level=1, frame=frame),
        -model.chi_s + 2.0 * model.chi_sr + model.kerr_s,
        atol=1.0e-12,
    )
    assert np.isclose(
        model.readout_transition_frequency(qubit_level=1, storage_level=2, readout_level=1, frame=frame),
        -model.chi_r + 2.0 * model.chi_sr + model.kerr_r,
        atol=1.0e-12,
    )
