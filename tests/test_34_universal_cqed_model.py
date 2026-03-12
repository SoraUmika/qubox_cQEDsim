from __future__ import annotations

import numpy as np
import pytest

from cqed_sim import (
    BosonicModeSpec,
    CrossKerrSpec,
    DispersiveCouplingSpec,
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    FrameSpec,
    NoiseSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
    collapse_operators,
    default_observables,
)


def test_universal_two_mode_matches_dispersive_transmon_cavity_wrapper():
    frame = FrameSpec(omega_c_frame=1.3, omega_q_frame=4.1)
    wrapper = DispersiveTransmonCavityModel(
        omega_c=1.8,
        omega_q=4.7,
        alpha=-0.23,
        chi=-0.04,
        chi_higher=(0.005,),
        kerr=-0.02,
        kerr_higher=(0.003,),
        n_cav=5,
        n_tr=4,
    )
    universal = UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=wrapper.omega_q,
            dim=wrapper.n_tr,
            alpha=wrapper.alpha,
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=wrapper.omega_c,
                dim=wrapper.n_cav,
                kerr=wrapper.kerr,
                kerr_higher=wrapper.kerr_higher,
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
        ),
        dispersive_couplings=(
            DispersiveCouplingSpec(
                mode="storage",
                chi=wrapper.chi,
                chi_higher=wrapper.chi_higher,
                transmon="qubit",
            ),
        ),
    )

    assert universal.subsystem_dims == wrapper.subsystem_dims
    assert (universal.static_hamiltonian(frame) - wrapper.static_hamiltonian(frame)).norm() < 1.0e-12
    assert (universal.transmon_lowering() - wrapper.transmon_lowering()).norm() < 1.0e-12
    assert (universal.cavity_annihilation() - wrapper.cavity_annihilation()).norm() < 1.0e-12

    for q_level in range(wrapper.n_tr):
        for cavity_level in range(wrapper.n_cav):
            assert np.isclose(
                universal.basis_energy(q_level, cavity_level, frame=frame),
                wrapper.basis_energy(q_level, cavity_level, frame=frame),
                atol=1.0e-12,
            )
            assert (universal.basis_state(q_level, cavity_level) - wrapper.basis_state(q_level, cavity_level)).norm() < 1.0e-12

    assert np.isclose(
        universal.transmon_transition_frequency(mode_levels={"storage": 2}, lower_level=1, upper_level=2, frame=frame),
        wrapper.transmon_transition_frequency(cavity_level=2, lower_level=1, upper_level=2, frame=frame),
        atol=1.0e-12,
    )
    assert np.isclose(
        universal.sideband_transition_frequency(
            mode="storage",
            mode_levels={"storage": 1},
            lower_level=0,
            upper_level=2,
            sideband="red",
            frame=frame,
        ),
        wrapper.sideband_transition_frequency(cavity_level=1, lower_level=0, upper_level=2, sideband="red", frame=frame),
        atol=1.0e-12,
    )


def test_universal_three_mode_matches_readout_wrapper():
    frame = FrameSpec(omega_c_frame=1.9, omega_q_frame=4.5, omega_r_frame=6.8)
    wrapper = DispersiveReadoutTransmonStorageModel(
        omega_s=2.3,
        omega_r=7.4,
        omega_q=5.1,
        alpha=-0.31,
        chi_s=-0.05,
        chi_r=0.07,
        chi_sr=0.015,
        kerr_s=-0.01,
        kerr_r=0.025,
        n_storage=4,
        n_readout=3,
        n_tr=4,
    )
    universal = UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=wrapper.omega_q,
            dim=wrapper.n_tr,
            alpha=wrapper.alpha,
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=wrapper.omega_s,
                dim=wrapper.n_storage,
                kerr=wrapper.kerr_s,
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
            BosonicModeSpec(
                label="readout",
                omega=wrapper.omega_r,
                dim=wrapper.n_readout,
                kerr=wrapper.kerr_r,
                aliases=("readout",),
                frame_channel="r",
            ),
        ),
        dispersive_couplings=(
            DispersiveCouplingSpec(mode="storage", chi=wrapper.chi_s, transmon="qubit"),
            DispersiveCouplingSpec(mode="readout", chi=wrapper.chi_r, transmon="qubit"),
        ),
        cross_kerr_terms=(CrossKerrSpec("storage", "readout", wrapper.chi_sr),),
    )

    assert universal.subsystem_dims == wrapper.subsystem_dims
    assert (universal.static_hamiltonian(frame) - wrapper.static_hamiltonian(frame)).norm() < 1.0e-12
    assert (universal.storage_annihilation() - wrapper.storage_annihilation()).norm() < 1.0e-12
    assert (universal.readout_annihilation() - wrapper.readout_annihilation()).norm() < 1.0e-12

    for q_level in range(wrapper.n_tr):
        for storage_level in range(wrapper.n_storage):
            for readout_level in range(wrapper.n_readout):
                assert np.isclose(
                    universal.basis_energy(q_level, storage_level, readout_level, frame=frame),
                    wrapper.basis_energy(q_level, storage_level, readout_level, frame=frame),
                    atol=1.0e-12,
                )

    assert np.isclose(
        universal.transmon_transition_frequency(
            mode_levels={"storage": 2, "readout": 1},
            lower_level=1,
            upper_level=2,
            frame=frame,
        ),
        wrapper.transmon_transition_frequency(storage_level=2, readout_level=1, lower_level=1, upper_level=2, frame=frame),
        atol=1.0e-12,
    )
    assert np.isclose(
        universal.mode_transition_frequency(
            "storage",
            mode_levels={"storage": 1, "readout": 2},
            transmon_level=1,
            frame=frame,
        ),
        wrapper.storage_transition_frequency(qubit_level=1, storage_level=1, readout_level=2, frame=frame),
        atol=1.0e-12,
    )
    assert np.isclose(
        universal.mode_transition_frequency(
            "readout",
            mode_levels={"storage": 2, "readout": 0},
            transmon_level=1,
            frame=frame,
        ),
        wrapper.readout_transition_frequency(qubit_level=1, storage_level=2, readout_level=0, frame=frame),
        atol=1.0e-12,
    )


def test_universal_transmon_only_model_exposes_multilevel_anharmonic_spectrum():
    model = UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=5.6,
            dim=5,
            alpha=-0.24,
            label="ancilla",
            aliases=("ancilla", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(),
    )

    ge = model.transmon_transition_frequency(lower_level=0, upper_level=1)
    ef = model.transmon_transition_frequency(lower_level=1, upper_level=2)

    assert np.isclose(ge, 5.6, atol=1.0e-12)
    assert np.isclose(ef, 5.6 - 0.24, atol=1.0e-12)
    assert ef < ge
    assert (model.hamiltonian() - model.hamiltonian().dag()).norm() < 1.0e-12


def test_universal_cavity_only_model_supports_observables_and_bosonic_noise():
    model = UniversalCQEDModel(
        transmon=None,
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=2.1,
                dim=5,
                kerr=-0.03,
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
        ),
    )

    observables = default_observables(model)
    assert "P_e" not in observables
    assert {"n_c", "x_c", "p_c"}.issubset(observables)

    c_ops = collapse_operators(model, NoiseSpec(kappa=0.12))
    assert len(c_ops) == 1
    assert c_ops[0].dims == model.hamiltonian().dims

    with pytest.raises(ValueError, match="without a transmon subsystem"):
        collapse_operators(model, NoiseSpec(t1=40.0))
