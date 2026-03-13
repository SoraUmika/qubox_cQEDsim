from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from cqed_sim import (
    BosonicModeSpec,
    CrossKerrSpec,
    DispersiveCouplingSpec,
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    FrameSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
    compute_energy_spectrum,
)
from cqed_sim.plotting import plot_energy_levels


def test_vacuum_reference_sets_bare_vacuum_energy_to_zero():
    model = DispersiveTransmonCavityModel(
        omega_c=5.0,
        omega_q=7.0,
        alpha=-0.4,
        n_cav=3,
        n_tr=4,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

    spectrum = compute_energy_spectrum(model, frame=frame)

    vacuum = spectrum.find_level("|g,0>")
    assert np.isclose(vacuum.energy, 0.0, atol=1.0e-12)
    assert np.isclose(spectrum.vacuum_energy, 0.0, atol=1.0e-12)
    assert spectrum.vacuum_level_index is not None
    assert spectrum.vacuum_level_overlap > 1.0 - 1.0e-12
    assert spectrum.vacuum_residual_norm < 1.0e-12


def test_energy_spectrum_orders_raw_energies_and_returns_qutip_eigenstates():
    model = DispersiveReadoutTransmonStorageModel(
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
    frame = FrameSpec(omega_c_frame=1.9, omega_q_frame=4.5, omega_r_frame=6.8)

    spectrum = model.energy_spectrum(frame=frame, levels=8)

    assert len(spectrum.levels) == 8
    assert np.all(np.diff(spectrum.raw_energies) >= -1.0e-12)
    hamiltonian = model.hamiltonian(frame)
    for level in spectrum.levels:
        assert isinstance(level.eigenstate, qt.Qobj)
        assert level.eigenstate.dims[0] == hamiltonian.dims[0]


def test_energy_spectrum_wrapper_matches_universal_model():
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

    wrapper_spectrum = wrapper.energy_spectrum(frame=frame, levels=10)
    universal_spectrum = universal.energy_spectrum(frame=frame, levels=10)

    assert np.allclose(wrapper_spectrum.energies, universal_spectrum.energies, atol=1.0e-12)
    assert [level.dominant_basis_label for level in wrapper_spectrum.levels] == [
        level.dominant_basis_label for level in universal_spectrum.levels
    ]


def test_diagonal_cavity_only_model_has_expected_bare_labels_and_energies():
    model = UniversalCQEDModel(
        transmon=None,
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=2.0,
                dim=4,
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
        ),
    )

    spectrum = compute_energy_spectrum(model)

    assert np.allclose(spectrum.energies, np.array([0.0, 2.0, 4.0, 6.0]), atol=1.0e-12)
    assert [level.dominant_basis_label for level in spectrum.levels] == ["|0>", "|1>", "|2>", "|3>"]


def test_lab_frame_level_spacing_matches_sideband_transition_frequency():
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=2.3,
        omega_r=7.4,
        omega_q=5.1,
        alpha=-0.31,
        chi_s=-0.05,
        chi_r=0.07,
        chi_sr=0.015,
        kerr_s=-0.01,
        kerr_r=0.025,
        n_storage=3,
        n_readout=2,
        n_tr=3,
    )

    spectrum = compute_energy_spectrum(model)
    gf_storage = spectrum.find_level("|f,0_storage,0_readout>")
    g_storage_1 = spectrum.find_level("|g,1_storage,0_readout>")

    expected = model.sideband_transition_frequency(
        mode="storage",
        storage_level=0,
        readout_level=0,
        lower_level=0,
        upper_level=2,
    )
    observed = gf_storage.energy - g_storage_1.energy

    assert np.isclose(observed, expected, atol=1.0e-12)


def test_plot_energy_levels_returns_matplotlib_figure():
    model = UniversalCQEDModel(
        transmon=None,
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=2.0,
                dim=4,
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
        ),
    )
    spectrum = compute_energy_spectrum(model)

    fig = plot_energy_levels(spectrum, max_levels=4, energy_unit_label="arb.")
    try:
        assert len(fig.axes) == 1
        assert fig.axes[0].collections
    finally:
        plt.close(fig)
