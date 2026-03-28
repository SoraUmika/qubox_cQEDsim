from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core import coherent_state, fock_state
from examples.workflows.kerr_free_evolution import (
    build_kerr_free_evolution_model,
    resolve_kerr_parameter_set,
    run_kerr_free_evolution,
    times_us_to_seconds,
)


def test_parameter_set_switch_changes_coefficients():
    phase = resolve_kerr_parameter_set("phase_evolution")
    alt = resolve_kerr_parameter_set("value_2")
    assert phase.kerr_hz != alt.kerr_hz
    assert phase.chi_hz != alt.chi_hz
    assert phase.chi2_hz != alt.chi2_hz


def test_kerr_free_evolution_trace_preserved_for_reduced_cavity_state():
    result = run_kerr_free_evolution(
        times_us_to_seconds([0.0, 1.0, 4.0]),
        cavity_state=coherent_state(0.8 + 0.2j),
        parameter_set="phase_evolution",
        n_cav=18,
        wigner_times_s=[],
    )
    for snapshot in result.snapshots:
        assert np.isclose(float(np.real(snapshot.cavity_state.tr())), 1.0, atol=1.0e-9)


def test_kerr_free_evolution_t0_is_identity():
    result = run_kerr_free_evolution(
        times_us_to_seconds([0.0]),
        cavity_state=coherent_state(0.35 - 0.15j),
        parameter_set="phase_evolution",
        n_cav=16,
        wigner_times_s=[],
    )
    assert (result.snapshots[0].joint_state - result.initial_state).norm() < 1.0e-12


def test_fock_state_undergoes_phase_only_evolution():
    n = 4
    time_s = times_us_to_seconds([6.0])[0]
    result = run_kerr_free_evolution(
        [time_s],
        cavity_state=fock_state(n),
        parameter_set="phase_evolution",
        n_cav=12,
        wigner_times_s=[],
    )
    snapshot = result.snapshots[0]
    overlap = result.initial_state.overlap(snapshot.joint_state)
    expected = np.exp(-1j * result.model.basis_energy(0, n, frame=result.frame) * time_s)
    assert np.isclose(np.abs(overlap), 1.0, atol=1.0e-9)
    assert np.allclose(overlap / np.abs(overlap), expected, atol=1.0e-8)
    target = qt.basis(result.model.n_cav, n).proj()
    assert (snapshot.cavity_state - target).norm() < 1.0e-10


def test_coherent_state_evolution_shows_nonlinear_distortion():
    result = run_kerr_free_evolution(
        times_us_to_seconds([12.0]),
        cavity_state=coherent_state(1.8),
        parameter_set="phase_evolution",
        n_cav=30,
        wigner_times_s=[],
    )
    snapshot = result.snapshots[0]
    reference = qt.coherent_dm(result.model.n_cav, snapshot.cavity_mean)
    fidelity = qt.fidelity(snapshot.cavity_state, reference)
    assert fidelity < 0.995


def test_default_wigner_snapshots_use_alpha_coordinates():
    result = run_kerr_free_evolution(
        times_us_to_seconds([0.0]),
        cavity_state=coherent_state(2.0),
        parameter_set="phase_evolution",
        n_cav=30,
        wigner_times_s=times_us_to_seconds([0.0]),
        wigner_n_points=91,
        wigner_extent=4.6,
    )
    snapshot = result.snapshots[0]
    assert result.metadata["wigner_coordinate"] == "alpha"
    assert snapshot.wigner is not None

    peak_index = np.unravel_index(np.argmax(snapshot.wigner["w"]), snapshot.wigner["w"].shape)
    peak_x = float(snapshot.wigner["xvec"][peak_index[1]])
    peak_y = float(snapshot.wigner["yvec"][peak_index[0]])
    assert np.isclose(peak_x, 2.0, atol=0.15)
    assert np.isclose(peak_y, 0.0, atol=0.15)


def test_value_2_model_uses_alternate_coefficients():
    model = build_kerr_free_evolution_model("value_2", n_cav=10)
    phase = resolve_kerr_parameter_set("phase_evolution")
    assert not np.isclose(model.kerr, 2.0 * np.pi * phase.kerr_hz)
