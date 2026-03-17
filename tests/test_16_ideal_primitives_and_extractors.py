from __future__ import annotations

import time

import numpy as np
import qutip as qt

from cqed_sim.core.ideal_gates import (
    cavity_block_phase_op,
    displacement_op,
    embed_cavity_op,
    embed_qubit_op,
    logical_block_phase_op,
    qubit_rotation_axis,
    qubit_rotation_xy,
    snap_op,
    sqr_op,
)
from cqed_sim.core.conventions import qubit_cavity_block_indices
from cqed_sim.sim.extractors import (
    bloch_xyz_from_joint,
    bloch_xyz_from_qubit_state,
    cavity_moments,
    cavity_wigner,
    conditioned_bloch_xyz,
    conditioned_qubit_state,
)


def _joint_g(n_cav: int = 8) -> qt.Qobj:
    return qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0))


def test_ideal_rotation_mapping_x90_and_y90():
    start = time.perf_counter()
    psi = _joint_g(4)
    ux = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "x"), 4)
    uy = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "y"), 4)
    x = bloch_xyz_from_joint(ux * psi)
    y = bloch_xyz_from_joint(uy * psi)
    assert np.allclose(x, (0.0, -1.0, 0.0), atol=2e-3)
    assert np.allclose(y, (1.0, 0.0, 0.0), atol=2e-3)
    assert (time.perf_counter() - start) < 0.8


def test_ideal_rotation_composition_x90_x90_equals_x180():
    start = time.perf_counter()
    psi = _joint_g(4)
    u90 = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "x"), 4)
    u180 = embed_qubit_op(qubit_rotation_axis(np.pi, "x"), 4)
    assert abs(((u90 * u90 * psi).overlap(u180 * psi))) > 1 - 1e-10
    assert (time.perf_counter() - start) < 0.6


def test_ideal_rotation_inverse_cancels():
    start = time.perf_counter()
    psi = _joint_g(4)
    u = embed_qubit_op(qubit_rotation_xy(0.73, 0.41), 4)
    ui = embed_qubit_op(qubit_rotation_xy(-0.73, 0.41), 4)
    assert abs((ui * u * psi).overlap(psi)) > 1 - 1e-10
    assert (time.perf_counter() - start) < 0.6


def test_ideal_rotation_noncommuting_axes():
    start = time.perf_counter()
    psi = _joint_g(4)
    ux = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "x"), 4)
    uy = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "y"), 4)
    f = abs((ux * uy * psi).overlap(uy * ux * psi)) ** 2
    assert f < 0.95
    assert (time.perf_counter() - start) < 0.6


def test_displacement_vacuum_to_coherent_moments():
    start = time.perf_counter()
    n_cav = 20
    alpha = 0.45 + 0.2j
    d = displacement_op(n_cav, alpha)
    vac = qt.basis(n_cav, 0)
    psi = d * vac
    a = qt.destroy(n_cav)
    assert np.isclose(qt.expect(a, psi), alpha, atol=3e-3)
    assert np.isclose(qt.expect(a.dag() * a, psi), abs(alpha) ** 2, atol=6e-3)
    # Identity D^\dag a D = a + alpha on a low-lying state (finite truncation safe).
    low = (qt.basis(n_cav, 0) + 0.4 * qt.basis(n_cav, 1) - 0.2j * qt.basis(n_cav, 2)).unit()
    lhs = qt.expect(d.dag() * a * d, low)
    rhs = qt.expect(a + alpha * qt.qeye(n_cav), low)
    assert np.isclose(lhs, rhs, atol=5e-3)
    assert (time.perf_counter() - start) < 1.0


def test_displacement_composition_and_phase():
    start = time.perf_counter()
    n_cav = 20
    a = 0.2 + 0.1j
    b = -0.15 + 0.22j
    vac = qt.basis(n_cav, 0)
    psi1 = displacement_op(n_cav, a) * displacement_op(n_cav, b) * vac
    psi2 = displacement_op(n_cav, a + b) * vac
    op = qt.destroy(n_cav)
    assert np.isclose(qt.expect(op, psi1), qt.expect(op, psi2), atol=5e-3)
    assert np.isclose(qt.expect(op.dag() * op, psi1), qt.expect(op.dag() * op, psi2), atol=7e-3)
    assert (time.perf_counter() - start) < 0.9


def test_displacement_undo():
    start = time.perf_counter()
    n_cav = 16
    alpha = 0.35 - 0.14j
    d = displacement_op(n_cav, alpha)
    dd = displacement_op(n_cav, -alpha)
    vac = qt.basis(n_cav, 0)
    assert abs((dd * d * vac).overlap(vac)) > 1 - 1e-10
    rho = qt.rand_dm(n_cav, seed=23)
    out = dd * d * rho * d.dag() * dd.dag()
    assert (out - rho).norm() < 1e-10
    assert (time.perf_counter() - start) < 0.9


def test_snap_acts_diagonal_in_fock():
    start = time.perf_counter()
    n_cav = 6
    phases = np.array([0.2, -0.3, 0.7, 0.0, 0.0, 0.0])
    u = snap_op(phases)
    psi = (qt.basis(n_cav, 0) + qt.basis(n_cav, 1) + qt.basis(n_cav, 2)).unit()
    out = u * psi
    pops0 = np.abs(psi.full().ravel()) ** 2
    pops1 = np.abs(out.full().ravel()) ** 2
    assert np.allclose(pops0, pops1, atol=1e-12)
    amps0 = psi.full().ravel()
    amps1 = out.full().ravel()
    for n in [0, 1, 2]:
        dphi = np.angle(amps1[n] / amps0[n])
        assert abs(np.angle(np.exp(1j * (dphi - phases[n])))) < 1e-10
    assert (time.perf_counter() - start) < 0.8


def test_snap_identity_when_all_phases_zero():
    n_cav = 8
    u = snap_op(np.zeros(n_cav))
    assert (u - qt.qeye(n_cav)).norm() < 1e-12


def test_snap_global_phase_irrelevance():
    n_cav = 8
    ph = np.linspace(0.0, 0.7, n_cav)
    psi = qt.rand_ket(n_cav, seed=33)
    o1 = snap_op(ph) * psi
    o2 = snap_op(ph + 0.4) * psi
    a = qt.destroy(n_cav)
    assert np.isclose(qt.expect(a, o1), qt.expect(a, o2), atol=1e-10)
    assert np.isclose(qt.expect(a.dag() * a, o1), qt.expect(a.dag() * a, o2), atol=1e-10)


def test_cavity_block_phase_targets_selected_levels_only():
    n_cav = 6
    u = cavity_block_phase_op((0.25, -0.4), fock_levels=(1, 4), cavity_dim=n_cav)
    diag = np.diag(np.asarray(u.full(), dtype=np.complex128))
    expected = np.ones(n_cav, dtype=np.complex128)
    expected[1] = np.exp(1j * 0.25)
    expected[4] = np.exp(1j * -0.4)
    assert np.allclose(diag, expected, atol=1e-12)


def test_logical_block_phase_embeds_identically_on_each_qubit_block():
    n_cav = 5
    phases = (0.1, -0.35)
    levels = (0, 3)
    u = logical_block_phase_op(phases, fock_levels=levels, cavity_dim=n_cav)
    full = np.asarray(u.full(), dtype=np.complex128)
    for phase, level in zip(phases, levels, strict=True):
        idx = qubit_cavity_block_indices(n_cav, level)
        block = full[np.ix_(idx, idx)]
        assert np.allclose(block, np.exp(1j * phase) * np.eye(2), atol=1e-12)
    untouched = qubit_cavity_block_indices(n_cav, 2)
    assert np.allclose(full[np.ix_(untouched, untouched)], np.eye(2), atol=1e-12)


def test_sqr_applies_rotation_only_on_target_fock():
    n_cav = 6
    n0 = 2
    thetas = np.zeros(n_cav)
    phis = np.zeros(n_cav)
    thetas[n0] = np.pi / 2
    u = sqr_op(thetas, phis)
    psi = qt.tensor( qt.basis(2, 0),qt.basis(n_cav, n0))
    out = u * psi
    x, y, z = bloch_xyz_from_joint(out)
    assert np.allclose((x, y, z), (0.0, -1.0, 0.0), atol=2e-3)
    # Mixed-n input: n=1 branch unchanged.
    rho = 0.5 * qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 1)).proj() + 0.5 * psi.proj()
    rho_out = u * rho * u.dag()
    x1, y1, z1, p1, _ = conditioned_bloch_xyz(rho_out, 1)
    assert np.isclose(p1, 0.5, atol=1e-8)
    assert np.allclose((x1, y1, z1), (0.0, 0.0, 1.0), atol=2e-3)


def test_sqr_linearity_on_superposition():
    n_cav = 5
    thetas = np.linspace(0, np.pi / 2, n_cav)
    phis = np.linspace(0.1, 0.5, n_cav)
    u = sqr_op(thetas, phis)
    psi = qt.tensor( (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),(qt.basis(n_cav, 0) + qt.basis(n_cav, 2)).unit())
    out1 = u * psi
    out2 = 0 * psi
    for n in range(n_cav):
        out2 += qt.tensor( qubit_rotation_xy(thetas[n], phis[n]),qt.basis(n_cav, n) * qt.basis(n_cav, n).dag()) * psi
    assert abs(out1.overlap(out2)) > 1 - 1e-10


def test_sqr_reduces_to_global_rotation_when_theta_phi_constant():
    n_cav = 6
    theta = np.pi / 3
    phi = 0.2
    u = sqr_op(np.full(n_cav, theta), np.full(n_cav, phi))
    ref = qt.tensor( qubit_rotation_xy(theta, phi),qt.qeye(n_cav))
    assert (u - ref).norm() < 1e-10


def test_bloch_extractor_for_known_pure_states():
    n_cav = 4
    g = qt.tensor(qt.basis(2, 0), qt.basis(n_cav, 0))
    e = qt.tensor(qt.basis(2, 1), qt.basis(n_cav, 0))
    plus_x = qt.tensor( (qt.basis(2, 0) + qt.basis(2, 1)).unit(),qt.basis(n_cav, 0))
    plus_y = qt.tensor( (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit(),qt.basis(n_cav, 0))
    assert np.allclose(bloch_xyz_from_joint(g), (0.0, 0.0, 1.0), atol=2e-3)
    assert np.allclose(bloch_xyz_from_joint(e), (0.0, 0.0, -1.0), atol=2e-3)
    assert np.allclose(bloch_xyz_from_joint(plus_x), (1.0, 0.0, 0.0), atol=2e-3)
    assert np.allclose(bloch_xyz_from_joint(plus_y), (0.0, 1.0, 0.0), atol=2e-3)


def test_bloch_y_matches_sigma_y_expectation():
    rho_q = ((qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()).proj()
    x, y, z = bloch_xyz_from_qubit_state(rho_q)
    assert np.allclose((x, y, z), (0.0, 1.0, 0.0), atol=1e-12)
    assert np.isclose(y, float(np.real((rho_q * qt.sigmay()).tr())), atol=1e-12)


def test_fock_weighted_bloch_contributions_sum_to_joint_trace():
    n_cav = 5
    psi_q0 = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
    psi_q2 = (qt.basis(2, 0) + 1j * qt.basis(2, 1)).unit()
    psi = (
        np.sqrt(0.35) * qt.tensor(psi_q0, qt.basis(n_cav, 0))
        + np.sqrt(0.65) * qt.tensor(psi_q2, qt.basis(n_cav, 2))
    ).unit()
    rho = psi.proj()
    total = np.array(bloch_xyz_from_joint(rho), dtype=float)
    weighted = np.zeros(3, dtype=float)
    population_sum = 0.0

    for n in range(n_cav):
        rho_q_n, p_n, valid = conditioned_qubit_state(rho, n=n, fallback="zero")
        population_sum += p_n
        if valid:
            weighted += p_n * np.array(bloch_xyz_from_qubit_state(rho_q_n), dtype=float)

    assert np.isclose(population_sum, 1.0, atol=1e-12)
    assert np.allclose(weighted, total, atol=1e-12)


def test_conditional_bloch_extractor_matches_postselection():
    n_cav = 4
    p = 0.3
    rho = p * qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0)).proj() + (1 - p) * qt.tensor( qt.basis(2, 1),qt.basis(n_cav, 1)).proj()
    x0, y0, z0, p0, _ = conditioned_bloch_xyz(rho, 0)
    x1, y1, z1, p1, _ = conditioned_bloch_xyz(rho, 1)
    assert np.isclose(p0, p, atol=1e-10) and np.isclose(p1, 1 - p, atol=1e-10)
    assert np.allclose((x0, y0, z0), (0.0, 0.0, 1.0), atol=1e-10)
    assert np.allclose((x1, y1, z1), (0.0, 0.0, -1.0), atol=1e-10)


def test_conditional_extractor_handles_zero_probability():
    n_cav = 5
    rho = qt.tensor( qt.basis(2, 0),qt.basis(n_cav, 0)).proj()
    x, y, z, p, valid = conditioned_bloch_xyz(rho, 3, fallback="nan")
    assert p == 0.0 and not valid
    assert np.isnan(x) and np.isnan(y) and np.isnan(z)


def test_wigner_vacuum_gaussian_and_normalized():
    n_cav = 18
    rho = qt.basis(n_cav, 0).proj()
    x, y, w = cavity_wigner(rho, n_points=41, extent=4.0)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    integ = np.sum(w) * dx * dy
    center = w[len(y) // 2, len(x) // 2]
    assert np.min(w) >= -2e-3
    assert center == np.max(w)
    assert np.isclose(integ, 1.0, atol=0.06)


def test_wigner_fock_state_negativity():
    n_cav = 18
    rho = qt.basis(n_cav, 1).proj()
    _, _, w = cavity_wigner(rho, n_points=41, extent=4.0)
    c = w.shape[0] // 2
    patch = w[c - 2 : c + 3, c - 2 : c + 3]
    assert np.min(patch) < -0.02


def test_wigner_displaced_vacuum_center_shifts():
    n_cav = 20
    alpha0 = 0.7 - 0.5j
    rho = (displacement_op(n_cav, alpha0) * qt.basis(n_cav, 0)).proj()
    x, y, w = cavity_wigner(rho, n_points=51, extent=4.0)
    iy, ix = np.unravel_index(np.argmax(w), w.shape)
    x_peak = x[ix]
    y_peak = y[iy]
    # QuTiP wigner uses x/sqrt(2)+i y/sqrt(2) scaling.
    assert np.isclose(x_peak, np.sqrt(2) * np.real(alpha0), atol=0.35)
    assert np.isclose(y_peak, np.sqrt(2) * np.imag(alpha0), atol=0.35)


def test_wigner_alpha_coordinates_center_match_coherent_amplitude():
    n_cav = 20
    alpha0 = 0.7 - 0.5j
    rho = (displacement_op(n_cav, alpha0) * qt.basis(n_cav, 0)).proj()
    x, y, w = cavity_wigner(rho, n_points=51, extent=3.0, coordinate="alpha")
    iy, ix = np.unravel_index(np.argmax(w), w.shape)
    x_peak = x[ix]
    y_peak = y[iy]
    assert np.isclose(x_peak, np.real(alpha0), atol=0.25)
    assert np.isclose(y_peak, np.imag(alpha0), atol=0.25)


def test_plot_helpers_return_correct_shapes_and_axes():
    n_cav = 12
    rho = qt.basis(n_cav, 0).proj()
    x, y, w = cavity_wigner(rho, n_points=31, extent=3.0)
    assert x.shape == (31,)
    assert y.shape == (31,)
    assert w.shape == (31, 31)
    # Moment helper smoke.
    joint = qt.tensor(qt.basis(2, 0).proj(), rho)
    moments = cavity_moments(joint)
    assert set(moments.keys()) == {"a", "adag_a", "n"}


def test_cavity_moments_handles_trivial_singleton_mode():
    joint = qt.tensor(qt.basis(2, 0), qt.basis(1, 0))
    moments = cavity_moments(joint)
    assert np.isclose(moments["a"], 0.0, atol=1.0e-12)
    assert np.isclose(moments["adag_a"], 0.0, atol=1.0e-12)
    assert np.isclose(moments["n"], 0.0, atol=1.0e-12)
