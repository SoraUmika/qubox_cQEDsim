from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import RotationGate
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, hamiltonian_time_slices, simulate_sequence
from cqed_sim.simulators.pulse_unitary import build_rotation_pulse


def _normalize_unitary(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    det = np.linalg.det(matrix)
    if abs(det) > 1.0e-15:
        matrix = matrix * np.exp(-0.5j * np.angle(det))
    return matrix


def _rotation_axis(unitary: np.ndarray) -> tuple[float, float, float]:
    u = _normalize_unitary(np.asarray(unitary, dtype=np.complex128))
    trace = np.trace(u)
    cos_half = float(np.clip(np.real(trace / 2.0), -1.0, 1.0))
    theta = float(2.0 * np.arccos(cos_half))
    if theta < 1.0e-12:
        return 1.0, 0.0, 0.0
    sin_half = float(np.sin(theta / 2.0))
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    sy = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    nx = float(np.real(1.0j * np.trace(sx @ u) / (2.0 * sin_half)))
    ny = float(np.real(1.0j * np.trace(sy @ u) / (2.0 * sin_half)))
    nz = float(np.real(1.0j * np.trace(sz @ u) / (2.0 * sin_half)))
    norm = float(np.sqrt(nx * nx + ny * ny + nz * nz))
    if norm > 1.0e-12:
        nx /= norm
        ny /= norm
        nz /= norm
    return nx, ny, nz


def _n0_block(unitary_qobj: qt.Qobj) -> np.ndarray:
    full = np.asarray(unitary_qobj.full(), dtype=np.complex128)
    n_cav = int(unitary_qobj.dims[0][1])
    return full[np.ix_([0, n_cav], [0, n_cav])]


def _two_level_model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)


def _simulate_gaussian(theta: float, phi: float, carrier: float = 0.0) -> tuple[np.ndarray, qt.Qobj]:
    model = _two_level_model()
    cfg = {"duration_rotation_s": 1.0, "rotation_sigma_fraction": 0.18}
    gate = RotationGate(index=0, name="rot", theta=float(theta), phi=float(phi))
    pulses, drive_ops, _ = build_rotation_pulse(gate, cfg)
    base = pulses[0]
    pulse = Pulse(
        channel=base.channel,
        t0=base.t0,
        duration=base.duration,
        envelope=base.envelope,
        carrier=float(carrier),
        phase=base.phase,
        amp=base.amp,
        drag=base.drag,
        label=base.label,
    )
    compiled = SequenceCompiler(dt=0.002).compile([pulse], t_end=1.002)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state( 0,0),
        drive_ops,
        config=SimulationConfig(frame=FrameSpec(), max_step=0.001),
    )
    rho_q = qt.ptrace(result.final_state, 0)
    bloch = np.array(
        [
            float(np.real((rho_q * qt.sigmax()).tr())),
            float(np.real((rho_q * qt.sigmay()).tr())),
            float(np.real((rho_q * qt.sigmaz()).tr())),
        ],
        dtype=float,
    )
    h = hamiltonian_time_slices(model, compiled, drive_ops, frame=FrameSpec())
    u = qt.propagator(h, compiled.tlist, options={"atol": 1.0e-8, "rtol": 1.0e-7, "max_step": 0.001}, tlist=compiled.tlist)[-1]
    return bloch, u


def test_gaussian_i_quadrature_is_x90_with_g_to_minus_y():
    bloch, _ = _simulate_gaussian(theta=np.pi / 2.0, phi=0.0)
    assert np.isclose(bloch[0], 0.0, atol=0.03)
    assert bloch[1] < -0.95
    assert abs(bloch[2]) < 0.03


def test_gaussian_q_quadrature_is_y90_with_g_to_plus_x():
    bloch, _ = _simulate_gaussian(theta=np.pi / 2.0, phi=np.pi / 2.0)
    assert bloch[0] > 0.95
    assert np.isclose(bloch[1], 0.0, atol=0.03)
    assert abs(bloch[2]) < 0.03


def test_gaussian_phase_sweep_matches_cos_phi_sin_phi_axis():
    for phi in (0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0):
        _, u = _simulate_gaussian(theta=np.pi / 3.0, phi=float(phi))
        nx, ny, nz = _rotation_axis(_n0_block(u))
        assert np.isclose(nx, np.cos(phi), atol=0.08)
        assert np.isclose(ny, np.sin(phi), atol=0.08)
        assert abs(nz) < 0.04


def test_detuning_sign_flips_z_error_axis():
    _, u_res = _simulate_gaussian(theta=np.pi / 2.0, phi=0.0, carrier=0.0)
    _, u_plus = _simulate_gaussian(theta=np.pi / 2.0, phi=0.0, carrier=2.0 * np.pi * 0.25)
    _, u_minus = _simulate_gaussian(theta=np.pi / 2.0, phi=0.0, carrier=-2.0 * np.pi * 0.25)
    err_plus = _normalize_unitary(_n0_block(u_res).conj().T @ _n0_block(u_plus))
    err_minus = _normalize_unitary(_n0_block(u_res).conj().T @ _n0_block(u_minus))
    _, _, nz_plus = _rotation_axis(err_plus)
    _, _, nz_minus = _rotation_axis(err_minus)
    assert np.sign(nz_plus) == -np.sign(nz_minus)


def test_single_tone_multitone_and_gaussian_match_iq_convention():
    model = _two_level_model()
    cfg = {"duration_rotation_s": 1.0, "rotation_sigma_fraction": 0.18}
    theta = np.pi / 2.0
    phi = np.pi / 2.0

    gate = RotationGate(index=0, name="rot", theta=float(theta), phi=float(phi))
    pulses, drive_ops, _ = build_rotation_pulse(gate, cfg)
    compiled_gaussian = SequenceCompiler(dt=0.002).compile(pulses, t_end=1.002)
    h_gaussian = hamiltonian_time_slices(model, compiled_gaussian, drive_ops, frame=FrameSpec())
    u_gaussian = qt.propagator(
        h_gaussian,
        compiled_gaussian.tlist,
        options={"atol": 1.0e-8, "rtol": 1.0e-7, "max_step": 0.001},
        tlist=compiled_gaussian.tlist,
    )[-1]

    def normalized_gaussian_env(t_rel: np.ndarray) -> np.ndarray:
        sigma = 0.18
        base = np.exp(-0.5 * ((np.asarray(t_rel, dtype=float) - 0.5) / sigma) ** 2).astype(np.complex128)
        area = float(np.trapz(np.real(base), np.asarray(t_rel, dtype=float)))
        return base / area if abs(area) > 1.0e-15 else base

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        t_rel = np.asarray(t_rel, dtype=float)
        env = normalized_gaussian_env(t_rel)
        coeff = (theta / 2.0) * np.exp(1j * phi)
        return env * coeff

    pulse_multitone = Pulse(channel="qubit", t0=0.0, duration=1.0, envelope=envelope, amp=1.0, phase=0.0, carrier=0.0)
    compiled_multitone = SequenceCompiler(dt=0.002).compile([pulse_multitone], t_end=1.002)
    h_multitone = hamiltonian_time_slices(model, compiled_multitone, {"qubit": "qubit"}, frame=FrameSpec())
    u_multitone = qt.propagator(
        h_multitone,
        compiled_multitone.tlist,
        options={"atol": 1.0e-8, "rtol": 1.0e-7, "max_step": 0.001},
        tlist=compiled_multitone.tlist,
    )[-1]

    overlap = np.trace(
        _normalize_unitary(_n0_block(u_gaussian)).conj().T
        @ _normalize_unitary(_n0_block(u_multitone))
    )
    process_fid = float(np.abs(overlap) ** 2 / 4.0)
    assert process_fid > 0.999
