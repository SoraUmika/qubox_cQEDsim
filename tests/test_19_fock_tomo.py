from __future__ import annotations

import time

import numpy as np
import pytest
import qutip as qt

from cqed_sim.core.frequencies import drive_frequency_from_internal_carrier
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.tomo.device import DeviceParameters
from cqed_sim.tomo.protocol import (
    QubitPulseCal,
    autocalibrate_all_xy,
    calibrate_leakage_matrix,
    run_all_xy,
    run_fock_resolved_tomo,
    selective_pi_pulse,
    selective_qubit_drive_frequency,
    selective_qubit_freq,
    true_fock_resolved_vectors,
)
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _toy_model(n_cav: int = 8, n_tr: int = 2) -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2 * np.pi * 0.020,
        chi_higher=(),
        kerr=0.0,
        n_cav=n_cav,
        n_tr=n_tr,
    )


def _rmse(v_hat: dict[str, np.ndarray], v_true: dict[str, np.ndarray]) -> float:
    x = np.concatenate([v_hat[a] - v_true[a] for a in ("x", "y", "z")])
    return float(np.sqrt(np.mean(x**2)))


def test_unit_convention_ringdown_ro_kappa_matches_choice():
    p = DeviceParameters()
    # Convention: convert Hz -> rad/ns for Hamiltonian-rate style quantities.
    kappa = p.hz_to_rad_per_ns(p.ro_kappa)
    assert np.isclose(kappa, 2 * np.pi * p.ro_kappa * 1e-9)


def test_all_xy_passes_after_autocal():
    start = time.perf_counter()
    model = _toy_model(n_cav=4, n_tr=2)
    bad = QubitPulseCal(amp90=QubitPulseCal.nominal().amp90 * 0.75, y_phase=np.pi / 2 + 0.35, drag=0.02)
    before = run_all_xy(model, bad, dt_ns=0.2)
    cal, after = autocalibrate_all_xy(model, bad, dt_ns=0.2, max_iter=10, target_rms=0.08)
    assert after["rms_error"] < before["rms_error"] * 0.6
    assert after["rms_error"] < 0.12
    assert cal.amp90 > 0.0
    assert (time.perf_counter() - start) < 6.0


def test_fock_tomo_ideal_recovers_exact_blocks():
    model = _toy_model(n_cav=6, n_tr=2)
    n_max = 3
    p = 0.35
    rho = p * model.basis_state( 0,0).proj() + (1 - p) * model.basis_state( 1,1).proj()
    tomo = run_fock_resolved_tomo(
        model=model,
        state_prep=lambda: rho,
        n_max=n_max,
        cal=QubitPulseCal.nominal(),
        ideal_tag=True,
        pre_rotation_mode="ideal",
        tag_duration_ns=400.0,
        dt_ns=1.0,
    )
    v_true = true_fock_resolved_vectors(rho, n_max=n_max)
    assert _rmse(tomo.v_hat, v_true) < 5e-3


def test_fock_tomo_realistic_baseline_on_known_states():
    model = _toy_model(n_cav=8, n_tr=2)
    n_max = 3
    alpha = 0.9
    r = np.array([1.0, 0.0, 0.0], dtype=float)
    rho_q = 0.5 * (qt.qeye(2) + r[0] * qt.sigmax() + r[1] * qt.sigmay() + r[2] * qt.sigmaz())
    rho = qt.tensor( rho_q,qt.coherent(model.n_cav, alpha).proj())
    tomo = run_fock_resolved_tomo(
        model=model,
        state_prep=lambda: rho,
        n_max=n_max,
        cal=QubitPulseCal.nominal(),
        ideal_tag=False,
        tag_duration_ns=1200.0,
        tag_amp=0.0014,
        dt_ns=1.0,
        noise=NoiseSpec(t1=9000.0, tphi=7000.0),
    )
    v_true = true_fock_resolved_vectors(rho, n_max=n_max)
    err = _rmse(tomo.v_hat, v_true)
    assert err < 0.35
    assert err > 0.01


@pytest.mark.slow
def test_leakage_calibration_unmixing_improves_rmse():
    start = time.perf_counter()
    model = _toy_model(n_cav=5, n_tr=2)
    n_max = 1
    cal = QubitPulseCal.nominal()
    alphas = [0.4, 1.0]
    bloch_cal = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    w, b, cond = calibrate_leakage_matrix(
        model=model,
        n_max=n_max,
        alphas=alphas,
        bloch_states=bloch_cal,
        cal=cal,
        tag_duration_ns=180.0,
        tag_amp=0.0015,
        dt_ns=1.0,
    )
    assert cond < 1e8

    # Science state: coherent cavity with +X qubit.
    rho_q = 0.5 * (qt.qeye(2) + qt.sigmax())
    rho = qt.tensor( rho_q,qt.coherent(model.n_cav, 0.8).proj())
    tomo_base = run_fock_resolved_tomo(
        model=model, state_prep=lambda: rho, n_max=n_max, cal=cal, ideal_tag=False, tag_duration_ns=180.0, tag_amp=0.0015, dt_ns=1.0
    )
    tomo_fix = run_fock_resolved_tomo(
        model=model,
        state_prep=lambda: rho,
        n_max=n_max,
        cal=cal,
        ideal_tag=False,
        tag_duration_ns=180.0,
        tag_amp=0.0015,
        dt_ns=1.0,
        leakage_cal=(w, b),
    )
    v_true = true_fock_resolved_vectors(rho, n_max=n_max)
    err_base = _rmse(tomo_base.v_hat, v_true)
    err_fix = _rmse(tomo_fix.v_rec, v_true)  # type: ignore[arg-type]
    assert err_fix < err_base
    assert (time.perf_counter() - start) < 100.0


def test_selective_pi_minimizes_offmanifold_action():
    model = _toy_model(n_cav=6, n_tr=2)
    # build once per target n
    for n in [0, 1, 2]:
        tag = selective_pi_pulse(n=n, t0_ns=0.0, duration_ns=1200.0, amp=0.0014, model=model, drag=0.0)
        comp = SequenceCompiler(dt=1.0).compile([tag], t_end=1201.0)
        for m in [0, 1, 2, 3]:
            if m == n:
                continue
            res = simulate_sequence(
                model,
                comp,
                model.basis_state( 0,m),
                {"q": "qubit"},
                SimulationConfig(frame=FrameSpec(omega_q_frame=model.omega_q)),
            )
            pe = float(np.real((qt.ptrace(res.final_state, 0) * (qt.basis(2, 1) * qt.basis(2, 1).dag())).tr()))
            assert pe < 0.12


def test_selective_qubit_drive_frequency_round_trips_through_raw_carrier():
    model = _toy_model(n_cav=6, n_tr=2)
    for n in [0, 1, 2]:
        drive_frequency = selective_qubit_drive_frequency(model, n)
        carrier = selective_qubit_freq(model, n)
        pulse = selective_pi_pulse(n=n, t0_ns=0.0, duration_ns=1200.0, amp=0.0014, model=model, drag=0.0)

        assert np.isclose(pulse.carrier, carrier, atol=1.0e-12)
        assert np.isclose(
            drive_frequency_from_internal_carrier(carrier, model.omega_q),
            drive_frequency,
            atol=1.0e-12,
        )


def test_fock_tomo_runtime_budget():
    start = time.perf_counter()
    model = _toy_model(n_cav=5, n_tr=2)
    n_max = 1
    rho = qt.tensor( qt.basis(2, 0).proj(),qt.coherent(model.n_cav, 0.6).proj())
    _ = run_fock_resolved_tomo(
        model=model,
        state_prep=lambda: rho,
        n_max=n_max,
        cal=QubitPulseCal.nominal(),
        ideal_tag=False,
        tag_duration_ns=250.0,
        tag_amp=0.0012,
        dt_ns=1.0,
        noise=NoiseSpec(t1=9000.0, tphi=7000.0),
    )
    assert (time.perf_counter() - start) < 8.0
