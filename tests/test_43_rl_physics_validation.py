"""Physics validation tests for the RL-ready cQED simulation stack.

These tests verify that the physics models built through the RL runtime layer
satisfy fundamental physical properties: no-drive stationarity, open-system
trace preservation, dispersive phase accumulation, Kerr evolution, cross-regime
consistency, and leakage behaviour under strong drives.

All tests use the HamiltonianModelFactory and RL simulation machinery directly,
exercising the same code paths the RL environment exercises during training.
"""
from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim import (
    FullPulseModelConfig,
    HybridSystemConfig,
    NormalPrior,
    QubitMeasurementSpec,
    ReducedDispersiveModelConfig,
)
from cqed_sim.sim import NoiseSpec
from cqed_sim.rl_control import (
    HamiltonianModelFactory,
    OpenSystemEngine,
    PulseGenerator,
    DistortionModel,
)
from cqed_sim.rl_control.action_spaces import WaitAction, CavityDisplacementAction, QubitGaussianAction
from cqed_sim.rl_control.metrics import (
    photon_number_distribution,
    parity_expectation,
    state_fidelity,
)


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _reduced_system(*, n_cav: int = 8, n_tr: int = 3, with_noise: bool = False) -> HybridSystemConfig:
    noise = NoiseSpec(t1=40.0e-6, tphi=80.0e-6, kappa=2.0 * np.pi * 1.0e3) if with_noise else None
    return HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=2.0 * np.pi * (-4.0e3),
            n_cav=n_cav,
            n_tr=n_tr,
        ),
        noise=noise,
        dt=4.0e-9,
        max_step=4.0e-9,
    )


def _full_system(*, n_cav: int = 6, n_tr: int = 4) -> HybridSystemConfig:
    return HybridSystemConfig(
        regime="full_pulse",
        full_model=FullPulseModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.0e9,
            alpha=2.0 * np.pi * (-200.0e6),
            exchange_g=2.0 * np.pi * 0.4e6,
            kerr=2.0 * np.pi * (-3.0e3),
            n_cav=n_cav,
            n_tr=n_tr,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )


def _build_bundle(system: HybridSystemConfig) -> "EpisodeModelBundle":
    return HamiltonianModelFactory.build(system)


def _propagate(system: HybridSystemConfig, initial_state: qt.Qobj, action):
    bundle = _build_bundle(system)
    generator = PulseGenerator()
    distortion = DistortionModel(bundle)
    engine = OpenSystemEngine(bundle)
    segment = generator.generate(action, bundle)
    if segment.duration > 0.0 or segment.pulses:
        compiled = distortion.compile(segment)
        result = engine.propagate_state(initial_state, compiled, segment.drive_ops)
        return result.final_state
    return initial_state


# ---------------------------------------------------------------------------
# Test: vacuum state is a fixed point under no-drive evolution
# ---------------------------------------------------------------------------

def test_vacuum_state_is_stationary_under_no_drive_reduced_regime() -> None:
    """
    The vacuum-ground state |g,0> should be essentially unchanged after a
    free-evolution wait in the model's own rotating frame.  Any residual
    drift signals a frame-convention error in the RL runtime.
    """
    system = _reduced_system()
    bundle = _build_bundle(system)
    initial = bundle.model.basis_state(0, 0)
    final = _propagate(system, initial, WaitAction(duration=100.0e-9))
    fid = state_fidelity(final, initial)
    assert fid > 0.9999, f"Vacuum state fidelity after free evolution: {fid:.6f} (expected > 0.9999)"


def test_vacuum_state_is_stationary_under_no_drive_full_regime() -> None:
    """Same test on the full pulse-level model."""
    system = _full_system()
    bundle = _build_bundle(system)
    initial = bundle.model.basis_state(0, 0)
    final = _propagate(system, initial, WaitAction(duration=100.0e-9))
    fid = state_fidelity(final, initial)
    assert fid > 0.9999, f"Full-model vacuum state fidelity after free evolution: {fid:.6f}"


# ---------------------------------------------------------------------------
# Test: cavity photon number decreases under cavity loss
# ---------------------------------------------------------------------------

def test_cavity_photon_number_decreases_under_cavity_loss() -> None:
    """
    Start in |g,1>.  Add a cavity loss rate kappa/2pi = 1 MHz and wait 1 µs.
    The expected photon number should drop below its initial value of 1.

    kappa * t = 2pi * 1e6 * 1e-6 = 2pi ~ 6.28.  After this waiting, the
    population in |1> has decayed substantially toward vacuum.
    """
    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=0.0,
            n_cav=6,
            n_tr=3,
        ),
        noise=NoiseSpec(t1=100.0e-6, tphi=200.0e-6, kappa=2.0 * np.pi * 1.0e6),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    bundle = _build_bundle(system)
    n_tr, n_cav = int(bundle.model.n_tr), int(bundle.model.n_cav)
    initial = qt.tensor(qt.basis(n_tr, 0), qt.basis(n_cav, 1))
    final = _propagate(system, initial, WaitAction(duration=1.0e-6))
    dist_initial = photon_number_distribution(initial)
    dist_final = photon_number_distribution(final)
    mean_n_initial = float(np.sum(np.arange(len(dist_initial)) * dist_initial))
    mean_n_final = float(np.sum(np.arange(len(dist_final)) * dist_final))
    assert mean_n_final < mean_n_initial, (
        f"Photon number should decrease under cavity loss. "
        f"Initial <n>={mean_n_initial:.3f}, final <n>={mean_n_final:.3f}."
    )
    assert mean_n_final < 0.1, (
        f"After kappa*t >> 1, photon number should be near zero. Got <n>={mean_n_final:.3f}."
    )


# ---------------------------------------------------------------------------
# Test: open-system density matrix remains trace-1
# ---------------------------------------------------------------------------

def test_density_matrix_trace_preserved_under_noisy_evolution() -> None:
    """
    Under Lindblad mesolve, the density matrix should remain trace-1
    at all times (trace preservation is a fundamental property of CPTP maps).
    """
    system = _reduced_system(n_cav=6, with_noise=True)
    bundle = _build_bundle(system)
    n_tr, n_cav = int(bundle.model.n_tr), int(bundle.model.n_cav)
    # Start in a non-trivial superposition
    initial = qt.tensor(
        (qt.basis(n_tr, 0) + qt.basis(n_tr, 1)).unit(),
        (qt.basis(n_cav, 0) + qt.basis(n_cav, 1)).unit(),
    )
    rho_initial = initial.proj()
    assert abs(float(rho_initial.tr()) - 1.0) < 1.0e-12

    final = _propagate(system, rho_initial, WaitAction(duration=200.0e-9))
    assert final.isoper, "Expected density matrix output under open-system evolution."
    trace_error = abs(float(final.tr()) - 1.0)
    assert trace_error < 1.0e-6, f"Trace-preservation error: {trace_error:.2e} (expected < 1e-6)."


# ---------------------------------------------------------------------------
# Test: dispersive phase accumulation in the reduced dispersive regime
# ---------------------------------------------------------------------------

def test_dispersive_phase_accumulation_at_pi_shift() -> None:
    """
    In the dispersive regime, the conditional frequency shift chi causes
    the |e,n=1> state to accumulate phase relative to |e,n=0>.

    After a wait time t = pi / |chi|, the relative phase is pi: the |e,0>
    and |e,1> components are pi out of phase.

    We verify this by preparing (|e,0> + |e,1>)/sqrt(2) and checking that
    after t = pi/|chi| the state has acquired the expected relative phase.
    """
    chi = 2.0 * np.pi * (-2.2e6)
    t_pi = abs(np.pi / chi)
    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=chi,
            kerr=0.0,
            n_cav=4,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    bundle = _build_bundle(system)
    n_tr, n_cav = int(bundle.model.n_tr), int(bundle.model.n_cav)
    # Prepare |e> tensor (|0> + |1>)/sqrt(2)
    cavity_sup = (qt.basis(n_cav, 0) + qt.basis(n_cav, 1)).unit()
    initial = qt.tensor(qt.basis(n_tr, 1), cavity_sup)
    final = _propagate(system, initial, WaitAction(duration=float(t_pi)))
    # After t_pi, the cavity superposition should have parity flipped or phase-shifted
    # The even parity contribution should change sign (for chi < 0: |1> component gains -pi phase)
    final_array = np.asarray(final.full(), dtype=np.complex128).reshape(-1)
    initial_array = np.asarray(initial.full(), dtype=np.complex128).reshape(-1)
    # The state must not return to the initial state (it has acquired a relative phase)
    # Fidelity should not be 1.0 (it should be ~0 if perfect pi-shift, but >=0 due to frame effects)
    overlap = abs(float(np.vdot(initial_array, final_array)))
    # If chi is pure dispersive, overlap = |cos(chi * t_pi * 1/2)| for the chosen superposition
    # For t_pi = pi/|chi|: the relative phase of |1> is e^{-i chi t} = e^{i*pi} = -1
    # The state becomes (|0> - |1>)/sqrt(2), overlap with initial = |1/2 - 1/2| * 2 = 0
    assert overlap < 0.1, (
        f"After dispersive pi shift, overlap with initial state should be near zero. "
        f"Got overlap={overlap:.4f}. "
        "This indicates the chi convention or rotation frame may be misconfigured."
    )


# ---------------------------------------------------------------------------
# Test: self-Kerr changes cavity state distribution
# ---------------------------------------------------------------------------

def test_self_kerr_modifies_state_fidelity_with_initial() -> None:
    """
    Cavity self-Kerr introduces a photon-number-dependent phase:
        H_Kerr = (K/2) * n_c * (n_c - 1)

    For a Fock-state superposition (|0> + |1>)/sqrt(2), the |1> component
    acquires phase exp(-i*K*0) = 1 (since K/2 * 1*0 = 0), and |2> acquires
    exp(-i*K*t).  But |1> accrues no extra phase from Kerr since n(n-1)=0
    for n=1.  For (|0> + |2>)/sqrt(2), |2> acquires exp(-i*K*t) phase.

    We use a superposition that changes under Kerr and verify the state
    is no longer the same as the initial state (fidelity < 1).

    We use (|1> + |2>)/sqrt(2).  The |2> component accumulates extra phase
    exp(-i * K/2 * 2 * t) relative to |1> (since n(n-1) for n=2 gives 2).
    After t = pi/K, the relative phase is pi, making the state orthogonal to
    the initial.
    """
    kerr = 2.0 * np.pi * (-100.0e3)   # exaggerated Kerr for fast test
    t_pi = abs(np.pi / kerr)           # K/2 * 2 * t_pi = pi => t_pi = pi/K
    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=0.0,   # turn off dispersive shift to isolate Kerr
            kerr=kerr,
            n_cav=6,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    bundle = _build_bundle(system)
    n_tr, n_cav = int(bundle.model.n_tr), int(bundle.model.n_cav)
    # Prepare |g> tensor (|1> + |2>)/sqrt(2)
    cavity_sup = (qt.basis(n_cav, 1) + qt.basis(n_cav, 2)).unit()
    initial = qt.tensor(qt.basis(n_tr, 0), cavity_sup)
    final = _propagate(system, initial, WaitAction(duration=float(t_pi)))
    fid = state_fidelity(final, initial)
    # After t_pi, the relative phase of |2> wrt |1> is pi, so state is
    # (|1> - |2>)/sqrt(2), orthogonal to the initial (|1> + |2>)/sqrt(2).
    # Fidelity should be near zero.
    assert fid < 0.1, (
        f"After Kerr pi-shift, state should be near-orthogonal to initial. "
        f"Got fidelity={fid:.4f} (expected < 0.1)."
    )


# ---------------------------------------------------------------------------
# Test: reduced and full models agree in the weak-drive, low-excitation regime
# ---------------------------------------------------------------------------

def test_reduced_and_full_model_agree_on_small_displacement() -> None:
    """
    For a very weak cavity displacement (|alpha| << 1), both the reduced
    dispersive model and the full pulse model should produce nearly identical
    final states when targeting a coherent state.

    This verifies that the two regimes share the same low-excitation physics.
    Tolerance is relaxed to allow for exchange coupling and finite truncation
    differences in the full model.
    """
    alpha = 0.15 + 0.0j
    duration = 40.0e-9

    # Target state for both models (constructed from reduced model dimensions)
    reduced_system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.0e9,
            alpha=2.0 * np.pi * (-200.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=2.0 * np.pi * (-3.0e3),
            n_cav=8,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    full_system = _full_system(n_cav=8, n_tr=3)

    r_bundle = _build_bundle(reduced_system)
    f_bundle = _build_bundle(full_system)

    r_initial = r_bundle.model.basis_state(0, 0)
    f_initial = f_bundle.model.basis_state(0, 0)

    action = CavityDisplacementAction(alpha=alpha, duration=duration)

    r_final = _propagate(reduced_system, r_initial, action)
    f_final = _propagate(full_system, f_initial, action)

    # Build the target coherent state independently for each model dimension
    r_dims = tuple(int(d) for d in r_bundle.model.subsystem_dims)
    f_dims = tuple(int(d) for d in f_bundle.model.subsystem_dims)
    n_tr_r, n_cav_r = r_dims[0], r_dims[1]
    n_tr_f, n_cav_f = f_dims[0], f_dims[1]

    target_r = qt.tensor(qt.basis(n_tr_r, 0), qt.coherent(n_cav_r, complex(alpha)))
    target_f = qt.tensor(qt.basis(n_tr_f, 0), qt.coherent(n_cav_f, complex(alpha)))

    fid_r = state_fidelity(r_final, target_r)
    fid_f = state_fidelity(f_final, target_f)

    assert fid_r > 0.95, f"Reduced model coherent displacement fidelity too low: {fid_r:.4f}."
    assert fid_f > 0.90, (
        f"Full model coherent displacement fidelity too low: {fid_f:.4f}. "
        "Exchange coupling and finite truncation are expected to slightly reduce agreement."
    )


# ---------------------------------------------------------------------------
# Test: transmon higher-level leakage under strong resonant drive
# ---------------------------------------------------------------------------

def test_transmon_leakage_under_strong_drive() -> None:
    """
    With a 4-level transmon (|g,e,f,h>), a strong resonant drive on the g-e
    transition will drive some population into the |f> level via off-resonant
    coupling to the e-f transition (DRAG coefficient = 0).

    We verify that the |f> level has nonzero population after a large-angle
    (> 4*pi) pulse without DRAG correction.
    """
    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=0.0,
            n_cav=4,
            n_tr=4,   # must be >= 4 for f-level leakage
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    bundle = _build_bundle(system)
    n_tr = int(bundle.model.n_tr)
    n_cav = int(bundle.model.n_cav)
    initial = qt.tensor(qt.basis(n_tr, 0), qt.basis(n_cav, 0))

    # Very large rotation angle, no DRAG correction -> leakage
    action = QubitGaussianAction(
        theta=8.0 * np.pi,
        phi=0.0,
        detuning=0.0,
        duration=32.0e-9,
        drag=0.0,
    )
    final = _propagate(system, initial, action)

    # Extract f-level (index 2) population of transmon subsystem
    f_projector_full = qt.tensor(qt.basis(n_tr, 2) * qt.basis(n_tr, 2).dag(), qt.qeye(n_cav))
    if final.isket:
        overlap = (final.dag() * f_projector_full * final)
        # overlap may be a Qobj or a scalar depending on QuTiP version
        if hasattr(overlap, "full"):
            f_pop = float(np.real(overlap.full().reshape(-1)[0]))
        else:
            f_pop = float(np.real(complex(overlap)))
    else:
        f_pop = float(np.real((f_projector_full * final).tr()))

    assert f_pop > 1.0e-4, (
        f"Expected nonzero f-level leakage under strong drive with no DRAG. "
        f"Got f-level population = {f_pop:.2e}."
    )


# ---------------------------------------------------------------------------
# Test: Fock state target construction from benchmark tasks
# ---------------------------------------------------------------------------

def test_fock_state_target_has_correct_photon_number() -> None:
    """
    The Fock state preparation task must produce a target state with the
    photon-number distribution peaked at the requested cavity level.
    """
    from cqed_sim import fock_state_preparation_task

    for cavity_level in (1, 2, 3):
        task = fock_state_preparation_task(cavity_level=cavity_level)
        system = _reduced_system(n_cav=8)
        bundle = _build_bundle(system)
        target = task.build_target_state(bundle.model)
        dist = photon_number_distribution(target)
        peak = int(np.argmax(dist))
        assert peak == cavity_level, (
            f"Fock-{cavity_level} target state has peak at n={peak}, expected n={cavity_level}."
        )
        assert dist[cavity_level] > 0.999, (
            f"Fock-{cavity_level} target should have P(n={cavity_level}) ≈ 1. "
            f"Got {dist[cavity_level]:.6f}."
        )


# ---------------------------------------------------------------------------
# Test: cat state parity is correct
# ---------------------------------------------------------------------------

def test_even_cat_state_has_positive_parity() -> None:
    """Even cat state |cat+> = N(|alpha> + |-alpha>) should have positive parity."""
    from cqed_sim import even_cat_preparation_task, odd_cat_preparation_task

    system = _reduced_system(n_cav=12)
    bundle = _build_bundle(system)

    even_cat = even_cat_preparation_task(alpha=1.5 + 0.0j).build_target_state(bundle.model)
    odd_cat = odd_cat_preparation_task(alpha=1.5 + 0.0j).build_target_state(bundle.model)

    even_parity = parity_expectation(even_cat)
    odd_parity = parity_expectation(odd_cat)

    assert even_parity > 0.9, f"Even cat state should have positive parity. Got {even_parity:.4f}."
    assert odd_parity < -0.9, f"Odd cat state should have negative parity. Got {odd_parity:.4f}."


__all__ = []
