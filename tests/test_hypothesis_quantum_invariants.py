"""Property-based tests for quantum mechanical invariants using Hypothesis.

These tests verify structural guarantees that must hold for *any* valid
parameters — unitarity of propagators, trace preservation of Lindblad maps,
Hermiticity of static Hamiltonians, and fidelity bounds.  They complement the
existing deterministic tests by auto-generating edge-case inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from scipy.linalg import expm

from cqed_sim.backends.numpy_backend import NumPyBackend
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_hermitian(rng: np.random.Generator, dim: int) -> np.ndarray:
    M = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return (M + M.conj().T) / 2.0


def _random_pure_state(rng: np.random.Generator, dim: int) -> np.ndarray:
    psi = rng.standard_normal(dim) + 1j * rng.standard_normal(dim)
    return psi / np.linalg.norm(psi)


def _random_density_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    psi = _random_pure_state(rng, dim)
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Category 1: Propagator unitarity
# ---------------------------------------------------------------------------

@given(
    dim=st.integers(min_value=2, max_value=8),
    dt=st.floats(min_value=1e-4, max_value=2.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=60, deadline=8000, suppress_health_check=[HealthCheck.too_slow])
def test_propagator_is_unitary(dim: int, dt: float, seed: int) -> None:
    """expm(-i*dt*H) must be unitary for any Hermitian H."""
    rng = np.random.default_rng(seed)
    H = _random_hermitian(rng, dim)
    U = expm(-1j * dt * H)
    residual = np.max(np.abs(U @ U.conj().T - np.eye(dim)))
    assert residual < 1e-9, f"Propagator non-unitary: max residual={residual:.2e}, dim={dim}, dt={dt}"


@given(
    dim=st.integers(min_value=2, max_value=6),
    n_steps=st.integers(min_value=1, max_value=10),
    dt=st.floats(min_value=1e-5, max_value=0.5, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
def test_product_of_unitaries_is_unitary(dim: int, n_steps: int, dt: float, seed: int) -> None:
    """Product of piecewise-constant unitaries must remain unitary."""
    rng = np.random.default_rng(seed)
    U_total = np.eye(dim, dtype=complex)
    for _ in range(n_steps):
        H = _random_hermitian(rng, dim)
        U_total = expm(-1j * dt * H) @ U_total
    residual = np.max(np.abs(U_total @ U_total.conj().T - np.eye(dim)))
    assert residual < 1e-8, f"Product propagator non-unitary: residual={residual:.2e}"


# ---------------------------------------------------------------------------
# Category 2: Lindbladian trace preservation (NumPyBackend)
# ---------------------------------------------------------------------------

@given(
    dim=st.integers(min_value=2, max_value=5),
    dt=st.floats(min_value=1e-4, max_value=0.05, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=15000, suppress_health_check=[HealthCheck.too_slow])
def test_lindbladian_preserves_trace(dim: int, dt: float, seed: int) -> None:
    """Lindblad propagation must preserve tr(rho) = 1."""
    rng = np.random.default_rng(seed)
    backend = NumPyBackend()
    H = backend.asarray(_random_hermitian(rng, dim))
    # Small collapse operator (amplitude ~ 0.1 so exp doesn't blow up)
    c_raw = (rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) * 0.1
    c = backend.asarray(c_raw)
    rho = backend.asarray(_random_density_matrix(rng, dim))

    L = backend.lindbladian(H, [c])
    prop = backend.expm(L * dt)

    rho_vec = np.asarray(rho, dtype=np.complex128).T.reshape(-1)
    rho_final_vec = np.asarray(prop, dtype=np.complex128) @ rho_vec
    rho_final = rho_final_vec.reshape(dim, dim).T

    trace_residual = abs(np.trace(rho_final) - 1.0)
    assert trace_residual < 1e-7, f"Trace not preserved: |tr - 1| = {trace_residual:.2e}, dim={dim}, dt={dt}"


@given(
    dim=st.integers(min_value=2, max_value=4),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=30, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
def test_lindbladian_has_zero_trace_eigenvector(dim: int, seed: int) -> None:
    """The Lindbladian must have a zero-trace eigenvector structure (trace of density col = 0)."""
    rng = np.random.default_rng(seed)
    backend = NumPyBackend()
    H = backend.asarray(_random_hermitian(rng, dim))
    c = backend.asarray((rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))) * 0.1)
    L = np.asarray(backend.lindbladian(H, [c]))
    # Trace-one density matrix vectorized: rho_ii contribute to trace
    # All trace-zero rho must be in the kernel of the trace map
    # Concretely: sum of diagonal of L @ vec(rho) should be zero for trace-preserving L
    rho = _random_density_matrix(rng, dim)
    rho_vec = rho.T.reshape(-1)
    L_rho_vec = L @ rho_vec
    L_rho = L_rho_vec.reshape(dim, dim).T
    assert abs(np.trace(L_rho)) < 1e-9


# ---------------------------------------------------------------------------
# Category 3: Static Hamiltonian Hermiticity
# ---------------------------------------------------------------------------

@given(
    n_cav=st.integers(min_value=2, max_value=6),
    n_tr=st.integers(min_value=2, max_value=3),
    chi=st.floats(min_value=-0.5, max_value=0.5, allow_nan=False, allow_infinity=False),
    alpha=st.floats(min_value=-2.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    kerr=st.floats(min_value=-0.1, max_value=0.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=40, deadline=10000, suppress_health_check=[HealthCheck.too_slow])
def test_static_hamiltonian_is_hermitian(
    n_cav: int, n_tr: int, chi: float, alpha: float, kerr: float
) -> None:
    """H_static must be Hermitian for any valid model parameters."""
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=alpha,
        chi=chi,
        kerr=kerr,
        n_cav=n_cav,
        n_tr=n_tr,
    )
    H = model.static_hamiltonian(FrameSpec())
    H_arr = np.asarray(H.full(), dtype=np.complex128)
    residual = np.max(np.abs(H_arr - H_arr.conj().T))
    assert residual < 1e-11, (
        f"H not Hermitian: max|H - H†| = {residual:.2e}, "
        f"n_cav={n_cav}, n_tr={n_tr}, chi={chi:.3g}, alpha={alpha:.3g}"
    )


@given(
    omega_q=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    omega_c=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    chi=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=8000, suppress_health_check=[HealthCheck.too_slow])
def test_static_hamiltonian_is_real_diagonal_for_zero_coupling(
    omega_q: float, omega_c: float, chi: float
) -> None:
    """Diagonal elements of H_static must be real."""
    model = DispersiveTransmonCavityModel(
        omega_c=omega_c, omega_q=omega_q, alpha=-0.3, chi=chi, kerr=0.0, n_cav=3, n_tr=2
    )
    H = model.static_hamiltonian(FrameSpec())
    H_arr = np.asarray(H.full(), dtype=np.complex128)
    diag_imag = np.max(np.abs(np.imag(np.diag(H_arr))))
    assert diag_imag < 1e-12, f"Diagonal has imaginary part: {diag_imag:.2e}"


# ---------------------------------------------------------------------------
# Category 4: Quantum fidelity bounds [0, 1]
# ---------------------------------------------------------------------------

@given(
    dim=st.integers(min_value=2, max_value=16),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=100, deadline=2000)
def test_state_fidelity_is_in_unit_interval(dim: int, seed: int) -> None:
    """|<ψ₁|ψ₂>|² must lie in [0, 1] for any two normalized states."""
    rng = np.random.default_rng(seed)
    psi1 = _random_pure_state(rng, dim)
    psi2 = _random_pure_state(rng, dim)
    fidelity = abs(np.vdot(psi1, psi2)) ** 2
    assert 0.0 - 1e-12 <= fidelity <= 1.0 + 1e-12, (
        f"Fidelity out of [0,1]: {fidelity:.6f}, dim={dim}"
    )


@given(
    dim=st.integers(min_value=2, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=60, deadline=4000)
def test_self_fidelity_is_one(dim: int, seed: int) -> None:
    """Self-fidelity |<ψ|ψ>|² must be exactly 1."""
    rng = np.random.default_rng(seed)
    psi = _random_pure_state(rng, dim)
    fidelity = abs(np.vdot(psi, psi)) ** 2
    assert abs(fidelity - 1.0) < 1e-12, f"Self-fidelity != 1: {fidelity:.15f}"


@given(
    dim=st.integers(min_value=2, max_value=8),
    dt=st.floats(min_value=1e-4, max_value=1.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=8000, suppress_health_check=[HealthCheck.too_slow])
def test_unitary_evolution_preserves_norm(dim: int, dt: float, seed: int) -> None:
    """Unitary evolution must preserve state norm (‖U|ψ>‖ = 1)."""
    rng = np.random.default_rng(seed)
    H = _random_hermitian(rng, dim)
    psi = _random_pure_state(rng, dim)
    U = expm(-1j * dt * H)
    psi_evolved = U @ psi
    norm = np.linalg.norm(psi_evolved)
    assert abs(norm - 1.0) < 1e-9, f"Norm not preserved: ‖U|ψ>‖ = {norm:.12f}"


# ---------------------------------------------------------------------------
# Category 5: NumPyBackend expm consistency
# ---------------------------------------------------------------------------

@given(
    dim=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=40, deadline=6000, suppress_health_check=[HealthCheck.too_slow])
def test_numpy_backend_expm_matches_scipy(dim: int, seed: int) -> None:
    """NumPyBackend.expm must agree with scipy.linalg.expm to machine precision."""
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    backend = NumPyBackend()
    result_backend = np.asarray(backend.expm(M))
    result_scipy = expm(M.astype(np.complex128))
    np.testing.assert_allclose(result_backend, result_scipy, atol=1e-12)
