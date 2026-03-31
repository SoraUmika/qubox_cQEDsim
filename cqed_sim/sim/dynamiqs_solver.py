"""dynamiqs-based ODE solver for cqed_sim.

Provides a drop-in replacement for the QuTiP ``sesolve``/``mesolve`` path that
uses JAX (via diffrax) for JIT-compiled, GPU-friendly time evolution.

Activated by setting ``SimulationConfig.dynamiqs_solver`` to a solver name
such as ``"Tsit5"``.  The default (``None``) leaves the existing QuTiP path
completely unchanged.

Supported solver names
----------------------
``"Tsit5"`` (default, explicit Runge-Kutta 4/5, recommended)
``"Dopri5"`` (Dormand-Prince 4/5)
``"Dopri8"`` (Dormand-Prince 7/8)
``"Kvaerno3"`` (implicit, good for stiff systems)
``"Kvaerno5"`` (implicit, higher-order stiff)
``"Euler"`` (explicit Euler, for debugging only)
``"Expm"`` (matrix-exponential step, piecewise-constant)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import qutip as qt

# Lazy import guard — module is only loaded when dynamiqs_solver is not None.
try:
    import jax
    import jax.numpy as jnp
    import dynamiqs as dq
    _HAS_DYNAMIQS = True
except ImportError:
    _HAS_DYNAMIQS = False

# Import the result type and coefficient helper from the sibling solver module.
from .solver import DenseSolverResult, _coeff_samples


_SOLVER_MAP: dict[str, object] = {}  # populated lazily


def _get_solver(name: str):
    """Return the dynamiqs solver object for *name*."""
    if not _SOLVER_MAP:
        _SOLVER_MAP.update({
            "tsit5": dq.method.Tsit5(),
            "dopri5": dq.method.Dopri5(),
            "dopri8": dq.method.Dopri8(),
            "kvaerno3": dq.method.Kvaerno3(),
            "kvaerno5": dq.method.Kvaerno5(),
            "expm": dq.method.Expm(),
        })
    key = str(name).lower()
    if key not in _SOLVER_MAP:
        raise ValueError(
            f"Unknown dynamiqs solver '{name}'. "
            f"Supported: {sorted(_SOLVER_MAP.keys())}"
        )
    return _SOLVER_MAP[key]


def _operator_to_jax(op: qt.Qobj) -> "jnp.ndarray":
    return jnp.asarray(np.asarray(op.full(), dtype=np.complex128), dtype=jnp.complex128)


def _build_dynamiqs_hamiltonian(hamiltonian: list, tlist: np.ndarray):
    """Convert the cqed_sim hamiltonian list to a dynamiqs H representation.

    The cqed_sim format is::

        hamiltonian = [H_static,  [H_1, coeff_1],  [H_2, coeff_2], ...]

    where each ``coeff_i`` is a scalar, a numpy array of length ``len(tlist)``,
    or a callable ``coeff(tlist) -> array``.

    For a static-only Hamiltonian (single qt.Qobj element), returns a plain JAX
    array.  For time-dependent Hamiltonians, returns a ``dynamiqs.TimeQArray``
    sum.
    """
    H_static_jax = _operator_to_jax(hamiltonian[0])

    if len(hamiltonian) == 1:
        # Static Hamiltonian — plain JAX array accepted by dynamiqs solvers.
        return H_static_jax

    # Build pwc terms for each drive channel.
    # tlist has N_t points → N_t − 1 piecewise-constant intervals.
    # dq.pwc requires N_t boundary times and N_t−1 values.
    td_terms = []
    tlist_arr = np.asarray(tlist, dtype=float)

    for pair in hamiltonian[1:]:
        H_i_qt, coeff_i = pair
        H_i_jax = _operator_to_jax(H_i_qt)
        # Sample coefficient at every tlist point, then drop the last one
        # (it belongs to the half-open interval boundary, not a full interval).
        coeff_samples = _coeff_samples(coeff_i, tlist_arr)  # shape (N_t,)
        values = jnp.asarray(coeff_samples[:-1], dtype=jnp.complex128)  # (N_t−1,)
        times = jnp.asarray(tlist_arr, dtype=jnp.float64)               # (N_t,)
        td_terms.append(dq.pwc(times, values, H_i_jax))

    # Sum all terms: dynamiqs TimeQArray supports + with plain arrays.
    H = H_static_jax
    for term in td_terms:
        H = H + term
    return H


def solve_with_dynamiqs(
    hamiltonian: list,
    tlist: np.ndarray,
    initial_state: qt.Qobj,
    *,
    observables: Sequence[qt.Qobj] = (),
    collapse_ops: Sequence[qt.Qobj] = (),
    store_states: bool = False,
    solver: str = "Tsit5",
    atol: float = 1e-8,
    rtol: float = 1e-6,
    jax_device: str | None = None,
) -> DenseSolverResult:
    """Solve the Schrödinger or Lindblad master equation using dynamiqs.

    Parameters
    ----------
    hamiltonian:
        cqed_sim Hamiltonian list: ``[H_static, [H_i, coeff_i], ...]``
    tlist:
        Time array (same as QuTiP/cqed_sim convention).
    initial_state:
        Initial state as a ``qt.Qobj`` (ket or density matrix).
    observables:
        Operators whose expectation values are returned.
    collapse_ops:
        Lindblad jump operators.  If non-empty, ``mesolve`` is used.
    store_states:
        If ``True``, return all intermediate states.
    solver:
        dynamiqs solver name (``"Tsit5"``, ``"Dopri5"``, ``"Expm"``, etc.).
    atol, rtol:
        Absolute and relative tolerance for adaptive-step solvers.
    jax_device:
        ``"cpu"`` or ``"gpu"`` device placement for JAX arrays.
        ``None`` uses the default device.

    Returns
    -------
    DenseSolverResult
        Same return type as ``solve_with_backend``, compatible with the
        existing ``SimulationSession.run()`` result-processing code.
    """
    if not _HAS_DYNAMIQS:
        raise ImportError(
            "dynamiqs is not installed.  "
            "Install with: pip install dynamiqs"
        )

    # Enable 64-bit precision (idempotent).
    jax.config.update("jax_enable_x64", True)

    tlist_arr = np.asarray(tlist, dtype=float)
    tsave = jnp.asarray(tlist_arr, dtype=jnp.float64)

    is_open = bool(collapse_ops) or bool(initial_state.isoper)

    # --- Build dynamiqs H --------------------------------------------------
    H_dq = _build_dynamiqs_hamiltonian(hamiltonian, tlist_arr)

    # --- Convert initial state ---------------------------------------------
    state_template = initial_state
    if is_open:
        rho0_np = (
            np.asarray(initial_state.full(), dtype=np.complex128)
            if initial_state.isoper
            else np.outer(
                np.asarray(initial_state.full(), dtype=np.complex128).ravel(),
                np.asarray(initial_state.full(), dtype=np.complex128).ravel().conj(),
            )
        )
        state0 = jnp.asarray(rho0_np, dtype=jnp.complex128)
    else:
        psi0_np = np.asarray(initial_state.full(), dtype=np.complex128)
        state0 = jnp.asarray(psi0_np, dtype=jnp.complex128)  # shape (n, 1)

    if jax_device is not None:
        devices = [d for d in jax.devices() if d.platform == jax_device]
        if devices:
            state0 = jax.device_put(state0, devices[0])
            H_dq = jax.device_put(H_dq, devices[0])

    # --- Convert observables -----------------------------------------------
    exp_ops_jax = [
        jnp.asarray(np.asarray(op.full(), dtype=np.complex128), dtype=jnp.complex128)
        for op in observables
    ]

    # --- Build solver options ----------------------------------------------
    solver_obj = _get_solver(solver)
    options = dq.Options(
        save_states=bool(store_states),
        progress_meter=None,
    )

    # --- Solve -------------------------------------------------------------
    if is_open:
        jump_ops_jax = [
            jnp.asarray(np.asarray(c.full(), dtype=np.complex128), dtype=jnp.complex128)
            for c in collapse_ops
        ]
        result = dq.mesolve(
            H_dq,
            state0,
            tsave,
            jump_ops=jump_ops_jax,
            exp_ops=exp_ops_jax if exp_ops_jax else None,
            method=solver_obj,
            options=options,
        )
    else:
        result = dq.sesolve(
            H_dq,
            state0,
            tsave,
            exp_ops=exp_ops_jax if exp_ops_jax else None,
            method=solver_obj,
            options=options,
        )

    # --- Extract final state -----------------------------------------------
    dims = initial_state.dims
    final_arr = np.asarray(result.states[-1], dtype=np.complex128)
    if is_open:
        final_state = qt.Qobj(final_arr, dims=dims)
    else:
        final_state = qt.Qobj(final_arr.reshape(-1, 1), dims=dims)

    # --- Extract stored states ---------------------------------------------
    states: list[qt.Qobj] = []
    if store_states and result.states is not None:
        for s in result.states:
            s_arr = np.asarray(s, dtype=np.complex128)
            if is_open:
                states.append(qt.Qobj(s_arr, dims=dims))
            else:
                states.append(qt.Qobj(s_arr.reshape(-1, 1), dims=dims))

    # --- Extract expectation values ----------------------------------------
    # dynamiqs returns expects as array of shape (n_ops, ntsave)
    expect_list: list[np.ndarray] = []
    if exp_ops_jax:
        expects_arr = np.asarray(result.expects, dtype=np.complex128)
        # Shape can be (n_ops, ntsave) or (ntsave,) for single op
        if expects_arr.ndim == 1:
            expects_arr = expects_arr[np.newaxis, :]
        for i in range(len(exp_ops_jax)):
            expect_list.append(expects_arr[i])

    return DenseSolverResult(
        final_state=final_state,
        states=states,
        expect=expect_list,
        backend_name="dynamiqs",
    )


__all__ = ["solve_with_dynamiqs"]
