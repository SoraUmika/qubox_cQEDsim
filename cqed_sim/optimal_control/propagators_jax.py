"""JAX-accelerated GRAPE propagation with automatic differentiation.

Provides JIT-compiled forward propagation and automatic gradient
computation via ``jax.value_and_grad``, replacing the manual adjoint
method and ``scipy.linalg.expm_frechet`` calls used in the NumPy path.

Advantages
----------
- **JIT compilation**: Forward propagation compiled once via XLA, reused
  across all L-BFGS-B iterations within a single GRAPE solve.
- **Automatic differentiation**: Reverse-mode AD through
  ``jax.scipy.linalg.expm`` eliminates manual adjoint bookkeeping and
  per-step ``expm_frechet`` calls.
- **GPU support**: When JAX is configured with a GPU device (e.g.
  ``pip install jax[cuda12]``), propagation and gradient computation
  run on GPU with no code changes.

Requirements
------------
``jax`` and ``jaxlib`` must be installed (``pip install jax``).
For GPU: ``pip install jax[cuda12]``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.scipy.linalg as jla

    jax.config.update("jax_enable_x64", True)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False


def is_available() -> bool:
    """Return True if JAX is installed and functional."""
    return HAS_JAX


def _to_jax(arr: Any, dtype: Any, device: str | None = None) -> Any:
    """Convert *arr* to a JAX array, optionally placed on *device*."""
    a = jnp.asarray(arr, dtype=dtype)
    if device is None:
        return a
    devices = [d for d in jax.devices() if d.platform == device]
    return jax.device_put(a, devices[0]) if devices else a


def build_jax_evaluator(
    drift_hamiltonian: np.ndarray,
    control_operators: tuple[np.ndarray, ...],
    step_durations_s: np.ndarray,
    n_time_slices: int,
    objective_pair_counts: tuple[int, ...],
    leakage_metrics: tuple[str, ...] = (),
    device: str | None = None,
):
    """Build a JIT-compiled cost + gradient evaluator for one system.

    The returned callable performs:

    1.  Piecewise-constant forward propagation via ``jax.lax.scan`` and
        ``jax.scipy.linalg.expm``.
    2.  Probe-state fidelity cost computation.
    3.  Optional leakage penalty cost.
    4.  Reverse-mode automatic differentiation for the gradient of the
        total cost with respect to the physical control values.

    Parameters
    ----------
    drift_hamiltonian : (dim, dim) array
    control_operators : tuple of (dim, dim) arrays
    step_durations_s : (n_time_slices,) array
    n_time_slices : int
    objective_pair_counts : tuple[int, ...]
        Number of probe-state pairs per objective.
    leakage_metrics : tuple[str, ...]
        ``"worst"`` or ``"mean"`` per leakage penalty.
    device : str or None
        JAX device platform (``"cpu"``, ``"gpu"``, or ``None``).

    Returns
    -------
    evaluate : callable
        ``(physical_values, initial_states, target_states, state_weights,
        obj_weights, projectors, leak_penalty_weights,
        leak_pair_weights) -> (cost, gradient, U_final, fidelities)``
    """
    if not HAS_JAX:
        raise RuntimeError(
            "JAX is required for engine='jax'. Install with: pip install jax"
        )

    drift_H = _to_jax(drift_hamiltonian, jnp.complex128, device)
    ops = jnp.stack(
        [_to_jax(op, jnp.complex128, device) for op in control_operators]
    )
    dt = _to_jax(step_durations_s, jnp.float64, device)
    dim = int(drift_hamiltonian.shape[0])
    n_ctrl = len(control_operators)
    eye = jnp.eye(dim, dtype=jnp.complex128)
    n_leak = len(leakage_metrics)

    # Pre-compute pair offsets (Python-level, baked into the JIT trace)
    offsets: list[tuple[int, int]] = []
    cum = 0
    for count in objective_pair_counts:
        offsets.append((cum, cum + int(count)))
        cum += int(count)
    n_objs = len(objective_pair_counts)

    # ---- differentiable cost function -----------------------------------
    def _cost_fn(pv_flat, init_s, tgt_s, s_wt, o_wt, projs, l_wt, lp_wt):
        pv = pv_flat.reshape(n_ctrl, n_time_slices)
        H_steps = drift_H + jnp.einsum("ct,cij->tij", pv, ops)

        def _scan(U_acc, hd):
            H, d = hd
            U_step = jla.expm(-1j * d * H)
            return U_step @ U_acc, None

        U_final, _ = jax.lax.scan(_scan, eye, (H_steps, dt))

        final = init_s @ U_final.T
        overlaps = jnp.sum(jnp.conj(tgt_s) * final, axis=1)
        fids = jnp.abs(overlaps) ** 2

        # Per-objective infidelity cost (loop unrolled at trace time)
        total = jnp.float64(0.0)
        for idx in range(n_objs):
            start, end = offsets[idx]
            obj_f = jax.lax.dynamic_slice(fids, (start,), (end - start,))
            obj_w = jax.lax.dynamic_slice(s_wt, (start,), (end - start,))
            total = total + o_wt[idx] * (1.0 - jnp.sum(obj_w * obj_f))

        # Leakage penalty cost (loop unrolled at trace time)
        for j in range(n_leak):
            proj = projs[j]
            kept = final @ proj.T
            kp = jnp.real(jnp.sum(jnp.conj(final) * kept, axis=1))
            leaks = jnp.clip(1.0 - kp, 0.0, 1.0)
            if leakage_metrics[j] == "worst":
                total = total + l_wt[j] * jnp.max(leaks)
            else:
                norm = jnp.maximum(jnp.sum(lp_wt[j]), 1e-18)
                total = total + l_wt[j] * jnp.sum(lp_wt[j] * leaks) / norm

        return total, (U_final, fids)

    # JIT-compile the combined value_and_grad
    _compiled = jax.jit(jax.value_and_grad(_cost_fn, has_aux=True))

    # ---- public evaluation wrapper --------------------------------------
    def evaluate(
        physical_values: np.ndarray,
        initial_states: np.ndarray,
        target_states: np.ndarray,
        state_weights: np.ndarray,
        obj_weights: np.ndarray,
        projectors: list[np.ndarray] | None = None,
        leak_penalty_weights: np.ndarray | None = None,
        leak_pair_weights: list[np.ndarray] | None = None,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate cost and gradient.

        Returns
        -------
        cost : float
        gradient : (n_controls, n_time_slices) ndarray
        U_final : (dim, dim) ndarray
        fidelities : (total_pairs,) ndarray
        """
        n_pairs = int(initial_states.shape[0])
        pv_j = jnp.asarray(physical_values.reshape(-1), dtype=jnp.float64)
        init_j = jnp.asarray(initial_states, dtype=jnp.complex128)
        tgt_j = jnp.asarray(target_states, dtype=jnp.complex128)
        sw_j = jnp.asarray(state_weights, dtype=jnp.float64)
        ow_j = jnp.asarray(obj_weights, dtype=jnp.float64)

        if n_leak > 0 and projectors is not None:
            p_j = jnp.stack(
                [jnp.asarray(p, dtype=jnp.complex128) for p in projectors]
            )
            lw_j = jnp.asarray(leak_penalty_weights, dtype=jnp.float64)
            lpw_j = jnp.stack(
                [jnp.asarray(w, dtype=jnp.float64) for w in leak_pair_weights]
            )
        else:
            p_j = jnp.zeros((0, dim, dim), dtype=jnp.complex128)
            lw_j = jnp.zeros((0,), dtype=jnp.float64)
            lpw_j = jnp.zeros((0, max(n_pairs, 1)), dtype=jnp.float64)

        (cost_val, (U_final, fids)), grad = _compiled(
            pv_j, init_j, tgt_j, sw_j, ow_j, p_j, lw_j, lpw_j
        )

        return (
            float(cost_val),
            np.asarray(grad, dtype=float).reshape(n_ctrl, n_time_slices),
            np.asarray(U_final, dtype=np.complex128),
            np.asarray(fids, dtype=float),
        )

    return evaluate


__all__ = ["is_available", "build_jax_evaluator"]
