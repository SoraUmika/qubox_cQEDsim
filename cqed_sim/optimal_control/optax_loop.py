"""Optax-based gradient-descent optimization loop for the JAX GRAPE path.

This module is an *optional* replacement for ``scipy.optimize.minimize`` when
``GrapeConfig.engine == "jax"`` and the chosen ``optimizer_method`` is an
Optax optimizer name (e.g. ``"adam"``, ``"adagrad"``).

Only activated when both conditions are met; the default L-BFGS-B path via
scipy is completely unchanged.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

# Recognised Optax method names (lower-cased).
_OPTAX_METHODS: frozenset[str] = frozenset({"adam", "adagrad", "sgd", "adamw"})


def is_optax_method(name: str) -> bool:
    """Return True if *name* is a recognised Optax optimizer."""
    return str(name).lower() in _OPTAX_METHODS


def _build_optimizer(method: str, learning_rate: float):
    """Return an ``optax.GradientTransformation`` for *method*."""
    import optax  # deferred — optional dependency

    name = str(method).lower()
    builders = {
        "adam": optax.adam,
        "adagrad": optax.adagrad,
        "sgd": optax.sgd,
        "adamw": optax.adamw,
    }
    if name not in builders:
        raise ValueError(
            f"Unknown Optax method '{method}'. "
            f"Supported: {sorted(_OPTAX_METHODS)}"
        )
    return builders[name](learning_rate)


def run_optax_loop(
    evaluate_fn: Callable[[np.ndarray], tuple[float, np.ndarray]],
    x0: np.ndarray,
    *,
    method: str,
    maxiter: int,
    learning_rate: float,
    grad_clip: float | None,
    bounds: tuple[tuple[float, float], ...] | None,
    history_callback: Callable[[int, float, np.ndarray], Any],
    show_progress: bool,
) -> tuple[np.ndarray, bool, str, int]:
    """Run an Optax gradient-descent loop.

    Parameters
    ----------
    evaluate_fn:
        Callable ``(x: np.ndarray) -> (cost: float, gradient: np.ndarray)``.
        The JAX GRAPE evaluator already JIT-compiles the cost+gradient
        internally, so no double-JIT occurs here.
    x0:
        Initial parameter vector (flat, numpy).
    method:
        Optax method name: ``"adam"``, ``"adagrad"``, ``"sgd"``, or ``"adamw"``.
    maxiter:
        Maximum number of gradient-descent steps.
    learning_rate:
        Step size (learning rate) passed to the Optax optimizer.
    grad_clip:
        If not ``None``, clip gradients to this global L2 norm before the
        optimizer update.
    bounds:
        Optional ``(lo, hi)`` pairs per parameter.  Parameters are projected
        back into bounds after each update (clip projection).
    history_callback:
        Called every iteration with ``(iteration_index, cost, gradient)``.
    show_progress:
        If ``True``, display a ``tqdm`` progress bar (best-effort).

    Returns
    -------
    (final_x, converged, message, n_iterations)
    """
    import jax.numpy as jnp
    import optax  # noqa: F401 — confirms installation

    opt = _build_optimizer(method, learning_rate)
    if grad_clip is not None:
        opt = optax.chain(optax.clip_by_global_norm(float(grad_clip)), opt)

    x = jnp.asarray(x0, dtype=jnp.float64)
    opt_state = opt.init(x)

    # Pre-build bound arrays for fast clipping
    lo_arr = jnp.asarray([b[0] for b in bounds], dtype=jnp.float64) if bounds else None
    hi_arr = jnp.asarray([b[1] for b in bounds], dtype=jnp.float64) if bounds else None

    pbar = None
    if show_progress:
        try:
            from tqdm.auto import tqdm
            pbar = tqdm(total=int(maxiter), desc=f"GRAPE ({method})", unit="iter", dynamic_ncols=True)
        except Exception:
            pass

    last_cost = float("nan")
    try:
        for i in range(int(maxiter)):
            x_np = np.asarray(x, dtype=np.float64)
            cost, grad = evaluate_fn(x_np)
            last_cost = float(cost)

            grad_j = jnp.asarray(grad, dtype=jnp.float64)
            updates, opt_state = opt.update(grad_j, opt_state, x)
            x = optax.apply_updates(x, updates)

            if lo_arr is not None:
                x = jnp.clip(x, lo_arr, hi_arr)

            history_callback(i, last_cost, np.asarray(grad, dtype=np.float64))

            if pbar is not None:
                pbar.set_postfix(cost=f"{last_cost:.4g}", refresh=False)
                pbar.update(1)
    finally:
        if pbar is not None:
            pbar.close()

    message = f"optax.{str(method).lower()} completed {maxiter} iterations (final cost={last_cost:.6g})"
    return np.asarray(x, dtype=np.float64), True, message, int(maxiter)
