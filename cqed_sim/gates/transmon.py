"""Multilevel transmon transition-selective gate library for cqed_sim.

These gates act on a ``dim``-dimensional transmon Hilbert space, supporting
level transitions beyond the strict qubit (g–e) approximation.

Convention summary
------------------
- Level indices are 0-based: ``|g⟩ = |0⟩``, ``|e⟩ = |1⟩``, ``|f⟩ = |2⟩``, etc.
- Angles in radians.
- The general transition-selective rotation between adjacent levels ``j`` and
  ``j+1`` is

  .. math::

     R^{j,j+1}_{\\phi}(\\theta) =
     \\exp\\!\\left[-i\\frac{\\theta}{2}\\left(
       e^{-i\\phi}|j\\rangle\\langle j+1|
       + e^{i\\phi}|j+1\\rangle\\langle j|
     \\right)\\right]

  which leaves all other levels unchanged (acts as identity on the complement).

- Non-adjacent levels (``|level_a - level_b| > 1``) are supported with the
  same formula; physical realisability requires a mechanism to drive that
  transition directly (e.g., STIRAP or a direct EJ coupling), but the
  mathematical gate is well-defined.
"""

from __future__ import annotations

import numpy as np
import qutip as qt


def transition_rotation(
    dim: int,
    level_a: int,
    level_b: int,
    theta: float,
    phi: float = 0.0,
) -> qt.Qobj:
    """Transition-selective rotation between levels ``level_a`` and ``level_b``.

    Computes

    .. math::

       R^{a,b}_{\\phi}(\\theta)
       = \\exp\\!\\left[-i\\frac{\\theta}{2}\\left(
         e^{-i\\phi}|a\\rangle\\langle b|
         + e^{i\\phi}|b\\rangle\\langle a|
       \\right)\\right]

    acting on a ``dim``-dimensional Hilbert space.  All other levels are
    unaffected (identity on the complement of ``{|a⟩, |b⟩}``).

    Parameters
    ----------
    dim:
        Total Hilbert-space dimension (truncation).
    level_a, level_b:
        The two levels connected by the transition.  Must satisfy
        ``0 <= level_a, level_b < dim`` and ``level_a != level_b``.
    theta:
        Rotation angle in radians.
    phi:
        Drive phase in radians.  At ``phi=0`` the generator is purely real
        (X-like); at ``phi=π/2`` it is purely imaginary (Y-like).

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` unitary operator.
    """
    dim = int(dim)
    a = int(level_a)
    b = int(level_b)
    if not (0 <= a < dim and 0 <= b < dim):
        raise ValueError(
            f"Levels {a}, {b} must be in [0, {dim - 1}]."
        )
    if a == b:
        raise ValueError("level_a and level_b must be distinct.")

    ket_a = qt.basis(dim, a)
    ket_b = qt.basis(dim, b)
    # Generator: e^{-iφ}|a><b| + e^{iφ}|b><a|
    gen = np.exp(-1j * float(phi)) * ket_a * ket_b.dag() + \
          np.exp(+1j * float(phi)) * ket_b * ket_a.dag()
    return (-1j * float(theta) / 2.0 * gen).expm()


def r_ge(theta: float, phi: float = 0.0, dim: int = 3) -> qt.Qobj:
    """g–e selective rotation ``R^{0,1}_{φ}(θ)`` in a ``dim``-level transmon.

    Parameters
    ----------
    theta:
        Rotation angle in radians.
    phi:
        Drive phase in radians.
    dim:
        Hilbert-space dimension (default 3 to include the |f⟩ level).

    Notes
    -----
    At ``dim=2`` this is equivalent to ``rphi(theta, phi)`` from the qubit
    gate library.
    """
    return transition_rotation(dim, 0, 1, theta, phi)


def r_ef(theta: float, phi: float = 0.0, dim: int = 3) -> qt.Qobj:
    """e–f selective rotation ``R^{1,2}_{φ}(θ)`` in a ``dim``-level transmon.

    Parameters
    ----------
    theta:
        Rotation angle in radians.
    phi:
        Drive phase in radians.
    dim:
        Hilbert-space dimension (must be at least 3).
    """
    if int(dim) < 3:
        raise ValueError("dim must be >= 3 to include the |f⟩ level.")
    return transition_rotation(dim, 1, 2, theta, phi)


__all__ = [
    "transition_rotation",
    "r_ge",
    "r_ef",
]
