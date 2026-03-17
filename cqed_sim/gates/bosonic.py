"""Bosonic cavity gate library for cqed_sim.

All functions operate on a single bosonic mode with Fock-space dimension
``dim``, returning ``dim × dim`` QuTiP unitaries.

Convention summary
------------------
- Angles / parameters in standard units: angles in radians, times in seconds,
  Kerr/frequencies in rad/s.
- Displacement: ``D(α) = exp(α a† - α* a)`` — uses ``qt.displace`` directly.
- Oscillator rotation: ``R(θ) = exp(-i θ a†a)``, so ``|n⟩ → e^{-inθ} |n⟩``.
- Parity: ``Π = exp(i π a†a)``, so ``|n⟩ → (-1)^n |n⟩``.
- Squeezing: ``S(ζ) = exp(½ ζ* a² - ½ ζ (a†)²)`` — uses ``qt.squeeze``.
- Self-Kerr evolution (as a gate): ``U_K(t) = exp[-i K/2 t n̂(n̂-1)]``.
  Note: the Hamiltonian coupling ``self_kerr`` in ``cqed_sim.sim.couplings``
  returns the Hamiltonian term; this function returns the resulting unitary.
- SNAP: ``S({φ_n}) = Σ_n exp(i φ_n) |n⟩⟨n|``.  Accepts a dense list/array
  (phases for each Fock level 0, 1, …, len-1) or a sparse ``{n: phase}`` dict
  with unspecified levels receiving zero phase.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import qutip as qt

from cqed_sim.operators.cavity import destroy_cavity, number_operator


# ---------------------------------------------------------------------------
# Displacement
# ---------------------------------------------------------------------------

def displacement(alpha: complex, dim: int) -> qt.Qobj:
    """Bosonic displacement operator ``D(α) = exp(α a† - α* a)``.

    Parameters
    ----------
    alpha:
        Complex displacement amplitude.
    dim:
        Fock-space truncation dimension.

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` unitary.

    Notes
    -----
    Implemented via ``qt.displace(dim, alpha)``, which is exact within the
    finite truncated Hilbert space.  For large ``|α|``, a larger ``dim`` is
    required to avoid truncation errors.
    """
    return qt.displace(int(dim), complex(alpha))


# ---------------------------------------------------------------------------
# Oscillator phase rotation
# ---------------------------------------------------------------------------

def oscillator_rotation(theta: float, dim: int) -> qt.Qobj:
    """Oscillator phase-rotation gate ``R(θ) = exp(-i θ a†a)``.

    Acts as ``|n⟩ → exp(-i n θ) |n⟩`` on each Fock state.

    Parameters
    ----------
    theta:
        Rotation angle in radians.
    dim:
        Fock-space truncation dimension.

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` diagonal unitary.
    """
    n_vals = np.arange(int(dim), dtype=float)
    phases = np.exp(-1j * float(theta) * n_vals)
    return qt.Qobj(np.diag(phases), dims=[[int(dim)], [int(dim)]])


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------

def parity(dim: int) -> qt.Qobj:
    """Photon-parity gate ``Π = exp(i π a†a)``.

    Acts as ``|n⟩ → (-1)^n |n⟩`` on each Fock state.

    Parameters
    ----------
    dim:
        Fock-space truncation dimension.

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` diagonal unitary.

    Notes
    -----
    Equivalent to ``oscillator_rotation(-π, dim)``, since
    ``exp(i π n̂) = exp(-i (-π) n̂)``.
    """
    n_vals = np.arange(int(dim), dtype=float)
    phases = np.exp(1j * np.pi * n_vals)  # (-1)^n
    return qt.Qobj(np.diag(phases), dims=[[int(dim)], [int(dim)]])


# ---------------------------------------------------------------------------
# Squeezing
# ---------------------------------------------------------------------------

def squeeze(zeta: complex, dim: int) -> qt.Qobj:
    """Single-mode squeezing gate ``S(ζ) = exp(½ ζ* a² - ½ ζ (a†)²)``.

    Parameters
    ----------
    zeta:
        Complex squeezing parameter.  The magnitude controls the squeezing
        strength; the argument sets the quadrature angle.
    dim:
        Fock-space truncation dimension.

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` unitary.

    Notes
    -----
    Implemented via ``qt.squeeze(dim, zeta)``.  QuTiP's convention matches
    the definition above exactly.
    """
    return qt.squeeze(int(dim), complex(zeta))


# ---------------------------------------------------------------------------
# Self-Kerr evolution (unitary gate)
# ---------------------------------------------------------------------------

def kerr_evolution(kerr: float, time: float, dim: int) -> qt.Qobj:
    """Self-Kerr unitary ``U_K(t) = exp[-i K/2 t n̂(n̂-1)]``.

    The Hamiltonian that generates this gate is
    ``H_K = K/2 a†² a² = K/2 n̂(n̂-1)`` (in ``ħ=1`` / angular-frequency units).

    Parameters
    ----------
    kerr:
        Self-Kerr coupling ``K`` in rad/s.  Typically negative for a transmon.
    time:
        Evolution time in seconds.
    dim:
        Fock-space truncation dimension.

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` diagonal unitary.

    Notes
    -----
    This function returns the *unitary gate*.  For the *Hamiltonian term*
    ``K/2 a†² a²`` used during simulation assembly, see
    ``cqed_sim.sim.couplings.self_kerr``.
    """
    n_vals = np.arange(int(dim), dtype=float)
    # n̂(n̂-1) is the falling factorial of order 2
    phases = np.exp(-1j * 0.5 * float(kerr) * float(time) * n_vals * (n_vals - 1.0))
    return qt.Qobj(np.diag(phases), dims=[[int(dim)], [int(dim)]])


# ---------------------------------------------------------------------------
# SNAP gate
# ---------------------------------------------------------------------------

def snap(
    phases: dict[int, float] | Sequence[float] | np.ndarray,
    dim: int,
) -> qt.Qobj:
    """Selective Number-dependent Arbitrary Phase (SNAP) gate.

    .. math::

       S(\\{\\phi_n\\}) = \\sum_{n=0}^{N-1} e^{i\\phi_n} |n\\rangle\\langle n|

    Parameters
    ----------
    phases:
        Phase specification, one of:

        * **dict** ``{n: phase}``: sparse form; levels not present receive
          zero phase.
        * **list / array** of length ``≤ dim``: dense form; entry ``k`` is
          the phase for Fock level ``k``; remaining levels receive zero phase.

    dim:
        Fock-space truncation dimension ``N``.

    Returns
    -------
    qt.Qobj
        A ``dim × dim`` diagonal unitary.

    Examples
    --------
    >>> import numpy as np
    >>> from cqed_sim.gates.bosonic import snap
    >>> # Sparse dict form: only Fock levels 0 and 2 get a phase
    >>> U = snap({0: 0.0, 1: np.pi / 2, 2: np.pi}, dim=10)
    >>> # Dense array form
    >>> U = snap([0.0, np.pi / 2, np.pi], dim=10)
    """
    dim = int(dim)
    phase_array = np.zeros(dim, dtype=float)

    if isinstance(phases, dict):
        for n, phi in phases.items():
            n = int(n)
            if not (0 <= n < dim):
                raise ValueError(f"Fock level {n} is outside dim={dim}.")
            phase_array[n] = float(phi)
    else:
        arr = np.asarray(phases, dtype=float).reshape(-1)
        if arr.size > dim:
            raise ValueError(
                f"Dense phases array length {arr.size} exceeds dim={dim}."
            )
        phase_array[: arr.size] = arr

    diag = np.exp(1j * phase_array)
    return qt.Qobj(np.diag(diag), dims=[[dim], [dim]])


__all__ = [
    "displacement",
    "oscillator_rotation",
    "parity",
    "squeeze",
    "kerr_evolution",
    "snap",
]
