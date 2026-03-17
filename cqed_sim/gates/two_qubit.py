"""Two-qubit gate library for cqed_sim.

These gates act on a two-qubit Hilbert space with tensor ordering

    **qubit 0 (control) first, qubit 1 (target) second.**

The computational basis order is:

    ``|00⟩ = |g,g⟩``,  ``|01⟩ = |g,e⟩``,
    ``|10⟩ = |e,g⟩``,  ``|11⟩ = |e,e⟩``

where ``|g⟩ = basis(2, 0)`` and ``|e⟩ = basis(2, 1)``.

All gates return 4×4 (or generalized) QuTiP ``Qobj`` unitaries.

Gate definitions
----------------
- **CNOT**: ``|0⟩⟨0| ⊗ I + |1⟩⟨1| ⊗ X``
- **CZ**: ``diag(1, 1, 1, −1)``
- **Controlled-phase** CP(φ): ``diag(1, 1, 1, e^{iφ})``
- **SWAP**: ``|00⟩⟨00| + |01⟩⟨10| + |10⟩⟨01| + |11⟩⟨11|``
- **iSWAP**:

  .. math::

      \\begin{pmatrix}1&0&0&0\\\\0&0&i&0\\\\0&i&0&0\\\\0&0&0&1\\end{pmatrix}

- **√iSWAP**:

  .. math::

      \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1/\\sqrt{2} & i/\\sqrt{2} & 0 \\\\
        0 & i/\\sqrt{2} & 1/\\sqrt{2} & 0 \\\\
        0 & 0 & 0 & 1
      \\end{pmatrix}
"""

from __future__ import annotations

import numpy as np
import qutip as qt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _proj(level: int) -> qt.Qobj:
    """Projector onto ``basis(2, level)``."""
    k = qt.basis(2, int(level))
    return k * k.dag()


# ---------------------------------------------------------------------------
# CNOT
# ---------------------------------------------------------------------------

def cnot_gate() -> qt.Qobj:
    """Controlled-NOT (CNOT) gate.

    Control on ``|e⟩ = basis(2, 1)``; applies X to the target qubit.

    .. math::

        \\text{CNOT} = |g\\rangle\\langle g| \\otimes I + |e\\rangle\\langle e| \\otimes X

    Returns
    -------
    qt.Qobj
        A 4×4 unitary.
    """
    return (
        qt.tensor(_proj(0), qt.qeye(2))
        + qt.tensor(_proj(1), qt.sigmax())
    )


# ---------------------------------------------------------------------------
# CZ
# ---------------------------------------------------------------------------

def cz_gate() -> qt.Qobj:
    """Controlled-Z (CZ) gate.

    .. math::

        \\text{CZ} = \\operatorname{diag}(1, 1, 1, -1)

    Equivalent to applying a Z gate to the target when the control is
    ``|e⟩``.  Symmetric under interchange of control and target.

    Returns
    -------
    qt.Qobj
        A 4×4 unitary.
    """
    diag = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex128)
    return qt.Qobj(np.diag(diag), dims=[[2, 2], [2, 2]])


# ---------------------------------------------------------------------------
# Controlled-phase
# ---------------------------------------------------------------------------

def controlled_phase(phi: float) -> qt.Qobj:
    """Controlled-phase gate CP(φ).

    .. math::

        \\text{CP}(\\phi) = \\operatorname{diag}(1, 1, 1, e^{i\\phi})

    At ``phi=π`` this is equal to ``cz_gate()`` (up to global phase on the
    ``|e,e⟩`` entry, which here gives a sign flip).

    Parameters
    ----------
    phi:
        Phase angle in radians.

    Returns
    -------
    qt.Qobj
        A 4×4 unitary.
    """
    diag = np.array(
        [1.0, 1.0, 1.0, np.exp(1j * float(phi))], dtype=np.complex128
    )
    return qt.Qobj(np.diag(diag), dims=[[2, 2], [2, 2]])


# ---------------------------------------------------------------------------
# SWAP
# ---------------------------------------------------------------------------

def swap_gate() -> qt.Qobj:
    """SWAP gate.

    .. math::

        \\text{SWAP} =
        \\begin{pmatrix}
          1 & 0 & 0 & 0 \\\\
          0 & 0 & 1 & 0 \\\\
          0 & 1 & 0 & 0 \\\\
          0 & 0 & 0 & 1
        \\end{pmatrix}

    Exchanges the states of the two qubits.

    Returns
    -------
    qt.Qobj
        A 4×4 unitary.
    """
    mat = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    return qt.Qobj(mat, dims=[[2, 2], [2, 2]])


# ---------------------------------------------------------------------------
# iSWAP
# ---------------------------------------------------------------------------

def iswap_gate() -> qt.Qobj:
    """iSWAP gate.

    .. math::

        \\text{iSWAP} =
        \\begin{pmatrix}
          1 & 0 & 0 & 0 \\\\
          0 & 0 & i & 0 \\\\
          0 & i & 0 & 0 \\\\
          0 & 0 & 0 & 1
        \\end{pmatrix}

    Generated (up to single-qubit phases) by the exchange Hamiltonian
    ``H = J (σ+σ- + σ-σ+)`` for time ``t = π / (2J)``.  Commonly native
    in transmon-based processors with tunable coupling.

    Returns
    -------
    qt.Qobj
        A 4×4 unitary.
    """
    mat = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    return qt.Qobj(mat, dims=[[2, 2], [2, 2]])


# ---------------------------------------------------------------------------
# √iSWAP
# ---------------------------------------------------------------------------

def sqrt_iswap_gate() -> qt.Qobj:
    r"""Square-root-iSWAP gate.

    .. math::

        \sqrt{\text{iSWAP}} =
        \begin{pmatrix}
          1 & 0 & 0 & 0 \\
          0 & 1/\sqrt{2} & i/\sqrt{2} & 0 \\
          0 & i/\sqrt{2} & 1/\sqrt{2} & 0 \\
          0 & 0 & 0 & 1
        \end{pmatrix}

    Satisfies ``sqrt_iswap_gate() @ sqrt_iswap_gate() == iswap_gate()``
    (in exact arithmetic).

    Returns
    -------
    qt.Qobj
        A 4×4 unitary.
    """
    s = 1.0 / np.sqrt(2.0)
    mat = np.array(
        [
            [1, 0, 0, 0],
            [0, s, 1j * s, 0],
            [0, 1j * s, s, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.complex128,
    )
    return qt.Qobj(mat, dims=[[2, 2], [2, 2]])


__all__ = [
    "cnot_gate",
    "cz_gate",
    "controlled_phase",
    "swap_gate",
    "iswap_gate",
    "sqrt_iswap_gate",
]
