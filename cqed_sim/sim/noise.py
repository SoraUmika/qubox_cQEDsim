from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import qutip as qt


def pure_dephasing_time_from_t1_t2(*, t1_s: float | None, t2_s: float | None) -> float | None:
    """Infer ``T_phi`` from ``T1`` and ``T2`` in seconds.

    Returns ``None`` when ``T2`` is not supplied or when the inferred extra pure-dephasing
    rate is non-positive. Callers treat that ``None`` value as "no additional dephasing
    term" rather than as a literal zero-valued time.
    """

    if t2_s is None:
        return None
    if t1_s is None:
        return float(t2_s)
    inv_tphi = max(0.0, 1.0 / float(t2_s) - 1.0 / (2.0 * float(t1_s)))
    if inv_tphi <= 0.0:
        return None
    return 1.0 / inv_tphi


@dataclass(frozen=True)
class NoiseSpec:
    """Lindblad noise parameters in SI-style units.

    The library is unit-coherent: it does not enforce specific physical units for
    frequencies or times. Any internally consistent unit system is valid (for
    example, rad/s with times in seconds, or rad/ns with times in nanoseconds).
    The recommended convention used in the main examples and calibration function
    naming is rad/s and seconds; field names ending in ``_s`` nominally imply
    seconds in that convention.

    Dephasing-rate convention
    ------------------------
    The **transmon** (qubit) pure-dephasing Lindblad operator is ``sigma_z`` for
    a two-level qubit (or ``n_q`` for multilevel), and the rate is::

        gamma_phi = 1 / (2 * T_phi)

    The factor of 2 arises because ``sigma_z`` has eigenvalues ±1, so the dephasing
    super-operator strength must be halved to produce ``1/T_phi`` total decay of
    off-diagonal coherence elements.

    The **bosonic** (storage/readout) pure-dephasing Lindblad operator is the number
    operator ``n``, and the rate is::

        gamma_phi_bosonic = 1 / T_phi

    There is no factor of 2 because the dephasing super-operator for the bosonic
    ``n`` operator already maps directly to ``1/T_phi`` per-photon coherence decay.

    These two conventions are *not* interchangeable.  A given ``T_phi`` value has
    different physical meaning for qubit vs bosonic dephasing.  This matches the
    standard Lindblad treatment:

    - Qubit: ``Gamma_phi * D[sigma_z](rho)`` with ``Gamma_phi = 1/(2 T_phi)``
    - Bosonic: ``Gamma_phi * D[n](rho)`` with ``Gamma_phi = 1/T_phi``
    """

    t1: float | None = None
    transmon_t1: tuple[float | None, ...] | None = None
    tphi: float | None = None
    tphi_storage: float | None = None
    tphi_readout: float | None = None
    kappa: float | None = None
    nth: float = 0.0
    kappa_storage: float | None = None
    kappa_readout: float | None = None
    nth_storage: float | None = None
    nth_readout: float | None = None

    @property
    def gamma1(self) -> float:
        return 0.0 if self.t1 is None else 1.0 / self.t1

    @property
    def gamma_phi(self) -> float:
        """Transmon pure-dephasing rate: ``1 / (2 * T_phi)``.

        The factor of 2 is required because the Lindblad operator is ``sigma_z``
        (eigenvalues ±1) for a two-level qubit, or ``n_q`` (projector) for
        multilevel.  See the class docstring for the full convention.
        """
        return 0.0 if self.tphi is None else 1.0 / (2.0 * self.tphi)

    @property
    def gamma_phi_storage(self) -> float:
        """Storage-mode pure-dephasing rate: ``1 / T_phi``.

        No factor of 2: the bosonic Lindblad operator is the number operator
        ``n_s``, which maps directly to ``1/T_phi`` per-photon coherence decay.
        """
        return 0.0 if self.tphi_storage is None else 1.0 / float(self.tphi_storage)

    @property
    def gamma_phi_readout(self) -> float:
        """Readout-mode pure-dephasing rate: ``1 / T_phi``.

        Same convention as ``gamma_phi_storage``: no factor of 2 for bosonic modes.
        """
        return 0.0 if self.tphi_readout is None else 1.0 / float(self.tphi_readout)


def _total_identity(subsystem_dims: tuple[int, ...]) -> qt.Qobj:
    return qt.tensor(*(qt.qeye(dim) for dim in subsystem_dims))


def _append_bosonic_loss(
    c_ops: list[qt.Qobj],
    lowering: qt.Qobj,
    raising: qt.Qobj,
    kappa: float | None,
    nth: float | None,
    *,
    seen_targets: set[int] | None = None,
) -> None:
    if kappa is None or kappa <= 0.0:
        return
    target_key = id(lowering)
    if seen_targets is not None and target_key in seen_targets:
        return
    nth = max(0.0, 0.0 if nth is None else float(nth))
    c_ops.append(np.sqrt(float(kappa) * (nth + 1.0)) * lowering)
    if nth > 0.0:
        c_ops.append(np.sqrt(float(kappa) * nth) * raising)
    if seen_targets is not None:
        seen_targets.add(target_key)


def _append_bosonic_dephasing(c_ops: list[qt.Qobj], number: qt.Qobj, tphi: float | None) -> None:
    if tphi is None or tphi <= 0.0:
        return
    c_ops.append(np.sqrt(1.0 / float(tphi)) * number)


def _embed_transmon_relaxation(subsystem_dims: tuple[int, ...], upper_level: int) -> qt.Qobj:
    upper_level = int(upper_level)
    lowering = qt.basis(subsystem_dims[0], upper_level - 1) * qt.basis(subsystem_dims[0], upper_level).dag()
    factors = [lowering]
    factors.extend(qt.qeye(dim) for dim in subsystem_dims[1:])
    return qt.tensor(*factors)


def collapse_operators(model: Any, noise: NoiseSpec | None) -> list[qt.Qobj]:
    if noise is None:
        return []
    ops = model.operators()
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    c_ops: list[qt.Qobj] = []
    has_transmon = bool(getattr(model, "has_transmon", "n_q" in ops and "b" in ops))
    seen_bosonic_targets: set[int] = set()

    if noise.transmon_t1 is not None:
        if not has_transmon:
            raise ValueError("transmon_t1 noise was requested for a model without a transmon subsystem.")
        for upper_level, lifetime in enumerate(noise.transmon_t1, start=1):
            if upper_level >= dims[0]:
                break
            if lifetime is None or lifetime <= 0.0:
                continue
            c_ops.append(np.sqrt(1.0 / float(lifetime)) * _embed_transmon_relaxation(dims, upper_level))
    elif noise.gamma1 > 0.0:
        if not has_transmon:
            raise ValueError("t1 noise was requested for a model without a transmon subsystem.")
        c_ops.append(np.sqrt(noise.gamma1) * ops["b"])

    if noise.gamma_phi > 0.0:
        if not has_transmon:
            raise ValueError("tphi noise was requested for a model without a transmon subsystem.")
        if dims[0] == 2:
            sigma_z = _total_identity(dims) - 2.0 * ops["n_q"]
            c_ops.append(np.sqrt(noise.gamma_phi) * sigma_z)
        else:
            c_ops.append(np.sqrt(noise.gamma_phi) * ops["n_q"])

    if "a" in ops:
        _append_bosonic_loss(c_ops, ops["a"], ops["adag"], noise.kappa, noise.nth, seen_targets=seen_bosonic_targets)

    if "a_s" in ops:
        kappa_storage = noise.kappa_storage if noise.kappa_storage is not None else noise.kappa
        nth_storage = noise.nth_storage if noise.nth_storage is not None else noise.nth
        _append_bosonic_loss(
            c_ops,
            ops["a_s"],
            ops["adag_s"],
            kappa_storage,
            nth_storage,
            seen_targets=seen_bosonic_targets,
        )
        _append_bosonic_dephasing(c_ops, ops["n_s"], noise.tphi_storage)

    if "a_r" in ops:
        _append_bosonic_loss(
            c_ops,
            ops["a_r"],
            ops["adag_r"],
            noise.kappa_readout,
            noise.nth_readout,
            seen_targets=seen_bosonic_targets,
        )
        _append_bosonic_dephasing(c_ops, ops["n_r"], noise.tphi_readout)

    return c_ops
