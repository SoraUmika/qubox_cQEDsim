from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import qutip as qt


@dataclass(frozen=True)
class NoiseSpec:
    """Lindblad noise parameters in SI-style units.

    Times are in seconds and rates are in 1/s. Internally these are used with the
    repository convention H / hbar in rad/s and t in seconds.
    """

    t1: float | None = None
    transmon_t1: tuple[float | None, ...] | None = None
    tphi: float | None = None
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
        return 0.0 if self.tphi is None else 1.0 / (2.0 * self.tphi)


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

    if "a_r" in ops:
        _append_bosonic_loss(
            c_ops,
            ops["a_r"],
            ops["adag_r"],
            noise.kappa_readout,
            noise.nth_readout,
            seen_targets=seen_bosonic_targets,
        )

    return c_ops
