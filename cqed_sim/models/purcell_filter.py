from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import qutip as qt

from cqed_sim.models.multilevel_cqed import HamiltonianData, IQPulse, MultilevelCQEDModel, ReadoutFrame


@dataclass(frozen=True)
class ExplicitPurcellFilterMode:
    """Explicit filter resonator mode.

    ``frequency``, ``coupling`` and ``kappa`` are angular frequencies.  The
    output port collapse operator is ``sqrt(kappa) * f``.
    """

    frequency: float
    levels: int
    coupling: float
    kappa: float
    label: str = "filter"

    def __post_init__(self) -> None:
        if int(self.levels) < 1:
            raise ValueError("levels must be positive.")
        if float(self.kappa) < 0.0:
            raise ValueError("kappa must be nonnegative.")
        object.__setattr__(self, "frequency", float(self.frequency))
        object.__setattr__(self, "levels", int(self.levels))
        object.__setattr__(self, "coupling", float(self.coupling))
        object.__setattr__(self, "kappa", float(self.kappa))


@dataclass
class FilteredMultilevelCQEDModel:
    """Multilevel transmon, readout resonator, and one explicit Purcell filter.

    Tensor ordering is ``|q, n_r, n_f>``.  When the filter is explicit, the
    output field is taken from the filter mode and Purcell decay should not be
    added as a separate qubit collapse channel.
    """

    base: MultilevelCQEDModel
    filter_mode: ExplicitPurcellFilterMode
    _operators_cache: dict[str, qt.Qobj] | None = field(default=None, init=False, repr=False)

    @property
    def subsystem_dims(self) -> tuple[int, int, int]:
        nq, nr = self.base.subsystem_dims
        return (nq, nr, int(self.filter_mode.levels))

    @property
    def has_transmon(self) -> bool:
        return True

    def _embed(self, op: qt.Qobj, index: int) -> qt.Qobj:
        factors = [qt.qeye(dim) for dim in self.subsystem_dims]
        factors[int(index)] = op
        return qt.tensor(*factors)

    def operators(self) -> dict[str, qt.Qobj]:
        if self._operators_cache is not None:
            return dict(self._operators_cache)
        nq, nr, nf = self.subsystem_dims
        b = qt.destroy(nq)
        a = qt.destroy(nr) if nr > 1 else qt.Qobj(np.zeros((1, 1), dtype=np.complex128), dims=[[1], [1]])
        f = qt.destroy(nf) if nf > 1 else qt.Qobj(np.zeros((1, 1), dtype=np.complex128), dims=[[1], [1]])
        q_local = qt.Qobj(np.asarray(self.base.coupling_matrix, dtype=np.complex128), dims=[[nq], [nq]])
        q_up_data = np.zeros((nq, nq), dtype=np.complex128)
        q_data = np.asarray(q_local.full(), dtype=np.complex128)
        for row in range(nq):
            for col in range(nq):
                if row > col:
                    q_up_data[row, col] = q_data[row, col]
        q_up = qt.Qobj(q_up_data, dims=[[nq], [nq]])
        ops = {
            "b": self._embed(b, 0),
            "bdag": self._embed(b.dag(), 0),
            "n_q": self._embed(b.dag() * b, 0),
            "q_coupling": self._embed(q_local, 0),
            "q_coupling_up": self._embed(q_up, 0),
            "q_coupling_down": self._embed(q_up.dag(), 0),
            "a": self._embed(a, 1),
            "adag": self._embed(a.dag(), 1),
            "n_r": self._embed(a.dag() * a, 1),
            "readout": self._embed(a, 1),
            "readout_dag": self._embed(a.dag(), 1),
            "f": self._embed(f, 2),
            "fdag": self._embed(f.dag(), 2),
            "n_f": self._embed(f.dag() * f, 2),
            "filter": self._embed(f, 2),
            "filter_dag": self._embed(f.dag(), 2),
        }
        self._operators_cache = ops
        return dict(ops)

    def basis_state(self, q_level: int, resonator_level: int = 0, filter_level: int = 0) -> qt.Qobj:
        nq, nr, nf = self.subsystem_dims
        return qt.tensor(qt.basis(nq, int(q_level)), qt.basis(nr, int(resonator_level)), qt.basis(nf, int(filter_level)))

    def transmon_level_projector(self, level: int) -> qt.Qobj:
        nq, _nr, _nf = self.subsystem_dims
        return self._embed(qt.basis(nq, int(level)) * qt.basis(nq, int(level)).dag(), 0)

    def transmon_transition_operators(self, lower_level: int, upper_level: int) -> tuple[qt.Qobj, qt.Qobj]:
        nq, _nr, _nf = self.subsystem_dims
        up = qt.basis(nq, int(upper_level)) * qt.basis(nq, int(lower_level)).dag()
        embedded = self._embed(up, 0)
        return embedded, embedded.dag()

    def static_hamiltonian(self, frame: ReadoutFrame | None = None) -> qt.Qobj:
        frame = self.base.rotating_frame if frame is None else frame
        nq, _nr, _nf = self.subsystem_dims
        q_levels = np.arange(nq, dtype=float)
        q_energies = np.asarray(self.base.transmon_energies, dtype=float) - float(frame.transmon_frequency) * q_levels
        h = self._embed(qt.Qobj(np.diag(q_energies), dims=[[nq], [nq]]), 0)
        ops = self.operators()
        h += (self.base.resonator_frequency - float(frame.resonator_frequency)) * ops["n_r"]
        h += (self.filter_mode.frequency - float(frame.filter_frequency)) * ops["n_f"]
        if self.base.counter_rotating:
            h += ops["q_coupling"] * (ops["a"] + ops["adag"])
        else:
            h += ops["q_coupling_up"] * ops["a"] + ops["q_coupling_down"] * ops["adag"]
        h += self.filter_mode.coupling * (ops["adag"] * ops["f"] + ops["a"] * ops["fdag"])
        return h

    def drive_coupling_operators(self) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
        ops = self.operators()
        return {
            "readout": (ops["adag"], ops["a"]),
            "resonator": (ops["adag"], ops["a"]),
            "filter": (ops["fdag"], ops["f"]),
            "qubit": (ops["bdag"], ops["b"]),
            "transmon": (ops["bdag"], ops["b"]),
        }

    def build_hamiltonian(self, pulse: IQPulse | object, *, drive_target: str = "readout") -> HamiltonianData:
        tlist, coeff = self.base._pulse_arrays(pulse)
        ops = self.operators()
        if str(drive_target) in {"filter", "f"}:
            raising, lowering = ops["fdag"], ops["f"]
        else:
            raising, lowering = ops["adag"], ops["a"]
        output = self.output_operator()
        return HamiltonianData(
            hamiltonian=[
                self.static_hamiltonian(),
                [raising, np.asarray(coeff, dtype=np.complex128)],
                [lowering, np.conjugate(np.asarray(coeff, dtype=np.complex128))],
            ],
            tlist=np.asarray(tlist, dtype=float),
            drive_coefficients=np.asarray(coeff, dtype=np.complex128),
            output_operator=output,
            metadata={
                "model": "filtered_multilevel_cqed",
                "tensor_order": "|q,n_r,n_f>",
                "output_channel": "filter",
                "purcell_decay_explicit": True,
            },
        )

    def output_operator(self) -> qt.Qobj:
        return np.sqrt(max(float(self.filter_mode.kappa), 0.0)) * self.operators()["f"]

    def collapse_operators(
        self,
        *,
        kappa_r_internal: float = 0.0,
        transmon_t1: Sequence[float] | None = None,
        transmon_tphi: float | None = None,
        include_purcell_qubit_decay: bool = False,
    ) -> tuple[qt.Qobj, ...]:
        if include_purcell_qubit_decay:
            raise ValueError(
                "Purcell decay is already represented by the explicit filter output channel; "
                "do not add a separate Purcell qubit collapse operator."
            )
        ops = self.operators()
        c_ops: list[qt.Qobj] = [self.output_operator()]
        if float(kappa_r_internal) > 0.0:
            c_ops.append(np.sqrt(float(kappa_r_internal)) * ops["a"])
        if transmon_t1 is not None:
            nq, _nr, _nf = self.subsystem_dims
            for lower, time in enumerate(tuple(transmon_t1)):
                upper = lower + 1
                if upper >= nq:
                    break
                if float(time) > 0.0 and np.isfinite(float(time)):
                    _up, down = self.transmon_transition_operators(lower, upper)
                    c_ops.append(np.sqrt(1.0 / float(time)) * down)
        if transmon_tphi is not None and float(transmon_tphi) > 0.0 and np.isfinite(float(transmon_tphi)):
            c_ops.append(np.sqrt(1.0 / float(transmon_tphi)) * ops["n_q"])
        return tuple(c_ops)


def add_explicit_purcell_filter(
    model: MultilevelCQEDModel,
    filter_mode: ExplicitPurcellFilterMode,
) -> FilteredMultilevelCQEDModel:
    return FilteredMultilevelCQEDModel(base=model, filter_mode=filter_mode)


__all__ = [
    "ExplicitPurcellFilterMode",
    "FilteredMultilevelCQEDModel",
    "add_explicit_purcell_filter",
]
