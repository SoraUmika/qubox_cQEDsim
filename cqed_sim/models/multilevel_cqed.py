from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.models.transmon import TransmonSpectrum


EnvelopeLike = np.ndarray | complex | float | Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ReadoutFrame:
    """Rotating-frame frequencies for strong-readout Hamiltonians.

    All fields are angular frequencies.  Use zeros for lab-frame operation.
    """

    transmon_frequency: float = 0.0
    resonator_frequency: float = 0.0
    filter_frequency: float = 0.0


@dataclass
class IQPulse:
    """Complex IQ envelope for a readout drive.

    ``samples`` are complex angular-frequency amplitudes.  ``drive_frequency``
    is the positive lab-frame angular frequency of the tone.  The samples are
    interpreted directly in the selected rotating frame.
    """

    samples: np.ndarray
    dt: float
    drive_frequency: float
    t0: float = 0.0
    phase: float = 0.0
    label: str = "readout"

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.complex128).reshape(-1)
        if samples.size < 1:
            raise ValueError("IQPulse.samples must contain at least one sample.")
        if float(self.dt) <= 0.0:
            raise ValueError("IQPulse.dt must be positive.")
        self.samples = samples
        self.dt = float(self.dt)
        self.drive_frequency = float(self.drive_frequency)
        self.t0 = float(self.t0)
        self.phase = float(self.phase)

    @property
    def duration(self) -> float:
        return float(self.samples.size * self.dt)

    @property
    def tlist(self) -> np.ndarray:
        return self.t0 + np.arange(self.samples.size + 1, dtype=float) * self.dt

    @property
    def coefficients(self) -> np.ndarray:
        if abs(self.phase) <= 0.0:
            return np.concatenate([self.samples, self.samples[-1:]])
        return np.concatenate([self.samples * np.exp(1j * self.phase), self.samples[-1:] * np.exp(1j * self.phase)])


@dataclass
class HamiltonianData:
    hamiltonian: list
    tlist: np.ndarray
    drive_coefficients: np.ndarray
    output_operator: qt.Qobj | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def as_qobjevo(self) -> qt.QobjEvo | qt.Qobj:
        if len(self.hamiltonian) == 1:
            return self.hamiltonian[0]
        return qt.QobjEvo(self.hamiltonian, tlist=self.tlist)


def _as_qobj_matrix(matrix: np.ndarray, levels: int) -> qt.Qobj:
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.shape != (levels, levels):
        raise ValueError(f"Expected a ({levels}, {levels}) matrix, got {arr.shape}.")
    return qt.Qobj(arr, dims=[[levels], [levels]])


def _transition_splitting(operator: qt.Qobj) -> tuple[qt.Qobj, qt.Qobj]:
    data = np.asarray(operator.full(), dtype=np.complex128)
    up = np.zeros_like(data)
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            if row > col:
                up[row, col] = data[row, col]
    q_up = qt.Qobj(up, dims=operator.dims)
    return q_up, q_up.dag()


@dataclass
class MultilevelCQEDModel:
    """Driven multilevel transmon plus readout resonator.

    Tensor ordering is transmon first, readout resonator second:
    ``|q, n_r>``.  Energies and rates are angular frequencies.
    """

    transmon_energies: Sequence[float]
    resonator_frequency: float
    resonator_levels: int
    coupling_matrix: np.ndarray
    rotating_frame: ReadoutFrame = field(default_factory=ReadoutFrame)
    counter_rotating: bool = True
    label: str = "multilevel_cqed"

    _operators_cache: dict[str, qt.Qobj] | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_transmon_spectrum(
        cls,
        spectrum: TransmonSpectrum,
        *,
        resonator_frequency: float,
        resonator_levels: int,
        coupling_strength: float,
        rotating_frame: ReadoutFrame | None = None,
        counter_rotating: bool = True,
    ) -> "MultilevelCQEDModel":
        coupling = float(coupling_strength) * np.asarray(spectrum.n_matrix, dtype=np.complex128)
        return cls(
            transmon_energies=np.asarray(spectrum.shifted_energies, dtype=float),
            resonator_frequency=float(resonator_frequency),
            resonator_levels=int(resonator_levels),
            coupling_matrix=coupling,
            rotating_frame=ReadoutFrame() if rotating_frame is None else rotating_frame,
            counter_rotating=counter_rotating,
        )

    def __post_init__(self) -> None:
        energies = np.asarray(self.transmon_energies, dtype=float).reshape(-1)
        if energies.size < 2:
            raise ValueError("At least two transmon levels are required.")
        if int(self.resonator_levels) < 1:
            raise ValueError("resonator_levels must be positive.")
        coupling = np.asarray(self.coupling_matrix, dtype=np.complex128)
        if coupling.shape != (energies.size, energies.size):
            raise ValueError("coupling_matrix must match the number of transmon levels.")
        self.transmon_energies = energies
        self.resonator_frequency = float(self.resonator_frequency)
        self.resonator_levels = int(self.resonator_levels)
        self.coupling_matrix = coupling

    @property
    def subsystem_dims(self) -> tuple[int, int]:
        return (int(len(self.transmon_energies)), int(self.resonator_levels))

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
        nq, nr = self.subsystem_dims
        b = qt.destroy(nq)
        a = qt.destroy(nr) if nr > 1 else qt.Qobj(np.zeros((1, 1), dtype=np.complex128), dims=[[1], [1]])
        q_coupling = _as_qobj_matrix(self.coupling_matrix, nq)
        q_up, q_down = _transition_splitting(q_coupling)
        ops = {
            "b": self._embed(b, 0),
            "bdag": self._embed(b.dag(), 0),
            "n_q": self._embed(b.dag() * b, 0),
            "q_coupling": self._embed(q_coupling, 0),
            "q_coupling_up": self._embed(q_up, 0),
            "q_coupling_down": self._embed(q_down, 0),
            "a": self._embed(a, 1),
            "adag": self._embed(a.dag(), 1),
            "n_r": self._embed(a.dag() * a, 1),
            "readout": self._embed(a, 1),
            "readout_dag": self._embed(a.dag(), 1),
        }
        self._operators_cache = ops
        return dict(ops)

    def basis_state(self, q_level: int, resonator_level: int = 0) -> qt.Qobj:
        nq, nr = self.subsystem_dims
        return qt.tensor(qt.basis(nq, int(q_level)), qt.basis(nr, int(resonator_level)))

    def transmon_level_projector(self, level: int) -> qt.Qobj:
        nq, _nr = self.subsystem_dims
        return self._embed(qt.basis(nq, int(level)) * qt.basis(nq, int(level)).dag(), 0)

    def transmon_transition_operators(self, lower_level: int, upper_level: int) -> tuple[qt.Qobj, qt.Qobj]:
        nq, _nr = self.subsystem_dims
        up = qt.basis(nq, int(upper_level)) * qt.basis(nq, int(lower_level)).dag()
        embedded = self._embed(up, 0)
        return embedded, embedded.dag()

    def static_hamiltonian(self, frame: ReadoutFrame | None = None) -> qt.Qobj:
        frame = self.rotating_frame if frame is None else frame
        nq, nr = self.subsystem_dims
        q_levels = np.arange(nq, dtype=float)
        q_energies = np.asarray(self.transmon_energies, dtype=float) - float(frame.transmon_frequency) * q_levels
        hq = self._embed(qt.Qobj(np.diag(q_energies), dims=[[nq], [nq]]), 0)
        a_ops = self.operators()
        delta_r = self.resonator_frequency - float(frame.resonator_frequency)
        h = hq + delta_r * a_ops["n_r"]
        if self.counter_rotating:
            h += a_ops["q_coupling"] * (a_ops["a"] + a_ops["adag"])
        else:
            h += a_ops["q_coupling_up"] * a_ops["a"] + a_ops["q_coupling_down"] * a_ops["adag"]
        return h

    def drive_coupling_operators(self) -> dict[str, tuple[qt.Qobj, qt.Qobj]]:
        ops = self.operators()
        return {
            "readout": (ops["adag"], ops["a"]),
            "resonator": (ops["adag"], ops["a"]),
            "qubit": (ops["bdag"], ops["b"]),
            "transmon": (ops["bdag"], ops["b"]),
        }

    def _pulse_arrays(self, pulse: IQPulse | object) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(pulse, IQPulse):
            return pulse.tlist, pulse.coefficients
        if hasattr(pulse, "tlist") and hasattr(pulse, "samples"):
            samples = np.asarray(getattr(pulse, "samples"), dtype=np.complex128).reshape(-1)
            tlist = np.asarray(getattr(pulse, "tlist"), dtype=float).reshape(-1)
            if tlist.size == samples.size:
                coeff = samples
            elif tlist.size == samples.size + 1:
                coeff = np.concatenate([samples, samples[-1:]])
            else:
                raise ValueError("Pulse tlist must have len(samples) or len(samples)+1.")
            return tlist, coeff
        raise TypeError("build_hamiltonian expects an IQPulse or an object with samples and tlist attributes.")

    def build_hamiltonian(self, pulse: IQPulse | object) -> HamiltonianData:
        """Return QuTiP-compatible time-dependent Hamiltonian data."""

        tlist, coeff = self._pulse_arrays(pulse)
        ops = self.operators()
        hamiltonian = [
            self.static_hamiltonian(),
            [ops["adag"], np.asarray(coeff, dtype=np.complex128)],
            [ops["a"], np.conjugate(np.asarray(coeff, dtype=np.complex128))],
        ]
        return HamiltonianData(
            hamiltonian=hamiltonian,
            tlist=np.asarray(tlist, dtype=float),
            drive_coefficients=np.asarray(coeff, dtype=np.complex128),
            output_operator=None,
            metadata={
                "model": self.label,
                "tensor_order": "|q,n_r>",
                "counter_rotating": bool(self.counter_rotating),
            },
        )

    def collapse_operators(
        self,
        *,
        kappa_r: float = 0.0,
        transmon_t1: Sequence[float] | None = None,
        transmon_tphi: float | None = None,
    ) -> tuple[qt.Qobj, ...]:
        ops = self.operators()
        c_ops: list[qt.Qobj] = []
        if float(kappa_r) > 0.0:
            c_ops.append(np.sqrt(float(kappa_r)) * ops["a"])
        if transmon_t1 is not None:
            nq, _nr = self.subsystem_dims
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


__all__ = [
    "EnvelopeLike",
    "HamiltonianData",
    "IQPulse",
    "MultilevelCQEDModel",
    "ReadoutFrame",
]
