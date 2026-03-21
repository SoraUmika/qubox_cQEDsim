from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import prod
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import qutip as qt

from .backends import SimulationResult, simulate_sequence
from .sequence import (
    BlueSidebandExchange,
    CavityBlockPhase,
    ConditionalDisplacement,
    ConditionalPhaseSQR,
    Displacement,
    DriftPhaseModel,
    FreeEvolveCondPhase,
    GateSequence,
    JaynesCummingsExchange,
    PrimitiveGate,
    QubitRotation,
    SNAP,
    SQR,
    _compile_waveform,
    _final_unitary_from_compiled,
    _runtime_simulation_config,
    _simulate_waveform_states,
    _waveform_payload,
)
from .subspace import Subspace
from .targets import (
    ObservableTarget,
    TargetChannel,
    TargetIsometry,
    TargetReducedStateMapping,
    TargetStateMapping,
    TargetUnitary,
    TrajectoryTarget,
)

_CQED_GATE_TYPES = (
    QubitRotation,
    SQR,
    CavityBlockPhase,
    SNAP,
    Displacement,
    ConditionalDisplacement,
    JaynesCummingsExchange,
    BlueSidebandExchange,
    ConditionalPhaseSQR,
    FreeEvolveCondPhase,
)
_CQED_GATE_NAMES = {
    "QubitRotation",
    "SQR",
    "CavityBlockPhase",
    "LogicalBlockPhase",
    "BlockPhase",
    "SNAP",
    "Displacement",
    "ConditionalDisplacement",
    "JaynesCummingsExchange",
    "BlueSidebandExchange",
    "ConditionalPhaseSQR",
    "CondPhaseSQR",
    "ConditionalPhase",
    "FreeEvolveCondPhase",
    "FreeCondPhaseWait",
}


def _normalize_gate_name(name: str) -> str:
    aliases = {
        "CondPhaseSQR": "ConditionalPhaseSQR",
        "ConditionalPhase": "ConditionalPhaseSQR",
        "FreeCondPhaseWait": "FreeEvolveCondPhase",
        "LogicalBlockPhase": "CavityBlockPhase",
        "BlockPhase": "CavityBlockPhase",
    }
    return aliases.get(str(name), str(name))


def _infer_state_mapping_dim(target: TargetStateMapping) -> int:
    for state in list(target.initial_states) + list(target.target_states):
        if hasattr(state, "shape"):
            shape = getattr(state, "shape")
            if len(shape) == 1:
                return int(shape[0])
            if len(shape) == 2:
                return int(shape[0])
        arr = np.asarray(state, dtype=np.complex128)
        if arr.ndim == 1:
            return int(arr.size)
        if arr.ndim == 2:
            return int(arr.shape[0])
    raise ValueError("Could not infer a Hilbert-space dimension from the target state mapping.")


def _coerce_states_with_dims(states: Sequence[qt.Qobj | np.ndarray], dims: Sequence[int] | None) -> list[qt.Qobj]:
    out: list[qt.Qobj] = []
    dims_list = None if dims is None else [int(dim) for dim in dims]
    for state in states:
        if isinstance(state, qt.Qobj):
            out.append(state)
            continue
        arr = np.asarray(state, dtype=np.complex128)
        if arr.ndim == 1:
            if dims_list is None:
                out.append(qt.Qobj(arr.reshape(-1)))
            else:
                out.append(qt.Qobj(arr.reshape(-1), dims=[dims_list, [1] * len(dims_list)]))
            continue
        if arr.ndim == 2:
            if dims_list is None:
                out.append(qt.Qobj(arr))
            else:
                out.append(qt.Qobj(arr, dims=[dims_list, dims_list]))
            continue
        raise ValueError("States must be vectors or density matrices.")
    return out


def _legacy_cqed_sequence(
    *,
    gateset: Sequence[str],
    full_dim: int,
    n_cav: int,
    default_duration: float,
    optimize_times: bool,
    time_bounds_for: Callable[[str], tuple[float, float]],
    include_conditional_phase_in_sqr: bool,
    drift_model: DriftPhaseModel,
) -> GateSequence:
    gates = []
    for i, raw_name in enumerate(gateset):
        name = _normalize_gate_name(raw_name)
        bounds = time_bounds_for(name)
        kw = {
            "name": f"{name}_{i}",
            "duration": float(default_duration),
            "optimize_time": bool(optimize_times),
            "time_bounds": bounds,
            "duration_ref": float(default_duration),
        }
        if name == "QubitRotation":
            gates.append(QubitRotation(theta=0.3, phi=0.0, **kw))
        elif name == "SQR":
            gates.append(
                SQR(
                    theta_n=[0.1] * int(n_cav),
                    phi_n=[0.0] * int(n_cav),
                    tones=int(n_cav),
                    tone_freqs=[],
                    include_conditional_phase=bool(include_conditional_phase_in_sqr),
                    drift_model=drift_model,
                    **kw,
                )
            )
        elif name == "CavityBlockPhase":
            gates.append(CavityBlockPhase(phases=[0.0] * int(n_cav), **kw))
        elif name == "SNAP":
            gates.append(SNAP(phases=[0.0] * int(n_cav), **kw))
        elif name == "Displacement":
            gates.append(Displacement(alpha=0.0 + 0.0j, **kw))
        elif name == "ConditionalPhaseSQR":
            gates.append(ConditionalPhaseSQR(phases_n=[0.0] * int(n_cav), drift_model=drift_model, **kw))
        elif name == "FreeEvolveCondPhase":
            gates.append(FreeEvolveCondPhase(drift_model=drift_model, **kw))
        else:
            raise ValueError(f"Unsupported gate in gateset: {raw_name}")
    return GateSequence(gates=gates, n_cav=int(n_cav), full_dim=int(full_dim))


class QuantumSystem(ABC):
    """Backend interface used by UnitarySynthesizer.

    The synthesizer talks to this abstraction rather than to a raw cQED model.
    Concrete systems can expose native simulation capabilities while the
    optimizer only depends on Hilbert-space and propagation interfaces.
    """

    @abstractmethod
    def hilbert_dimension(
        self,
        *,
        sequence: GateSequence | None = None,
        primitive: PrimitiveGate | None = None,
        subspace: Subspace | None = None,
        target: TargetUnitary | TargetStateMapping | TargetReducedStateMapping | TargetIsometry | TargetChannel | None = None,
    ) -> int | None:
        raise NotImplementedError

    def subsystem_dimensions(
        self,
        *,
        sequence: GateSequence | None = None,
        full_dim: int | None = None,
        subspace: Subspace | None = None,
    ) -> tuple[int, ...]:
        dim = int(full_dim) if full_dim is not None else self.hilbert_dimension(sequence=sequence, subspace=subspace)
        if dim is None:
            raise ValueError("This system could not infer subsystem dimensions.")
        return (int(dim),)

    def infer_n_cav(
        self,
        *,
        sequence: GateSequence | None = None,
        full_dim: int | None = None,
        subspace: Subspace | None = None,
    ) -> int | None:
        return None

    def configure_sequence(self, sequence: GateSequence, *, subspace: Subspace | None = None) -> GateSequence:
        if sequence.full_dim is None:
            if subspace is not None:
                sequence.full_dim = int(subspace.full_dim)
            else:
                dim = self.hilbert_dimension(sequence=sequence, subspace=subspace)
                if dim is not None:
                    sequence.full_dim = int(dim)
        if sequence.n_cav is None:
            n_cav = self.infer_n_cav(sequence=sequence, full_dim=sequence.full_dim, subspace=subspace)
            if n_cav is not None:
                sequence.n_cav = int(n_cav)
        return sequence

    def build_sequence_from_gateset(
        self,
        gateset: Sequence[str],
        *,
        subspace: Subspace,
        default_duration: float,
        optimize_times: bool,
        time_bounds_for: Callable[[str], tuple[float, float]],
        include_conditional_phase_in_sqr: bool,
        drift_model: DriftPhaseModel,
    ) -> GateSequence:
        raise ValueError(
            f"{self.__class__.__name__} does not provide a default gateset builder. "
            "Pass explicit primitives or supply a system adapter that knows how to build the requested gates."
        )

    def runtime_model(self) -> Any | None:
        return None

    def with_model(self, model: Any) -> QuantumSystem:
        if model is None:
            return self
        raise ValueError(f"{self.__class__.__name__} does not support external model overrides.")

    def simulate_unitary(
        self,
        sequence: GateSequence,
        *,
        backend: str = "ideal",
        **backend_settings: Any,
    ) -> np.ndarray:
        settings = dict(backend_settings)
        settings["system"] = self
        return sequence.unitary(backend=backend, backend_settings=settings)

    def simulate_state(
        self,
        sequence: GateSequence,
        psi0: qt.Qobj | np.ndarray,
        *,
        backend: str = "ideal",
        **backend_settings: Any,
    ) -> qt.Qobj:
        return self.simulate_states(sequence, [psi0], backend=backend, **backend_settings)[0]

    def simulate_states(
        self,
        sequence: GateSequence,
        states: Sequence[qt.Qobj | np.ndarray],
        *,
        backend: str = "ideal",
        **backend_settings: Any,
    ) -> list[qt.Qobj]:
        settings = dict(backend_settings)
        settings["system"] = self
        return sequence.propagate_states(list(states), backend=backend, backend_settings=settings)

    def simulate_sequence(
        self,
        sequence: GateSequence,
        subspace: Subspace | None,
        *,
        backend: str = "ideal",
        target_subspace: np.ndarray | None = None,
        leakage_weight: float = 0.0,
        gauge: str = "global",
        block_slices: Sequence[slice | Sequence[int] | np.ndarray] | None = None,
        state_inputs: Sequence[qt.Qobj | np.ndarray] | None = None,
        need_operator: bool = True,
        **backend_settings: Any,
    ) -> SimulationResult:
        return simulate_sequence(
            sequence=sequence,
            subspace=subspace,
            backend=backend,
            target_subspace=target_subspace,
            leakage_weight=leakage_weight,
            gauge=gauge,
            block_slices=block_slices,
            state_inputs=state_inputs,
            need_operator=need_operator,
            system=self,
            **backend_settings,
        )

    def simulate_primitive_unitary(self, primitive: PrimitiveGate, *, settings: Mapping[str, Any]) -> np.ndarray:
        raise ValueError(f"{self.__class__.__name__} does not support waveform primitive simulation.")

    def simulate_primitive_states(
        self,
        primitive: PrimitiveGate,
        states: Sequence[qt.Qobj],
        *,
        settings: Mapping[str, Any],
    ) -> list[qt.Qobj]:
        if settings.get("c_ops") is not None or settings.get("noise") is not None:
            raise ValueError(f"{self.__class__.__name__} does not support open-system waveform primitive propagation.")
        operator = np.asarray(self.simulate_primitive_unitary(primitive, settings=settings), dtype=np.complex128)
        dims = list(self.subsystem_dimensions(full_dim=int(operator.shape[0])))
        op = qt.Qobj(operator, dims=[dims, dims])
        outputs: list[qt.Qobj] = []
        for state in _coerce_states_with_dims(states, dims):
            outputs.append(op * state * op.dag() if state.isoper else op * state)
        return outputs

    def to_record(self) -> dict[str, Any]:
        model = self.runtime_model()
        dims = None
        if model is not None and hasattr(model, "subsystem_dims"):
            dims = [int(dim) for dim in getattr(model, "subsystem_dims")]
        return {
            "kind": self.__class__.__name__,
            "model_class": None if model is None else model.__class__.__name__,
            "subsystem_dims": dims,
        }


@dataclass(frozen=True)
class GenericQuantumSystem(QuantumSystem):
    dimension: int | None = None

    def hilbert_dimension(
        self,
        *,
        sequence: GateSequence | None = None,
        primitive: PrimitiveGate | None = None,
        subspace: Subspace | None = None,
        target: TargetUnitary | TargetStateMapping | TargetReducedStateMapping | TargetIsometry | TargetChannel | None = None,
    ) -> int | None:
        if self.dimension is not None:
            return int(self.dimension)
        if subspace is not None:
            return int(subspace.full_dim)
        if isinstance(target, TargetUnitary):
            return int(target.dim)
        if isinstance(target, TargetStateMapping):
            return _infer_state_mapping_dim(target)
        if isinstance(target, TargetReducedStateMapping):
            return int(target.infer_dimension())
        if isinstance(target, TargetIsometry):
            return int(target.infer_dimension())
        if isinstance(target, TargetChannel):
            full_dim = target.infer_full_dimension()
            return None if full_dim is None else int(full_dim)
        if isinstance(target, ObservableTarget):
            return int(target.infer_dimension())
        if isinstance(target, TrajectoryTarget):
            return int(target.infer_dimension())
        if primitive is not None:
            try:
                return int(primitive.resolved_dimension(model=None))
            except Exception:
                return None
        if sequence is not None:
            if sequence.full_dim is not None:
                return int(sequence.full_dim)
            for gate in sequence.gates:
                if isinstance(gate, PrimitiveGate):
                    try:
                        return int(gate.resolved_dimension(model=None))
                    except Exception:
                        continue
        return None


@dataclass(frozen=True)
class _CQEDIdealSystem(GenericQuantumSystem):
    def to_record(self) -> dict[str, Any]:
        row = super().to_record()
        row["kind"] = "CQEDIdealSystem"
        return row

    def infer_n_cav(
        self,
        *,
        sequence: GateSequence | None = None,
        full_dim: int | None = None,
        subspace: Subspace | None = None,
    ) -> int | None:
        for source in (subspace,):
            if source is None or not isinstance(source.metadata, dict):
                continue
            if "n_cav" in source.metadata:
                return int(source.metadata["n_cav"])
        dim = None
        if full_dim is not None:
            dim = int(full_dim)
        elif sequence is not None and sequence.full_dim is not None:
            dim = int(sequence.full_dim)
        elif self.dimension is not None:
            dim = int(self.dimension)
        if dim is not None and dim % 2 == 0:
            return int(dim // 2)
        return None

    def configure_sequence(self, sequence: GateSequence, *, subspace: Subspace | None = None) -> GateSequence:
        sequence = super().configure_sequence(sequence, subspace=subspace)
        if sequence.full_dim is None and sequence.n_cav is not None:
            sequence.full_dim = int(2 * sequence.n_cav)
        if sequence.n_cav is None:
            n_cav = self.infer_n_cav(sequence=sequence, full_dim=sequence.full_dim, subspace=subspace)
            if n_cav is not None:
                sequence.n_cav = int(n_cav)
        return sequence

    def build_sequence_from_gateset(
        self,
        gateset: Sequence[str],
        *,
        subspace: Subspace,
        default_duration: float,
        optimize_times: bool,
        time_bounds_for: Callable[[str], tuple[float, float]],
        include_conditional_phase_in_sqr: bool,
        drift_model: DriftPhaseModel,
    ) -> GateSequence:
        n_cav = self.infer_n_cav(full_dim=subspace.full_dim, subspace=subspace)
        if n_cav is None:
            raise ValueError("Could not infer n_cav for the legacy cQED gateset. Provide a cQED-compatible subspace.")
        return _legacy_cqed_sequence(
            gateset=gateset,
            full_dim=int(subspace.full_dim),
            n_cav=int(n_cav),
            default_duration=float(default_duration),
            optimize_times=bool(optimize_times),
            time_bounds_for=time_bounds_for,
            include_conditional_phase_in_sqr=bool(include_conditional_phase_in_sqr),
            drift_model=drift_model,
        )


@dataclass(frozen=True, kw_only=True)
class CQEDSystemAdapter(_CQEDIdealSystem):
    model: Any

    def hilbert_dimension(
        self,
        *,
        sequence: GateSequence | None = None,
        primitive: PrimitiveGate | None = None,
        subspace: Subspace | None = None,
        target: TargetUnitary | TargetStateMapping | None = None,
    ) -> int | None:
        if hasattr(self.model, "subsystem_dims"):
            return int(prod(int(dim) for dim in getattr(self.model, "subsystem_dims")))
        return super().hilbert_dimension(sequence=sequence, primitive=primitive, subspace=subspace, target=target)

    def subsystem_dimensions(
        self,
        *,
        sequence: GateSequence | None = None,
        full_dim: int | None = None,
        subspace: Subspace | None = None,
    ) -> tuple[int, ...]:
        if hasattr(self.model, "subsystem_dims"):
            return tuple(int(dim) for dim in getattr(self.model, "subsystem_dims"))
        return super().subsystem_dimensions(sequence=sequence, full_dim=full_dim, subspace=subspace)

    def infer_n_cav(
        self,
        *,
        sequence: GateSequence | None = None,
        full_dim: int | None = None,
        subspace: Subspace | None = None,
    ) -> int | None:
        dims = self.subsystem_dimensions(sequence=sequence, full_dim=full_dim, subspace=subspace)
        if len(dims) >= 2 and dims[0] == 2:
            return int(dims[1])
        return super().infer_n_cav(sequence=sequence, full_dim=full_dim, subspace=subspace)

    def runtime_model(self) -> Any | None:
        return self.model

    def with_model(self, model: Any) -> QuantumSystem:
        return CQEDSystemAdapter(model=model)

    def to_record(self) -> dict[str, Any]:
        row = super().to_record()
        row["kind"] = "CQEDSystemAdapter"
        row["model_class"] = self.model.__class__.__name__
        row["subsystem_dims"] = [int(dim) for dim in self.subsystem_dimensions()]
        return row

    def simulate_primitive_unitary(self, primitive: PrimitiveGate, *, settings: Mapping[str, Any]) -> np.ndarray:
        if primitive.waveform is None:
            raise ValueError("PrimitiveGate is not waveform defined.")
        cache_key = primitive._operator_cache_key(self.model, settings)
        cached = primitive._operator_cache.get(cache_key)
        if cached is not None:
            return cached

        pulses, drive_ops, meta = _waveform_payload(primitive.waveform(primitive.runtime_parameters(), self.model))
        dt = float(settings.get("dt", 1.0e-9))
        compiled = _compile_waveform(
            pulses,
            dt=dt,
            hardware=settings.get("compiler_hardware"),
            crosstalk=settings.get("compiler_crosstalk"),
            t_end=meta.get("t_end"),
        )
        cfg = _runtime_simulation_config(settings)
        if settings.get("c_ops") or settings.get("noise"):
            raise ValueError(
                "Waveform primitive unitary extraction only supports closed-system simulation. "
                "Use state mappings for dissipative synthesis."
            )
        operator = _final_unitary_from_compiled(
            self.model,
            compiled,
            drive_ops,
            frame=cfg.frame,
            config=cfg,
        )
        primitive._operator_cache[cache_key] = operator
        if primitive.hilbert_dim is None:
            primitive.hilbert_dim = int(operator.shape[0])
        return operator

    def simulate_primitive_states(
        self,
        primitive: PrimitiveGate,
        states: Sequence[qt.Qobj],
        *,
        settings: Mapping[str, Any],
    ) -> list[qt.Qobj]:
        if primitive.waveform is None:
            raise ValueError("PrimitiveGate is not waveform defined.")
        if settings.get("c_ops") is None and settings.get("noise") is None:
            return super().simulate_primitive_states(primitive, states, settings=settings)

        pulses, drive_ops, meta = _waveform_payload(primitive.waveform(primitive.runtime_parameters(), self.model))
        dt = float(settings.get("dt", 1.0e-9))
        compiled = _compile_waveform(
            pulses,
            dt=dt,
            hardware=settings.get("compiler_hardware"),
            crosstalk=settings.get("compiler_crosstalk"),
            t_end=meta.get("t_end"),
        )
        cfg = _runtime_simulation_config(settings)
        return _simulate_waveform_states(
            self.model,
            compiled,
            drive_ops,
            _coerce_states_with_dims(states, self.subsystem_dimensions()),
            config=cfg,
            c_ops=None if settings.get("c_ops") is None else list(settings["c_ops"]),
            noise=settings.get("noise"),
            max_workers=int(settings.get("state_workers", 1)),
        )


def requires_cqed_system(
    *,
    primitives: Sequence[Any] | None = None,
    gateset: Sequence[str] | None = None,
) -> bool:
    if gateset is not None and any(_normalize_gate_name(name) in _CQED_GATE_NAMES for name in gateset):
        return True
    if primitives is not None and any(isinstance(gate, _CQED_GATE_TYPES) for gate in primitives):
        return True
    return False


def resolve_quantum_system(
    *,
    system: QuantumSystem | None = None,
    model: Any | None = None,
    subspace: Subspace | None = None,
    primitives: Sequence[Any] | None = None,
    gateset: Sequence[str] | None = None,
) -> QuantumSystem:
    if system is not None and model is not None:
        raise ValueError("Pass either system=... or model=..., not both.")
    if system is not None:
        return system
    if model is not None:
        return CQEDSystemAdapter(model=model)
    if requires_cqed_system(primitives=primitives, gateset=gateset):
        dim = None if subspace is None else int(subspace.full_dim)
        return _CQEDIdealSystem(dimension=dim)
    dim = None if subspace is None else int(subspace.full_dim)
    return GenericQuantumSystem(dimension=dim)
