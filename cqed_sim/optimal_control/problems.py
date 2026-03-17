from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

from cqed_sim.core import FrameSpec
from cqed_sim.core.drive_targets import SidebandDriveSpec, TransmonTransitionDriveSpec

from .utils import as_square_matrix, quadrature_operators

if TYPE_CHECKING:
    from .parameterizations import PiecewiseConstantParameterization


@dataclass(frozen=True)
class ControlTerm:
    name: str
    operator: np.ndarray
    amplitude_bounds: tuple[float, float] = (-float("inf"), float("inf"))
    export_channel: str | None = None
    drive_target: str | TransmonTransitionDriveSpec | SidebandDriveSpec | None = None
    quadrature: str = "I"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operator = as_square_matrix(self.operator, name=f"ControlTerm[{self.name}].operator")
        object.__setattr__(self, "operator", operator)
        lower, upper = float(self.amplitude_bounds[0]), float(self.amplitude_bounds[1])
        if lower > upper:
            raise ValueError("ControlTerm amplitude bounds must satisfy lower <= upper.")
        object.__setattr__(self, "amplitude_bounds", (lower, upper))
        quadrature = str(self.quadrature).upper()
        if quadrature not in {"I", "Q", "SCALAR"}:
            raise ValueError("ControlTerm.quadrature must be 'I', 'Q', or 'SCALAR'.")
        object.__setattr__(self, "quadrature", quadrature)


@dataclass(frozen=True)
class ControlSystem:
    drift_hamiltonian: np.ndarray
    control_operators: tuple[np.ndarray, ...]
    weight: float = 1.0
    label: str = "nominal"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        drift = as_square_matrix(self.drift_hamiltonian, name=f"ControlSystem[{self.label}].drift_hamiltonian")
        operators = tuple(as_square_matrix(operator, name=f"ControlSystem[{self.label}].control_operator") for operator in self.control_operators)
        if not operators:
            raise ValueError("ControlSystem requires at least one control operator.")
        for operator in operators:
            if operator.shape != drift.shape:
                raise ValueError("All control operators must have the same shape as the drift Hamiltonian.")
        if float(self.weight) <= 0.0:
            raise ValueError("ControlSystem.weight must be positive.")
        object.__setattr__(self, "drift_hamiltonian", drift)
        object.__setattr__(self, "control_operators", operators)


@dataclass(frozen=True)
class ModelControlChannelSpec:
    name: str
    target: str | TransmonTransitionDriveSpec | SidebandDriveSpec
    quadratures: tuple[str, ...] = ("I", "Q")
    amplitude_bounds: tuple[float, float] = (-float("inf"), float("inf"))
    export_channel: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        quadratures = tuple(str(value).upper() for value in self.quadratures)
        if not quadratures:
            raise ValueError("ModelControlChannelSpec requires at least one quadrature.")
        if any(value not in {"I", "Q"} for value in quadratures):
            raise ValueError("ModelControlChannelSpec.quadratures must contain only 'I' and 'Q'.")
        lower, upper = float(self.amplitude_bounds[0]), float(self.amplitude_bounds[1])
        if lower > upper:
            raise ValueError("ModelControlChannelSpec amplitude bounds must satisfy lower <= upper.")
        object.__setattr__(self, "quadratures", quadratures)
        object.__setattr__(self, "amplitude_bounds", (lower, upper))


@dataclass(frozen=True)
class ModelEnsembleMember:
    model: Any
    label: str
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.weight) <= 0.0:
            raise ValueError("ModelEnsembleMember.weight must be positive.")


@dataclass
class ControlProblem:
    parameterization: Any
    systems: tuple[ControlSystem, ...]
    objectives: tuple[Any, ...]
    penalties: tuple[Any, ...] = ()
    ensemble_aggregate: str = "mean"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.systems:
            raise ValueError("ControlProblem requires at least one control system.")
        if not self.objectives:
            raise ValueError("ControlProblem requires at least one objective.")
        if self.ensemble_aggregate not in {"mean", "worst"}:
            raise ValueError("ControlProblem.ensemble_aggregate must be 'mean' or 'worst'.")
        parameterization = self.parameterization
        if not hasattr(parameterization, "control_terms") or not hasattr(parameterization, "time_grid"):
            raise TypeError("ControlProblem.parameterization must expose control_terms and time_grid.")
        n_controls = len(parameterization.control_terms)
        if n_controls <= 0:
            raise ValueError("ControlProblem.parameterization must define at least one control term.")
        expected_shape = self.systems[0].drift_hamiltonian.shape
        for system in self.systems:
            if len(system.control_operators) != n_controls:
                raise ValueError("Each control system must supply one control operator per parameterized control term.")
            if system.drift_hamiltonian.shape != expected_shape:
                raise ValueError("All control systems must share a common Hilbert-space dimension.")

    @property
    def full_dim(self) -> int:
        return int(self.systems[0].drift_hamiltonian.shape[0])

    @property
    def control_terms(self):
        return self.parameterization.control_terms

    @property
    def time_grid(self):
        return self.parameterization.time_grid

    @property
    def n_controls(self) -> int:
        return int(self.parameterization.n_controls)

    @property
    def n_slices(self) -> int:
        return int(self.parameterization.n_slices)

    def zero_schedule(self):
        return self.parameterization.zero_schedule()


def _resolve_drive_pair(model: Any, target: str | TransmonTransitionDriveSpec | SidebandDriveSpec):
    if isinstance(target, str):
        if not hasattr(model, "drive_coupling_operators"):
            raise ValueError("Model does not expose drive_coupling_operators().")
        couplings = model.drive_coupling_operators()
        if target not in couplings:
            raise ValueError(f"Unsupported drive target '{target}'.")
        return couplings[target]
    if isinstance(target, TransmonTransitionDriveSpec):
        if not hasattr(model, "transmon_transition_operators"):
            raise ValueError("Model does not expose transmon_transition_operators(...).")
        return model.transmon_transition_operators(target.lower_level, target.upper_level)
    if isinstance(target, SidebandDriveSpec):
        if not hasattr(model, "sideband_drive_operators"):
            raise ValueError("Model does not expose sideband_drive_operators(...).")
        return model.sideband_drive_operators(
            mode=target.mode,
            lower_level=target.lower_level,
            upper_level=target.upper_level,
            sideband=target.sideband,
        )
    raise TypeError(f"Unsupported drive target type '{type(target).__name__}'.")


def build_control_terms_from_model(
    model: Any,
    channel_specs: Sequence[ModelControlChannelSpec],
) -> tuple[ControlTerm, ...]:
    terms: list[ControlTerm] = []
    for spec in channel_specs:
        raising, lowering = _resolve_drive_pair(model, spec.target)
        i_term, q_term = quadrature_operators(raising, lowering)
        export_channel = spec.export_channel or str(spec.name)
        if "I" in spec.quadratures:
            terms.append(
                ControlTerm(
                    name=f"{spec.name}_I",
                    operator=i_term,
                    amplitude_bounds=spec.amplitude_bounds,
                    export_channel=export_channel,
                    drive_target=spec.target,
                    quadrature="I",
                    metadata={"source": str(spec.name), **dict(spec.metadata)},
                )
            )
        if "Q" in spec.quadratures:
            terms.append(
                ControlTerm(
                    name=f"{spec.name}_Q",
                    operator=q_term,
                    amplitude_bounds=spec.amplitude_bounds,
                    export_channel=export_channel,
                    drive_target=spec.target,
                    quadrature="Q",
                    metadata={"source": str(spec.name), **dict(spec.metadata)},
                )
            )
    return tuple(terms)


def build_control_system_from_model(
    model: Any,
    *,
    frame: FrameSpec | None,
    channel_specs: Sequence[ModelControlChannelSpec],
    weight: float = 1.0,
    label: str = "nominal",
    metadata: Mapping[str, Any] | None = None,
) -> ControlSystem:
    drift = np.asarray(model.static_hamiltonian(frame=frame).full(), dtype=np.complex128)
    terms = build_control_terms_from_model(model, channel_specs)
    return ControlSystem(
        drift_hamiltonian=drift,
        control_operators=tuple(term.operator for term in terms),
        weight=float(weight),
        label=str(label),
        metadata={} if metadata is None else dict(metadata),
    )


def build_control_problem_from_model(
    model: Any,
    *,
    frame: FrameSpec | None,
    time_grid,
    channel_specs: Sequence[ModelControlChannelSpec],
    objectives: Sequence[Any],
    penalties: Sequence[Any] = (),
    ensemble_members: Sequence[ModelEnsembleMember] = (),
    ensemble_aggregate: str = "mean",
    metadata: Mapping[str, Any] | None = None,
) -> ControlProblem:
    from .parameterizations import PiecewiseConstantParameterization

    control_terms = build_control_terms_from_model(model, channel_specs)
    parameterization = PiecewiseConstantParameterization(time_grid=time_grid, control_terms=control_terms)
    systems = [
        build_control_system_from_model(
            model,
            frame=frame,
            channel_specs=channel_specs,
            weight=1.0,
            label="nominal",
            metadata={"model_type": type(model).__name__},
        )
    ]
    for member in ensemble_members:
        systems.append(
            build_control_system_from_model(
                member.model,
                frame=frame,
                channel_specs=channel_specs,
                weight=float(member.weight),
                label=str(member.label),
                metadata=dict(member.metadata),
            )
        )
    return ControlProblem(
        parameterization=parameterization,
        systems=tuple(systems),
        objectives=tuple(objectives),
        penalties=tuple(penalties),
        ensemble_aggregate=str(ensemble_aggregate),
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "ControlTerm",
    "ControlSystem",
    "ModelControlChannelSpec",
    "ModelEnsembleMember",
    "ControlProblem",
    "build_control_terms_from_model",
    "build_control_system_from_model",
    "build_control_problem_from_model",
]