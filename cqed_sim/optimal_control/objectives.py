from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np
import qutip as qt

from cqed_sim.unitary_synthesis.subspace import Subspace

from .utils import as_density_matrix, as_square_matrix, as_state_vector


@dataclass(frozen=True)
class CustomObjectiveContext:
    problem: Any
    system: Any
    schedule: Any
    resolved_waveforms: Any
    propagation: Any
    final_unitary: np.ndarray


@dataclass(frozen=True)
class CustomObjectiveEvaluation:
    cost: float
    gradient_physical: np.ndarray
    metrics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cost", float(self.cost))
        object.__setattr__(self, "gradient_physical", np.asarray(self.gradient_physical, dtype=float))


@dataclass(frozen=True)
class CustomControlObjective:
    evaluator: Callable[[CustomObjectiveContext], CustomObjectiveEvaluation] = field(repr=False)
    weight: float = 1.0
    name: str = "custom_objective"

    def __post_init__(self) -> None:
        if not callable(self.evaluator):
            raise TypeError("CustomControlObjective.evaluator must be callable.")
        if float(self.weight) <= 0.0:
            raise ValueError("CustomControlObjective.weight must be positive.")

    def evaluate(self, context: CustomObjectiveContext) -> CustomObjectiveEvaluation:
        evaluation = self.evaluator(context)
        if not isinstance(evaluation, CustomObjectiveEvaluation):
            raise TypeError("CustomControlObjective.evaluator must return a CustomObjectiveEvaluation.")
        return evaluation


@dataclass(frozen=True)
class StateTransferPair:
    initial_state: qt.Qobj | np.ndarray | Any
    target_state: qt.Qobj | np.ndarray | Any
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        if float(self.weight) <= 0.0:
            raise ValueError("StateTransferPair.weight must be positive.")


@dataclass(frozen=True)
class StateTransferObjective:
    pairs: tuple[StateTransferPair, ...]
    weight: float = 1.0
    name: str = "state_transfer"

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("StateTransferObjective requires at least one state pair.")
        if float(self.weight) <= 0.0:
            raise ValueError("StateTransferObjective.weight must be positive.")

    @classmethod
    def single(
        cls,
        initial_state: qt.Qobj | np.ndarray | Any,
        target_state: qt.Qobj | np.ndarray | Any,
        *,
        weight: float = 1.0,
        label: str | None = None,
        name: str = "state_preparation",
    ) -> "StateTransferObjective":
        return cls(pairs=(StateTransferPair(initial_state=initial_state, target_state=target_state, label=label),), weight=weight, name=name)

    def resolved_pairs(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        initial_states: list[np.ndarray] = []
        target_states: list[np.ndarray] = []
        weights: list[float] = []
        labels: list[str] = []
        for index, pair in enumerate(self.pairs):
            initial_states.append(
                as_state_vector(pair.initial_state, full_dim=full_dim, subspace=subspace, name=f"{self.name}.initial[{index}]")
            )
            target_states.append(
                as_state_vector(pair.target_state, full_dim=full_dim, subspace=subspace, name=f"{self.name}.target[{index}]")
            )
            weights.append(float(pair.weight))
            labels.append(pair.label or f"pair_{index}")
        weight_array = np.asarray(weights, dtype=float)
        weight_array = weight_array / np.sum(weight_array)
        return (
            np.asarray(initial_states, dtype=np.complex128),
            np.asarray(target_states, dtype=np.complex128),
            weight_array,
            tuple(labels),
        )


@dataclass(frozen=True)
class DensityMatrixTransferPair:
    initial_state: qt.Qobj | np.ndarray | Any
    target_state: qt.Qobj | np.ndarray | Any
    weight: float = 1.0
    label: str | None = None

    def __post_init__(self) -> None:
        if float(self.weight) <= 0.0:
            raise ValueError("DensityMatrixTransferPair.weight must be positive.")


@dataclass(frozen=True)
class DensityMatrixTransferObjective:
    pairs: tuple[DensityMatrixTransferPair, ...]
    weight: float = 1.0
    name: str = "density_matrix_transfer"

    def __post_init__(self) -> None:
        if not self.pairs:
            raise ValueError("DensityMatrixTransferObjective requires at least one state pair.")
        if float(self.weight) <= 0.0:
            raise ValueError("DensityMatrixTransferObjective.weight must be positive.")

    @classmethod
    def single(
        cls,
        initial_state: qt.Qobj | np.ndarray | Any,
        target_state: qt.Qobj | np.ndarray | Any,
        *,
        weight: float = 1.0,
        label: str | None = None,
        name: str = "density_matrix_transfer",
    ) -> "DensityMatrixTransferObjective":
        return cls(
            pairs=(
                DensityMatrixTransferPair(
                    initial_state=initial_state,
                    target_state=target_state,
                    label=label,
                ),
            ),
            weight=weight,
            name=name,
        )

    def resolved_pairs(
        self,
        *,
        full_dim: int,
        subspace: Subspace | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        initial_states: list[np.ndarray] = []
        target_states: list[np.ndarray] = []
        weights: list[float] = []
        labels: list[str] = []
        for index, pair in enumerate(self.pairs):
            initial_states.append(
                as_density_matrix(
                    pair.initial_state,
                    full_dim=full_dim,
                    subspace=subspace,
                    name=f"{self.name}.initial[{index}]",
                )
            )
            target_states.append(
                as_density_matrix(
                    pair.target_state,
                    full_dim=full_dim,
                    subspace=subspace,
                    name=f"{self.name}.target[{index}]",
                )
            )
            weights.append(float(pair.weight))
            labels.append(pair.label or f"pair_{index}")
        weight_array = np.asarray(weights, dtype=float)
        weight_array = weight_array / np.sum(weight_array)
        return (
            np.asarray(initial_states, dtype=np.complex128),
            np.asarray(target_states, dtype=np.complex128),
            weight_array,
            tuple(labels),
        )


@dataclass(frozen=True)
class UnitaryObjective:
    target_operator: qt.Qobj | np.ndarray | Any
    subspace: Subspace | None = None
    ignore_global_phase: bool = True
    allow_diagonal_phase: bool = False
    phase_blocks: tuple[tuple[int, ...], ...] | None = None
    probe_states: tuple[qt.Qobj | np.ndarray | Any, ...] = field(default_factory=tuple)
    probe_strategy: str = "basis_plus_uniform"
    weight: float = 1.0
    name: str = "unitary"

    def __post_init__(self) -> None:
        matrix = as_square_matrix(self.target_operator, name="target_operator")
        object.__setattr__(self, "target_operator", matrix)
        if float(self.weight) <= 0.0:
            raise ValueError("UnitaryObjective.weight must be positive.")
        if self.phase_blocks is not None:
            normalized = tuple(tuple(int(index) for index in block) for block in self.phase_blocks)
            object.__setattr__(self, "phase_blocks", normalized)

    @property
    def target_dim(self) -> int:
        return int(np.asarray(self.target_operator).shape[0])

    def gauge(self) -> str:
        if self.phase_blocks:
            return "block"
        if self.allow_diagonal_phase:
            return "diagonal"
        if self.ignore_global_phase:
            return "global"
        return "none"

    def resolved_pairs(self, *, full_dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
        from cqed_sim.unitary_synthesis.targets import TargetUnitary

        target = TargetUnitary(
            matrix=np.asarray(self.target_operator, dtype=np.complex128),
            ignore_global_phase=bool(self.ignore_global_phase),
            allow_diagonal_phase=bool(self.allow_diagonal_phase),
            phase_blocks=self.phase_blocks,
            probe_states=self.probe_states,
            open_system_probe_strategy=str(self.probe_strategy),
        )
        initial_qobjs, target_qobjs = target.resolved_probe_pairs(full_dim=full_dim, subspace=self.subspace)
        initial_states = np.asarray(
            [as_state_vector(state, full_dim=full_dim, name=f"{self.name}.probe_initial[{index}]") for index, state in enumerate(initial_qobjs)],
            dtype=np.complex128,
        )
        target_states = np.asarray(
            [as_state_vector(state, full_dim=full_dim, name=f"{self.name}.probe_target[{index}]") for index, state in enumerate(target_qobjs)],
            dtype=np.complex128,
        )
        weights = np.full(initial_states.shape[0], 1.0 / max(initial_states.shape[0], 1), dtype=float)
        labels = tuple(f"probe_{index}" for index in range(initial_states.shape[0]))
        return initial_states, target_states, weights, labels


def state_preparation_objective(
    initial_state: qt.Qobj | np.ndarray | Any,
    target_state: qt.Qobj | np.ndarray | Any,
    *,
    weight: float = 1.0,
    label: str | None = None,
) -> StateTransferObjective:
    return StateTransferObjective.single(
        initial_state,
        target_state,
        weight=weight,
        label=label,
        name="state_preparation",
    )


def multi_state_transfer_objective(
    initial_states: Sequence[qt.Qobj | np.ndarray | Any],
    target_states: Sequence[qt.Qobj | np.ndarray | Any],
    *,
    pair_weights: Sequence[float] | None = None,
    weight: float = 1.0,
    name: str = "state_transfer",
) -> StateTransferObjective:
    if len(initial_states) != len(target_states):
        raise ValueError("initial_states and target_states must have the same length.")
    if pair_weights is None:
        weights = [1.0] * len(initial_states)
    else:
        weights = list(pair_weights)
        if len(weights) != len(initial_states):
            raise ValueError("pair_weights must match the number of state pairs.")
    pairs = tuple(
        StateTransferPair(initial_state=psi_in, target_state=psi_out, weight=float(pair_weight), label=f"pair_{index}")
        for index, (psi_in, psi_out, pair_weight) in enumerate(zip(initial_states, target_states, weights, strict=True))
    )
    return StateTransferObjective(pairs=pairs, weight=weight, name=name)


def objective_from_unitary_synthesis_target(
    target: Any,
    *,
    subspace: Subspace | None = None,
    weight: float = 1.0,
) -> StateTransferObjective | UnitaryObjective:
    from cqed_sim.unitary_synthesis.targets import TargetStateMapping, TargetUnitary

    if isinstance(target, TargetUnitary):
        return UnitaryObjective(
            target_operator=np.asarray(target.matrix, dtype=np.complex128),
            subspace=subspace,
            ignore_global_phase=bool(target.ignore_global_phase),
            allow_diagonal_phase=bool(target.allow_diagonal_phase),
            phase_blocks=target.phase_blocks,
            probe_states=target.probe_states,
            probe_strategy=target.open_system_probe_strategy,
            weight=weight,
        )
    if isinstance(target, TargetStateMapping):
        pairs = tuple(
            StateTransferPair(initial_state=psi_in, target_state=psi_out, weight=float(pair_weight), label=f"pair_{index}")
            for index, (psi_in, psi_out, pair_weight) in enumerate(zip(target.initial_states, target.target_states, target.weights, strict=True))
        )
        return StateTransferObjective(pairs=pairs, weight=weight, name="state_transfer")
    raise TypeError(f"Unsupported synthesis target type '{type(target).__name__}'.")


__all__ = [
    "CustomObjectiveContext",
    "CustomObjectiveEvaluation",
    "CustomControlObjective",
    "DensityMatrixTransferPair",
    "DensityMatrixTransferObjective",
    "StateTransferPair",
    "StateTransferObjective",
    "UnitaryObjective",
    "state_preparation_objective",
    "multi_state_transfer_objective",
    "objective_from_unitary_synthesis_target",
]