from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import qutip as qt

from cqed_sim.core.conventions import qubit_cavity_index
from cqed_sim.unitary_synthesis.subspace import Subspace

from .action_spaces import CavityDisplacementAction, PrimitiveAction, SidebandAction, WaitAction
from .configs import TaskKind


def _two_mode_dims(model: Any) -> tuple[int, int]:
    dims = tuple(int(dim) for dim in getattr(model, "subsystem_dims"))
    if len(dims) != 2:
        raise ValueError("The first-pass RL tasks currently target the two-mode qubit-storage path.")
    return dims[0], dims[1]


def _ground_vacuum_state(model: Any) -> qt.Qobj:
    _two_mode_dims(model)
    return model.basis_state(0, 0)


def _coherent_joint_state(model: Any, alpha: complex) -> qt.Qobj:
    n_tr, n_cav = _two_mode_dims(model)
    return qt.tensor(qt.basis(n_tr, 0), qt.coherent(n_cav, complex(alpha)))


def _fock_joint_state(model: Any, cavity_level: int) -> qt.Qobj:
    n_tr, n_cav = _two_mode_dims(model)
    if not 0 <= int(cavity_level) < n_cav:
        raise ValueError(f"cavity_level must lie in [0, {n_cav - 1}].")
    return qt.tensor(qt.basis(n_tr, 0), qt.basis(n_cav, int(cavity_level)))


def _storage_superposition_state(model: Any, coefficients: tuple[complex, ...]) -> qt.Qobj:
    n_tr, n_cav = _two_mode_dims(model)
    if not coefficients:
        raise ValueError("coefficients must not be empty.")
    cavity = sum(complex(weight) * qt.basis(n_cav, level) for level, weight in enumerate(coefficients[:n_cav]))
    norm = float(cavity.norm())
    if norm <= 1.0e-15:
        raise ValueError("The requested cavity superposition has zero norm under the current truncation.")
    return qt.tensor(qt.basis(n_tr, 0), cavity / norm)


def _cat_state(model: Any, alpha: complex, *, parity: str) -> qt.Qobj:
    n_tr, n_cav = _two_mode_dims(model)
    plus = qt.coherent(n_cav, complex(alpha))
    minus = qt.coherent(n_cav, -complex(alpha))
    if parity == "even":
        cavity = (plus + minus).unit()
    elif parity == "odd":
        cavity = (plus - minus).unit()
    else:
        raise ValueError("parity must be 'even' or 'odd'.")
    return qt.tensor(qt.basis(n_tr, 0), cavity)


def _bell_state(model: Any) -> qt.Qobj:
    _two_mode_dims(model)
    return (model.basis_state(0, 0) - 1j * model.basis_state(1, 1)).unit()


def _conditional_phase_target(_model: Any, phase: float) -> np.ndarray:
    del _model
    return np.diag([1.0, 1.0, 1.0, np.exp(1j * float(phase))]).astype(np.complex128)


def _lowest_qubit_cavity_block_subspace(model: Any, *, n_match: int) -> Subspace:
    n_tr, n_cav = _two_mode_dims(model)
    if n_tr < 2:
        raise ValueError("Conditional-phase tasks require at least two transmon levels.")
    indices: list[int] = []
    labels: list[str] = []
    for cavity_level in range(int(n_match) + 1):
        indices.extend([
            qubit_cavity_index(n_cav, 0, cavity_level),
            qubit_cavity_index(n_cav, 1, cavity_level),
        ])
        labels.extend([f"|g,{cavity_level}>", f"|e,{cavity_level}>"])
    return Subspace(
        full_dim=int(n_tr * n_cav),
        indices=tuple(indices),
        labels=tuple(labels),
        kind="qubit_cavity_block",
        metadata={"n_match": int(n_match), "n_cav": int(n_cav)},
    )


def _basis_state_from_index(model: Any, flat_index: int) -> qt.Qobj:
    dims = [int(dim) for dim in getattr(model, "subsystem_dims")]
    vector = np.zeros(int(np.prod(dims)), dtype=np.complex128)
    vector[int(flat_index)] = 1.0
    return qt.Qobj(vector.reshape((-1, 1)), dims=[dims, [1] * len(dims)])


@dataclass
class HybridBenchmarkTask:
    name: str
    tier: int
    kind: TaskKind
    description: str
    initial_state_factory: Callable[[Any], qt.Qobj]
    target_state_factory: Callable[[Any], qt.Qobj] | None = None
    target_operator_factory: Callable[[Any], np.ndarray | qt.Qobj] | None = None
    subspace_factory: Callable[[Any], Subspace] | None = None
    horizon: int = 4
    success_threshold: float = 0.95
    baseline_actions: tuple[Any, ...] = ()
    gauge: str = "global"
    target_ancilla_level: int = 0
    recommended_action_mode: str = "parametric"
    recommended_observation_mode: str = "ideal_summary"
    expected_diagnostics: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def build_initial_state(self, model: Any) -> qt.Qobj:
        return self.initial_state_factory(model)

    def build_target_state(self, model: Any) -> qt.Qobj:
        if self.target_state_factory is None:
            raise ValueError(f"Task '{self.name}' does not define a target state.")
        return self.target_state_factory(model)

    def build_target_operator(self, model: Any) -> np.ndarray | None:
        if self.target_operator_factory is None:
            return None
        operator = self.target_operator_factory(model)
        return np.asarray(operator.full() if isinstance(operator, qt.Qobj) else operator, dtype=np.complex128)

    def build_subspace(self, model: Any) -> Subspace | None:
        return None if self.subspace_factory is None else self.subspace_factory(model)

    def build_probe_states(self, model: Any) -> list[qt.Qobj]:
        subspace = self.build_subspace(model)
        if subspace is None:
            return [self.build_initial_state(model)]
        return [_basis_state_from_index(model, index) for index in subspace.indices]

    def build_target_probe_states(self, model: Any) -> list[qt.Qobj]:
        subspace = self.build_subspace(model)
        operator = self.build_target_operator(model)
        if subspace is None or operator is None:
            raise ValueError(f"Task '{self.name}' does not define a subspace target operator.")
        outputs: list[qt.Qobj] = []
        dims = [int(dim) for dim in getattr(model, "subsystem_dims")]
        for basis_vector in np.eye(subspace.dim, dtype=np.complex128).T:
            transformed = subspace.embed(operator @ basis_vector)
            outputs.append(qt.Qobj(transformed.reshape((-1, 1)), dims=[dims, [1] * len(dims)]))
        return outputs


def vacuum_preservation_task() -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="vacuum_preservation",
        tier=1,
        kind="state_preparation",
        description="Keep the ancilla-storage system in |g,0> while minimizing unnecessary control effort.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=_ground_vacuum_state,
        horizon=1,
        success_threshold=0.999,
        baseline_actions=(WaitAction(duration=0.0),),
        expected_diagnostics=("ancilla_populations", "photon_number_distribution"),
    )


def coherent_state_preparation_task(alpha: complex = 0.8 + 0.0j, duration: float = 120.0e-9) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="coherent_state_preparation",
        tier=1,
        kind="state_preparation",
        description="Prepare |g> tensor |alpha> using a single calibrated cavity displacement as a baseline.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=lambda model: _coherent_joint_state(model, alpha),
        horizon=1,
        success_threshold=0.99,
        baseline_actions=(CavityDisplacementAction(alpha=complex(alpha), duration=float(duration)),),
        recommended_action_mode="primitive",
        expected_diagnostics=("reduced_cavity_state", "photon_number_distribution", "wigner"),
        metadata={"alpha": complex(alpha)},
    )


def fock_state_preparation_task(cavity_level: int = 1) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name=f"fock_{int(cavity_level)}_preparation",
        tier=1,
        kind="state_preparation",
        description="Prepare a storage Fock state while returning the ancilla to |g>.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=lambda model: _fock_joint_state(model, cavity_level),
        horizon=2,
        success_threshold=0.92,
        recommended_action_mode="parametric",
        expected_diagnostics=("photon_number_distribution", "reduced_cavity_state", "wigner"),
        metadata={"cavity_level": int(cavity_level)},
    )


def storage_superposition_task(coefficients: tuple[complex, ...] = (1.0 + 0.0j, 1.0 + 0.0j)) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="storage_superposition_preparation",
        tier=1,
        kind="state_preparation",
        description="Prepare a cavity basis superposition with the ancilla parked in |g>.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=lambda model: _storage_superposition_state(model, coefficients),
        horizon=2,
        success_threshold=0.92,
        recommended_action_mode="parametric",
        expected_diagnostics=("photon_number_distribution", "reduced_cavity_state", "wigner"),
        metadata={"coefficients": tuple(complex(value) for value in coefficients)},
    )


def even_cat_preparation_task(alpha: complex = 1.2 + 0.0j) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="even_cat_preparation",
        tier=2,
        kind="state_preparation",
        description="Prepare an even cat state in the storage cavity while returning the ancilla to |g>.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=lambda model: _cat_state(model, alpha, parity="even"),
        horizon=3,
        success_threshold=0.90,
        recommended_action_mode="parametric",
        expected_diagnostics=("wigner", "parity", "ancilla_populations"),
        metadata={"alpha": complex(alpha), "parity": "even"},
    )


def odd_cat_preparation_task(alpha: complex = 1.2 + 0.0j) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="odd_cat_preparation",
        tier=2,
        kind="state_preparation",
        description="Prepare an odd cat state in the storage cavity while returning the ancilla to |g>.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=lambda model: _cat_state(model, alpha, parity="odd"),
        horizon=3,
        success_threshold=0.88,
        recommended_action_mode="parametric",
        expected_diagnostics=("wigner", "parity", "ancilla_populations"),
        metadata={"alpha": complex(alpha), "parity": "odd"},
    )


def ancilla_storage_bell_task(duration: float = 80.0e-9, amplitude: float = 2.0 * np.pi * 1.5e6) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="ancilla_storage_bell",
        tier=3,
        kind="state_preparation",
        description="Prepare a hybrid ancilla-storage entangled state using a blue-sideband baseline pulse.",
        initial_state_factory=_ground_vacuum_state,
        target_state_factory=_bell_state,
        horizon=1,
        success_threshold=0.85,
        baseline_actions=(
            SidebandAction(
                amplitude=float(amplitude),
                detuning=0.0,
                duration=float(duration),
                lower_level=0,
                upper_level=1,
                sideband="blue",
            ),
        ),
        recommended_action_mode="primitive",
        expected_diagnostics=("joint_state", "reduced_cavity_state", "ancilla_populations"),
    )


def conditional_phase_gate_task(phase: float = np.pi) -> HybridBenchmarkTask:
    return HybridBenchmarkTask(
        name="conditional_phase_gate",
        tier=4,
        kind="unitary_synthesis",
        description="Implement a conditional phase on the |e,1> basis state of the lowest qubit-cavity block.",
        initial_state_factory=_ground_vacuum_state,
        target_operator_factory=lambda model: _conditional_phase_target(model, phase),
        subspace_factory=lambda model: _lowest_qubit_cavity_block_subspace(model, n_match=1),
        horizon=1,
        success_threshold=0.99,
        baseline_actions=(WaitAction(duration=float(-phase / getattr(model_placeholder, "chi", -phase))),),
        gauge="block",
        recommended_action_mode="primitive",
        expected_diagnostics=("process_fidelity", "leakage", "block_phase"),
        metadata={"phase": float(phase)},
    )


class _ModelPlaceholder:
    chi = -1.0


model_placeholder = _ModelPlaceholder()


def finalize_conditional_phase_baseline(task: HybridBenchmarkTask, model: Any) -> HybridBenchmarkTask:
    if task.name != "conditional_phase_gate":
        return task
    chi = float(getattr(model, "chi", 0.0))
    phase = float(task.metadata.get("phase", np.pi))
    if abs(chi) <= 1.0e-15:
        wait_duration = 0.0
    else:
        wait_duration = max(0.0, float(-phase / chi))
    task.baseline_actions = (WaitAction(duration=wait_duration),)
    return task


def benchmark_task_suite() -> dict[str, HybridBenchmarkTask]:
    return {
        "vacuum_preservation": vacuum_preservation_task(),
        "coherent_state_preparation": coherent_state_preparation_task(),
        "fock_1_preparation": fock_state_preparation_task(1),
        "storage_superposition_preparation": storage_superposition_task(),
        "even_cat_preparation": even_cat_preparation_task(),
        "odd_cat_preparation": odd_cat_preparation_task(),
        "ancilla_storage_bell": ancilla_storage_bell_task(),
        "conditional_phase_gate": conditional_phase_gate_task(),
    }


__all__ = [
    "HybridBenchmarkTask",
    "vacuum_preservation_task",
    "coherent_state_preparation_task",
    "fock_state_preparation_task",
    "storage_superposition_task",
    "even_cat_preparation_task",
    "odd_cat_preparation_task",
    "ancilla_storage_bell_task",
    "conditional_phase_gate_task",
    "finalize_conditional_phase_baseline",
    "benchmark_task_suite",
]