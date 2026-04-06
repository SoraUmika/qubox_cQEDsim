"""Validated catalog of supported map-synthesis optimization use cases.

This script runs a compact set of deterministic examples against the public
``cqed_sim.map_synthesis`` API and summarizes which optimization workflows are
covered by the current surface:

- subspace unitary targets with gate-duration optimization
- state-ensemble mapping targets
- reduced-state targets that ignore spectator subsystems
- channel/process targets
- isometry/encoding targets
- observable targets
- trajectory/checkpoint targets
- robust optimization under sampled parameter uncertainty
- leakage, checkpoint-leakage, and edge-projector regularizers
- warm starts, Pareto exploration, and gate-order search

The generated figure is saved to the unitary-synthesis tutorial asset path by
default so the documentation and checked-in MkDocs site can reuse the exact
artifact produced by this example.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import LinAlgWarning

from cqed_sim.map_synthesis import (
    ExecutionOptions,
    GateOrderConfig,
    GateOrderOptimizer,
    LeakagePenalty,
    MultiObjective,
    ObservableTarget,
    ParameterDistribution,
    PrimitiveGate,
    QuantumMapSynthesizer,
    Subspace,
    SynthesisConstraints,
    TargetChannel,
    TargetIsometry,
    TargetReducedStateMapping,
    TargetStateMapping,
    TargetUnitary,
    TrajectoryCheckpoint,
    TrajectoryTarget,
    make_gate_from_matrix,
)
from cqed_sim.map_synthesis.metrics import subspace_unitary_fidelity


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLOT_PATH = (
    REPO_ROOT
    / "documentations"
    / "assets"
    / "images"
    / "tutorials"
    / "unitary_synthesis_optimization_catalog.png"
)


@dataclass
class CaseSummary:
    label: str
    api_surface: str
    primary_metric: str
    primary_value: float
    validation_score: float
    objective: float
    duration_ns: float
    execution_engine: str
    note: str = ""


@dataclass
class WorkflowSummary:
    label: str
    api_surface: str
    result: str
    note: str = ""


@dataclass
class _RobustAngleModel:
    scale: float = 1.0
    subsystem_dims: tuple[int, ...] = (2,)


class _FixedSamples:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.index = 0

    def sample(self, rng) -> float:  # pragma: no cover - exercised by the synthesizer.
        del rng
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return float(value)

    def nominal(self) -> float:
        return float(self.values[0])


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def hadamard() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)


def pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def cnot() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )


def rotation_primitive(
    *,
    name: str,
    theta: float,
    duration: float,
) -> PrimitiveGate:
    return PrimitiveGate(
        name=name,
        duration=duration,
        matrix=lambda params, model: rotation_y(float(params["theta"])),
        parameters={"theta": theta, "duration": duration},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 80.0e-9)},
        hilbert_dim=2,
    )


def robust_rotation_primitive(*, theta: float, duration: float) -> PrimitiveGate:
    return PrimitiveGate(
        name="robust_ry",
        duration=duration,
        matrix=lambda params, model: rotation_y(float(params["theta"]) * float(getattr(model, "scale", 1.0))),
        parameters={"theta": theta, "duration": duration},
        parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
        hilbert_dim=2,
    )


def validation_score(metric_name: str, metric_value: float) -> float:
    lower = metric_name.lower()
    magnitude = abs(float(metric_value))
    if any(token in lower for token in ("error", "loss", "infidelity", "leakage")):
        return max(0.0, 1.0 - min(magnitude, 1.0))
    return max(0.0, min(magnitude, 1.0))


def summarize_case(
    *,
    label: str,
    api_surface: str,
    metric_name: str,
    metric_value: float,
    result,
    note: str = "",
) -> CaseSummary:
    execution = result.report.get("execution", {})
    duration_ns = float(result.sequence.total_duration() * 1.0e9)
    return CaseSummary(
        label=label,
        api_surface=api_surface,
        primary_metric=metric_name,
        primary_value=float(metric_value),
        validation_score=validation_score(metric_name, float(metric_value)),
        objective=float(result.objective),
        duration_ns=duration_ns,
        execution_engine=str(execution.get("selected_engine", "legacy")),
        note=note,
    )


def run_unitary_time_policy_case() -> CaseSummary:
    primitive = PrimitiveGate(
        name="idle",
        duration=80.0e-9,
        matrix=np.eye(2, dtype=np.complex128),
        parameters={"duration": 80.0e-9},
        parameter_bounds={"duration": (20.0e-9, 100.0e-9)},
        hilbert_dim=2,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=TargetUnitary(np.eye(2, dtype=np.complex128), ignore_global_phase=True),
        objectives=MultiObjective(fidelity_weight=1.0, duration_weight=1.0),
        synthesis_constraints=SynthesisConstraints(max_duration=40.0e-9, duration_mode="hard"),
        optimizer="powell",
        execution=ExecutionOptions(engine="numpy"),
        seed=1,
    ).fit(maxiter=6)
    return summarize_case(
        label="Unitary target + duration constraint",
        api_surface="TargetUnitary + SynthesisConstraints(max_duration)",
        metric_name="fidelity",
        metric_value=result.report["metrics"]["fidelity"],
        result=result,
        note=f"duration={result.sequence.total_duration() * 1.0e9:.1f} ns",
    )


def run_state_mapping_case() -> CaseSummary:
    psi_g = np.array([1.0, 0.0], dtype=np.complex128)
    psi_e = np.array([0.0, 1.0], dtype=np.complex128)
    primitive = rotation_primitive(name="ry_state", theta=0.2, duration=20.0e-9)
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=TargetStateMapping(
            initial_states=[psi_g, psi_e],
            target_states=[psi_e, psi_g],
            weights=[0.6, 0.4],
        ),
        objectives=MultiObjective(task_weight=1.0, duration_weight=0.05, gate_count_weight=0.05),
        execution=ExecutionOptions(engine="numpy"),
        optimizer="powell",
        optimize_times=False,
        seed=5,
    ).fit(maxiter=50)
    return summarize_case(
        label="State-ensemble mapping",
        api_surface="TargetStateMapping",
        metric_name="state_fidelity_mean",
        metric_value=result.report["metrics"]["state_fidelity_mean"],
        result=result,
    )


def run_reduced_state_case() -> CaseSummary:
    primitive = PrimitiveGate(
        name="ix",
        duration=20.0e-9,
        matrix=np.kron(np.eye(2, dtype=np.complex128), pauli_x()),
        hilbert_dim=4,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=TargetReducedStateMapping(
            initial_states=[
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
            ],
            target_states=[
                np.array([1.0, 0.0], dtype=np.complex128),
                np.array([0.0, 1.0], dtype=np.complex128),
            ],
            retained_subsystems=(0,),
            subsystem_dims=(2, 2),
        ),
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=11,
    ).fit(maxiter=1)
    return summarize_case(
        label="Reduced-state target",
        api_surface="TargetReducedStateMapping",
        metric_name="reduced_state_fidelity_mean",
        metric_value=result.report["metrics"]["reduced_state_fidelity_mean"],
        result=result,
        note="retained_subsystems=(0,)",
    )


def run_channel_case() -> CaseSummary:
    primitive = rotation_primitive(name="ry_channel", theta=0.1, duration=20.0e-9)
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=TargetChannel(unitary=rotation_y(np.pi / 2.0), enforce_cptp=True),
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=7,
    ).fit(maxiter=40)
    return summarize_case(
        label="Channel / process target",
        api_surface="TargetChannel",
        metric_name="channel_overlap",
        metric_value=result.report["metrics"]["channel_overlap"],
        result=result,
        note=f"choi_error={result.report['metrics']['channel_choi_error']:.3e}",
    )


def run_isometry_case() -> CaseSummary:
    encoder = cnot() @ np.kron(hadamard(), np.eye(2, dtype=np.complex128))
    primitive = PrimitiveGate(
        name="encoder",
        duration=30.0e-9,
        matrix=encoder,
        hilbert_dim=4,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[primitive],
        target=TargetIsometry(encoder[:, :2]),
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=13,
    ).fit(maxiter=1)
    return summarize_case(
        label="Isometry / encoding target",
        api_surface="TargetIsometry",
        metric_name="isometry_coherent_fidelity",
        metric_value=result.report["metrics"]["isometry_coherent_fidelity"],
        result=result,
        note=f"basis_fidelity={result.report['metrics']['isometry_basis_fidelity']:.6f}",
    )


def run_observable_case() -> CaseSummary:
    primitive = rotation_primitive(name="ry_observable", theta=0.1, duration=20.0e-9)
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=[primitive],
        target=ObservableTarget(
            initial_state=np.array([1.0, 0.0], dtype=np.complex128),
            observable=np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128),
            target_expectation=-1.0,
        ),
        objectives=MultiObjective(task_weight=1.0, duration_weight=0.05),
        execution=ExecutionOptions(engine="numpy"),
        optimizer="powell",
        optimize_times=False,
        seed=8,
    ).fit(maxiter=40)
    return summarize_case(
        label="Observable target",
        api_surface="ObservableTarget",
        metric_name="weighted_observable_error",
        metric_value=result.report["metrics"]["weighted_observable_error"],
        result=result,
    )


def run_trajectory_case() -> CaseSummary:
    primitives = [
        rotation_primitive(name="ry_first", theta=0.2, duration=20.0e-9),
        rotation_primitive(name="ry_second", theta=-0.2, duration=20.0e-9),
    ]
    psi_g = np.array([1.0, 0.0], dtype=np.complex128)
    psi_e = np.array([0.0, 1.0], dtype=np.complex128)
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=primitives,
        target=TrajectoryTarget(
            initial_states=[psi_g],
            checkpoints=[
                TrajectoryCheckpoint(step=1, target_states=(psi_e,), weight=1.0, label="after_first"),
                TrajectoryCheckpoint(step=2, target_states=(psi_g,), weight=1.0, label="after_second"),
            ],
        ),
        objectives=MultiObjective(task_weight=1.0),
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=12,
    ).fit(maxiter=60)
    return summarize_case(
        label="Trajectory checkpoints",
        api_surface="TrajectoryTarget",
        metric_name="trajectory_task_loss",
        metric_value=result.report["metrics"]["trajectory_task_loss"],
        result=result,
    )


def run_robust_case() -> CaseSummary:
    scales = [0.7, 0.8, 0.9, 1.0]
    target = TargetUnitary(rotation_y(np.pi / 2.0), ignore_global_phase=True)

    nominal = QuantumMapSynthesizer(
        model=_RobustAngleModel(),
        subspace=Subspace.custom(2, range(2)),
        primitives=[robust_rotation_primitive(theta=0.2, duration=20.0e-9)],
        target=target,
        optimizer="powell",
        optimize_times=False,
        execution=ExecutionOptions(engine="numpy"),
        seed=9,
    ).fit(maxiter=40)

    robust = QuantumMapSynthesizer(
        model=_RobustAngleModel(),
        subspace=Subspace.custom(2, range(2)),
        primitives=[robust_rotation_primitive(theta=0.2, duration=20.0e-9)],
        target=target,
        optimizer="powell",
        optimize_times=False,
        objectives=MultiObjective(fidelity_weight=1.0, robustness_weight=4.0),
        parameter_distribution=ParameterDistribution(
            sample_count=len(scales) - 1,
            include_nominal=False,
            aggregate="worst",
            scale=_FixedSamples(scales),
        ),
        execution=ExecutionOptions(engine="numpy"),
        seed=9,
    ).fit(maxiter=40)

    def worst_case_fidelity(theta: float) -> float:
        return min(
            subspace_unitary_fidelity(rotation_y(theta * scale), target.matrix, gauge="global")
            for scale in scales
        )

    nominal_theta = float(nominal.sequence.gates[0].parameters["theta"])
    robust_theta = float(robust.sequence.gates[0].parameters["theta"])
    nominal_worst = worst_case_fidelity(nominal_theta)
    robust_worst = worst_case_fidelity(robust_theta)
    return summarize_case(
        label="Robust parameter uncertainty",
        api_surface="ParameterDistribution + robustness_weight",
        metric_name="worst_case_fidelity",
        metric_value=robust_worst,
        result=robust,
        note=f"nominal_worst={nominal_worst:.6f}",
    )


def run_final_leakage_case() -> CaseSummary:
    phi = 0.5
    c = np.cos(phi)
    s = np.sin(phi)
    primitive = PrimitiveGate(
        name="leak",
        duration=20.0e-9,
        matrix=np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.complex128,
        ),
        parameters={"duration": 20.0e-9},
        parameter_bounds={"duration": (19.0e-9, 21.0e-9)},
        hilbert_dim=3,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(3, [0, 1]),
        primitives=[primitive],
        target=TargetStateMapping(
            initial_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=1.0),
        objectives=MultiObjective(fidelity_weight=1.0, leakage_weight=1.0),
        execution=ExecutionOptions(engine="numpy"),
        optimize_times=False,
        seed=2,
    ).fit(maxiter=1)
    return summarize_case(
        label="Final leakage penalty",
        api_surface="LeakagePenalty(weight=...)",
        metric_name="leakage_worst",
        metric_value=result.report["metrics"]["leakage_worst"],
        result=result,
        note=f"leakage_term={result.report['objective']['leakage_term']:.6f}",
    )


def run_checkpoint_leakage_case() -> CaseSummary:
    leak = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )
    primitives = [
        PrimitiveGate(name="leak", duration=10.0e-9, matrix=leak, hilbert_dim=3),
        PrimitiveGate(name="unleak", duration=10.0e-9, matrix=leak.conj().T, hilbert_dim=3),
    ]
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(3, [0, 1]),
        primitives=primitives,
        target=TargetStateMapping(
            initial_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=0.0, checkpoint_weight=1.0, checkpoints=(1,)),
        execution=ExecutionOptions(engine="numpy"),
        optimize_times=False,
        seed=7,
    ).fit(maxiter=1)
    return summarize_case(
        label="Checkpoint leakage penalty",
        api_surface="LeakagePenalty(checkpoint_weight=..., checkpoints=...)",
        metric_name="checkpoint_leakage_worst",
        metric_value=result.report["metrics"]["checkpoint_leakage_worst"],
        result=result,
        note=f"term={result.report['objective']['checkpoint_leakage_term']:.6f}",
    )


def run_edge_projector_case() -> CaseSummary:
    swap_02 = np.array(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    result = QuantumMapSynthesizer(
        subspace=Subspace.custom(4, range(4)),
        primitives=[PrimitiveGate(name="swap_02", duration=10.0e-9, matrix=swap_02, hilbert_dim=4)],
        target=TargetStateMapping(
            initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=0.0, checkpoint_weight=0.0, edge_weight=0.5, edge_projector=[2]),
        execution=ExecutionOptions(engine="numpy"),
        optimize_times=False,
        seed=9,
    ).fit(maxiter=1)
    return summarize_case(
        label="Edge-projector penalty",
        api_surface="LeakagePenalty(edge_weight=..., edge_projector=...)",
        metric_name="edge_population_worst",
        metric_value=result.report["metrics"]["edge_population_worst"],
        result=result,
        note=f"term={result.report['objective']['edge_population_term']:.6f}",
    )


def run_warm_start_case() -> WorkflowSummary:
    def idle_primitive() -> PrimitiveGate:
        return PrimitiveGate(
            name="idle",
            duration=80.0e-9,
            matrix=np.eye(2, dtype=np.complex128),
            parameters={"duration": 80.0e-9},
            parameter_bounds={"duration": (20.0e-9, 100.0e-9)},
            hilbert_dim=2,
        )

    first = QuantumMapSynthesizer(
        primitives=[idle_primitive()],
        target=TargetUnitary(np.eye(2, dtype=np.complex128), ignore_global_phase=True),
        objectives=MultiObjective(fidelity_weight=1.0, duration_weight=1.0),
        synthesis_constraints=SynthesisConstraints(max_duration=40.0e-9, duration_mode="hard"),
        execution=ExecutionOptions(engine="numpy"),
        seed=3,
    ).fit(maxiter=6)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "warm_start.json"
        first.save(save_path)
        resumed = QuantumMapSynthesizer(
            primitives=[idle_primitive()],
            target=TargetUnitary(np.eye(2, dtype=np.complex128), ignore_global_phase=True),
            warm_start=save_path,
            execution=ExecutionOptions(engine="numpy"),
            seed=3,
        ).fit(maxiter=1)

    return WorkflowSummary(
        label="Warm start",
        api_surface="warm_start=path | payload | SynthesisResult",
        result=(
            f"saved_duration={first.sequence.total_duration() * 1.0e9:.1f} ns, "
            f"resumed_duration={resumed.sequence.total_duration() * 1.0e9:.1f} ns"
        ),
    )


def run_pareto_case() -> WorkflowSummary:
    primitive = PrimitiveGate(
        name="idle",
        duration=80.0e-9,
        matrix=np.eye(2, dtype=np.complex128),
        parameters={"duration": 80.0e-9},
        parameter_bounds={"duration": (20.0e-9, 100.0e-9)},
        hilbert_dim=2,
    )
    front = QuantumMapSynthesizer(
        primitives=[primitive],
        target=TargetUnitary(np.eye(2, dtype=np.complex128), ignore_global_phase=True),
        execution=ExecutionOptions(engine="numpy"),
        seed=9,
    ).explore_pareto(
        [
            MultiObjective(fidelity_weight=1.0, duration_weight=0.0),
            MultiObjective(fidelity_weight=1.0, duration_weight=1.0),
        ],
        maxiter=6,
    )
    durations = [result.sequence.total_duration() * 1.0e9 for result in front.results]
    return WorkflowSummary(
        label="Pareto exploration",
        api_surface="explore_pareto(weight_sets, ...)",
        result=f"durations_ns={[round(value, 1) for value in durations]}",
        note=f"nondominated={len(front.nondominated_indices)}",
    )


def run_gate_order_case() -> WorkflowSummary:
    pool = [
        make_gate_from_matrix("H", hadamard(), duration=10.0e-9),
        make_gate_from_matrix("X", pauli_x(), duration=10.0e-9),
    ]
    order_result = GateOrderOptimizer(
        gate_pool=pool,
        order_config=GateOrderConfig(
            search_strategy="exhaustive",
            allow_repetitions=False,
            min_sequence_length=2,
            max_sequence_length=2,
            seed=17,
            early_stop_infidelity=1.0e-12,
        ),
        synthesizer_kwargs=dict(
            subspace=Subspace.custom(2, range(2)),
            optimize_times=False,
            execution=ExecutionOptions(engine="numpy"),
            optimizer="powell",
            seed=17,
        ),
    ).search(target=TargetUnitary(pauli_x() @ hadamard(), ignore_global_phase=True))
    best = [gate.name for gate in order_result.best_ordering]
    return WorkflowSummary(
        label="Gate-order search",
        api_surface="GateOrderOptimizer + GateOrderConfig",
        result=f"best_ordering={best}, objective={order_result.best_result.objective:.3e}",
        note=f"tried={order_result.n_orderings_tried}",
    )


def save_summary_plot(
    core_cases: list[CaseSummary],
    regularization_cases: list[CaseSummary],
    workflow_cases: list[WorkflowSummary],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.0, 8.0))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.4, 2.2])
    ax = fig.add_subplot(grid[0, 0])
    ax_info = fig.add_subplot(grid[1, 0])

    labels = [case.label for case in core_cases]
    scores = [case.validation_score for case in core_cases]
    colors = plt.cm.tab20(np.linspace(0.05, 0.85, len(core_cases)))

    ax.barh(labels, scores, color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.02)
    ax.set_xlabel("Validation score")
    ax.set_title("Validated map-synthesis optimization use cases")
    ax.grid(axis="x", alpha=0.25)

    for index, case in enumerate(core_cases):
        text_x = min(case.validation_score + 0.01, 0.985)
        ax.text(
            text_x,
            index,
            f"{case.primary_metric}={case.primary_value:.6f}",
            va="center",
            ha="left",
            fontsize=8,
        )

    ax_info.axis("off")
    regularization_lines = [
        f"- {case.label}: {case.primary_metric}={case.primary_value:.6f}; {case.note}" for case in regularization_cases
    ]
    workflow_lines = [f"- {case.label}: {case.result}; {case.note}" for case in workflow_cases]
    info_text = (
        "Regularization and constraint checks\n"
        + "\n".join(regularization_lines)
        + "\n\nWorkflow helper checks\n"
        + "\n".join(workflow_lines)
    )
    ax_info.text(0.0, 1.0, info_text, va="top", ha="left", fontsize=9, family="monospace")

    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def main() -> None:
    warnings.filterwarnings("ignore", category=LinAlgWarning)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PLOT_PATH,
        help="Path for the optimization-catalog summary figure.",
    )
    args = parser.parse_args()

    core_cases = [
        run_unitary_time_policy_case(),
        run_state_mapping_case(),
        run_reduced_state_case(),
        run_channel_case(),
        run_isometry_case(),
        run_observable_case(),
        run_trajectory_case(),
        run_robust_case(),
    ]
    regularization_cases = [
        run_final_leakage_case(),
        run_checkpoint_leakage_case(),
        run_edge_projector_case(),
    ]
    workflow_cases = [
        run_warm_start_case(),
        run_pareto_case(),
        run_gate_order_case(),
    ]
    save_summary_plot(core_cases, regularization_cases, workflow_cases, args.output)

    payload = {
        "core_cases": [asdict(case) for case in core_cases],
        "regularization_cases": [asdict(case) for case in regularization_cases],
        "workflow_cases": [asdict(case) for case in workflow_cases],
        "plot_path": str(args.output),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()