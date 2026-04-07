"""Physics experiment harness: three-agent orchestration over cqed_sim optimization workflows.

The three agents are:
  - Planner: chooses optimization strategy and candidate gate times (LLM optional).
  - Executor: runs the chosen workflow via cqed_sim (pure Python, never uses an LLM).
  - Evaluator: checks fidelity/leakage against acceptance criteria (LLM optional).

When both planner_backend and evaluator_backend are None the harness runs deterministically
without any LLM calls.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    from cqed_sim.optimal_control.grape import GrapeConfig
    from cqed_sim.optimal_control.problems import ControlProblem
    from cqed_sim.optimal_control.structured import StructuredControlConfig
    from cqed_sim.optimal_control.workflows import (
        GateTimeOptimizationConfig,
        GateTimeOptimizationResult,
        StructuredToGrapeResult,
    )

    from .backends import AgentBackend


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class ExperimentTaskSpec:
    """Declarative specification for a single gate-optimization experiment campaign."""

    title: str
    gate_target: str
    problem: "ControlProblem"
    gate_time_range_ns: tuple[float, float]

    gate_time_steps: int = 5
    fidelity_threshold: float = 0.999
    leakage_budget: float | None = None
    optimization_strategy: str = "auto"  # "grape" | "structured_then_grape" | "structured" | "auto"
    grape_config: "GrapeConfig | None" = None
    structured_config: "StructuredControlConfig | None" = None
    gate_time_config: "GateTimeOptimizationConfig | None" = None
    max_rounds: int = 3
    backend_profile: str | None = None
    run_directory: Path | None = None


@dataclass
class ExperimentRunState:
    """Mutable run state for an experiment campaign, persisted as JSON."""

    run_id: str
    task_title: str
    gate_target: str
    current_round: int = 0
    status: str = "initialized"  # initialized | running | complete | failed | escalated
    best_fidelity: float | None = None
    best_duration_ns: float | None = None
    planner_strategy: str | None = None
    optimization_history: list[dict[str, Any]] = field(default_factory=list)
    blocking_reason: str | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_title": self.task_title,
            "gate_target": self.gate_target,
            "current_round": self.current_round,
            "status": self.status,
            "best_fidelity": self.best_fidelity,
            "best_duration_ns": self.best_duration_ns,
            "planner_strategy": self.planner_strategy,
            "optimization_history": self.optimization_history,
            "blocking_reason": self.blocking_reason,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def write(self, path: Path) -> None:
        self.updated_at = _utc_now()
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")


class ExperimentOrchestrator:
    """Three-agent loop over a cqed_sim optimization pipeline.

    Usage (no LLM)::

        from agent_workflow import ExperimentOrchestrator, ExperimentTaskSpec
        task = ExperimentTaskSpec(
            title="CNOT gate time sweep",
            gate_target="CNOT",
            problem=my_control_problem,
            gate_time_range_ns=(40.0, 120.0),
            fidelity_threshold=0.999,
        )
        state = ExperimentOrchestrator(task=task).run()
        print(state.status, state.best_fidelity, state.best_duration_ns)

    Usage (with LLM planner and evaluator)::

        from agent_workflow.backends import AnthropicBackend
        planner = AnthropicBackend(name="planner", model="claude-opus-4-6")
        evaluator = AnthropicBackend(name="evaluator", model="claude-opus-4-6")
        state = ExperimentOrchestrator(task=task, planner_backend=planner, evaluator_backend=evaluator).run()
    """

    def __init__(
        self,
        *,
        task: ExperimentTaskSpec,
        run_directory: Path | None = None,
        planner_backend: "AgentBackend | None" = None,
        evaluator_backend: "AgentBackend | None" = None,
        verbose: bool = False,
    ) -> None:
        self.task = task
        self.planner_backend = planner_backend
        self.evaluator_backend = evaluator_backend
        self.verbose = verbose
        self._run_dir: Path | None = run_directory or task.run_directory
        self._state: ExperimentRunState | None = None
        self._repo_root: Path | None = None

    def run(self) -> ExperimentRunState:
        """Execute the three-agent loop and return the final state."""
        import uuid

        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]
        self._state = ExperimentRunState(
            run_id=run_id,
            task_title=self.task.title,
            gate_target=self.task.gate_target,
        )
        if self._run_dir is not None:
            self._run_dir = Path(self._run_dir)
            self._run_dir.mkdir(parents=True, exist_ok=True)
            self._state.write(self._run_dir / "EXPERIMENT_STATE.json")

        self._state.status = "running"
        self._log(f"Starting experiment '{self.task.title}' for gate '{self.task.gate_target}'")

        last_verdict: dict[str, Any] = {}
        for round_idx in range(self.task.max_rounds):
            self._state.current_round = round_idx + 1
            self._log(f"Round {self._state.current_round}/{self.task.max_rounds}")

            # ── Planner ──────────────────────────────────────────────────────
            plan = self._plan(last_verdict)
            self._state.planner_strategy = plan.get("strategy")
            self._log(f"  strategy={plan['strategy']}, candidates={len(plan['durations_ns'])}")

            # ── Executor (pure Python) ────────────────────────────────────────
            result = self._execute(plan)
            round_fidelity, round_duration_s = _extract_best(result)
            round_duration_ns = round_duration_s * 1e9 if round_duration_s is not None else None

            # ── Evaluator ────────────────────────────────────────────────────
            verdict = self._evaluate(result, round_fidelity)
            self._log(f"  fidelity={round_fidelity:.6f}  verdict={verdict['verdict']}")

            # Record round history
            round_record: dict[str, Any] = {
                "round": self._state.current_round,
                "strategy": plan.get("strategy"),
                "durations_ns": plan.get("durations_ns"),
                "best_fidelity": round_fidelity,
                "best_duration_ns": round_duration_ns,
                "verdict": verdict["verdict"],
                "unmet_criteria": verdict.get("unmet_criteria", []),
                "feedback": verdict.get("feedback", []),
            }
            self._state.optimization_history.append(round_record)

            # Update best across rounds
            if round_fidelity is not None:
                if self._state.best_fidelity is None or round_fidelity > self._state.best_fidelity:
                    self._state.best_fidelity = round_fidelity
                    self._state.best_duration_ns = round_duration_ns

            self._save_state()

            if verdict["verdict"] == "accept":
                self._state.status = "complete"
                break
            if verdict["verdict"] == "escalate":
                self._state.status = "escalated"
                self._state.blocking_reason = "; ".join(verdict.get("feedback", ["escalated by evaluator"]))
                break
            # "retry" → update last verdict so planner can adapt
            last_verdict = verdict
        else:
            self._state.status = "failed"

        self._save_state()
        self._log(f"Experiment finished: status={self._state.status}, best_fidelity={self._state.best_fidelity}")
        return self._state

    # ─── Planner ─────────────────────────────────────────────────────────────

    def _plan(self, last_verdict: dict[str, Any]) -> dict[str, Any]:
        if self.planner_backend is not None:
            return self._llm_plan(last_verdict)
        return self._deterministic_plan(last_verdict)

    def _deterministic_plan(self, last_verdict: dict[str, Any]) -> dict[str, Any]:
        strategy = self.task.optimization_strategy
        if strategy == "auto":
            strategy = self._auto_strategy()

        # On retry, try switching strategy
        if last_verdict.get("verdict") == "retry" and self._state is not None and self._state.current_round > 1:
            if strategy == "grape":
                strategy = "structured_then_grape"
            elif strategy == "structured_then_grape":
                strategy = "grape"

        min_ns, max_ns = self.task.gate_time_range_ns
        durations_ns = list(np.linspace(min_ns, max_ns, self.task.gate_time_steps))
        return {"strategy": strategy, "durations_ns": durations_ns}

    def _auto_strategy(self) -> str:
        try:
            from cqed_sim.optimal_control.structured import StructuredPulseParameterization  # type: ignore[import]
            if isinstance(self.task.problem.parameterization, StructuredPulseParameterization):
                return "structured_then_grape"
        except ImportError:
            pass
        return "grape"

    def _llm_plan(self, last_verdict: dict[str, Any]) -> dict[str, Any]:
        from .backends import AgentRequest, extract_json_payload
        from .prompts import load_prompt_template, render_prompt

        assert self.planner_backend is not None and self._state is not None
        repo_root = self._repo_root or Path.cwd()
        template = load_prompt_template(repo_root, "experiment_planner")
        context: dict[str, Any] = {
            "GATE_TARGET": self.task.gate_target,
            "GATE_TIME_RANGE_NS": f"{self.task.gate_time_range_ns[0]} – {self.task.gate_time_range_ns[1]}",
            "GATE_TIME_STEPS": str(self.task.gate_time_steps),
            "FIDELITY_THRESHOLD": str(self.task.fidelity_threshold),
            "LEAKAGE_BUDGET": str(self.task.leakage_budget) if self.task.leakage_budget is not None else "not set",
            "CURRENT_ROUND": str(self._state.current_round),
            "MAX_ROUNDS": str(self.task.max_rounds),
            "OPTIMIZATION_HISTORY": json.dumps(self._state.optimization_history, indent=2),
            "LAST_FEEDBACK": json.dumps(last_verdict.get("feedback", []), indent=2),
        }
        prompt_text = render_prompt(template, context)
        dummy_path = Path("EXPERIMENT_PLANNER_PROMPT.txt")
        request = AgentRequest(
            role="planner",
            phase="plan",
            iteration=self._state.current_round,
            prompt=prompt_text,
            prompt_path=dummy_path,
            context=context,
            context_path=dummy_path,
            working_directory=self._run_dir or Path.cwd(),
            run_directory=self._run_dir or Path.cwd(),
        )
        response = self.planner_backend.run(request)
        structured = response.structured or extract_json_payload(response.content)
        if structured and "strategy" in structured and "durations_ns" in structured:
            return structured
        self._log("  LLM planner returned invalid response; falling back to deterministic plan.")
        return self._deterministic_plan(last_verdict)

    # ─── Executor (always pure Python) ───────────────────────────────────────

    def _execute(self, plan: dict[str, Any]) -> "GateTimeOptimizationResult | StructuredToGrapeResult":
        from cqed_sim.optimal_control.workflows import (  # type: ignore[import]
            optimize_gate_time_with_grape,
            optimize_gate_time_with_structured_control,
            solve_structured_then_grape,
        )

        strategy = str(plan.get("strategy", "grape"))
        durations_ns: list[float] = list(plan.get("durations_ns", []))
        durations_s = [d * 1e-9 for d in durations_ns]

        if strategy == "grape":
            return optimize_gate_time_with_grape(
                self.task.problem,
                durations_s=durations_s,
                config=self.task.grape_config,
                gate_time_config=self.task.gate_time_config,
            )
        if strategy == "structured":
            return optimize_gate_time_with_structured_control(
                self.task.problem,
                durations_s=durations_s,
                config=self.task.structured_config,
                gate_time_config=self.task.gate_time_config,
            )
        if strategy == "structured_then_grape":
            # Use the best duration from the sweep as the single duration
            best_duration_s = _pick_best_duration_s(durations_s)
            return solve_structured_then_grape(
                self.task.problem,
                structured_config=self.task.structured_config,
                grape_config=self.task.grape_config,
            )
        raise ValueError(f"Unknown optimization strategy '{strategy}'. Use 'grape', 'structured', or 'structured_then_grape'.")

    # ─── Evaluator ────────────────────────────────────────────────────────────

    def _evaluate(self, result: Any, best_fidelity: float | None) -> dict[str, Any]:
        if self.evaluator_backend is not None:
            return self._llm_evaluate(result, best_fidelity)
        return self._deterministic_evaluate(best_fidelity)

    def _deterministic_evaluate(self, best_fidelity: float | None) -> dict[str, Any]:
        if best_fidelity is None:
            return {
                "verdict": "retry",
                "feedback": ["Optimization returned no valid result."],
                "unmet_criteria": ["no result"],
            }
        unmet: list[str] = []
        if best_fidelity < self.task.fidelity_threshold:
            unmet.append(f"fidelity {best_fidelity:.6f} < threshold {self.task.fidelity_threshold}")
        if unmet:
            return {"verdict": "retry", "feedback": unmet, "unmet_criteria": unmet}
        return {"verdict": "accept", "feedback": [], "unmet_criteria": []}

    def _llm_evaluate(self, result: Any, best_fidelity: float | None) -> dict[str, Any]:
        from .backends import AgentRequest, extract_json_payload
        from .prompts import load_prompt_template, render_prompt

        assert self.evaluator_backend is not None and self._state is not None
        repo_root = self._repo_root or Path.cwd()
        template = load_prompt_template(repo_root, "experiment_evaluator")
        metrics = getattr(result, "metrics", {})
        context: dict[str, Any] = {
            "GATE_TARGET": self.task.gate_target,
            "FIDELITY_THRESHOLD": str(self.task.fidelity_threshold),
            "LEAKAGE_BUDGET": str(self.task.leakage_budget) if self.task.leakage_budget is not None else "not set",
            "BEST_FIDELITY": str(best_fidelity) if best_fidelity is not None else "N/A",
            "RESULT_METRICS": json.dumps(metrics, indent=2),
            "CURRENT_ROUND": str(self._state.current_round),
            "MAX_ROUNDS": str(self.task.max_rounds),
            "OPTIMIZATION_HISTORY": json.dumps(self._state.optimization_history, indent=2),
        }
        prompt_text = render_prompt(template, context)
        dummy_path = Path("EXPERIMENT_EVALUATOR_PROMPT.txt")
        request = AgentRequest(
            role="evaluator",
            phase="evaluate",
            iteration=self._state.current_round,
            prompt=prompt_text,
            prompt_path=dummy_path,
            context=context,
            context_path=dummy_path,
            working_directory=self._run_dir or Path.cwd(),
            run_directory=self._run_dir or Path.cwd(),
        )
        response = self.evaluator_backend.run(request)
        structured = response.structured or extract_json_payload(response.content)
        if structured and "verdict" in structured:
            return structured
        self._log("  LLM evaluator returned invalid response; falling back to deterministic evaluation.")
        return self._deterministic_evaluate(best_fidelity)

    # ─── Utilities ────────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        if self._run_dir is not None and self._state is not None:
            self._state.write(self._run_dir / "EXPERIMENT_STATE.json")

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)


# ─── Module-level helpers ─────────────────────────────────────────────────────


def _extract_best(result: Any) -> tuple[float | None, float | None]:
    """Extract (best_fidelity, best_duration_s) from any workflow result type."""
    # GateTimeOptimizationResult
    if hasattr(result, "best_result") and hasattr(result, "best_duration_s"):
        metrics = getattr(result.best_result, "metrics", {}) or {}
        fidelity = _fidelity_from_metrics(metrics)
        duration_s = float(result.best_duration_s)
        return fidelity, duration_s

    # StructuredToGrapeResult
    if hasattr(result, "grape_result"):
        grape_metrics = getattr(result.grape_result, "metrics", {}) or {}
        fidelity = _fidelity_from_metrics(grape_metrics) or _fidelity_from_metrics(result.metrics or {})
        return fidelity, None

    return None, None


def _fidelity_from_metrics(metrics: dict[str, Any]) -> float | None:
    for key in ("nominal_physical_fidelity", "nominal_command_fidelity", "nominal_fidelity", "best_nominal_fidelity", "grape_nominal_fidelity"):
        value = metrics.get(key)
        if value is not None:
            return float(value)
    return None


def _pick_best_duration_s(durations_s: list[float]) -> float:
    """Pick the middle duration as a starting point for structured_then_grape."""
    if not durations_s:
        raise ValueError("durations_s must be non-empty.")
    mid = len(durations_s) // 2
    return float(durations_s[mid])
