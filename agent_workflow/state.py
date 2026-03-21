from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RunState:
    run_id: str
    task_name: str
    task_path: str
    run_dir: str
    current_phase: str = "initialize"
    status: str = "initialized"
    iteration_count: int = 0
    max_iterations: int = 1
    human_plan_supplied: bool = False
    review_required: bool = True
    acceptance_criteria_satisfied: bool = False
    blocking_reason: str | None = None
    last_error: str | None = None
    active_working_directory: str | None = None
    last_workspace_snapshot: str | None = None
    last_tested_snapshot: str | None = None
    code_changed_since_last_test: bool = False
    completion_flags: dict[str, bool] = field(default_factory=dict)
    repair_instructions: list[str] = field(default_factory=list)
    phase_history: list[dict[str, Any]] = field(default_factory=list)
    iterations: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    completed_at: str | None = None

    def __post_init__(self) -> None:
        defaults = {
            "initialized": False,
            "implementation_complete": False,
            "tests_complete": False,
            "review_complete": False,
            "documentation_complete": False,
            "summary_complete": False,
            "run_complete": False,
        }
        defaults.update(self.completion_flags)
        self.completion_flags = defaults

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        task_name: str,
        task_path: str,
        run_dir: str,
        max_iterations: int,
        human_plan_supplied: bool,
        review_required: bool,
        active_working_directory: str,
    ) -> "RunState":
        return cls(
            run_id=run_id,
            task_name=task_name,
            task_path=task_path,
            run_dir=run_dir,
            max_iterations=max_iterations,
            human_plan_supplied=human_plan_supplied,
            review_required=review_required,
            active_working_directory=active_working_directory,
        )

    @classmethod
    def from_path(cls, path: Path) -> "RunState":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_name": self.task_name,
            "task_path": self.task_path,
            "run_dir": self.run_dir,
            "current_phase": self.current_phase,
            "status": self.status,
            "iteration_count": self.iteration_count,
            "max_iterations": self.max_iterations,
            "human_plan_supplied": self.human_plan_supplied,
            "review_required": self.review_required,
            "acceptance_criteria_satisfied": self.acceptance_criteria_satisfied,
            "blocking_reason": self.blocking_reason,
            "last_error": self.last_error,
            "active_working_directory": self.active_working_directory,
            "last_workspace_snapshot": self.last_workspace_snapshot,
            "last_tested_snapshot": self.last_tested_snapshot,
            "code_changed_since_last_test": self.code_changed_since_last_test,
            "completion_flags": self.completion_flags,
            "repair_instructions": self.repair_instructions,
            "phase_history": self.phase_history,
            "iterations": self.iterations,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
        }

    def write(self, path: Path) -> None:
        self.updated_at = utc_now()
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    def ensure_iteration(self, iteration: int) -> dict[str, Any]:
        while len(self.iterations) < iteration:
            self.iterations.append(
                {
                    "iteration": len(self.iterations) + 1,
                    "execution_status": "pending",
                    "execution_summary": "",
                    "execution_output": "",
                    "test_status": "pending",
                    "test_commands": [],
                    "test_log": "",
                    "review_status": "pending",
                    "review_summary": "",
                    "changed_files": [],
                    "workspace_snapshot": None,
                    "repair_instructions": [],
                }
            )
        return self.iterations[iteration - 1]

    def record_phase(self, phase: str, status: str, *, iteration: int | None = None, note: str | None = None, artifact: str | None = None) -> None:
        self.phase_history.append(
            {
                "phase": phase,
                "status": status,
                "iteration": iteration,
                "note": note or "",
                "artifact": artifact or "",
                "timestamp": utc_now(),
            }
        )
        self.current_phase = phase
        self.status = status

    def set_snapshot(self, snapshot: str | None, *, tested: bool = False) -> None:
        self.last_workspace_snapshot = snapshot
        self.code_changed_since_last_test = snapshot != self.last_tested_snapshot
        if tested:
            self.last_tested_snapshot = snapshot
            self.code_changed_since_last_test = False

    def mark_complete(self) -> None:
        self.status = "complete"
        self.current_phase = "complete"
        self.blocking_reason = None
        self.last_error = None
        self.completion_flags["run_complete"] = True
        self.completed_at = utc_now()

    def mark_blocked(self, reason: str) -> None:
        self.status = "blocked"
        self.blocking_reason = reason
        self.last_error = reason

    def mark_incomplete(self, reason: str | None = None) -> None:
        self.status = "incomplete"
        if reason:
            self.last_error = reason
