from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .simple_yaml import dump_simple_yaml, load_yaml_or_json


@dataclass(frozen=True)
class TaskSpec:
    title: str
    goal: str
    human_plan: str = ""
    context_files: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    deliverables: tuple[str, ...] = ()
    tests_to_run: tuple[str, ...] = ()
    docs_to_update: tuple[str, ...] = ()
    acceptance_criteria: tuple[str, ...] = ()
    max_iterations: int = 2
    strict_context: bool = False
    allow_repo_edits: bool = True
    review_required: bool = True
    auto_resume: bool = True
    stop_on_blocking_error: bool = True
    backend_profile: str | None = None
    bootstrap_workspace_from: str | None = None
    working_directory: str | None = None
    source_path: Path | None = None

    @property
    def slug(self) -> str:
        chunks: list[str] = []
        for char in self.title.lower():
            if char.isalnum():
                chunks.append(char)
            elif not chunks or chunks[-1] != "_":
                chunks.append("_")
        return "".join(chunks).strip("_") or "task"

    @property
    def human_plan_supplied(self) -> bool:
        return bool(self.human_plan.strip())

    def with_source_path(self, path: Path) -> "TaskSpec":
        return TaskSpec(
            title=self.title,
            goal=self.goal,
            human_plan=self.human_plan,
            context_files=self.context_files,
            constraints=self.constraints,
            deliverables=self.deliverables,
            tests_to_run=self.tests_to_run,
            docs_to_update=self.docs_to_update,
            acceptance_criteria=self.acceptance_criteria,
            max_iterations=self.max_iterations,
            strict_context=self.strict_context,
            allow_repo_edits=self.allow_repo_edits,
            review_required=self.review_required,
            auto_resume=self.auto_resume,
            stop_on_blocking_error=self.stop_on_blocking_error,
            backend_profile=self.backend_profile,
            bootstrap_workspace_from=self.bootstrap_workspace_from,
            working_directory=self.working_directory,
            source_path=path,
        )

    def to_mapping(self, *, repo_root: Path | None = None, run_dir: Path | None = None, max_iterations_override: int | None = None) -> dict[str, Any]:
        mapping: dict[str, Any] = {
            "title": self.title,
            "goal": self.goal,
            "human_plan": self.human_plan,
            "context_files": list(self.context_files),
            "constraints": list(self.constraints),
            "deliverables": list(self.deliverables),
            "tests_to_run": list(self.tests_to_run),
            "docs_to_update": list(self.docs_to_update),
            "acceptance_criteria": list(self.acceptance_criteria),
            "max_iterations": int(max_iterations_override or self.max_iterations),
            "strict_context": self.strict_context,
            "allow_repo_edits": self.allow_repo_edits,
            "review_required": self.review_required,
            "auto_resume": self.auto_resume,
            "stop_on_blocking_error": self.stop_on_blocking_error,
            "backend_profile": self.backend_profile or "",
            "bootstrap_workspace_from": self.bootstrap_workspace_from or "",
            "working_directory": self.working_directory or "",
            "human_plan_supplied": self.human_plan_supplied,
        }
        if self.source_path is not None:
            mapping["task_file"] = str(self.source_path)
        if repo_root is not None:
            mapping["repo_root"] = str(repo_root)
        if run_dir is not None:
            mapping["run_dir"] = str(run_dir)
            mapping["resolved_context_files"] = [str(resolve_path(value, repo_root or run_dir.parent, run_dir)) for value in self.context_files]
            mapping["resolved_docs_to_update"] = [str(resolve_path(value, repo_root or run_dir.parent, run_dir)) for value in self.docs_to_update]
            active_dir = resolve_path(self.working_directory or "{repo_root}", repo_root or run_dir.parent, run_dir)
            mapping["active_working_directory"] = str(active_dir)
        return mapping

    def dump_resolved_yaml(self, *, repo_root: Path, run_dir: Path, max_iterations_override: int | None = None) -> str:
        return dump_simple_yaml(self.to_mapping(repo_root=repo_root, run_dir=run_dir, max_iterations_override=max_iterations_override))


REQUIRED_STRING_FIELDS = ("title", "goal")
LIST_FIELDS = (
    "context_files",
    "constraints",
    "deliverables",
    "tests_to_run",
    "docs_to_update",
    "acceptance_criteria",
)


def load_task_spec(path: str | Path) -> TaskSpec:
    resolved = Path(path).resolve()
    raw = load_yaml_or_json(resolved)
    for field_name in REQUIRED_STRING_FIELDS:
        value = raw.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Task spec field '{field_name}' must be a non-empty string.")
    list_values = {field_name: _normalize_string_list(raw.get(field_name, []), field_name) for field_name in LIST_FIELDS}
    spec = TaskSpec(
        title=str(raw["title"]).strip(),
        goal=str(raw["goal"]).strip(),
        human_plan=str(raw.get("human_plan", "")),
        context_files=tuple(list_values["context_files"]),
        constraints=tuple(list_values["constraints"]),
        deliverables=tuple(list_values["deliverables"]),
        tests_to_run=tuple(list_values["tests_to_run"]),
        docs_to_update=tuple(list_values["docs_to_update"]),
        acceptance_criteria=tuple(list_values["acceptance_criteria"]),
        max_iterations=max(1, int(raw.get("max_iterations", 2))),
        strict_context=bool(raw.get("strict_context", False)),
        allow_repo_edits=bool(raw.get("allow_repo_edits", True)),
        review_required=bool(raw.get("review_required", True)),
        auto_resume=bool(raw.get("auto_resume", True)),
        stop_on_blocking_error=bool(raw.get("stop_on_blocking_error", True)),
        backend_profile=_normalize_optional_string(raw.get("backend_profile")),
        bootstrap_workspace_from=_normalize_optional_string(raw.get("bootstrap_workspace_from")),
        working_directory=_normalize_optional_string(raw.get("working_directory")),
    )
    return spec.with_source_path(resolved)


def resolve_path(raw_value: str, repo_root: Path, run_dir: Path) -> Path:
    substituted = (
        raw_value.replace("{repo_root}", str(repo_root))
        .replace("{run_dir}", str(run_dir))
        .replace("{task_dir}", str(run_dir))
    )
    path = Path(substituted)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _normalize_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise ValueError(f"Task spec field '{field_name}' must be a string or list of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, (str, int, float, bool)):
            raise ValueError(f"Task spec field '{field_name}' contains a non-scalar list item.")
        normalized.append(str(item))
    return normalized
