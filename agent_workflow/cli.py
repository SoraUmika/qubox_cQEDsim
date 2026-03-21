from __future__ import annotations

import argparse
from pathlib import Path

from .orchestrator import OrchestratorOptions, WorkflowOrchestrator, resolve_latest_resume_task
from .task_spec import load_task_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the repo-side semi-autonomous two-agent workflow.")
    parser.add_argument("--task", type=Path, default=None, help="Path to a YAML or JSON task spec.")
    parser.add_argument("--resume", action="store_true", help="Resume the latest incomplete run for the selected task.")
    parser.add_argument("--resume-last", action="store_true", help="Resume the latest run seen in agent_runs/, regardless of task.")
    parser.add_argument("--force-restart", action="store_true", help="Ignore resumable runs and start a fresh run directory.")
    parser.add_argument("--max-iterations", type=int, default=None, help="Override the task spec iteration limit.")
    parser.add_argument("--dry-run", action="store_true", help="Render prompts and initialize artifacts without invoking backends or test commands.")
    parser.add_argument("--verbose", action="store_true", help="Print extra workflow status to stdout.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parent.parent
    task_path = _resolve_task_path(repo_root, args.task, resume_last=bool(args.resume_last))
    if task_path is None:
        parser.error("A task path is required unless --resume-last can resolve one from an existing run.")
    task = load_task_spec(task_path)
    orchestrator = WorkflowOrchestrator(
        repo_root=repo_root,
        task=task,
        options=OrchestratorOptions(
            resume=bool(args.resume),
            resume_last=bool(args.resume_last),
            force_restart=bool(args.force_restart),
            max_iterations_override=args.max_iterations,
            dry_run=bool(args.dry_run),
            verbose=bool(args.verbose),
        ),
    )
    exit_code = orchestrator.run()
    if args.verbose:
        print(f"Task: {task.title}")
        print(f"Exit code: {exit_code}")
    return exit_code


def _resolve_task_path(repo_root: Path, explicit_task: Path | None, *, resume_last: bool) -> Path | None:
    if explicit_task is not None:
        task_path = explicit_task
        if not task_path.is_absolute():
            task_path = (repo_root / task_path).resolve()
        return task_path
    if resume_last:
        return resolve_latest_resume_task(repo_root)
    return None
