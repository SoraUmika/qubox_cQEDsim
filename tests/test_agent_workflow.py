from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from agent_workflow.artifacts import find_latest_run, slugify
from agent_workflow.cli import main as workflow_main
from agent_workflow.task_spec import load_task_spec


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "agent_runs"
VALIDATION_TEMPLATE = (REPO_ROOT / "agent_workflow" / "tasks" / "validation_demo_task.yaml").read_text(encoding="utf-8")


def test_validation_demo_task_spec_loads() -> None:
    task = load_task_spec(REPO_ROOT / "agent_workflow" / "tasks" / "validation_demo_task.yaml")
    assert task.backend_profile == "validation_demo"
    assert task.bootstrap_workspace_from == "agent_workflow/validation_fixture/seed_project"
    assert task.working_directory == "{run_dir}/workspace"
    assert task.human_plan_supplied is True
    assert task.max_iterations == 2


def test_validation_demo_marks_run_incomplete_when_iteration_cap_is_hit(tmp_path: Path) -> None:
    title = f"Workflow Validation Incomplete {uuid.uuid4().hex[:8]}"
    task_path = _write_validation_task(tmp_path, title)
    try:
        exit_code = workflow_main([
            "--task",
            str(task_path),
            "--force-restart",
            "--max-iterations",
            "1",
        ])
        assert exit_code == 2
        run_dir = _latest_run_dir_for_title(title)
        state = json.loads((run_dir / "RUN_STATE.json").read_text(encoding="utf-8"))
        assert state["status"] == "incomplete"
        assert state["current_phase"] == "execute"
        assert state["iteration_count"] == 1
        assert (run_dir / "REVIEW_OUTPUT_iter01.txt").exists()
        assert not (run_dir / "DOC_OUTPUT.md").exists()
    finally:
        _cleanup_runs_for_title(title)


def test_validation_demo_resume_completes_run(tmp_path: Path) -> None:
    title = f"Workflow Validation Resume {uuid.uuid4().hex[:8]}"
    task_path = _write_validation_task(tmp_path, title)
    try:
        first_exit = workflow_main([
            "--task",
            str(task_path),
            "--force-restart",
            "--max-iterations",
            "1",
        ])
        assert first_exit == 2
        resume_exit = workflow_main([
            "--task",
            str(task_path),
            "--resume",
            "--max-iterations",
            "3",
        ])
        assert resume_exit == 0
        run_dir = _latest_run_dir_for_title(title)
        state = json.loads((run_dir / "RUN_STATE.json").read_text(encoding="utf-8"))
        assert state["status"] == "complete"
        assert state["current_phase"] == "complete"
        workspace = run_dir / "workspace"
        assert "return a + b" in (workspace / "demo_math.py").read_text(encoding="utf-8")
        assert "arithmetic sum" in (workspace / "demo_docs.md").read_text(encoding="utf-8")
        assert (run_dir / "DOC_OUTPUT.md").exists()
        assert (run_dir / "FINAL_SUMMARY.md").exists()
    finally:
        _cleanup_runs_for_title(title)


def _write_validation_task(tmp_path: Path, title: str) -> Path:
    content = VALIDATION_TEMPLATE.replace(
        "title: Validation demo for semi-autonomous repair loop",
        f"title: {title}",
        1,
    )
    task_path = tmp_path / "validation_demo_task.yaml"
    task_path.write_text(content, encoding="utf-8")
    return task_path


def _latest_run_dir_for_title(title: str) -> Path:
    run_dir = find_latest_run(RUNS_ROOT, task_slug=slugify(title))
    assert run_dir is not None
    return run_dir


def _cleanup_runs_for_title(title: str) -> None:
    slug = slugify(title)
    for run_dir in RUNS_ROOT.glob(f"*_{slug}"):
        shutil.rmtree(run_dir, ignore_errors=True)
