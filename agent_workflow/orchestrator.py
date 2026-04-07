from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .artifacts import (
    append_markdown_log,
    copy_tree,
    create_run_directory,
    find_latest_incomplete_run,
    find_latest_run,
    write_json,
    write_text,
)
from .backends import AgentRequest, UnavailableBackend, build_agent_backends, build_role_backends
from .prompts import default_prompt_context, load_prompt_template, render_prompt
from .state import RunState
from .task_spec import TaskSpec, resolve_path


@dataclass(frozen=True)
class OrchestratorOptions:
    resume: bool = False
    resume_last: bool = False
    force_restart: bool = False
    max_iterations_override: int | None = None
    dry_run: bool = False
    verbose: bool = False


class WorkflowOrchestrator:
    def __init__(self, *, repo_root: Path, task: TaskSpec, options: OrchestratorOptions) -> None:
        self.repo_root = repo_root.resolve()
        self.task = task
        self.options = options
        self.runs_root = self.repo_root / "agent_runs"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.run_dir: Path | None = None
        self.run_state_path: Path | None = None
        self.implementation_log_path: Path | None = None
        self.task_resolved_path: Path | None = None
        self.state: RunState | None = None
        self.codex_backend = None
        self.opus_backend = None
        self.backends: dict[str, Any] = {}
        self.backend_profile_name: str | None = None

    def run(self) -> int:
        self._prepare_run()
        assert self.state is not None
        assert self.run_state_path is not None
        if self.options.dry_run:
            self._prepare_dry_run_artifacts()
            self.state.write(self.run_state_path)
            return 0
        while True:
            phase = self.state.current_phase
            if phase == "initialize":
                self._initialize_phase()
            elif phase == "plan":
                self._plan_phase()
            elif phase == "execute":
                self._execute_phase()
            elif phase == "test":
                self._test_phase()
            elif phase == "review":
                self._review_phase()
            elif phase == "docs":
                self._docs_phase()
            elif phase == "summary":
                self._summary_phase()
            elif phase == "complete":
                break
            else:
                raise ValueError(f"Unknown workflow phase '{phase}'.")
            self.state.write(self.run_state_path)
            if self.state.status in {"blocked", "incomplete", "complete"}:
                break
        return self._exit_code()

    def _prepare_run(self) -> None:
        run_dir, state = self._resolve_run_directory_and_state()
        self.run_dir = run_dir
        self.run_state_path = run_dir / "RUN_STATE.json"
        self.implementation_log_path = run_dir / "IMPLEMENTATION_LOG.md"
        self.task_resolved_path = run_dir / "task_resolved.yaml"
        self.state = state
        codex, opus, profile_name = build_role_backends(self.repo_root, self.task.backend_profile)
        self.codex_backend = codex
        self.opus_backend = opus
        self.backends, _ = build_agent_backends(self.repo_root, self.task.backend_profile)
        self.backend_profile_name = profile_name
        self.state.max_iterations = int(self.options.max_iterations_override or self.task.max_iterations)
        self.state.write(self.run_state_path)

    def _resolve_run_directory_and_state(self) -> tuple[Path, RunState]:
        task_slug = self.task.slug
        existing_run: Path | None = None
        if not self.options.force_restart:
            if self.options.resume:
                existing_run = find_latest_incomplete_run(self.runs_root, task_slug=task_slug)
                if existing_run is None:
                    raise FileNotFoundError(f"No incomplete run found for task '{self.task.title}'.")
            elif self.options.resume_last:
                existing_run = find_latest_incomplete_run(self.runs_root)
                if existing_run is None:
                    raise FileNotFoundError("No incomplete run found to resume.")
            elif self.task.auto_resume:
                existing_run = find_latest_incomplete_run(self.runs_root, task_slug=task_slug)
        if existing_run is not None:
            state = RunState.from_path(existing_run / "RUN_STATE.json")
            if self.options.max_iterations_override is not None:
                state.max_iterations = int(self.options.max_iterations_override)
            return existing_run, state

        run_dir = create_run_directory(self.runs_root, self.task.title)
        active_working_directory = self._resolve_active_working_directory(run_dir)
        state = RunState.new(
            run_id=run_dir.name,
            task_name=self.task.title,
            task_path=str(self.task.source_path) if self.task.source_path is not None else "",
            run_dir=str(run_dir),
            max_iterations=int(self.options.max_iterations_override or self.task.max_iterations),
            human_plan_supplied=self.task.human_plan_supplied,
            review_required=self.task.review_required,
            active_working_directory=str(active_working_directory),
        )
        return run_dir, state

    def _initialize_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None and self.task_resolved_path is not None
        bootstrap_source = self._resolve_bootstrap_source(self.run_dir)
        if bootstrap_source is not None:
            workspace_dir = self.run_dir / "workspace"
            copy_tree(bootstrap_source, workspace_dir)
        active_working_directory = self._resolve_active_working_directory(self.run_dir)
        self.state.active_working_directory = str(active_working_directory)
        resolved_yaml = self.task.dump_resolved_yaml(
            repo_root=self.repo_root,
            run_dir=self.run_dir,
            max_iterations_override=self.state.max_iterations,
        )
        write_text(self.task_resolved_path, resolved_yaml)
        self.state.completion_flags["initialized"] = True
        self.state.record_phase(
            "plan",
            "running",
            note=f"Initialized run with backend profile '{self.backend_profile_name}'.",
            artifact=self.task_resolved_path.name,
        )
        initial_snapshot, _ = self._capture_workspace_snapshot(active_working_directory)
        self.state.set_snapshot(initial_snapshot)
        append_markdown_log(
            self.implementation_log_path,
            "Initialize",
            (
                f"Run directory: {self.run_dir}\n\n"
                f"Active working directory: {active_working_directory}\n\n"
                f"Backend profile: {self.backend_profile_name}\n\n"
                f"Task file: {self.task.source_path}"
            ),
        )

    def _prepare_dry_run_artifacts(self) -> None:
        assert self.state is not None and self.run_dir is not None and self.task_resolved_path is not None
        if self.state.current_phase == "initialize":
            self._initialize_phase()
        iteration = max(1, self.state.iteration_count + 1)
        prompt_text, prompt_path, context_path = self._render_phase_prompt("execute", iteration)
        _ = prompt_text
        write_text(self.run_dir / "DRY_RUN_NOTE.txt", "Dry run completed. No backends or test commands were executed.\n")
        self.state.status = "dry-run"
        self.state.record_phase(
            "execute",
            "dry-run",
            iteration=iteration,
            note="Rendered the next execution prompt without invoking a backend.",
            artifact=prompt_path.name,
        )
        append_markdown_log(
            self.implementation_log_path,
            "Dry Run",
            f"Execution prompt rendered to {prompt_path.name}. Context written to {context_path.name}.",
        )

    def _plan_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None
        # Skip if human plan is supplied or planner is explicitly disabled
        if self.task.human_plan_supplied or self.task.skip_planner:
            self.state.plan = _wrap_human_plan(self.task.human_plan or self.task.goal)
            self.state.record_phase(
                "execute",
                "running",
                note="Plan phase skipped (human_plan supplied or skip_planner=True).",
            )
            append_markdown_log(
                self.implementation_log_path,
                "Plan",
                "Plan phase skipped; using human plan or goal as single subtask.",
            )
            return

        planner = self.backends.get("planner")
        if planner is None or isinstance(planner, UnavailableBackend):
            self.state.plan = _single_subtask_plan(self.task.goal, self.task.acceptance_criteria)
            self.state.record_phase(
                "execute",
                "running",
                note="Planner unavailable; using single-subtask fallback.",
            )
            append_markdown_log(
                self.implementation_log_path,
                "Plan",
                "Planner backend unavailable; goal wrapped as single subtask.",
            )
            return

        prompt_text, prompt_path, context_path = self._render_phase_prompt("plan", 0)
        request = self._make_request("planner", "plan", 0, prompt_text, prompt_path, context_path)
        response = planner.run(request)
        plan_output_path = self.run_dir / "PLAN.json"
        write_text(plan_output_path, response.content + "\n")

        if response.structured and "subtasks" in response.structured:
            plan = response.structured
        else:
            plan = _single_subtask_plan(self.task.goal, self.task.acceptance_criteria)

        self.state.plan = plan
        subtask_count = len(plan.get("subtasks", []))
        self.state.record_phase(
            "execute",
            "running",
            note=f"Planner produced {subtask_count} subtask(s).",
            artifact=plan_output_path.name,
        )
        append_markdown_log(
            self.implementation_log_path,
            "Plan",
            f"Planner produced {subtask_count} subtask(s). Plan written to {plan_output_path.name}.",
        )

    def _execute_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None and self.codex_backend is not None
        iteration = self.state.iteration_count + 1
        # Resolve current subtask for context injection (handled inside _render_phase_prompt)
        prompt_text, prompt_path, context_path = self._render_phase_prompt("execute", iteration)
        request = self._make_request("codex", "execute", iteration, prompt_text, prompt_path, context_path)
        response = self.codex_backend.run(request)
        output_path = self.run_dir / f"EXECUTION_OUTPUT_iter{iteration:02d}.txt"
        write_text(output_path, response.content + ("\n" + response.stderr if response.stderr else "\n"))
        payload = response.structured or {"status": "completed" if response.success else "incomplete", "summary": response.content.strip()}
        record = self.state.ensure_iteration(iteration)
        record["execution_status"] = str(payload.get("status", "completed"))
        record["execution_summary"] = str(payload.get("summary", ""))
        record["execution_output"] = output_path.name
        snapshot, changed_files = self._capture_workspace_snapshot(self.active_working_directory)
        record["workspace_snapshot"] = snapshot
        record["changed_files"] = list(changed_files)
        self.state.set_snapshot(snapshot)
        self.state.iteration_count = iteration
        changed_files_path = self.run_dir / f"CHANGED_FILES_iter{iteration:02d}.txt"
        write_text(changed_files_path, "\n".join(changed_files) + ("\n" if changed_files else ""))
        append_markdown_log(
            self.implementation_log_path,
            f"Execute Iteration {iteration}",
            f"Status: {record['execution_status']}\n\nSummary: {record['execution_summary']}\n\nChanged files artifact: {changed_files_path.name}",
        )
        self.state.record_phase(
            "test",
            "running",
            iteration=iteration,
            note=record["execution_summary"],
            artifact=output_path.name,
        )
        if str(payload.get("status", "")).lower() == "blocked":
            reason = str(payload.get("summary") or "Execution backend reported a blocking issue.")
            self.state.mark_blocked(reason)
            self.state.record_phase("execute", "blocked", iteration=iteration, note=reason, artifact=output_path.name)

    def _test_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None
        iteration = self.state.iteration_count
        record = self.state.ensure_iteration(iteration)
        log_path = self.run_dir / f"TEST_LOG_iter{iteration:02d}.txt"
        snapshot = self.state.last_workspace_snapshot
        if not self.task.tests_to_run:
            log = "No explicit test commands were provided. Tests were waived by task configuration.\n"
            write_text(log_path, log)
            record["test_status"] = "waived"
            record["test_log"] = log_path.name
            self.state.completion_flags["tests_complete"] = True
            self.state.record_phase("review", "running", iteration=iteration, note="Tests waived.", artifact=log_path.name)
            append_markdown_log(self.implementation_log_path, f"Tests Iteration {iteration}", "Tests were waived by task configuration.")
            return
        if snapshot is not None and snapshot == self.state.last_tested_snapshot:
            log = "Workspace snapshot has not changed since the last successful test run. Tests were skipped.\n"
            write_text(log_path, log)
            record["test_status"] = "skipped_unchanged"
            record["test_log"] = log_path.name
            self.state.completion_flags["tests_complete"] = True
            self.state.record_phase("review", "running", iteration=iteration, note="Tests skipped because code is unchanged.", artifact=log_path.name)
            append_markdown_log(self.implementation_log_path, f"Tests Iteration {iteration}", "Tests were skipped because the workspace snapshot was unchanged.")
            return
        command_mapping = self._command_placeholder_mapping()
        log_chunks: list[str] = []
        overall_success = True
        rendered_commands: list[str] = []
        for command in self.task.tests_to_run:
            rendered_command = self._render_command_with_placeholders(command, command_mapping)
            rendered_commands.append(rendered_command)
            completed = subprocess.run(
                rendered_command,
                cwd=self.active_working_directory,
                capture_output=True,
                text=True,
                shell=True,
                check=False,
            )
            log_chunks.append(f"$ {rendered_command}\n")
            log_chunks.append(f"[exit={completed.returncode}]\n")
            if completed.stdout:
                log_chunks.append(completed.stdout)
                if not completed.stdout.endswith("\n"):
                    log_chunks.append("\n")
            if completed.stderr:
                log_chunks.append("--- STDERR ---\n")
                log_chunks.append(completed.stderr)
                if not completed.stderr.endswith("\n"):
                    log_chunks.append("\n")
            log_chunks.append("\n")
            if completed.returncode != 0:
                overall_success = False
        write_text(log_path, "".join(log_chunks))
        record["test_status"] = "passed" if overall_success else "failed"
        record["test_commands"] = rendered_commands
        record["test_log"] = log_path.name
        self.state.set_snapshot(snapshot, tested=True)
        self.state.completion_flags["tests_complete"] = overall_success
        note = "All configured tests passed." if overall_success else "One or more test commands failed."
        self.state.record_phase("review", "running", iteration=iteration, note=note, artifact=log_path.name)
        append_markdown_log(self.implementation_log_path, f"Tests Iteration {iteration}", note)

    def _review_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None and self.opus_backend is not None
        iteration = self.state.iteration_count
        if not self.task.review_required:
            self.state.acceptance_criteria_satisfied = self.state.completion_flags["tests_complete"]
            self.state.completion_flags["review_complete"] = True
            self.state.record_phase("docs", "running", iteration=iteration, note="Review skipped by task configuration.")
            append_markdown_log(self.implementation_log_path, f"Review Iteration {iteration}", "Review was skipped by task configuration.")
            return
        prompt_text, prompt_path, context_path = self._render_phase_prompt("review", iteration)
        request = self._make_request("opus", "review", iteration, prompt_text, prompt_path, context_path)
        response = self.opus_backend.run(request)
        output_path = self.run_dir / f"REVIEW_OUTPUT_iter{iteration:02d}.txt"
        write_text(output_path, response.content + ("\n" + response.stderr if response.stderr else "\n"))
        payload = response.structured or {"status": "accepted" if response.success else "blocked", "summary": response.content.strip()}
        record = self.state.ensure_iteration(iteration)
        record["review_status"] = str(payload.get("status", "accepted"))
        record["review_summary"] = str(payload.get("summary", ""))
        record["repair_instructions"] = [str(item) for item in payload.get("repair_instructions", [])]
        self.state.repair_instructions = list(record["repair_instructions"])
        append_markdown_log(
            self.implementation_log_path,
            f"Review Iteration {iteration}",
            f"Status: {record['review_status']}\n\nSummary: {record['review_summary']}",
        )
        status = str(payload.get("status", "accepted")).lower()
        if status == "blocked":
            reason = str(payload.get("summary") or "Review backend reported a blocking issue.")
            self.state.mark_blocked(reason)
            self.state.record_phase("review", "blocked", iteration=iteration, note=reason, artifact=output_path.name)
            return
        if status == "needs_repair":
            if iteration >= self.state.max_iterations:
                reason = "Maximum iterations reached before review acceptance."
                self.state.mark_incomplete(reason)
                self.state.record_phase("review", "incomplete", iteration=iteration, note=reason, artifact=output_path.name)
                self.state.current_phase = "execute"
                return
            self.state.status = "needs_repair"
            self.state.current_phase = "execute"
            self.state.record_phase("execute", "needs_repair", iteration=iteration + 1, note=record["review_summary"], artifact=output_path.name)
            return
        self.state.acceptance_criteria_satisfied = bool(self.state.completion_flags["tests_complete"] or not self.task.tests_to_run)
        self.state.completion_flags["review_complete"] = True
        # Advance subtask index; if more subtasks remain, loop back to execute
        if self.state.plan:
            subtasks = self.state.plan.get("subtasks", [])
            self.state.current_subtask_index += 1
            if self.state.current_subtask_index < len(subtasks):
                self.state.acceptance_criteria_satisfied = False
                self.state.completion_flags["review_complete"] = False
                self.state.record_phase(
                    "execute",
                    "running",
                    iteration=iteration + 1,
                    note=f"Subtask {self.state.current_subtask_index} accepted; proceeding to next subtask.",
                    artifact=output_path.name,
                )
                return
        self.state.record_phase("docs", "running", iteration=iteration, note=record["review_summary"], artifact=output_path.name)

    def _docs_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None and self.opus_backend is not None
        prompt_text, prompt_path, context_path = self._render_phase_prompt("docs", self.state.iteration_count)
        request = self._make_request("opus", "docs", self.state.iteration_count, prompt_text, prompt_path, context_path)
        response = self.opus_backend.run(request)
        output_path = self.run_dir / "DOC_OUTPUT.md"
        write_text(output_path, response.content if response.content.endswith("\n") else response.content + "\n")
        self.state.completion_flags["documentation_complete"] = True
        self.state.record_phase("summary", "running", iteration=self.state.iteration_count, note="Documentation artifact generated.", artifact=output_path.name)
        append_markdown_log(self.implementation_log_path, "Documentation", f"Documentation artifact written to {output_path.name}.")

    def _summary_phase(self) -> None:
        assert self.state is not None and self.run_dir is not None and self.opus_backend is not None
        prompt_text, prompt_path, context_path = self._render_phase_prompt("summary", self.state.iteration_count)
        request = self._make_request("opus", "summary", self.state.iteration_count, prompt_text, prompt_path, context_path)
        response = self.opus_backend.run(request)
        output_path = self.run_dir / "FINAL_SUMMARY.md"
        write_text(output_path, response.content if response.content.endswith("\n") else response.content + "\n")
        self.state.completion_flags["summary_complete"] = True
        implementation_done = self.state.iteration_count > 0 or not self.task.allow_repo_edits
        tests_done = self.state.completion_flags["tests_complete"] or not self.task.tests_to_run
        review_done = self.state.completion_flags["review_complete"] or not self.task.review_required
        docs_done = self.state.completion_flags["documentation_complete"]
        summary_done = self.state.completion_flags["summary_complete"]
        if implementation_done and tests_done and review_done and docs_done and summary_done and self.state.acceptance_criteria_satisfied:
            self.state.completion_flags["implementation_complete"] = True
            self.state.mark_complete()
            self.state.record_phase("complete", "complete", iteration=self.state.iteration_count, note="Workflow completed successfully.", artifact=output_path.name)
        else:
            self.state.mark_incomplete("Final completion criteria were not satisfied.")
            self.state.record_phase("summary", "incomplete", iteration=self.state.iteration_count, note="Final completion criteria were not satisfied.", artifact=output_path.name)
        append_markdown_log(self.implementation_log_path, "Summary", f"Final summary written to {output_path.name}.")

    def _render_phase_prompt(self, phase: str, iteration: int) -> tuple[str, Path, Path]:
        assert self.state is not None and self.run_dir is not None and self.run_state_path is not None and self.implementation_log_path is not None
        template = load_prompt_template(self.repo_root, phase)
        task_mapping = self.task.to_mapping(repo_root=self.repo_root, run_dir=self.run_dir, max_iterations_override=self.state.max_iterations)
        task_mapping["task_file"] = str(self.task.source_path) if self.task.source_path is not None else ""
        command_mapping = self._command_placeholder_mapping()
        task_mapping["tests_to_run"] = [
            self._render_command_with_placeholders(command, command_mapping) for command in task_mapping.get("tests_to_run", [])
        ]
        task_mapping["docs_to_update"] = [
            self._render_command_with_placeholders(item, command_mapping) for item in task_mapping.get("docs_to_update", [])
        ]
        state_mapping = self.state.to_dict()
        state_mapping["run_state_path"] = str(self.run_state_path)
        state_mapping["implementation_log_path"] = str(self.implementation_log_path)
        # Inject current subtask into state mapping for prompt rendering
        current_subtask: dict[str, Any] = {}
        subtask_index = 0
        subtask_count = 0
        if self.state.plan:
            subtasks = self.state.plan.get("subtasks", [])
            subtask_count = len(subtasks)
            subtask_index = self.state.current_subtask_index
            if subtask_index < subtask_count:
                current_subtask = subtasks[subtask_index]
        state_mapping["current_subtask"] = current_subtask
        state_mapping["current_subtask_index"] = subtask_index
        state_mapping["plan"] = self.state.plan

        prompt_context = default_prompt_context(
            task=task_mapping,
            state=state_mapping,
            working_directory=self.active_working_directory,
        )
        prompt_context["BACKEND_PROFILE"] = self.backend_profile_name or ""
        prompt_text = render_prompt(template, prompt_context)
        context_payload = {
            "repo_root": str(self.repo_root),
            "run_dir": str(self.run_dir),
            "working_directory": str(self.active_working_directory),
            "task": task_mapping,
            "state": state_mapping,
            "backend_profile": self.backend_profile_name,
        }
        if phase in {"execute", "review", "evaluate"}:
            prompt_name = f"{phase.upper()}_PROMPT_iter{iteration:02d}.txt"
            context_name = f"{phase.upper()}_CONTEXT_iter{iteration:02d}.json"
        elif phase == "plan":
            prompt_name = "PLAN_PROMPT.txt"
            context_name = "PLAN_CONTEXT.json"
        else:
            prompt_name = f"{phase.upper()}_PROMPT.txt"
            context_name = f"{phase.upper()}_CONTEXT.json"
        prompt_path = self.run_dir / prompt_name
        context_path = self.run_dir / context_name
        write_text(prompt_path, prompt_text if prompt_text.endswith("\n") else prompt_text + "\n")
        write_json(context_path, context_payload)
        return prompt_text, prompt_path, context_path

    def _make_request(self, role: str, phase: str, iteration: int, prompt: str, prompt_path: Path, context_path: Path) -> AgentRequest:
        return AgentRequest(
            role=role,
            phase=phase,
            iteration=iteration,
            prompt=prompt,
            prompt_path=prompt_path,
            context=json.loads(context_path.read_text(encoding="utf-8")),
            context_path=context_path,
            working_directory=self.active_working_directory,
            run_directory=self.run_dir,
        )

    def _resolve_bootstrap_source(self, run_dir: Path) -> Path | None:
        if not self.task.bootstrap_workspace_from:
            return None
        return resolve_path(self.task.bootstrap_workspace_from, self.repo_root, run_dir)

    def _resolve_active_working_directory(self, run_dir: Path) -> Path:
        if self.task.working_directory:
            return resolve_path(self.task.working_directory, self.repo_root, run_dir)
        if self.task.bootstrap_workspace_from:
            return run_dir / "workspace"
        return self.repo_root

    @property
    def active_working_directory(self) -> Path:
        assert self.state is not None and self.state.active_working_directory is not None
        return Path(self.state.active_working_directory)

    def _capture_workspace_snapshot(self, working_directory: Path) -> tuple[str | None, list[str]]:
        if not self.task.allow_repo_edits:
            return self._filesystem_snapshot(working_directory)
        git_probe = subprocess.run(
            ["git", "-C", str(working_directory), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            check=False,
        )
        if git_probe.returncode == 0 and git_probe.stdout.strip().lower() == "true":
            status = subprocess.run(
                ["git", "-C", str(working_directory), "status", "--porcelain=v1", "--untracked-files=all"],
                capture_output=True,
                text=True,
                check=False,
            )
            raw = status.stdout
            changed_files = [line[3:] for line in raw.splitlines() if len(line) >= 4]
            return "git\n" + raw, changed_files
        return self._filesystem_snapshot(working_directory)

    @staticmethod
    def _filesystem_snapshot(working_directory: Path) -> tuple[str | None, list[str]]:
        snapshot_lines: list[str] = []
        file_paths: list[str] = []
        for path in sorted(working_directory.rglob("*")):
            if path.is_dir():
                if path.name in {".git", "__pycache__", ".pytest_cache", ".mypy_cache"}:
                    continue
                continue
            relative = path.relative_to(working_directory).as_posix()
            if any(part in {".git", "__pycache__", ".pytest_cache", ".mypy_cache"} for part in path.parts):
                continue
            digest = hashlib.sha1(path.read_bytes()).hexdigest()
            snapshot_lines.append(f"{relative}\t{digest}")
            file_paths.append(relative)
        return "fs\n" + "\n".join(snapshot_lines), file_paths

    def _command_placeholder_mapping(self) -> dict[str, str]:
        assert self.run_dir is not None
        return {
            "repo_root": str(self.repo_root),
            "run_dir": str(self.run_dir),
            "working_directory": str(self.active_working_directory),
            "task_file": str(self.task.source_path) if self.task.source_path is not None else "",
            "python_executable": sys.executable,
        }

    @staticmethod
    def _render_command_with_placeholders(command: str, mapping: dict[str, str]) -> str:
        rendered = command
        for key, value in mapping.items():
            rendered = rendered.replace(f"{{{key}}}", value)
        return rendered

    def _exit_code(self) -> int:
        assert self.state is not None
        if self.state.status == "complete":
            return 0
        if self.state.status == "blocked":
            return 1
        return 2


def _wrap_human_plan(plan_text: str) -> dict[str, Any]:
    """Wrap a free-text human plan as a single-subtask plan dict."""
    return {
        "subtasks": [
            {
                "id": "1",
                "description": plan_text.strip(),
                "test_criteria": [],
                "expected_files": [],
            }
        ],
        "planning_notes": "Human-supplied plan wrapped as single subtask.",
        "risks": [],
    }


def _single_subtask_plan(goal: str, acceptance_criteria: tuple[str, ...]) -> dict[str, Any]:
    """Wrap a goal as a single-subtask plan (planner unavailable fallback)."""
    return {
        "subtasks": [
            {
                "id": "1",
                "description": goal.strip(),
                "test_criteria": list(acceptance_criteria),
                "expected_files": [],
            }
        ],
        "planning_notes": "Automatically generated single-subtask plan (no planner available).",
        "risks": [],
    }


def resolve_latest_resume_task(repo_root: Path) -> Path | None:
    latest = find_latest_incomplete_run(repo_root / "agent_runs")
    if latest is None:
        return None
    state_path = latest / "RUN_STATE.json"
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    task_path = payload.get("task_path")
    return Path(task_path) if task_path else None
