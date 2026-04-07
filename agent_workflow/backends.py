from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Protocol


@dataclass(frozen=True)
class AgentRequest:
    role: str
    phase: str
    iteration: int
    prompt: str
    prompt_path: Path
    context: dict[str, Any]
    context_path: Path
    working_directory: Path
    run_directory: Path


@dataclass
class AgentResponse:
    role: str
    phase: str
    success: bool
    content: str
    structured: dict[str, Any] | None = None
    returncode: int | None = None
    stderr: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentBackend(Protocol):
    name: str

    def run(self, request: AgentRequest) -> AgentResponse:
        ...


class UnavailableBackend:
    def __init__(self, message: str) -> None:
        self.name = "unavailable"
        self._message = message

    def run(self, request: AgentRequest) -> AgentResponse:
        payload = {
            "status": "blocked",
            "summary": self._message,
            "blocking_issues": [self._message],
        }
        return AgentResponse(
            role=request.role,
            phase=request.phase,
            success=False,
            content=json.dumps(payload, indent=2),
            structured=payload,
            returncode=None,
            stderr=self._message,
        )


class CommandTemplateBackend:
    def __init__(
        self,
        *,
        name: str,
        command: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | None = None,
    ) -> None:
        self.name = name
        self._command = command
        self._cwd = cwd
        self._env = env or {}
        self._timeout_s = timeout_s

    def run(self, request: AgentRequest) -> AgentResponse:
        mapping = _placeholder_mapping(request)
        command = [_format_token(token, mapping) for token in self._command]
        cwd = Path(_format_token(self._cwd, mapping)) if self._cwd else request.working_directory
        env = os.environ.copy()
        env.update({key: _format_token(value, mapping) for key, value in self._env.items()})
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
                check=False,
                env=env,
            )
        except FileNotFoundError as exc:
            message = f"Backend command not found: {command[0]}"
            payload = {"status": "blocked", "summary": message, "blocking_issues": [str(exc)]}
            return AgentResponse(
                role=request.role,
                phase=request.phase,
                success=False,
                content=json.dumps(payload, indent=2),
                structured=payload,
                stderr=str(exc),
                metadata={"command": command},
            )
        except subprocess.TimeoutExpired as exc:
            message = f"Backend command timed out after {self._timeout_s} seconds."
            payload = {"status": "incomplete", "summary": message, "blocking_issues": []}
            return AgentResponse(
                role=request.role,
                phase=request.phase,
                success=False,
                content=json.dumps(payload, indent=2),
                structured=payload,
                stderr=exc.stderr or "",
                metadata={"command": command, "timeout_s": self._timeout_s},
            )
        output = completed.stdout.strip()
        stderr = completed.stderr.strip()
        structured = extract_json_payload(output)
        success = completed.returncode == 0
        if not output and structured is None:
            if success:
                output = json.dumps({"status": "completed", "summary": "Command completed without structured output."}, indent=2)
            else:
                output = json.dumps({"status": "incomplete", "summary": "Command exited without structured output."}, indent=2)
                structured = extract_json_payload(output)
        return AgentResponse(
            role=request.role,
            phase=request.phase,
            success=success,
            content=output,
            structured=structured,
            returncode=completed.returncode,
            stderr=stderr,
            metadata={"command": command, "cwd": str(cwd)},
        )


class ScriptedBackend:
    def __init__(self, *, script_path: Path, role: str) -> None:
        self.name = f"scripted:{role}"
        self._script_path = script_path
        self._role = role
        self._payload = json.loads(script_path.read_text(encoding="utf-8"))

    def run(self, request: AgentRequest) -> AgentResponse:
        role_block = self._payload.get(self._role, {})
        key = f"{request.phase}:final" if request.phase in {"docs", "summary"} else f"{request.phase}:{request.iteration}"
        step = role_block.get(key)
        if step is None:
            payload = {
                "status": "blocked",
                "summary": f"No scripted response configured for {self._role}:{key}.",
                "blocking_issues": [f"Missing scripted step {key} in {self._script_path.name}."],
            }
            return AgentResponse(
                role=request.role,
                phase=request.phase,
                success=False,
                content=json.dumps(payload, indent=2),
                structured=payload,
                stderr=payload["summary"],
            )
        for action in step.get("actions", []):
            _apply_scripted_action(action, request)
        structured = step.get("response")
        if structured is not None:
            content = json.dumps(structured, indent=2)
        else:
            content = str(step.get("content", ""))
        return AgentResponse(
            role=request.role,
            phase=request.phase,
            success=bool(step.get("success", True)),
            content=content,
            structured=structured if isinstance(structured, dict) else None,
            returncode=0,
            metadata={"script_path": str(self._script_path), "step_key": key},
        )


class AnthropicBackend:
    def __init__(
        self,
        *,
        name: str,
        model: str,
        max_tokens: int = 8096,
        system_prompt: str | None = None,
    ) -> None:
        self.name = name
        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

    def run(self, request: AgentRequest) -> AgentResponse:
        try:
            import anthropic  # optional dependency
        except ImportError:
            payload = {
                "status": "blocked",
                "summary": "anthropic SDK not installed. Run: pip install anthropic",
                "blocking_issues": ["pip install anthropic"],
            }
            return AgentResponse(
                role=request.role,
                phase=request.phase,
                success=False,
                content=json.dumps(payload, indent=2),
                structured=payload,
                stderr="anthropic SDK not installed",
            )
        client = anthropic.Anthropic()
        messages: list[dict[str, Any]] = [{"role": "user", "content": request.prompt}]
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        response = client.messages.create(**kwargs)
        raw_content = "".join(block.text for block in response.content if hasattr(block, "text"))
        structured = extract_json_payload(raw_content)
        success = bool(raw_content.strip())
        usage = {}
        if hasattr(response, "usage") and response.usage is not None:
            usage = {
                "input_tokens": getattr(response.usage, "input_tokens", None),
                "output_tokens": getattr(response.usage, "output_tokens", None),
            }
        return AgentResponse(
            role=request.role,
            phase=request.phase,
            success=success,
            content=raw_content,
            structured=structured,
            metadata={"model": self._model, "usage": usage},
        )


class _RoleBackend:
    def __init__(self, role: str, backend: AgentBackend) -> None:
        self.role = role
        self.name = f"{role}:{backend.name}"
        self._backend = backend

    def run(self, request: AgentRequest) -> AgentResponse:
        effective = AgentRequest(
            role=self.role,
            phase=request.phase,
            iteration=request.iteration,
            prompt=request.prompt,
            prompt_path=request.prompt_path,
            context=request.context,
            context_path=request.context_path,
            working_directory=request.working_directory,
            run_directory=request.run_directory,
        )
        return self._backend.run(effective)


class CodexBackend(_RoleBackend):
    def __init__(self, backend: AgentBackend) -> None:
        super().__init__("codex", backend)


class OpusBackend(_RoleBackend):
    def __init__(self, backend: AgentBackend) -> None:
        super().__init__("opus", backend)


class PlannerBackend(_RoleBackend):
    def __init__(self, backend: AgentBackend) -> None:
        super().__init__("planner", backend)


class ExecutorBackend(_RoleBackend):
    def __init__(self, backend: AgentBackend) -> None:
        super().__init__("executor", backend)


class EvaluatorBackend(_RoleBackend):
    def __init__(self, backend: AgentBackend) -> None:
        super().__init__("evaluator", backend)


def load_workflow_config(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "agent_workflow" / "config.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_role_backends(repo_root: Path, profile_name: str | None) -> tuple[CodexBackend, OpusBackend, str]:
    """Legacy two-role builder. Prefer build_agent_backends() for new code."""
    config = load_workflow_config(repo_root)
    selected = profile_name or config.get("default_backend_profile") or "unconfigured"
    profiles = config.get("backend_profiles", {})
    if selected not in profiles:
        raise ValueError(f"Unknown backend profile '{selected}'.")
    profile = profiles[selected]
    codex = CodexBackend(_build_backend(repo_root, profile.get("codex", {}), role="codex"))
    opus = OpusBackend(_build_backend(repo_root, profile.get("opus", {}), role="opus"))
    return codex, opus, selected


def build_agent_backends(repo_root: Path, profile_name: str | None) -> tuple[dict[str, "AgentBackend"], str]:
    """Return a dict keyed by logical role: 'planner', 'executor', 'evaluator'.

    New-format profiles define those keys directly. Old profiles with 'codex'/'opus'
    are mapped: codex→executor, opus→evaluator, opus→planner (fallback).
    """
    config = load_workflow_config(repo_root)
    selected = profile_name or config.get("default_backend_profile") or "unconfigured"
    profiles = config.get("backend_profiles", {})
    if selected not in profiles:
        raise ValueError(f"Unknown backend profile '{selected}'.")
    profile = profiles[selected]

    def _role(key: str, fallback_key: str | None = None) -> "AgentBackend":
        cfg = profile.get(key)
        if cfg is None and fallback_key is not None:
            cfg = profile.get(fallback_key)
        return _build_backend(repo_root, cfg or {}, role=key)

    # Support both new-format (planner/executor/evaluator) and old-format (codex/opus)
    if "executor" in profile or "planner" in profile or "evaluator" in profile:
        executor_backend = _role("executor")
        evaluator_backend = _role("evaluator", fallback_key="executor")
        planner_backend = _role("planner", fallback_key="evaluator")
    else:
        # Old profile: map codex→executor, opus→evaluator/planner
        executor_backend = _build_backend(repo_root, profile.get("codex", {}), role="executor")
        evaluator_backend = _build_backend(repo_root, profile.get("opus", {}), role="evaluator")
        planner_backend = _build_backend(repo_root, profile.get("opus", {}), role="planner")

    return {
        "planner": PlannerBackend(planner_backend),
        "executor": ExecutorBackend(executor_backend),
        "evaluator": EvaluatorBackend(evaluator_backend),
    }, selected


def extract_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None
    candidates: list[str] = [stripped]
    if "```json" in stripped:
        marker = stripped.split("```json", 1)[1]
        fence_content = marker.split("```", 1)[0].strip()
        candidates.append(fence_content)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_backend(repo_root: Path, config: dict[str, Any], *, role: str) -> AgentBackend:
    backend_type = str(config.get("type", "unavailable")).lower()
    if backend_type == "command":
        command = config.get("command")
        if not isinstance(command, list) or not all(isinstance(token, str) for token in command):
            raise ValueError(f"Command backend for role '{role}' requires a string-list command.")
        return CommandTemplateBackend(
            name=str(config.get("name", f"command:{role}")),
            command=command,
            cwd=config.get("cwd"),
            env={str(key): str(value) for key, value in dict(config.get("env", {})).items()},
            timeout_s=int(config["timeout_s"]) if config.get("timeout_s") is not None else None,
        )
    if backend_type == "anthropic":
        model = str(config.get("model", "claude-sonnet-4-6"))
        return AnthropicBackend(
            name=str(config.get("name", f"anthropic:{role}")),
            model=model,
            max_tokens=int(config.get("max_tokens", 8096)),
            system_prompt=config.get("system_prompt") or None,
        )
    if backend_type == "scripted":
        script = config.get("script")
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"Scripted backend for role '{role}' requires a script path.")
        return ScriptedBackend(script_path=(repo_root / script).resolve(), role=role)
    message = str(
        config.get(
            "message",
            "No executable backend is configured. Use the validation_demo profile for the deterministic demo or configure a command backend in agent_workflow/config.json.",
        )
    )
    return UnavailableBackend(message)


def _apply_scripted_action(action: dict[str, Any], request: AgentRequest) -> None:
    action_type = str(action.get("type", "")).lower()
    mapping = _placeholder_mapping(request)
    raw_path = str(action.get("path", ""))
    if not raw_path:
        raise ValueError("Scripted backend actions require a 'path'.")
    path = Path(_format_token(raw_path, mapping))
    if not path.is_absolute():
        path = request.working_directory / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if action_type == "replace_text":
        old = str(action.get("old", ""))
        new = str(action.get("new", ""))
        content = path.read_text(encoding="utf-8")
        if old not in content:
            raise ValueError(f"Scripted replace_text could not find target text in {path}.")
        path.write_text(content.replace(old, new, 1), encoding="utf-8")
        return
    if action_type == "write_text":
        path.write_text(str(action.get("content", "")), encoding="utf-8")
        return
    if action_type == "append_text":
        with path.open("a", encoding="utf-8") as handle:
            handle.write(str(action.get("content", "")))
        return
    raise ValueError(f"Unsupported scripted action type '{action_type}'.")


def _placeholder_mapping(request: AgentRequest) -> dict[str, str]:
    repo_root = request.context.get("repo_root") or request.run_directory.parent
    return {
        "prompt_file": str(request.prompt_path),
        "context_file": str(request.context_path),
        "working_directory": str(request.working_directory),
        "run_dir": str(request.run_directory),
        "repo_root": str(repo_root),
        "iteration": str(request.iteration),
        "phase": request.phase,
        "role": request.role,
        "task_title": str(request.context.get("task_title", "")),
        "python_executable": sys.executable,
    }


def _format_token(value: str | None, mapping: dict[str, str]) -> str:
    if value is None:
        return ""
    rendered = value
    for key, replacement in mapping.items():
        rendered = rendered.replace(f"{{{key}}}", replacement)
    return rendered
