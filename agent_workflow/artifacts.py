from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any


def slugify(value: str) -> str:
    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif not chars or chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "task"


def timestamped_run_id(title: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{slugify(title)}"


def create_run_directory(runs_root: Path, title: str) -> Path:
    run_dir = runs_root / timestamped_run_id(title)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def list_run_directories(runs_root: Path, *, task_slug: str | None = None) -> list[Path]:
    if not runs_root.exists():
        return []
    directories = [path for path in runs_root.iterdir() if path.is_dir()]
    if task_slug is not None:
        directories = [path for path in directories if path.name.endswith(f"_{task_slug}")]
    return sorted(directories)


def find_latest_run(runs_root: Path, *, task_slug: str | None = None) -> Path | None:
    runs = list_run_directories(runs_root, task_slug=task_slug)
    return runs[-1] if runs else None


def find_latest_incomplete_run(runs_root: Path, *, task_slug: str | None = None) -> Path | None:
    for run_dir in reversed(list_run_directories(runs_root, task_slug=task_slug)):
        state_path = run_dir / "RUN_STATE.json"
        if not state_path.exists():
            continue
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "complete":
            return run_dir
    return None


def write_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def append_markdown_log(path: Path, heading: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"## {heading}\n\n{body.rstrip()}\n\n")


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    shutil.copytree(source, destination)
