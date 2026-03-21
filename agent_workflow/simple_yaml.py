from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SimpleYAMLParseError(ValueError):
    """Raised when a task spec uses unsupported YAML syntax."""


def load_yaml_or_json(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError:
            data = load_simple_yaml(text)
        else:
            data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SimpleYAMLParseError(f"Expected a mapping at the top level of {path}.")
    return data


def load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse a constrained YAML subset used by the workflow task specs.

    Supported forms:
    - top-level key: scalar
    - top-level key: []
    - top-level key:\n  - item\n  - item
    - top-level key: |\n  multi-line block

    More advanced YAML features require PyYAML to be installed.
    """

    normalized = text.lstrip("\ufeff")
    lines = normalized.splitlines()
    result: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw = lines[index].rstrip()
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if raw.startswith(" ") or raw.startswith("\t"):
            raise SimpleYAMLParseError(
                "Only top-level mappings are supported without PyYAML. "
                f"Unsupported indentation on line {index + 1}."
            )
        if ":" not in raw:
            raise SimpleYAMLParseError(f"Expected 'key: value' syntax on line {index + 1}.")
        key, remainder = raw.split(":", 1)
        key = key.strip()
        value = remainder.strip()
        if not key:
            raise SimpleYAMLParseError(f"Missing key on line {index + 1}.")
        if value == "|":
            index += 1
            block: list[str] = []
            while index < len(lines):
                block_raw = lines[index].rstrip("\n")
                if block_raw.strip() == "":
                    block.append("")
                    index += 1
                    continue
                if not block_raw.startswith("  "):
                    break
                block.append(block_raw[2:])
                index += 1
            result[key] = "\n".join(block).rstrip("\n")
            continue
        if value == "":
            index += 1
            items: list[Any] = []
            while index < len(lines):
                item_raw = lines[index].rstrip()
                if not item_raw.strip():
                    index += 1
                    continue
                if item_raw.startswith("  - "):
                    items.append(_parse_scalar(item_raw[4:].strip()))
                    index += 1
                    continue
                break
            result[key] = items
            continue
        result[key] = _parse_scalar(value)
        index += 1
    return result


def dump_simple_yaml(mapping: dict[str, Any]) -> str:
    lines: list[str] = []
    for key, value in mapping.items():
        if isinstance(value, str) and "\n" in value:
            lines.append(f"{key}: |")
            for block_line in value.splitlines():
                lines.append(f"  {block_line}")
            if value.endswith("\n") and (not value.splitlines() or value.splitlines()[-1] != ""):
                lines.append("  ")
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                lines.append(f"{key}: []")
                continue
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {_format_scalar(item)}")
            continue
        lines.append(f"{key}: {_format_scalar(value)}")
    return "\n".join(lines) + "\n"


def _parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none", "~"}:
        return None
    if raw == "[]":
        return []
    if raw.startswith(("'", '"')) and raw.endswith(("'", '"')) and len(raw) >= 2:
        return raw[1:-1]
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _format_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        if value == "":
            return '""'
        if any(ch in value for ch in [":", "#", "[", "]"]) or value != value.strip():
            return json.dumps(value)
        return value
    return json.dumps(value)
