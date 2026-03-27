from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class DisplacementGate:
    index: int
    name: str
    re: float
    im: float

    @property
    def type(self) -> str:
        return "Displacement"

    @property
    def target(self) -> str:
        return "storage"

    @property
    def alpha(self) -> complex:
        return complex(self.re, self.im)

    @property
    def params(self) -> dict[str, float]:
        return {"re": self.re, "im": self.im}


@dataclass(frozen=True)
class RotationGate:
    index: int
    name: str
    theta: float
    phi: float

    @property
    def type(self) -> str:
        return "Rotation"

    @property
    def target(self) -> str:
        return "qubit"

    @property
    def params(self) -> dict[str, float]:
        return {"theta": self.theta, "phi": self.phi}


@dataclass(frozen=True)
class SQRGate:
    index: int
    name: str
    theta: tuple[float, ...]
    phi: tuple[float, ...]

    @property
    def type(self) -> str:
        return "SQR"

    @property
    def target(self) -> str:
        return "qubit"

    @property
    def params(self) -> dict[str, list[float]]:
        return {"theta": list(self.theta), "phi": list(self.phi)}


Gate = DisplacementGate | RotationGate | SQRGate


def gate_to_record(gate: Gate) -> dict[str, Any]:
    return {
        "index": gate.index,
        "type": gate.type,
        "target": gate.target,
        "name": gate.name,
        "params": gate.params,
    }


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key not in seen:
            out.append(path)
            seen.add(key)
    return out


def gate_path_candidates(path_like: str | Path) -> list[Path]:
    path = Path(path_like).expanduser()
    candidates = [path]
    suffix = path.suffix.lower()
    if suffix == ".josn":
        candidates.extend([path.with_suffix(".json"), path.with_suffix("")])
    elif suffix == ".json":
        candidates.extend([path.with_suffix(".josn"), path.with_suffix("")])
    elif suffix == "":
        candidates.extend([path.with_suffix(".json"), path.with_suffix(".josn")])
    else:
        candidates.extend([path.with_suffix(".json"), path.with_suffix(".josn"), path.with_suffix("")])
    return _dedupe_paths(candidates)


def _ensure_number(value: Any, gate_index: int, gate_type: str, field_name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(
        f"Gate {gate_index} ({gate_type}) expected numeric field '{field_name}', "
        f"got {type(value).__name__}."
    )


def _ensure_number_list(value: Any, gate_index: int, gate_type: str, field_name: str) -> list[float]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(
            f"Gate {gate_index} ({gate_type}) expected array field '{field_name}', "
            f"got {type(value).__name__}."
        )
    return [_ensure_number(item, gate_index, gate_type, f"{field_name}[{idx}]") for idx, item in enumerate(value)]


def validate_gate_entry(entry: dict[str, Any], gate_index: int) -> Gate:
    if not isinstance(entry, dict):
        raise TypeError(f"Gate {gate_index} must be a dictionary, got {type(entry).__name__}.")
    for key in ("type", "target", "params"):
        if key not in entry:
            raise KeyError(f"Gate {gate_index} is missing required key '{key}'.")
    if not isinstance(entry["params"], dict):
        raise TypeError(f"Gate {gate_index} field 'params' must be a dictionary.")

    gate_type = str(entry["type"])
    target = str(entry["target"])
    params = dict(entry["params"])
    name = str(entry.get("name", f"{gate_type}_{gate_index}"))

    if gate_type == "Displacement":
        if target != "storage":
            raise ValueError(f"Gate {gate_index} Displacement must target 'storage', got '{target}'.")
        return DisplacementGate(
            index=gate_index,
            name=name,
            re=_ensure_number(params.get("re"), gate_index, gate_type, "re"),
            im=_ensure_number(params.get("im"), gate_index, gate_type, "im"),
        )

    if gate_type == "Rotation":
        if target != "qubit":
            raise ValueError(f"Gate {gate_index} Rotation must target 'qubit', got '{target}'.")
        return RotationGate(
            index=gate_index,
            name=name,
            theta=_ensure_number(params.get("theta"), gate_index, gate_type, "theta"),
            phi=_ensure_number(params.get("phi"), gate_index, gate_type, "phi"),
        )

    if gate_type == "SQR":
        if target != "qubit":
            raise ValueError(f"Gate {gate_index} SQR must target 'qubit', got '{target}'.")
        theta_key = "theta" if "theta" in params else "thetas"
        phi_key = "phi" if "phi" in params else "phis"
        if theta_key not in params or phi_key not in params:
            raise KeyError(
                f"Gate {gate_index} SQR must include theta/phi arrays. Found keys: {sorted(params)}"
            )
        return SQRGate(
            index=gate_index,
            name=name,
            theta=tuple(_ensure_number_list(params[theta_key], gate_index, gate_type, "theta")),
            phi=tuple(_ensure_number_list(params[phi_key], gate_index, gate_type, "phi")),
        )

    raise ValueError(f"Gate {gate_index} has unsupported type '{gate_type}'.")


def load_gate_sequence(path_like: str | Path) -> tuple[Path, list[Gate]]:
    candidates = gate_path_candidates(path_like)
    existing = [path for path in candidates if path.exists()]
    if not existing:
        tried = "\n".join(f"  - {path}" for path in candidates)
        raise FileNotFoundError("Gate JSON file not found. Tried:\n" + tried)
    chosen = existing[0]
    raw = json.loads(chosen.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError(f"Gate JSON must be a list of gate dictionaries, got {type(raw).__name__}.")
    return chosen, [validate_gate_entry(entry, gate_index=i) for i, entry in enumerate(raw)]


def gate_summary_text(gate: Gate) -> str:
    if isinstance(gate, DisplacementGate):
        alpha = gate.alpha
        return f"alpha={alpha.real:+.3f}{alpha.imag:+.3f}j"
    if isinstance(gate, RotationGate):
        return f"theta={gate.theta:+.3f}, phi={gate.phi:+.3f}"
    active = int(np.count_nonzero(np.abs(np.asarray(gate.theta, dtype=float)) > 1.0e-12))
    return f"tones={len(gate.theta)}, active={active}"


def render_gate_table(gates: list[Gate], max_rows: int = 20) -> None:
    header = f"{'#':>3}  {'Type':<12} {'Target':<8} {'Name':<40} Params"
    print(header)
    print("-" * len(header))
    for gate in gates[:max_rows]:
        print(
            f"{gate.index:>3}  {gate.type:<12} {gate.target:<8} "
            f"{gate.name[:40]:<40} {gate_summary_text(gate)}"
        )
    if len(gates) > max_rows:
        print(f"... ({len(gates) - max_rows} more gates)")
