from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .parameterizations import ControlSchedule
from .utils import json_ready


@dataclass(frozen=True)
class GrapeIterationRecord:
    evaluation: int
    objective: float
    gradient_norm: float
    elapsed_s: float
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ControlResult:
    success: bool
    message: str
    schedule: ControlSchedule
    objective_value: float
    metrics: dict[str, Any]
    system_metrics: tuple[dict[str, Any], ...]
    history: list[GrapeIterationRecord] = field(default_factory=list)
    nominal_final_unitary: np.ndarray | None = None
    optimizer_summary: dict[str, Any] = field(default_factory=dict)
    command_values: np.ndarray | None = None
    physical_values: np.ndarray | None = None
    time_boundaries_s: np.ndarray | None = None
    parameterization_metrics: dict[str, Any] = field(default_factory=dict)
    hardware_metrics: dict[str, Any] = field(default_factory=dict)
    hardware_reports: tuple[dict[str, Any], ...] = ()
    backend: str = "unknown"

    def to_pulses(self, *, waveform: str = "command"):
        mode = str(waveform).lower()
        if mode == "command":
            waveform_values = self.schedule.command_values() if self.command_values is None else self.command_values
        elif mode == "physical":
            if self.physical_values is not None:
                waveform_values = self.physical_values
            elif self.command_values is not None:
                waveform_values = self.command_values
            else:
                waveform_values = self.schedule.command_values()
        else:
            raise ValueError("ControlResult.to_pulses waveform must be 'command' or 'physical'.")
        return self.schedule.to_pulses(waveform_values=np.asarray(waveform_values, dtype=float))

    def evaluate_with_simulator(self, problem, **kwargs):
        from .evaluation import evaluate_control_with_simulator

        return evaluate_control_with_simulator(problem, self.schedule, **kwargs)

    def to_payload(self) -> dict[str, Any]:
        command_values = self.schedule.command_values() if self.command_values is None else np.asarray(self.command_values, dtype=float)
        physical_values = command_values if self.physical_values is None else np.asarray(self.physical_values, dtype=float)
        payload = {
            "backend": str(self.backend),
            "success": bool(self.success),
            "message": str(self.message),
            "objective_value": float(self.objective_value),
            "time_grid_s": [float(value) for value in self.schedule.parameterization.time_grid.step_durations_s],
            "time_boundaries_s": None if self.time_boundaries_s is None else np.asarray(self.time_boundaries_s, dtype=float),
            "control_terms": [term.name for term in self.schedule.parameterization.control_terms],
            "parameter_values": np.asarray(self.schedule.values, dtype=float),
            "command_values": command_values,
            "physical_values": physical_values,
            "metrics": self.metrics,
            "system_metrics": list(self.system_metrics),
            "parameterization_metrics": dict(self.parameterization_metrics),
            "hardware_metrics": dict(self.hardware_metrics),
            "hardware_reports": list(self.hardware_reports),
            "history": [
                {
                    "evaluation": int(record.evaluation),
                    "objective": float(record.objective),
                    "gradient_norm": float(record.gradient_norm),
                    "elapsed_s": float(record.elapsed_s),
                    "metrics": record.metrics,
                }
                for record in self.history
            ],
            "nominal_final_unitary": None if self.nominal_final_unitary is None else np.asarray(self.nominal_final_unitary, dtype=np.complex128),
            "optimizer_summary": dict(self.optimizer_summary),
        }
        return json_ready(payload)

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_payload(), indent=2), encoding="utf-8")
        return output_path


@dataclass
class GrapeResult(ControlResult):
    backend: str = "grape"


__all__ = ["GrapeIterationRecord", "ControlResult", "GrapeResult"]