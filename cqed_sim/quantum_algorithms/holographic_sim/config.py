from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import json_ready, normalize_state_vector


@dataclass(frozen=True)
class BurnInConfig:
    steps: int = 0
    label: str | None = None

    def __post_init__(self) -> None:
        if int(self.steps) < 0:
            raise ValueError("BurnInConfig.steps must be non-negative.")
        object.__setattr__(self, "steps", int(self.steps))

    def to_record(self) -> dict[str, Any]:
        return {"steps": int(self.steps), "label": self.label}


@dataclass(frozen=True)
class BoundaryCondition:
    right_state: np.ndarray | None = None
    postselect: bool = False
    label: str | None = None

    def __post_init__(self) -> None:
        if self.right_state is None:
            return
        right_state = normalize_state_vector(self.right_state)
        object.__setattr__(self, "right_state", right_state)

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "right_state": None if self.right_state is None else np.asarray(self.right_state, dtype=np.complex128),
                "postselect": bool(self.postselect),
                "label": self.label,
            }
        )


@dataclass(frozen=True)
class SamplingConfig:
    shots: int = 1_000
    seed: int | None = None
    show_progress: bool = False
    store_samples: bool = False

    def __post_init__(self) -> None:
        if int(self.shots) <= 0:
            raise ValueError("SamplingConfig.shots must be positive.")
        object.__setattr__(self, "shots", int(self.shots))

    def to_record(self) -> dict[str, Any]:
        return {
            "shots": int(self.shots),
            "seed": None if self.seed is None else int(self.seed),
            "show_progress": bool(self.show_progress),
            "store_samples": bool(self.store_samples),
        }

