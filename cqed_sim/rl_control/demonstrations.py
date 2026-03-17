from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


@dataclass(frozen=True)
class DemonstrationRollout:
    task_name: str
    actions: tuple[Any, ...]
    total_reward: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_records(self) -> list[dict[str, Any]]:
        return [
            {
                "task": self.task_name,
                "step": int(index),
                "action": action,
            }
            for index, action in enumerate(self.actions)
        ]


def scripted_demonstration(task: Any) -> DemonstrationRollout:
    return DemonstrationRollout(task_name=str(task.name), actions=tuple(task.baseline_actions), metadata={"source": "baseline"})


def rollout_records(task: Any, actions: Sequence[Any]) -> list[dict[str, Any]]:
    return DemonstrationRollout(task_name=str(task.name), actions=tuple(actions)).to_records()


__all__ = ["DemonstrationRollout", "scripted_demonstration", "rollout_records"]