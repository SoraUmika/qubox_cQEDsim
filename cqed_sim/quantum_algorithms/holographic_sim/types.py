from __future__ import annotations

from typing import Any, Protocol, Sequence, TypeAlias

import numpy as np


class SupportsFull(Protocol):
    def full(self) -> Any:
        ...


DenseLike: TypeAlias = np.ndarray | Sequence[complex] | Sequence[Sequence[complex]] | SupportsFull
ObservableInput: TypeAlias = DenseLike | "PhysicalObservable"

