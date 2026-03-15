from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RunningStats:
    store_samples: bool = False
    count: int = 0
    mean: complex = 0.0 + 0.0j
    _m2: float = 0.0
    _samples: list[complex] = field(default_factory=list, repr=False)

    def update(self, value: complex) -> None:
        sample = complex(value)
        self.count += 1
        delta = sample - self.mean
        self.mean += delta / self.count
        centered = sample - self.mean
        self._m2 += float(np.real(delta * np.conj(centered)))
        if self.store_samples:
            self._samples.append(sample)

    @property
    def variance(self) -> float:
        if self.count <= 1:
            return 0.0
        return float(self._m2 / (self.count - 1))

    @property
    def stderr(self) -> float:
        if self.count <= 0:
            return float("nan")
        return float(np.sqrt(self.variance / self.count))

    @property
    def samples(self) -> np.ndarray | None:
        if not self.store_samples:
            return None
        return np.asarray(self._samples, dtype=np.complex128)

    def to_record(self) -> dict[str, Any]:
        return {
            "count": int(self.count),
            "mean": complex(self.mean),
            "variance": float(self.variance),
            "stderr": float(self.stderr),
            "store_samples": bool(self.store_samples),
        }


def bootstrap_mean(
    samples: np.ndarray,
    *,
    resamples: int = 1_000,
    seed: int | None = None,
) -> tuple[complex, float]:
    values = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if values.size == 0:
        raise ValueError("bootstrap_mean requires at least one sample.")
    rng = np.random.default_rng(seed)
    draws = np.empty(int(resamples), dtype=np.complex128)
    for idx in range(int(resamples)):
        choice = rng.integers(0, values.size, size=values.size)
        draws[idx] = np.mean(values[choice])
    return complex(np.mean(draws)), float(np.std(draws, ddof=1))

