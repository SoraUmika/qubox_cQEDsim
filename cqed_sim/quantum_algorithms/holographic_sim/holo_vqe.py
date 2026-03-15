from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .results import CorrelatorEstimate, ExactCorrelatorResult
from .sampler import HolographicSampler
from .schedules import ObservableSchedule
from .utils import json_ready


@dataclass(frozen=True)
class EnergyTerm:
    """One local energy contribution represented by a holographic correlator."""

    coefficient: complex
    schedule: ObservableSchedule | Sequence[Any]
    label: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "coefficient", complex(self.coefficient))
        if not isinstance(self.schedule, ObservableSchedule):
            object.__setattr__(self, "schedule", ObservableSchedule(self.schedule))

    def to_record(self) -> dict[str, Any]:
        assert isinstance(self.schedule, ObservableSchedule)
        return {
            "label": self.label,
            "coefficient": complex(self.coefficient),
            "schedule": self.schedule.to_record(),
        }


@dataclass
class EnergyEstimate:
    """Scaffold result for future holoVQE optimization loops."""

    energy: complex
    stderr: float
    exact: bool
    term_records: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return json_ready(
            {
                "energy": complex(self.energy),
                "stderr": float(self.stderr),
                "exact": bool(self.exact),
                "terms": dict(self.term_records),
            }
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_record(), indent=2), encoding="utf-8")
        return path


@dataclass(frozen=True)
class HoloVQEObjective:
    """Minimal energy-objective wrapper built from correlator schedules."""

    terms: Sequence[EnergyTerm | Mapping[str, Any]]
    label: str | None = None

    def __post_init__(self) -> None:
        normalized: list[EnergyTerm] = []
        for idx, term in enumerate(self.terms):
            if isinstance(term, EnergyTerm):
                normalized.append(term)
                continue
            if "coefficient" not in term or "schedule" not in term:
                raise ValueError(f"Energy term {idx} must provide 'coefficient' and 'schedule'.")
            normalized.append(
                EnergyTerm(
                    coefficient=complex(term["coefficient"]),
                    schedule=term["schedule"],
                    label=term.get("label"),
                )
            )
        object.__setattr__(self, "terms", tuple(normalized))

    def estimate(
        self,
        sampler: HolographicSampler,
        *,
        shots: int = 1_000,
        exact: bool = False,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> EnergyEstimate:
        energy = 0.0 + 0.0j
        stderr_sq = 0.0
        records: dict[str, dict[str, Any]] = {}
        for idx, term in enumerate(self.terms):
            label = term.label if term.label is not None else f"term_{idx}"
            if exact:
                estimate: CorrelatorEstimate | ExactCorrelatorResult = sampler.enumerate_correlator(term.schedule)
                term_stderr = 0.0
            else:
                estimate = sampler.sample_correlator(
                    term.schedule,
                    shots=shots,
                    seed=None if seed is None else seed + idx,
                    show_progress=show_progress and idx == 0,
                )
                term_stderr = float(estimate.stderr)
            energy += complex(term.coefficient) * complex(estimate.mean)
            stderr_sq += abs(complex(term.coefficient)) ** 2 * term_stderr**2
            records[label] = {
                "coefficient": complex(term.coefficient),
                "estimate": estimate.to_record(),
            }
        return EnergyEstimate(
            energy=complex(energy),
            stderr=float(np.sqrt(stderr_sq)),
            exact=bool(exact),
            term_records=records,
        )
