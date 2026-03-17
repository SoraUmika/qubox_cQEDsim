from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from cqed_sim.unitary_synthesis import ExecutionOptions, PrimitiveGate, Subspace, UnitarySynthesizer


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def parameterized_rotation(params, model) -> np.ndarray:
    return rotation_y(float(params["theta"]))


def make_primitives() -> list[PrimitiveGate]:
    return [
        PrimitiveGate(
            name=f"ry_{index}",
            duration=20.0e-9,
            matrix=parameterized_rotation,
            parameters={"theta": 0.1 * (index + 1), "duration": 20.0e-9},
            parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
            hilbert_dim=2,
        )
        for index in range(4)
    ]


@dataclass
class BenchmarkRow:
    label: str
    runtime_s: float
    objective: float
    fidelity: float | None
    duration_s: float
    effective_gate_count: float
    execution_engine: str


def run_case(label: str, *, execution: ExecutionOptions, multistart: int, parallel: dict | None) -> BenchmarkRow:
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(2, range(2)),
        primitives=make_primitives(),
        target=rotation_y(np.pi / 2.0),
        optimizer="powell",
        optimize_times=False,
        execution=execution,
        parallel=parallel,
        seed=11,
    )
    t0 = time.perf_counter()
    result = synth.fit(maxiter=30, multistart=multistart, init_guess="random")
    elapsed = time.perf_counter() - t0
    metrics = result.report["metrics"]
    return BenchmarkRow(
        label=label,
        runtime_s=float(elapsed),
        objective=float(result.objective),
        fidelity=None if np.isnan(metrics.get("fidelity", np.nan)) else float(metrics["fidelity"]),
        duration_s=float(result.sequence.total_duration()),
        effective_gate_count=float(metrics.get("gate_count_metric", np.nan)),
        execution_engine=str(result.report["execution"].get("selected_engine", "legacy")),
    )


def main() -> None:
    rows = [
        run_case(
            "legacy_serial",
            execution=ExecutionOptions(engine="legacy"),
            multistart=1,
            parallel={"enabled": False, "n_jobs": 1, "backend": "multiprocessing"},
        ),
        run_case(
            "numpy_fast_serial",
            execution=ExecutionOptions(engine="numpy"),
            multistart=1,
            parallel={"enabled": False, "n_jobs": 1, "backend": "multiprocessing"},
        ),
        run_case(
            "jax_requested",
            execution=ExecutionOptions(engine="jax", fallback_engine="numpy"),
            multistart=1,
            parallel={"enabled": False, "n_jobs": 1, "backend": "multiprocessing"},
        ),
        run_case(
            "numpy_fast_parallel_multistart",
            execution=ExecutionOptions(engine="numpy"),
            multistart=4,
            parallel={"enabled": True, "n_jobs": 2, "backend": "multiprocessing"},
        ),
    ]
    print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()