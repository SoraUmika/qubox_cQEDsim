from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cqed_sim import FloquetConfig, FloquetProblem, SidebandDriveSpec, TransmonModeSpec, UniversalCQEDModel
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.floquet import build_target_drive_term, build_transmon_frequency_modulation_term, run_floquet_sweep, solve_floquet


def _single_transmon_model(*, omega_q: float, dim: int) -> UniversalCQEDModel:
    return UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=omega_q,
            dim=dim,
            alpha=0.0,
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(),
    )


def _time_and_memory(fn: Callable[[], Any], repeat: int) -> dict[str, Any]:
    timings: list[float] = []
    peaks_mb: list[float] = []
    for _ in range(max(1, int(repeat))):
        tracemalloc.start()
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        timings.append(float(elapsed))
        peaks_mb.append(float(peak / (1024.0 * 1024.0)))
    return {
        "best_s": float(min(timings)),
        "avg_s": float(sum(timings) / len(timings)),
        "runs_s": timings,
        "peak_python_memory_mb": float(max(peaks_mb)),
        "runs_peak_python_memory_mb": peaks_mb,
    }


def benchmark_single_transmon_cases(repeat: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, dim, n_time_samples, harmonic_cutoff in (
        ("small_dim2_t128", 2, 128, None),
        ("medium_dim4_t256", 4, 256, None),
        ("medium_dim4_t256_sambe3", 4, 256, 3),
    ):
        model = _single_transmon_model(omega_q=0.4, dim=dim)
        modulation_angular_frequency = 1.0
        problem = FloquetProblem(
            model=model,
            periodic_terms=(
                build_transmon_frequency_modulation_term(
                    model,
                    amplitude=0.7,
                    frequency=modulation_angular_frequency,
                    waveform="cos",
                ),
            ),
            period=2.0 * np.pi / modulation_angular_frequency,
        )
        config = FloquetConfig(
            n_time_samples=n_time_samples,
            sambe_harmonic_cutoff=harmonic_cutoff,
            sambe_n_time_samples=max(256, n_time_samples) if harmonic_cutoff is not None else None,
        )
        out[label] = {
            "dimension": int(problem.static_hamiltonian.shape[0]),
            "n_time_samples": int(n_time_samples),
            "sambe_harmonic_cutoff": harmonic_cutoff,
            **_time_and_memory(lambda p=problem, c=config: solve_floquet(p, c), repeat),
        }
    return out


def benchmark_sideband_case(repeat: int) -> dict[str, Any]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.05,
        omega_q=2.0 * np.pi * 6.25,
        alpha=2.0 * np.pi * (-0.25),
        chi=2.0 * np.pi * (-0.015),
        kerr=0.0,
        n_cav=3,
        n_tr=3,
    )
    frame = FrameSpec()
    center_frequency = model.sideband_transition_frequency(
        cavity_level=0,
        lower_level=0,
        upper_level=1,
        sideband="red",
        frame=frame,
    )
    sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
    problem = FloquetProblem(
        model=model,
        periodic_terms=(
            build_target_drive_term(
                model,
                sideband,
                amplitude=2.0 * np.pi * 0.03,
                frequency=center_frequency,
                waveform="cos",
            ),
        ),
        period=2.0 * np.pi / center_frequency,
    )
    config = FloquetConfig(n_time_samples=128, overlap_reference_time=0.17)
    return {
        "dimension": int(problem.static_hamiltonian.shape[0]),
        "n_time_samples": 128,
        **_time_and_memory(lambda p=problem, c=config: solve_floquet(p, c), repeat),
    }


def benchmark_sweep_cases(repeat: int) -> dict[str, Any]:
    model = _single_transmon_model(omega_q=0.4, dim=2)
    modulation_angular_frequency = 1.0
    sweep_out: dict[str, Any] = {}
    for label, points, n_time_samples in (
        ("transmon_sweep_9_points_t96", 9, 96),
        ("transmon_sweep_25_points_t128", 25, 128),
    ):
        amplitudes = np.linspace(0.0, 0.3, points)
        problems = [
            FloquetProblem(
                model=model,
                periodic_terms=(
                    build_target_drive_term(
                        model,
                        "qubit",
                        amplitude=float(amplitude),
                        frequency=modulation_angular_frequency,
                        waveform="cos",
                    ),
                ),
                period=2.0 * np.pi / modulation_angular_frequency,
            )
            for amplitude in amplitudes
        ]
        config = FloquetConfig(n_time_samples=n_time_samples, overlap_reference_time=0.13)
        sweep_out[label] = {
            "points": int(points),
            "dimension": int(problems[0].static_hamiltonian.shape[0]),
            "n_time_samples": int(n_time_samples),
            **_time_and_memory(
                lambda ps=problems, amps=amplitudes, cfg=config: run_floquet_sweep(ps, parameter_values=amps, config=cfg),
                repeat,
            ),
        }
    return sweep_out


def run_benchmarks(repeat: int) -> dict[str, Any]:
    return {
        "single_point": benchmark_single_transmon_cases(repeat),
        "sideband": benchmark_sideband_case(repeat),
        "sweeps": benchmark_sweep_cases(repeat),
        "notes": {
            "memory_metric": "peak Python allocation measured with tracemalloc; excludes some native-library allocations.",
            "interpretation": "Treat these results as machine-specific guidance for interactive versus batch-sized Floquet workloads.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Floquet-specific cqed_sim benchmarks.")
    parser.add_argument("--repeat", type=int, default=3, help="Repeat count for each benchmark case.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    results = run_benchmarks(repeat=max(1, int(args.repeat)))
    payload = json.dumps(results, indent=2)
    print(payload)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()