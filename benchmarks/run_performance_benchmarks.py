from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import qutip as qt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cqed_sim.calibration import conditional_loss
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.io import SQRGate
from cqed_sim.pulses import HardwareConfig, Pulse
from cqed_sim.pulses.builders import build_sqr_multitone_pulse
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import SimulationConfig, prepare_simulation, simulate_batch, simulate_sequence
from cqed_sim.tomo import QubitPulseCal, run_all_xy, run_fock_resolved_tomo
from cqed_sim.unitary_synthesis import GateSequence, QubitRotation, Subspace, UnitarySynthesizer


def _time_callable(fn: Callable[[], Any], repeat: int) -> dict[str, Any]:
    runs: list[float] = []
    for _ in range(max(1, int(repeat))):
        t0 = time.perf_counter()
        fn()
        runs.append(time.perf_counter() - t0)
    return {
        "best_s": float(min(runs)),
        "avg_s": float(sum(runs) / len(runs)),
        "runs_s": [float(value) for value in runs],
    }


def _gaussian(sigma: float) -> Callable[[np.ndarray], np.ndarray]:
    def envelope(t_rel: np.ndarray) -> np.ndarray:
        return gaussian_envelope(t_rel, sigma=sigma)

    return envelope


def benchmark_core_paths(repeat: int) -> dict[str, Any]:
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-2.0 * np.pi * 0.22,
        chi=2.0 * np.pi * 0.015,
        kerr=-2.0 * np.pi * 0.003,
        n_cav=5,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    initial = model.basis_state(0, 0)
    compiler = SequenceCompiler(
        dt=0.01,
        hardware={
            "q": HardwareConfig(zoh_samples=4, lowpass_bw=12.0, amplitude_bits=12),
            "c": HardwareConfig(zoh_samples=3, lowpass_bw=8.0),
        },
    )
    pulses: list[Pulse] = []
    for idx in range(20):
        channel = "q" if idx % 2 == 0 else "c"
        pulses.append(
            Pulse(
                channel,
                0.03 * idx,
                0.12,
                _gaussian(0.16 + 0.005 * (idx % 3)),
                amp=0.12 + 0.01 * (idx % 4),
                phase=0.1 * idx,
                carrier=2.0 * np.pi * (0.5 if channel == "q" else 0.2),
            )
        )

    compiled = compiler.compile(pulses, t_end=0.8)
    config = SimulationConfig(frame=frame, max_step=0.01)
    drive_ops = {"q": "qubit", "c": "cavity"}
    session = prepare_simulation(model, compiled, drive_ops, config=config, e_ops={})

    return {
        "compile_heavy_20": _time_callable(lambda: compiler.compile(pulses, t_end=0.8), repeat=max(3, repeat)),
        "repeat_simulate_20_raw": _time_callable(
            lambda: [simulate_sequence(model, compiled, initial, drive_ops, config=config, e_ops={}) for _ in range(20)],
            repeat=max(2, repeat),
        ),
        "repeat_simulate_20_prepared": _time_callable(
            lambda: [session.run(initial) for _ in range(20)],
            repeat=max(2, repeat),
        ),
    }


def benchmark_protocol_paths(repeat: int) -> dict[str, Any]:
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-2 * np.pi * 0.22,
        chi=2 * np.pi * 0.015,
        kerr=-2 * np.pi * 0.003,
        n_cav=8,
        n_tr=2,
    )
    cal = QubitPulseCal.nominal()
    rho_prep = qt.tensor(qt.basis(model.n_tr, 0).proj(), qt.coherent_dm(model.n_cav, 0.4))
    return {
        "run_all_xy": _time_callable(lambda: run_all_xy(model, cal, dt_ns=0.2), repeat=max(2, repeat)),
        "run_fock_resolved_tomo_n2": _time_callable(
            lambda: run_fock_resolved_tomo(
                model,
                state_prep=lambda rho=rho_prep: rho,
                n_max=2,
                cal=cal,
                tag_duration_ns=80.0,
                tag_amp=0.02,
                dt_ns=1.0,
                ideal_tag=False,
            ),
            repeat=max(1, min(2, repeat)),
        ),
    }


def benchmark_calibration_paths(repeat: int) -> dict[str, Any]:
    sqr_model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2.0 * np.pi * (-2.84e6),
        kerr=0.0,
        n_cav=6,
        n_tr=2,
    )
    sqr_config = {
        "duration_sqr_s": 1.0e-6,
        "sqr_sigma_fraction": 1.0 / 6.0,
        "sqr_theta_cutoff": 1.0e-10,
        "use_rotating_frame": True,
        "omega_c_hz": 0.0,
        "omega_q_hz": 0.0,
    }
    gate = SQRGate(index=0, name="bench_sqr", theta=tuple([0.1, 0.2, 0.1, 0.0, 0.0, 0.0]), phi=tuple([0.0] * 6))
    conditional_config = {
        "duration_sqr_s": 4.0e-7,
        "sqr_sigma_fraction": 1.0 / 6.0,
        "dt_s": 2.0e-9,
        "max_step_s": 2.0e-9,
        "qutip_nsteps_sqr_calibration": 20000,
    }
    params = np.array([0.02, -0.03, 2.0 * np.pi * 1.0e5], dtype=float)
    return {
        "build_sqr_multitone_pulse": _time_callable(lambda: build_sqr_multitone_pulse(gate, sqr_model, sqr_config), repeat=max(3, repeat)),
        "conditional_loss_12": _time_callable(
            lambda: [conditional_loss(params, n, 0.3, 0.1, conditional_config) for n in range(12)],
            repeat=max(2, repeat),
        ),
    }


def benchmark_synthesis_path(repeat: int) -> dict[str, Any]:
    subspace = Subspace.qubit_cavity_block(n_match=2)
    target = GateSequence(gates=[QubitRotation(name="t", theta=np.pi / 3, phi=0.2, duration=90e-9)], n_cav=3).unitary("ideal")

    def fit_small() -> None:
        synth = UnitarySynthesizer(
            subspace=subspace,
            backend="pulse",
            gateset=["QubitRotation"],
            optimize_times=True,
            time_policy={"default": {"optimize": True, "bounds": (20e-9, 120e-9), "init": 60e-9}},
            seed=123,
            parallel={"enabled": False, "n_jobs": 1, "backend": "multiprocessing"},
        )
        synth.fit(target=target, init_guess="random", multistart=1, maxiter=20)

    return {
        "unitary_fit_small": _time_callable(fit_small, repeat=max(1, min(2, repeat))),
    }


def benchmark_parallel_path() -> dict[str, Any]:
    def square(t_rel: np.ndarray) -> np.ndarray:
        return np.ones_like(t_rel, dtype=np.complex128)

    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    compiled = SequenceCompiler(dt=0.01).compile([Pulse("q", 0.0, 1.0, square, amp=np.pi / 4.0)], t_end=1.1)
    session = prepare_simulation(model, compiled, {"q": "qubit"}, config=SimulationConfig(), e_ops={})
    out: dict[str, Any] = {}
    for count in (4, 8, 16):
        states = [model.basis_state(idx % 2, 0) for idx in range(count)]
        serial_t0 = time.perf_counter()
        simulate_batch(session, states, max_workers=1)
        serial_s = time.perf_counter() - serial_t0

        parallel_t0 = time.perf_counter()
        simulate_batch(session, states, max_workers=2, mp_context="spawn")
        parallel_s = time.perf_counter() - parallel_t0
        out[str(count)] = {
            "serial_s": float(serial_s),
            "parallel_s": float(parallel_s),
            "speedup": float(serial_s / parallel_s) if parallel_s > 0.0 else float("nan"),
        }
    return out


def run_benchmarks(repeat: int) -> dict[str, Any]:
    return {
        "core": benchmark_core_paths(repeat),
        "protocols": benchmark_protocol_paths(repeat),
        "calibration": benchmark_calibration_paths(repeat),
        "synthesis": benchmark_synthesis_path(repeat),
        "parallel": benchmark_parallel_path(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible cqed_sim performance benchmarks.")
    parser.add_argument("--repeat", type=int, default=3, help="Default repeat count for scalar benchmarks.")
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
