from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim import (
    DomainRandomizer,
    HybridCQEDEnv,
    HybridEnvConfig,
    HybridSystemConfig,
    NormalPrior,
    PrimitiveActionSpace,
    QubitMeasurementSpec,
    ReducedDispersiveModelConfig,
    UniformPrior,
    benchmark_task_suite,
    build_observation_model,
    build_reward_model,
    coherent_state_preparation_task,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
FIGURE_PATH = OUTPUT_DIR / "rl_hybrid_control_rollout.png"
SUMMARY_PATH = OUTPUT_DIR / "rl_hybrid_control_rollout_summary.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return str(value)


def _build_environment() -> HybridCQEDEnv:
    nominal_chi = 2.0 * np.pi * (-2.25e6)
    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=nominal_chi,
            kerr=2.0 * np.pi * (-6.0e3),
            n_cav=10,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )
    randomizer = DomainRandomizer(
        model_priors_train={
            "chi": NormalPrior(nominal_chi, 2.0 * np.pi * 0.08e6),
            "kerr": NormalPrior(2.0 * np.pi * (-6.0e3), 2.0 * np.pi * 1.5e3),
        },
        measurement_priors_train={
            "iq_sigma": UniformPrior(0.03, 0.07),
        },
        drift_priors_train={
            "storage_amplitude_scale": NormalPrior(1.0, 0.03, low=0.9, high=1.1),
        },
        model_priors_eval={
            "chi": UniformPrior(nominal_chi - 2.0 * np.pi * 0.12e6, nominal_chi + 2.0 * np.pi * 0.12e6),
            "kerr": UniformPrior(2.0 * np.pi * (-9.0e3), 2.0 * np.pi * (-3.0e3)),
        },
        measurement_priors_eval={
            "iq_sigma": UniformPrior(0.05, 0.09),
        },
        drift_priors_eval={
            "storage_amplitude_scale": UniformPrior(0.88, 1.12),
        },
    )
    measurement = QubitMeasurementSpec(
        shots=256,
        iq_sigma=0.05,
        confusion_matrix=np.asarray([[0.97, 0.04], [0.03, 0.96]], dtype=float),
    )
    task = coherent_state_preparation_task(alpha=0.55 + 0.15j, duration=100.0e-9)
    action_space = PrimitiveActionSpace(primitives=("cavity_displacement", "wait", "measure"))
    return HybridCQEDEnv(
        HybridEnvConfig(
            system=system,
            task=task,
            action_space=action_space,
            observation_model=build_observation_model(
                "measurement_classifier_logits",
                action_dim=action_space.shape[0],
                history_length=2,
            ),
            reward_model=build_reward_model("measurement_proxy"),
            randomizer=randomizer,
            randomization_mode="train",
            measurement_spec=measurement,
            auto_measurement=True,
            episode_horizon=2,
            seed=17,
            diagnostics_wigner_points=31,
        )
    )


def _save_diagnostic_figure(diagnostics: dict[str, Any], output_path: Path) -> str:
    photon_distribution = np.asarray(diagnostics.get("photon_number_distribution", []), dtype=float)
    channel_payload = diagnostics.get("channels", {})
    first_channel = next(iter(channel_payload.values()), None)
    tlist = np.asarray(diagnostics.get("compiled_tlist", []), dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].bar(np.arange(photon_distribution.size), photon_distribution, color="#2a6f97")
    axes[0].set_xlabel("storage Fock level")
    axes[0].set_ylabel("population")
    axes[0].set_title("Final photon-number distribution")

    if first_channel is not None and tlist.size > 0:
        distorted = np.asarray(first_channel["distorted"], dtype=np.complex128)
        axes[1].plot(1.0e9 * tlist, distorted.real, label="I", linewidth=1.4)
        axes[1].plot(1.0e9 * tlist, distorted.imag, label="Q", linewidth=1.4)
        axes[1].legend()
    axes[1].set_xlabel("time [ns]")
    axes[1].set_ylabel("drive amplitude [rad/s]")
    axes[1].set_title("Compiled distorted pulse")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _write_summary(summary: dict[str, Any], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    return str(output_path)


def run_rollout_demo() -> dict[str, Any]:
    env = _build_environment()
    action_space = env.config.action_space

    initial_observation, initial_info = env.reset(seed=11)
    baseline = env.run_baseline(seed=11)
    diagnostics = env.render_diagnostics()

    rng = np.random.default_rng(23)
    random_actions = [action_space.sample(rng) for _ in range(2)]
    random_rollout = env.rollout(random_actions, seed=23)

    evaluation = env.estimate_metrics(
        env.task.baseline_actions,
        seeds=(101, 102, 103, 104),
        randomization_mode="eval",
    )

    figure_path = _save_diagnostic_figure(diagnostics, FIGURE_PATH)
    summary = {
        "task": env.task.name,
        "available_benchmarks": sorted(benchmark_task_suite().keys()),
        "python_version": sys.version.split()[0],
        "initial_observation_shape": tuple(int(value) for value in initial_observation.shape),
        "initial_randomization": initial_info["randomization"],
        "baseline_reward": float(baseline["total_reward"]),
        "baseline_metrics": baseline["final_metrics"],
        "random_rollout_reward": float(random_rollout["total_reward"]),
        "evaluation_summary": evaluation["summary"],
        "diagnostic_keys": sorted(diagnostics.keys()),
        "diagnostic_figure": figure_path,
    }
    summary["summary_path"] = _write_summary(summary, SUMMARY_PATH)
    return summary


def load_saved_summary(summary_path: Path | None = None) -> dict[str, Any]:
    resolved_path = SUMMARY_PATH if summary_path is None else Path(summary_path)
    return json.loads(resolved_path.read_text(encoding="utf-8"))


def main() -> None:
    summary = run_rollout_demo()
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()