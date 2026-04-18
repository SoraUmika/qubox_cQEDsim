"""Calibration → System-ID → RL training-loop pipeline.

Demonstrates the full device-characterisation-to-policy workflow:
  1. Run simulated calibration targets (spectroscopy, Rabi, T1)
  2. Convert the fitted results into CalibrationEvidence priors
  3. Convert evidence to a DomainRandomizer via system_id
  4. Build a HybridCQEDEnv wired to the randomizer
  5. Execute a short random rollout as a stand-in for RL training

This example focuses on wiring, not learning: any policy could be
substituted in step 5.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cqed_sim import (
    DispersiveTransmonCavityModel,
    HybridCQEDEnv,
    HybridEnvConfig,
    HybridSystemConfig,
    CalibrationEvidence,
    NormalPrior,
    PrimitiveActionSpace,
    ReducedDispersiveModelConfig,
    build_observation_model,
    build_reward_model,
    evidence_from_fit,
    fock_state_preparation_task,
    fit_rabi_trace,
    fit_ramsey_trace,
    fit_spectroscopy_trace,
    fit_t1_trace,
    merge_calibration_evidence,
)
from cqed_sim.calibration_targets import run_rabi, run_ramsey, run_spectroscopy, run_t1
from cqed_sim.system_id import randomizer_from_calibration


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "calibration_systemid_rl_summary.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return str(value)


# ── 1.  Simulated calibration ────────────────────────────────────
def run_calibration(model: DispersiveTransmonCavityModel) -> dict[str, Any]:
    """Execute standard characterisation targets and return results."""
    freq_span = np.linspace(
        model.omega_q / (2 * np.pi) - 100.0e6,
        model.omega_q / (2 * np.pi) + 100.0e6,
        201,
    )
    drive_frequencies = 2 * np.pi * freq_span

    rabi_amplitudes = np.linspace(0.0, 2.0, 201)
    delays = np.linspace(0.0, 60.0e-6, 201)

    targets = {
        "spectroscopy": run_spectroscopy(model, drive_frequencies),
        "rabi": run_rabi(model, rabi_amplitudes, duration=40.0e-9, omega_scale=2.0 * np.pi * 12.0e6),
        "ramsey": run_ramsey(model, delays, detuning=2.0 * np.pi * 0.6e6, t2_star=9.0e-6),
        "t1": run_t1(model, delays, t1=24.0e-6),
    }
    fits = {
        "spectroscopy": fit_spectroscopy_trace(
            targets["spectroscopy"].raw_data["drive_frequencies"],
            targets["spectroscopy"].raw_data["ground_response"],
        ),
        "rabi": fit_rabi_trace(
            targets["rabi"].raw_data["amplitudes"],
            targets["rabi"].raw_data["excited_population"],
            duration=targets["rabi"].fitted_parameters["duration"],
        ),
        "ramsey": fit_ramsey_trace(
            targets["ramsey"].raw_data["delays"],
            targets["ramsey"].raw_data["excited_population"],
        ),
        "t1": fit_t1_trace(
            targets["t1"].raw_data["delays"],
            targets["t1"].raw_data["excited_population"],
        ),
    }

    for name, result in fits.items():
        print(f"  {name.capitalize():<13}: {result.fitted_parameters}")
    return {"targets": targets, "fits": fits}


# ── 2.  Build CalibrationEvidence ─────────────────────────────────
def build_evidence(calibration: dict[str, Any]) -> CalibrationEvidence:
    """Wrap calibration results as prior distributions."""
    fits = calibration["fits"]
    fitted_evidence = merge_calibration_evidence(
        evidence_from_fit(
            fits["spectroscopy"],
            category="model",
            parameter_map={"omega_peak": "omega_q"},
            bounds={"omega_q": (0.0, None)},
            min_sigma={"omega_q": 2.0 * np.pi * 0.25e6},
        ),
        evidence_from_fit(
            fits["t1"],
            category="noise",
            bounds={"t1": (0.0, None)},
            min_sigma={"t1": 0.5e-6},
        ),
        evidence_from_fit(
            fits["ramsey"],
            category="noise",
            parameter_map={"t2_star": "t2_star"},
            bounds={"t2_star": (0.0, None)},
            min_sigma={"t2_star": 0.5e-6},
        ),
    )
    manual_model_evidence = CalibrationEvidence(
        model_posteriors={
            "chi": NormalPrior(mean=2 * np.pi * (-2.25e6), sigma=2 * np.pi * 0.08e6),
            "kerr": NormalPrior(mean=2 * np.pi * (-6.0e3), sigma=2 * np.pi * 1.5e3),
        },
    )
    return merge_calibration_evidence(
        fitted_evidence,
        manual_model_evidence,
        notes={"source": "simulated calibration pipeline example"},
    )


# ── 3 & 4.  Environment construction ─────────────────────────────
def build_env(evidence: CalibrationEvidence) -> HybridCQEDEnv:
    """Wire calibration evidence into the RL environment."""
    randomizer = randomizer_from_calibration(evidence)

    nominal_chi = 2 * np.pi * (-2.25e6)
    system = HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2 * np.pi * 5.0e9,
            omega_q=2 * np.pi * 6.0e9,
            alpha=2 * np.pi * (-220.0e6),
            chi=nominal_chi,
            kerr=2 * np.pi * (-6.0e3),
            n_cav=8,
            n_tr=2,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )

    task = fock_state_preparation_task(cavity_level=1)
    action_space = PrimitiveActionSpace(
        primitives=("qubit_gaussian", "cavity_displacement", "wait"),
    )

    return HybridCQEDEnv(
        HybridEnvConfig(
            system=system,
            task=task,
            action_space=action_space,
            observation_model=build_observation_model(
                "ideal_summary",
                action_dim=action_space.shape[0],
            ),
            reward_model=build_reward_model("state_preparation"),
            randomizer=randomizer,
            randomization_mode="train",
            episode_horizon=task.horizon,
            seed=42,
        )
    )


# ── 5.  Demo rollout ─────────────────────────────────────────────
def demo_rollout(env: HybridCQEDEnv, n_episodes: int = 5) -> list[dict[str, Any]]:
    """Run short random episodes and collect episode summaries."""
    rng = np.random.default_rng(99)
    action_space = env.config.action_space
    episodes: list[dict[str, Any]] = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31)))
        total_reward = 0.0
        steps = 0
        terminated = truncated = False

        while not (terminated or truncated):
            action = action_space.sample(rng)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1

        metrics = info.get("metrics", {})
        episodes.append({
            "episode": ep,
            "steps": steps,
            "total_reward": total_reward,
            "state_fidelity": float(metrics.get("state_fidelity", 0.0)),
            "leakage_worst": float(metrics.get("leakage_worst", 0.0)),
            "ancilla_return": float(metrics.get("ancilla_return", 0.0)),
            "randomization": info.get("randomization", {}),
        })
        print(
            f"  Episode {ep}: reward={total_reward:+.4f}  "
            f"fidelity={episodes[-1]['state_fidelity']:.4f}  "
            f"steps={steps}"
        )

    return episodes


# ── 6.  Baseline evaluation ──────────────────────────────────────
def run_baseline_evaluation(env: HybridCQEDEnv) -> dict[str, Any]:
    """Run the task's built-in baseline for comparison."""
    baseline = env.run_baseline(seed=55)
    print(
        f"  Baseline reward   : {baseline['total_reward']:+.4f}\n"
        f"  Baseline fidelity : {baseline['final_metrics'].get('state_fidelity', 'N/A')}"
    )
    return baseline


def main() -> None:
    # 1. Calibrate
    print("=" * 60)
    print("Step 1: Simulated calibration")
    print("=" * 60)
    model = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 5.0e9,
        omega_q=2 * np.pi * 6.0e9,
        alpha=2 * np.pi * (-220.0e6),
        chi=2 * np.pi * (-2.25e6),
        kerr=2 * np.pi * (-6.0e3),
        n_cav=8,
        n_tr=2,
    )
    calibration = run_calibration(model)
    print()

    # 2. Build evidence
    print("=" * 60)
    print("Step 2: Build CalibrationEvidence from fitted parameters")
    print("=" * 60)
    evidence = build_evidence(calibration)
    print(f"  Model posteriors : {sorted(evidence.model_posteriors.keys())}")
    print(f"  Noise posteriors : {sorted(evidence.noise_posteriors.keys())}")
    print()

    # 3/4. Build environment
    print("=" * 60)
    print("Step 3: Build RL environment with calibration-informed randomizer")
    print("=" * 60)
    env = build_env(evidence)
    print(f"  Task             : {env.task.name}")
    print(f"  Episode horizon  : {env.config.episode_horizon}")
    print(f"  Action dim       : {env.config.action_space.shape}")
    print()

    # 5. Random rollout
    print("=" * 60)
    print("Step 4: Random-agent rollout (stand-in for RL training)")
    print("=" * 60)
    episodes = demo_rollout(env, n_episodes=5)
    print()

    # 6. Baseline comparison
    print("=" * 60)
    print("Step 5: Baseline evaluation")
    print("=" * 60)
    baseline = run_baseline_evaluation(env)
    print()

    # Summary
    summary = {
        "calibration": {
            section: {
                name: {
                    "fitted": result.fitted_parameters,
                    "uncertainties": result.uncertainties,
                }
                for name, result in section_results.items()
            }
            for section, section_results in calibration.items()
        },
        "evidence_keys": {
            "model_posteriors": sorted(evidence.model_posteriors.keys()),
            "noise_posteriors": sorted(evidence.noise_posteriors.keys()),
        },
        "task": env.task.name,
        "episodes": episodes,
        "baseline_reward": float(baseline["total_reward"]),
        "baseline_fidelity": float(baseline["final_metrics"].get("state_fidelity", 0.0)),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=_json_default), encoding="utf-8")
    print(f"Summary saved to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
