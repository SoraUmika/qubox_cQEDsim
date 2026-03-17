from __future__ import annotations

import numpy as np

from cqed_sim import (
    DomainRandomizer,
    FixedPrior,
    FullPulseModelConfig,
    HybridCQEDEnv,
    HybridEnvConfig,
    HybridSystemConfig,
    NormalPrior,
    ParametricPulseActionSpace,
    PrimitiveActionSpace,
    QubitMeasurementSpec,
    ReducedDispersiveModelConfig,
    UniformPrior,
    build_observation_model,
    build_reward_model,
    coherent_state_preparation_task,
    conditional_phase_gate_task,
)


def _reduced_system() -> HybridSystemConfig:
    return HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=2.0 * np.pi * (-5.0e3),
            n_cav=6,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )


def _full_system() -> HybridSystemConfig:
    return HybridSystemConfig(
        regime="full_pulse",
        full_model=FullPulseModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.0e9,
            alpha=2.0 * np.pi * (-200.0e6),
            exchange_g=2.0 * np.pi * 0.4e6,
            kerr=2.0 * np.pi * (-3.0e3),
            cross_kerr=2.0 * np.pi * (-1.0e6),
            n_cav=4,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )


def _measurement_spec() -> QubitMeasurementSpec:
    return QubitMeasurementSpec(
        shots=64,
        iq_sigma=0.04,
        confusion_matrix=np.asarray([[0.96, 0.05], [0.04, 0.95]], dtype=float),
    )


def test_parametric_action_space_clips_extreme_values() -> None:
    space = ParametricPulseActionSpace(family="hybrid_block")
    action = space.parse(np.full(space.shape[0], 1.0e9, dtype=float))
    flattened = space.flatten(action)

    assert flattened.shape == space.shape
    assert flattened[3] <= space.high[3]
    assert flattened[8] <= space.high[8]
    assert flattened[-1] == 1.0


def test_domain_randomizer_uses_train_and_eval_priors() -> None:
    randomizer = DomainRandomizer(
        model_priors_train={"chi": FixedPrior(-1.0)},
        model_priors_eval={"chi": FixedPrior(-2.0)},
        measurement_priors_train={"iq_sigma": FixedPrior(0.03)},
        measurement_priors_eval={"iq_sigma": FixedPrior(0.08)},
    )

    train = randomizer.sample(seed=5, mode="train")
    eval_sample = randomizer.sample(seed=5, mode="eval")

    assert train.model_overrides["chi"] == -1.0
    assert eval_sample.model_overrides["chi"] == -2.0
    assert train.measurement_overrides["iq_sigma"] == 0.03
    assert eval_sample.measurement_overrides["iq_sigma"] == 0.08


def test_reset_is_deterministic_for_fixed_seed() -> None:
    randomizer = DomainRandomizer(
        model_priors_train={"chi": FixedPrior(2.0 * np.pi * (-2.1e6))},
        measurement_priors_train={"iq_sigma": FixedPrior(0.05)},
        drift_priors_train={"storage_amplitude_scale": FixedPrior(1.0)},
    )
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_reduced_system(),
            task=coherent_state_preparation_task(alpha=0.25 + 0.0j, duration=40.0e-9),
            action_space=PrimitiveActionSpace(primitives=("cavity_displacement", "wait", "measure")),
            observation_model=build_observation_model("measurement_iq"),
            reward_model=build_reward_model("state"),
            randomizer=randomizer,
            measurement_spec=_measurement_spec(),
            auto_measurement=True,
            episode_horizon=2,
            seed=9,
        )
    )

    obs_a, info_a = env.reset(seed=12)
    obs_b, info_b = env.reset(seed=12)

    assert np.allclose(obs_a, obs_b)
    assert info_a["randomization"] == info_b["randomization"]


def test_reduced_env_baseline_and_alias_methods_run() -> None:
    randomizer = DomainRandomizer(
        model_priors_train={"chi": NormalPrior(2.0 * np.pi * (-2.2e6), 2.0 * np.pi * 0.02e6)},
        measurement_priors_train={"iq_sigma": UniformPrior(0.03, 0.05)},
        drift_priors_train={"storage_amplitude_scale": UniformPrior(0.97, 1.03)},
    )
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_reduced_system(),
            task=coherent_state_preparation_task(alpha=0.35 + 0.10j, duration=60.0e-9),
            action_space=PrimitiveActionSpace(primitives=("cavity_displacement", "wait", "measure")),
            observation_model=build_observation_model("ideal_summary"),
            reward_model=build_reward_model("state"),
            randomizer=randomizer,
            measurement_spec=_measurement_spec(),
            auto_measurement=True,
            episode_horizon=2,
            seed=7,
        )
    )

    rollout = env.run_baseline(seed=4)
    diagnostics = env.render_diagnostics()
    metrics = env.estimate_metrics(env.task.baseline_actions, seeds=(21, 22), randomization_mode="eval")

    assert np.isfinite(rollout["total_reward"])
    assert 0.0 <= float(rollout["final_metrics"]["state_fidelity"]) <= 1.0
    assert "joint_state" in diagnostics
    assert "channels" in diagnostics
    assert "reward" in metrics["summary"]


def test_reduced_unitary_task_exposes_process_metrics() -> None:
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_reduced_system(),
            task=conditional_phase_gate_task(),
            action_space=PrimitiveActionSpace(primitives=("wait", "measure")),
            observation_model=build_observation_model("gate_metrics"),
            reward_model=build_reward_model("gate"),
            episode_horizon=1,
            seed=13,
        )
    )

    observation, _info = env.reset(seed=6)
    rollout = env.run_baseline(seed=6)

    assert observation.shape == (5,)
    assert np.isfinite(rollout["final_metrics"]["process_fidelity"])
    assert 0.0 <= float(rollout["final_metrics"]["process_fidelity"]) <= 1.0


def test_full_pulse_env_measurement_observation_runs() -> None:
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_full_system(),
            task=coherent_state_preparation_task(alpha=0.15 + 0.0j, duration=24.0e-9),
            action_space=ParametricPulseActionSpace(family="cavity_displacement", alpha_bounds=(-0.3, 0.3), duration_bounds=(8.0e-9, 40.0e-9)),
            observation_model=build_observation_model("measurement_iq"),
            reward_model=build_reward_model("state"),
            measurement_spec=_measurement_spec(),
            auto_measurement=True,
            episode_horizon=1,
            seed=19,
        )
    )

    observation, _info = env.reset(seed=8)
    next_observation, reward, terminated, truncated, step_info = env.step(np.asarray([0.12, 0.0, 0.0, 20.0e-9], dtype=float))
    diagnostics = env.render_diagnostics()

    assert np.all(np.isfinite(observation))
    assert np.all(np.isfinite(next_observation))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert step_info["measurement"] is not None
    assert diagnostics["measurement"] is not None