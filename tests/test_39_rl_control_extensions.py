from __future__ import annotations

import numpy as np
import pytest

from cqed_sim import (
    HybridCQEDEnv,
    HybridEnvConfig,
    HybridSystemConfig,
    ParametricPulseActionSpace,
    PrimitiveActionSpace,
    QubitMeasurementSpec,
    ReducedDispersiveModelConfig,
    benchmark_task_suite,
    build_observation_model,
    build_reward_model,
    coherent_state_preparation_task,
    fock_state_preparation_task,
    odd_cat_preparation_task,
    storage_superposition_task,
)
from cqed_sim.rl_control import HamiltonianModelFactory, parity_expectation, photon_number_distribution


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


def _measurement_spec() -> QubitMeasurementSpec:
    return QubitMeasurementSpec(
        shots=64,
        iq_sigma=0.04,
        confusion_matrix=np.asarray([[0.96, 0.05], [0.04, 0.95]], dtype=float),
    )


def test_extended_benchmark_targets_have_expected_structure() -> None:
    model = HamiltonianModelFactory.build(_reduced_system()).model
    suite = benchmark_task_suite()

    assert {"fock_1_preparation", "storage_superposition_preparation", "odd_cat_preparation"}.issubset(suite)

    fock_state = fock_state_preparation_task(cavity_level=1).build_target_state(model)
    fock_distribution = photon_number_distribution(fock_state)
    assert fock_distribution[1] == pytest.approx(1.0, abs=1.0e-12)

    superposition_state = storage_superposition_task().build_target_state(model)
    superposition_distribution = photon_number_distribution(superposition_state)
    assert superposition_distribution[0] == pytest.approx(0.5, abs=1.0e-12)
    assert superposition_distribution[1] == pytest.approx(0.5, abs=1.0e-12)

    odd_cat_state = odd_cat_preparation_task(alpha=1.0 + 0.0j).build_target_state(model)
    assert parity_expectation(odd_cat_state) < 0.0


def test_measurement_observation_aliases_proxy_reward_and_diagnostics_run() -> None:
    action_space = PrimitiveActionSpace(primitives=("cavity_displacement", "wait", "measure"))
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_reduced_system(),
            task=coherent_state_preparation_task(alpha=0.25 + 0.05j, duration=40.0e-9),
            action_space=action_space,
            observation_model=build_observation_model(
                "measurement_classifier_logits",
                action_dim=action_space.shape[0],
                history_length=2,
            ),
            reward_model=build_reward_model("measurement_proxy"),
            measurement_spec=_measurement_spec(),
            auto_measurement=True,
            episode_horizon=2,
            seed=31,
        )
    )

    observation, _info = env.reset(seed=5)
    next_observation, reward, terminated, truncated, step_info = env.step(
        {
            "primitive": "cavity_displacement",
            "alpha": 0.20 + 0.00j,
            "duration": 32.0e-9,
            "detuning": 0.0,
        }
    )
    diagnostics = env.render_diagnostics()

    assert observation.shape == (22,)
    assert next_observation.shape == (22,)
    assert np.all(np.isfinite(observation))
    assert np.all(np.isfinite(next_observation))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "measurement_assignment" in step_info["reward_breakdown"]
    assert diagnostics["segment_metadata"]["action_type"] == "cavity_displacement"
    assert len(diagnostics["pulse_summary"]) == 1
    assert diagnostics["regime"] == "reduced_dispersive"
    assert diagnostics["frame"]["omega_q_frame"] != 0.0

    count_model = build_observation_model("measurement_counts")
    count_observation = count_model.encode(measurement=env._last_measurement, metrics=env.last_metrics)
    outcome_model = build_observation_model("measurement_outcome")
    outcome_observation = outcome_model.encode(measurement=env._last_measurement, metrics=env.last_metrics)

    assert count_observation.shape == (4,)
    assert outcome_observation.shape == (4,)
    assert np.isclose(np.sum(outcome_observation[:2]), 1.0)


def test_parametric_env_with_new_tasks_remains_finite() -> None:
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_reduced_system(),
            task=odd_cat_preparation_task(alpha=0.9 + 0.0j),
            action_space=ParametricPulseActionSpace(
                family="hybrid_block",
                alpha_bounds=(-0.5, 0.5),
                duration_bounds=(8.0e-9, 80.0e-9),
            ),
            observation_model=build_observation_model("ideal_summary"),
            reward_model=build_reward_model("measurement_proxy"),
            measurement_spec=_measurement_spec(),
            auto_measurement=True,
            episode_horizon=1,
            seed=37,
        )
    )

    observation, _info = env.reset(seed=11)
    next_observation, reward, terminated, truncated, _step_info = env.step(np.zeros(env.config.action_space.shape[0], dtype=float))

    assert np.all(np.isfinite(observation))
    assert np.all(np.isfinite(next_observation))
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)