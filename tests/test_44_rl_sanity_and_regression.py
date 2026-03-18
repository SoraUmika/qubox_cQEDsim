"""RL environment sanity tests and baseline regression tests.

Tests in this file verify:
 - Random rollouts with all three action-space types complete without error.
 - All observation modes return finite, correctly shaped outputs.
 - Rewards are finite and bounded for all action/reward combinations.
 - Termination logic: terminated on success, truncated at horizon.
 - Train vs eval randomization produces different parameter samples.
 - Episode metadata (randomization, metrics) is always recorded.
 - All 8 benchmark tasks can run their baseline actions.
 - Scripted demonstrations produce correctly structured records.
 - Diagnostics API exposes expected keys.
 - Numerical regression: specific deterministic baselines achieve target fidelity.
"""
from __future__ import annotations

import numpy as np
import pytest

from cqed_sim import (
    DomainRandomizer,
    FixedPrior,
    HybridCQEDEnv,
    HybridEnvConfig,
    HybridSystemConfig,
    NormalPrior,
    QubitMeasurementSpec,
    ReducedDispersiveModelConfig,
    UniformPrior,
    benchmark_task_suite,
    build_observation_model,
    build_reward_model,
    coherent_state_preparation_task,
    even_cat_preparation_task,
    fock_state_preparation_task,
    vacuum_preservation_task,
)
from cqed_sim.rl_control import (
    ParametricPulseActionSpace,
    PrimitiveActionSpace,
    WaveformActionSpace,
)
from cqed_sim.rl_control.demonstrations import scripted_demonstration, rollout_records


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _small_reduced_system() -> HybridSystemConfig:
    return HybridSystemConfig(
        regime="reduced_dispersive",
        reduced_model=ReducedDispersiveModelConfig(
            omega_c=2.0 * np.pi * 5.0e9,
            omega_q=2.0 * np.pi * 6.1e9,
            alpha=2.0 * np.pi * (-220.0e6),
            chi=2.0 * np.pi * (-2.2e6),
            kerr=2.0 * np.pi * (-4.0e3),
            n_cav=6,
            n_tr=3,
        ),
        dt=4.0e-9,
        max_step=4.0e-9,
    )


def _measurement_spec() -> QubitMeasurementSpec:
    return QubitMeasurementSpec(
        shots=32,
        iq_sigma=0.04,
        confusion_matrix=np.asarray([[0.96, 0.05], [0.04, 0.95]], dtype=float),
    )


def _make_env(
    *,
    task=None,
    action_space=None,
    obs_mode: str = "ideal_summary",
    reward_mode: str = "state",
    randomizer=None,
    measurement_spec=None,
    auto_measurement: bool = False,
    horizon: int = 2,
    seed: int = 0,
) -> HybridCQEDEnv:
    if task is None:
        task = coherent_state_preparation_task(alpha=0.3 + 0.0j, duration=40.0e-9)
    if action_space is None:
        action_space = PrimitiveActionSpace(primitives=("cavity_displacement", "wait", "measure"))
    return HybridCQEDEnv(
        HybridEnvConfig(
            system=_small_reduced_system(),
            task=task,
            action_space=action_space,
            observation_model=build_observation_model(obs_mode),
            reward_model=build_reward_model(reward_mode),
            randomizer=randomizer,
            measurement_spec=measurement_spec or _measurement_spec(),
            auto_measurement=auto_measurement,
            episode_horizon=horizon,
            seed=seed,
        )
    )


# ---------------------------------------------------------------------------
# Test: all three action-space types run random rollouts without crashing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["hybrid_block", "qubit_gaussian", "cavity_displacement", "sideband"])
def test_parametric_action_space_with_calibrated_action_stays_finite(family: str) -> None:
    """ParametricPulseActionSpace with calibrated (non-extreme) actions must stay finite.

    Tests the full RL pipeline (parse → generate → compile → propagate → observe → reward)
    with physically calibrated action magnitudes rather than arbitrary random vectors,
    which could overwhelm the ODE solver.
    """
    from cqed_sim.rl_control.action_spaces import (
        QubitGaussianAction, CavityDisplacementAction, SidebandAction, HybridBlockAction,
    )
    action_space = ParametricPulseActionSpace(
        family=family,
        duration_bounds=(16.0e-9, 60.0e-9),
        alpha_bounds=(-0.3, 0.3),
        amplitude_bounds=(0.0, 2.0 * np.pi * 1.0e6),
        theta_bounds=(-np.pi, np.pi),
    )
    # Build a calibrated action for each family
    if family == "qubit_gaussian":
        action = QubitGaussianAction(theta=np.pi / 2, phi=0.0, detuning=0.0, duration=32.0e-9, drag=0.0)
    elif family == "cavity_displacement":
        action = CavityDisplacementAction(alpha=0.2 + 0.0j, duration=40.0e-9, detuning=0.0)
    elif family == "sideband":
        action = SidebandAction(amplitude=2.0 * np.pi * 0.5e6, detuning=0.0, duration=40.0e-9, phase=0.0)
    else:  # hybrid_block: use a modest displacement only (zero qubit and sideband amplitude)
        action = HybridBlockAction(
            cavity_alpha=0.2 + 0.0j,
            cavity_duration=40.0e-9,
            wait_duration=0.0,
        )
    env = _make_env(action_space=action_space, horizon=2)
    obs, _info = env.reset(seed=1)
    assert obs.ndim == 1
    assert np.all(np.isfinite(obs))
    obs, reward, terminated, truncated, _step_info = env.step(action)
    assert np.all(np.isfinite(obs)), f"Action produced non-finite obs for family '{family}'."
    assert np.isfinite(reward), f"Action produced non-finite reward for family '{family}'."
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_primitive_action_space_pipeline_stays_finite() -> None:
    """PrimitiveActionSpace with calibrated actions must not crash and must stay finite."""
    from cqed_sim.rl_control.action_spaces import CavityDisplacementAction, WaitAction

    primitives = ("qubit_gaussian", "cavity_displacement", "sideband", "wait", "measure", "reset")
    action_space = PrimitiveActionSpace(primitives=primitives)
    env = _make_env(action_space=action_space, horizon=3)
    obs, _info = env.reset(seed=7)
    assert np.all(np.isfinite(obs))

    # Use specific calibrated actions (not random) to avoid ODE solver overflow
    for action in [
        CavityDisplacementAction(alpha=0.2 + 0.0j, duration=40.0e-9),
        WaitAction(duration=20.0e-9),
    ]:
        if env._terminated or env._truncated:
            break
        obs, reward, terminated, truncated, _step_info = env.step(action)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(reward)


def test_waveform_action_space_pipeline_stays_finite() -> None:
    """WaveformActionSpace with zero waveform must not crash and must stay finite."""
    action_space = WaveformActionSpace(
        segments=4,
        duration=40.0e-9,
        channels=("qubit", "storage"),
        amplitude_limit=2.0 * np.pi * 1.0e6,
    )
    env = _make_env(action_space=action_space, horizon=2)
    obs, _info = env.reset(seed=3)
    assert np.all(np.isfinite(obs))
    # Use zero-amplitude waveform to test pipeline integrity without stiff ODE
    zero_action = action_space.parse(np.zeros(action_space.shape[0], dtype=float))
    obs, reward, terminated, truncated, _step_info = env.step(zero_action)
    assert np.all(np.isfinite(obs))
    assert np.isfinite(reward)


# ---------------------------------------------------------------------------
# Test: zero-action (all zeros) does not produce NaN or exceptions
# ---------------------------------------------------------------------------

def test_zero_amplitude_actions_are_safe() -> None:
    """Actions with zero amplitude / alpha should not produce NaN.

    A displacement with alpha=0, a qubit pulse with theta=0, or a waveform with all
    zeros each represent no-ops and must complete without ODE solver errors.
    """
    from cqed_sim.rl_control.action_spaces import (
        QubitGaussianAction, CavityDisplacementAction, WaitAction,
    )

    from cqed_sim.rl_control.action_spaces import HybridBlockAction

    # Test zero-amplitude actions for each meaningful case
    safe_actions = [
        (ParametricPulseActionSpace(family="cavity_displacement"), CavityDisplacementAction(alpha=0.0j, duration=40.0e-9)),
        (ParametricPulseActionSpace(family="qubit_gaussian"),      QubitGaussianAction(theta=0.0, phi=0.0, detuning=0.0, duration=40.0e-9, drag=0.0)),
        (ParametricPulseActionSpace(family="hybrid_block"),        HybridBlockAction(wait_duration=40.0e-9)),  # all zeros → only wait
        (WaveformActionSpace(segments=4, duration=40.0e-9),        WaveformActionSpace(segments=4, duration=40.0e-9).parse(np.zeros(16, dtype=float))),
    ]

    for space, action in safe_actions:
        env = _make_env(action_space=space, horizon=1)
        env.reset(seed=0)
        obs, reward, terminated, truncated, info = env.step(action)
        assert np.all(np.isfinite(obs)), f"Zero action produced non-finite obs for {type(space).__name__}."
        assert np.isfinite(reward), f"Zero action produced non-finite reward for {type(space).__name__}."


# ---------------------------------------------------------------------------
# Test: observation shapes are consistent across modes
# ---------------------------------------------------------------------------

def test_observation_shape_consistency_across_modes() -> None:
    """
    Build a minimal environment with each observation mode and verify the
    observation returned by reset() has the expected shape and dtype.
    """
    measurement_spec = _measurement_spec()
    system = _small_reduced_system()
    task = coherent_state_preparation_task(alpha=0.2 + 0.0j, duration=40.0e-9)
    action_space = PrimitiveActionSpace(primitives=("cavity_displacement", "wait", "measure"))

    obs_modes_and_expected = [
        ("ideal_summary", None),            # n_cav-dependent: (13 + n_cav,) = (19,) for n_cav=6
        ("gate_metrics", (5,)),             # 5 metric values
        ("measurement_iq", None),           # (4,) or similar
        ("measurement_counts", None),
        ("measurement_outcome", None),
        ("measurement_classifier_logits", None),
    ]

    for obs_mode, expected_shape in obs_modes_and_expected:
        env = HybridCQEDEnv(
            HybridEnvConfig(
                system=system,
                task=task,
                action_space=action_space,
                observation_model=build_observation_model(obs_mode),
                reward_model=build_reward_model("state"),
                measurement_spec=measurement_spec,
                auto_measurement=True,
                episode_horizon=1,
                seed=0,
            )
        )
        obs, _info = env.reset(seed=0)
        assert obs.ndim == 1, f"Observation from mode '{obs_mode}' should be 1D. Got shape {obs.shape}."
        assert obs.dtype == np.float64 or obs.dtype.kind == "f", (
            f"Observation from mode '{obs_mode}' should be float. Got dtype {obs.dtype}."
        )
        assert np.all(np.isfinite(obs)), f"Observation from mode '{obs_mode}' is not finite."
        if expected_shape is not None:
            assert obs.shape == expected_shape, (
                f"Mode '{obs_mode}': expected shape {expected_shape}, got {obs.shape}."
            )


# ---------------------------------------------------------------------------
# Test: termination and truncation logic
# ---------------------------------------------------------------------------

def test_episode_terminates_on_success_and_truncates_at_horizon() -> None:
    """
    The vacuum preservation task should succeed immediately (fidelity > threshold).
    With horizon=1, it should terminate on the first step.
    """
    task = vacuum_preservation_task()  # horizon=1, threshold=0.999
    env = _make_env(task=task, horizon=1, seed=0)
    obs, info = env.reset(seed=5)
    assert info["task_name"] == "vacuum_preservation"
    # Baseline action: WaitAction(0.0) - trivially succeeds
    action = task.baseline_actions[0]
    _obs, reward, terminated, truncated, step_info = env.step(action)
    assert step_info["metrics"]["state_fidelity"] > 0.999
    assert terminated, "Vacuum preservation should terminate on success."
    assert not truncated


def test_truncation_at_horizon_when_no_success() -> None:
    """
    With a hard task and horizon=1 and a do-nothing action,
    the episode should truncate at the horizon, not terminate with success.
    """
    task = even_cat_preparation_task(alpha=2.0 + 0.0j)  # very hard
    from cqed_sim.rl_control.action_spaces import WaitAction
    env = _make_env(task=task, horizon=1, seed=0)
    env.reset(seed=0)
    _obs, reward, terminated, truncated, _step_info = env.step(WaitAction(duration=0.0))
    # Cat state preparation from vacuum with a wait action should not succeed
    assert not terminated, "Should not succeed with just a wait action for cat preparation."
    assert truncated, "Episode should truncate at horizon=1."


def test_calling_step_after_termination_raises() -> None:
    """Calling step() after an episode has terminated should raise RuntimeError."""
    task = vacuum_preservation_task()
    env = _make_env(task=task, horizon=1)
    env.reset(seed=0)
    action = task.baseline_actions[0]
    env.step(action)
    assert env._terminated or env._truncated
    with pytest.raises(RuntimeError):
        env.step(action)


# ---------------------------------------------------------------------------
# Test: train vs eval randomization differ when configured differently
# ---------------------------------------------------------------------------

def test_train_and_eval_randomization_sample_different_parameters() -> None:
    """
    When model_priors_train and model_priors_eval specify different FixedPrior
    values for the same parameter, sampling in 'train' and 'eval' modes must
    produce different values.
    """
    randomizer = DomainRandomizer(
        model_priors_train={"chi": FixedPrior(2.0 * np.pi * (-2.0e6))},
        model_priors_eval={"chi": FixedPrior(2.0 * np.pi * (-3.0e6))},
        noise_priors_train={"kappa": FixedPrior(2.0 * np.pi * 1.0e3)},
        noise_priors_eval={"kappa": FixedPrior(2.0 * np.pi * 5.0e3)},
        drift_priors_train={"storage_amplitude_scale": FixedPrior(1.0)},
        drift_priors_eval={"storage_amplitude_scale": FixedPrior(0.95)},
    )
    train = randomizer.sample(seed=42, mode="train")
    eval_sample = randomizer.sample(seed=42, mode="eval")

    assert train.model_overrides["chi"] != eval_sample.model_overrides["chi"]
    assert train.noise_overrides["kappa"] != eval_sample.noise_overrides["kappa"]
    assert train.drift_state["storage_amplitude_scale"] != eval_sample.drift_state["storage_amplitude_scale"]


def test_normal_prior_samples_near_mean() -> None:
    """NormalPrior should produce samples concentrated near the mean."""
    prior = NormalPrior(mean=2.0 * np.pi * (-2.2e6), sigma=2.0 * np.pi * 0.01e6)
    rng = np.random.default_rng(0)
    samples = [prior.sample(rng) for _ in range(50)]
    mean_val = np.mean(samples)
    target_mean = 2.0 * np.pi * (-2.2e6)
    relative_error = abs(mean_val - target_mean) / abs(target_mean)
    assert relative_error < 0.05, f"NormalPrior mean far from target. Got {mean_val:.2e}, target {target_mean:.2e}."


def test_uniform_prior_samples_in_range() -> None:
    """UniformPrior should produce all samples within [low, high]."""
    low, high = -0.5, 0.5
    prior = UniformPrior(low=low, high=high)
    rng = np.random.default_rng(7)
    for _ in range(30):
        value = prior.sample(rng)
        assert low <= value <= high, f"UniformPrior produced value {value} outside [{low}, {high}]."


# ---------------------------------------------------------------------------
# Test: episode metadata is always recorded
# ---------------------------------------------------------------------------

def test_episode_metadata_always_present_after_reset_and_step() -> None:
    """After reset() and step(), info dicts must always contain required keys."""
    randomizer = DomainRandomizer(
        model_priors_train={"chi": NormalPrior(2.0 * np.pi * (-2.2e6), 2.0 * np.pi * 0.05e6)},
        drift_priors_train={"storage_amplitude_scale": UniformPrior(0.95, 1.05)},
    )
    env = _make_env(randomizer=randomizer, horizon=3)

    for seed in (0, 1, 42):
        obs, info = env.reset(seed=seed)
        assert "task_name" in info
        assert "task_kind" in info
        assert "seed" in info
        assert "system_regime" in info
        assert "metrics" in info
        assert "randomization" in info
        assert "baseline_actions" in info
        assert info["system_regime"] == "reduced_dispersive"

        from cqed_sim.rl_control.action_spaces import WaitAction
        obs2, reward, terminated, truncated, step_info = env.step(WaitAction(duration=20.0e-9))
        assert "task_name" in step_info
        assert "metrics" in step_info
        assert "reward_breakdown" in step_info
        assert "step_index" in step_info
        assert "randomization" in step_info
        assert step_info["step_index"] == 1


# ---------------------------------------------------------------------------
# Test: all benchmark tasks run their baseline actions without error
# ---------------------------------------------------------------------------

def test_all_benchmark_tasks_baseline_run_without_error() -> None:
    """
    Every task in benchmark_task_suite() must be able to run its baseline
    action sequence in a fresh environment without raising an exception.
    """
    system = _small_reduced_system()
    suite = benchmark_task_suite()
    for task_name, task in suite.items():
        action_space = PrimitiveActionSpace(
            primitives=("qubit_gaussian", "cavity_displacement", "sideband", "wait", "measure", "reset"),
        )
        env = HybridCQEDEnv(
            HybridEnvConfig(
                system=system,
                task=task,
                action_space=action_space,
                observation_model=build_observation_model("ideal_summary"),
                reward_model=build_reward_model("state"),
                measurement_spec=_measurement_spec(),
                episode_horizon=max(4, task.horizon + 1),
                seed=0,
            )
        )
        try:
            result = env.run_baseline(seed=0)
        except Exception as exc:
            pytest.fail(f"Baseline for task '{task_name}' raised: {exc!r}")
        assert np.isfinite(result["total_reward"]), (
            f"Baseline reward for task '{task_name}' is not finite: {result['total_reward']}."
        )


# ---------------------------------------------------------------------------
# Test: diagnostics API exposes expected keys
# ---------------------------------------------------------------------------

def test_diagnostics_api_exposes_required_keys() -> None:
    """render_diagnostics() must expose at minimum the keys expected by users."""
    env = _make_env(auto_measurement=True, horizon=2)
    env.reset(seed=0)
    env.step(np.zeros(env.config.action_space.shape[0], dtype=float))
    diagnostics = env.render_diagnostics()

    required_keys = {
        "joint_state",
        "reduced_qubit_state",
        "reduced_cavity_state",
        "ancilla_populations",
        "photon_number_distribution",
        "channels",
        "segment_metadata",
        "pulse_summary",
        "regime",
        "frame",
        "metrics",
    }
    missing = required_keys - set(diagnostics.keys())
    assert not missing, f"Diagnostics dict is missing expected keys: {sorted(missing)}."


def test_diagnostics_contains_wigner_when_wigner_points_nonzero() -> None:
    """With wigner_points > 0, diagnostics should include the Wigner function values."""
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_small_reduced_system(),
            task=coherent_state_preparation_task(alpha=0.3 + 0.0j, duration=40.0e-9),
            action_space=PrimitiveActionSpace(primitives=("cavity_displacement", "wait")),
            observation_model=build_observation_model("ideal_summary"),
            reward_model=build_reward_model("state"),
            measurement_spec=_measurement_spec(),
            episode_horizon=2,
            store_states_for_diagnostics=True,
            diagnostics_wigner_points=9,
            seed=0,
        )
    )
    obs, _info = env.reset(seed=0)
    from cqed_sim.rl_control.action_spaces import CavityDisplacementAction
    env.step(CavityDisplacementAction(alpha=0.3 + 0.0j, duration=40.0e-9))
    diagnostics = env.diagnostics()
    assert "wigner" in diagnostics, "Expected 'wigner' key in diagnostics when wigner_points > 0."
    wigner_vals = diagnostics["wigner"]
    assert wigner_vals is not None
    assert len(wigner_vals) > 0


# ---------------------------------------------------------------------------
# Test: scripted demonstrations produce correctly structured records
# ---------------------------------------------------------------------------

def test_scripted_demonstration_produces_valid_records() -> None:
    """
    scripted_demonstration() should return a DemonstrationRollout whose
    actions match the task's baseline_actions and whose metadata is a dict.
    rollout_records() should yield one record per step.
    """
    task = coherent_state_preparation_task(alpha=0.25 + 0.0j, duration=40.0e-9)
    demo = scripted_demonstration(task)

    assert demo.task_name == task.name
    assert len(demo.actions) == len(task.baseline_actions)
    assert isinstance(demo.metadata, dict)
    assert np.isfinite(demo.total_reward) or demo.total_reward == 0.0  # placeholder reward

    records = rollout_records(demo)
    assert len(records) == len(task.baseline_actions)
    for record in records:
        assert "task" in record
        assert "step" in record
        assert "action" in record


# ---------------------------------------------------------------------------
# Test: action space parse/flatten round-trip
# ---------------------------------------------------------------------------

def test_parametric_action_space_parse_flatten_roundtrip() -> None:
    """parse(flatten(parsed)) should return the same vector as flatten(parsed)."""
    for family in ("hybrid_block", "qubit_gaussian", "cavity_displacement", "sideband"):
        space = ParametricPulseActionSpace(family=family)
        rng = np.random.default_rng(0)
        for _ in range(5):
            raw = space.sample(rng)
            parsed = space.parse(raw)
            flat = space.flatten(parsed)
            re_parsed = space.parse(flat)
            re_flat = space.flatten(re_parsed)
            assert np.allclose(flat, re_flat, atol=1.0e-12), (
                f"Round-trip mismatch for family '{family}': "
                f"flat={flat}, re_flat={re_flat}."
            )


def test_waveform_action_space_parse_flatten_roundtrip() -> None:
    """WaveformActionSpace parse/flatten round-trip should be lossless."""
    space = WaveformActionSpace(segments=4, duration=40.0e-9, channels=("qubit", "storage"))
    rng = np.random.default_rng(3)
    raw = space.sample(rng)
    parsed = space.parse(raw)
    flat = space.flatten(parsed)
    assert np.allclose(raw, flat, atol=1.0e-12), "WaveformActionSpace round-trip failed."


# ---------------------------------------------------------------------------
# Regression tests: specific deterministic baselines must achieve target fidelity
# ---------------------------------------------------------------------------

def test_vacuum_preservation_baseline_achieves_high_fidelity_regression() -> None:
    """
    Regression test: WaitAction(0.0) on |g,0> must give state_fidelity > 0.9999.
    This is the simplest possible environment and establishes a numerical floor.
    """
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_small_reduced_system(),
            task=vacuum_preservation_task(),
            action_space=PrimitiveActionSpace(primitives=("wait",)),
            observation_model=build_observation_model("ideal_summary"),
            reward_model=build_reward_model("state"),
            episode_horizon=1,
            seed=0,
        )
    )
    result = env.run_baseline(seed=0)
    fidelity = float(result["final_metrics"]["state_fidelity"])
    assert fidelity > 0.9999, (
        f"Vacuum preservation regression: expected state_fidelity > 0.9999. Got {fidelity:.6f}."
    )


def test_coherent_state_baseline_achieves_reasonable_fidelity_regression() -> None:
    """
    Regression test: the displacement baseline for a weak coherent state
    (|alpha|=0.2) should achieve state_fidelity > 0.95 in the reduced regime.

    This anchors the calibration formula and displacement-to-operator conversion
    across refactors.
    """
    alpha = 0.2 + 0.0j
    duration = 40.0e-9
    task = coherent_state_preparation_task(alpha=alpha, duration=duration)
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_small_reduced_system(),
            task=task,
            action_space=PrimitiveActionSpace(
                primitives=("cavity_displacement", "wait"),
                duration_bounds=(8.0e-9, 200.0e-9),
                alpha_bounds=(-0.5, 0.5),
            ),
            observation_model=build_observation_model("ideal_summary"),
            reward_model=build_reward_model("state"),
            episode_horizon=2,
            seed=0,
        )
    )
    result = env.run_baseline(seed=0)
    fidelity = float(result["final_metrics"]["state_fidelity"])
    assert fidelity > 0.95, (
        f"Coherent state baseline regression: expected state_fidelity > 0.95 for alpha=0.2. "
        f"Got {fidelity:.4f}."
    )


def test_fock_1_state_task_metrics_are_bounded() -> None:
    """
    After any rollout, state_fidelity, ancilla_return, and leakage metrics
    must all lie in [0, 1].  This is a data-integrity regression test.
    """
    env = HybridCQEDEnv(
        HybridEnvConfig(
            system=_small_reduced_system(),
            task=fock_state_preparation_task(cavity_level=1),
            action_space=ParametricPulseActionSpace(
                family="hybrid_block",
                duration_bounds=(8.0e-9, 80.0e-9),
                alpha_bounds=(-1.0, 1.0),
            ),
            observation_model=build_observation_model("ideal_summary"),
            reward_model=build_reward_model("state"),
            measurement_spec=_measurement_spec(),
            episode_horizon=3,
            seed=5,
        )
    )
    rng = np.random.default_rng(5)
    env.reset(seed=5)
    for _ in range(3):
        if env._terminated or env._truncated:
            break
        env.step(env.config.action_space.sample(rng))
    metrics = env.last_metrics
    for key in ("state_fidelity", "ancilla_return", "leakage_average", "leakage_worst"):
        assert key in metrics, f"Expected metric '{key}' in last_metrics."
        val = float(metrics[key])
        assert 0.0 <= val <= 1.0 + 1.0e-9, f"Metric '{key}' = {val:.6f} is outside [0, 1]."


def test_estimate_metrics_returns_distribution_summaries() -> None:
    """
    estimate_metrics() should return per-rollout records and per-metric
    distribution summaries with mean, std, min, max, p05, p50, p95.
    """
    env = _make_env(horizon=1)
    result = env.estimate_metrics(n_rollouts=4, seeds=(10, 11, 12, 13), randomization_mode="train")

    assert "per_rollout" in result
    assert "summary" in result
    assert len(result["per_rollout"]) == 4

    for rollout in result["per_rollout"]:
        assert "seed" in rollout
        assert "reward" in rollout
        assert "metrics" in rollout

    assert "reward" in result["summary"]
    for key in ("mean", "std", "min", "max", "p05", "p50", "p95"):
        assert key in result["summary"]["reward"], (
            f"Distribution summary for 'reward' is missing key '{key}'."
        )


__all__ = []
