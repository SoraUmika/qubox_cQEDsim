from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import qutip as qt

from .configs import HybridEnvConfig
from .diagnostics import build_rollout_diagnostics
from .metrics import evaluate_state_task_metrics, evaluate_unitary_task_metrics, summarize_distribution
from .runtime import DistortionModel, HamiltonianModelFactory, MeasurementModel, OpenSystemEngine, PulseGenerator
from .tasks import HybridBenchmarkTask, benchmark_task_suite, finalize_conditional_phase_baseline


def _resolve_task(task_spec: Any) -> HybridBenchmarkTask:
    if isinstance(task_spec, HybridBenchmarkTask):
        return copy.deepcopy(task_spec)
    if isinstance(task_spec, str):
        suite = benchmark_task_suite()
        if task_spec not in suite:
            raise KeyError(f"Unknown benchmark task '{task_spec}'.")
        return copy.deepcopy(suite[task_spec])
    raise TypeError(f"Unsupported task specification '{type(task_spec).__name__}'.")


@dataclass
class StepRecord:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class HybridCQEDEnv:
    def __init__(self, config: HybridEnvConfig):
        self.config = config
        self._master_rng = np.random.default_rng(config.seed)
        self._task: HybridBenchmarkTask | None = None
        self._bundle: Any = None
        self._pulse_generator: PulseGenerator | None = None
        self._distortion_model: DistortionModel | None = None
        self._engine: OpenSystemEngine | None = None
        self._measurement_model: MeasurementModel | None = None
        self._rng = np.random.default_rng(config.seed)
        self._episode_seed: int | None = config.seed
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._current_state: qt.Qobj | None = None
        self._probe_states: list[qt.Qobj] | None = None
        self._last_measurement: Any = None
        self._last_metrics: dict[str, Any] = {}
        self._last_compiled: Any = None
        self._last_segment: Any = None
        self._randomization_metadata: dict[str, Any] = {}
        self._action_history: list[np.ndarray] = []
        self._observation_history: list[np.ndarray] = []

    @property
    def task(self) -> HybridBenchmarkTask:
        if self._task is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._task

    @property
    def last_metrics(self) -> dict[str, Any]:
        return dict(self._last_metrics)

    def _next_seed(self) -> int:
        return int(self._rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))

    def _resolve_observation_model(self) -> Any:
        return self.config.observation_model

    def _resolve_reward_model(self) -> Any:
        return self.config.reward_model

    def _observation_requires_measurement(self) -> bool:
        return bool(getattr(self._resolve_observation_model(), "requires_measurement", False))

    def _current_observation_state(self) -> qt.Qobj | None:
        if self._current_state is not None:
            return self._current_state
        if self._probe_states:
            return self._probe_states[0]
        return None

    def _build_metrics(self) -> dict[str, Any]:
        if self.task.kind == "state_preparation":
            if self._current_state is None:
                raise RuntimeError("State-preparation task has no current state.")
            return evaluate_state_task_metrics(self.task, self._bundle.model, self._current_state)
        if self._probe_states is None:
            raise RuntimeError("Unitary-synthesis task has no probe states.")
        return evaluate_unitary_task_metrics(self.task, self._bundle.model, self._probe_states)

    def _reset_internal_states(self) -> None:
        if self.task.kind == "state_preparation":
            self._current_state = self.task.build_initial_state(self._bundle.model)
            self._probe_states = None
        else:
            self._probe_states = self.task.build_probe_states(self._bundle.model)
            self._current_state = self.task.build_initial_state(self._bundle.model)

    def _maybe_measure(self) -> None:
        if not (self.config.auto_measurement or self._observation_requires_measurement()):
            return
        state = self._current_observation_state()
        if state is None:
            return
        self._last_measurement = self._measurement_model.observe(state, bundle=self._bundle, seed=self._next_seed())
        if self.config.collapse_on_measurement and self._current_state is not None:
            self._current_state = self._measurement_model.collapse_joint_state(self._current_state, self._last_measurement, seed=self._next_seed())

    def _build_observation(self) -> np.ndarray:
        observation = np.asarray(
            self._resolve_observation_model().encode(
                state=self._current_observation_state(),
                model=self._bundle.model,
                task=self.task,
                probe_states=self._probe_states,
                metrics=self._last_metrics,
                measurement=self._last_measurement,
                observation_history=self._observation_history,
                action_history=self._action_history,
            ),
            dtype=float,
        )
        return np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        options = {} if options is None else dict(options)
        if seed is None:
            seed = int(self._master_rng.integers(0, np.iinfo(np.uint32).max, endpoint=True))
        self._episode_seed = int(seed)
        self._rng = np.random.default_rng(self._episode_seed)

        task_spec = options.get("task", self.config.task)
        self._task = _resolve_task(task_spec)
        randomization_mode = str(options.get("randomization_mode", self.config.randomization_mode))
        randomization = None if self.config.randomizer is None else self.config.randomizer.sample(seed=self._next_seed(), mode=randomization_mode)
        self._bundle = HamiltonianModelFactory.build(self.config.system, randomization=randomization, measurement_spec=self.config.measurement_spec)
        self._task = finalize_conditional_phase_baseline(self._task, self._bundle.model)
        self._pulse_generator = PulseGenerator()
        self._distortion_model = DistortionModel(self._bundle)
        self._engine = OpenSystemEngine(self._bundle)
        self._measurement_model = MeasurementModel(self._bundle.measurement_spec, collapse_on_measurement=self.config.collapse_on_measurement)
        self._step_index = 0
        self._terminated = False
        self._truncated = False
        self._last_compiled = None
        self._last_segment = None
        self._last_measurement = None
        self._action_history.clear()
        self._observation_history.clear()
        self._randomization_metadata = {} if randomization is None else randomization.metadata
        self._reset_internal_states()
        self._maybe_measure()
        self._last_metrics = self._build_metrics()
        observation = self._build_observation()
        self._observation_history.append(observation.copy())
        info = {
            "task_name": self.task.name,
            "task_kind": self.task.kind,
            "seed": int(self._episode_seed),
            "system_regime": self._bundle.regime,
            "baseline_actions": tuple(self.task.baseline_actions),
            "metrics": dict(self._last_metrics),
            "randomization": dict(self._randomization_metadata),
        }
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._task is None or self._bundle is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        if self._terminated or self._truncated:
            raise RuntimeError("Episode has already terminated. Call reset() before stepping again.")

        parsed_action = self.config.action_space.parse(action)
        flat_action = np.asarray(self.config.action_space.flatten(parsed_action), dtype=float)
        segment = self._pulse_generator.generate(parsed_action, self._bundle)
        self._last_segment = segment
        self._last_compiled = None
        self._last_measurement = None

        if segment.reset_requested:
            self._reset_internal_states()
        elif segment.duration > 0.0 or segment.pulses:
            compiled = self._distortion_model.compile(segment)
            self._last_compiled = compiled
            if self.task.kind == "state_preparation":
                result = self._engine.propagate_state(
                    self._current_state,
                    compiled,
                    segment.drive_ops,
                    store_states=self.config.store_states_for_diagnostics,
                )
                self._current_state = result.final_state
            else:
                results = self._engine.propagate_states(self._probe_states or [], compiled, segment.drive_ops)
                self._probe_states = [result.final_state for result in results]
                if self._probe_states:
                    self._current_state = self._probe_states[0]

        if segment.measurement_requested or self.config.auto_measurement or self._observation_requires_measurement():
            state = self._current_observation_state()
            if state is not None:
                self._last_measurement = self._measurement_model.observe(state, bundle=self._bundle, seed=self._next_seed())
                if (self.config.collapse_on_measurement or bool(segment.metadata.get("collapse", False))) and self._current_state is not None:
                    self._current_state = self._measurement_model.collapse_joint_state(self._current_state, self._last_measurement, seed=self._next_seed())

        self._last_metrics = self._build_metrics()
        reward, reward_breakdown = self._resolve_reward_model().compute(
            state=self._current_observation_state(),
            probe_states=self._probe_states,
            task=self.task,
            model=self._bundle.model,
            metrics=self._last_metrics,
            measurement=self._last_measurement,
            segment=segment,
        )
        self._step_index += 1
        success = bool(float(self._last_metrics.get("success", 0.0)) >= 1.0)
        self._terminated = success
        self._truncated = self._step_index >= int(self.config.episode_horizon or self.task.horizon) and not self._terminated

        observation = self._build_observation()
        self._action_history.append(flat_action)
        self._observation_history.append(observation.copy())

        info = {
            "task_name": self.task.name,
            "metrics": dict(self._last_metrics),
            "reward_breakdown": reward_breakdown,
            "step_index": int(self._step_index),
            "measurement": None if self._last_measurement is None else {
                "probabilities": dict(self._last_measurement.probabilities),
                "observed_probabilities": dict(self._last_measurement.observed_probabilities),
                "counts": None if self._last_measurement.counts is None else dict(self._last_measurement.counts),
            },
            "randomization": dict(self._randomization_metadata),
        }
        return observation, float(reward), bool(self._terminated), bool(self._truncated), info

    def rollout(self, actions: Sequence[Any], *, seed: int | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        observation, info = self.reset(seed=seed, options=options)
        records: list[StepRecord] = []
        total_reward = 0.0
        for action in actions:
            observation, reward, terminated, truncated, step_info = self.step(action)
            total_reward += float(reward)
            records.append(
                StepRecord(
                    observation=np.asarray(observation, dtype=float),
                    reward=float(reward),
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                    info=dict(step_info),
                )
            )
            if terminated or truncated:
                break
        return {
            "initial_info": info,
            "final_observation": np.asarray(observation, dtype=float),
            "total_reward": float(total_reward),
            "records": records,
            "final_metrics": dict(self._last_metrics),
            "terminated": bool(self._terminated),
            "truncated": bool(self._truncated),
        }

    def run_baseline(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._task is None:
            task = _resolve_task(self.config.task)
        else:
            task = self.task
        return self.rollout(task.baseline_actions, seed=seed, options=options)

    def estimate_task_metrics(
        self,
        policy_or_actions: Callable[[np.ndarray, dict[str, Any]], Any] | Sequence[Any] | None = None,
        *,
        n_rollouts: int = 8,
        seeds: Sequence[int] | None = None,
        randomization_mode: str | None = None,
    ) -> dict[str, Any]:
        resolved_seeds = list(seeds) if seeds is not None else [int(self._master_rng.integers(0, np.iinfo(np.uint32).max, endpoint=True)) for _ in range(int(n_rollouts))]
        per_rollout: list[dict[str, Any]] = []
        for rollout_seed in resolved_seeds:
            options = {} if randomization_mode is None else {"randomization_mode": randomization_mode}
            observation, info = self.reset(seed=int(rollout_seed), options=options)
            total_reward = 0.0
            if policy_or_actions is None:
                actions = tuple(self.task.baseline_actions)
                for action in actions:
                    observation, reward, terminated, truncated, info = self.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
            elif callable(policy_or_actions):
                while not (self._terminated or self._truncated):
                    action = policy_or_actions(np.asarray(observation, dtype=float), dict(info))
                    observation, reward, terminated, truncated, info = self.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
            else:
                for action in policy_or_actions:
                    observation, reward, terminated, truncated, info = self.step(action)
                    total_reward += float(reward)
                    if terminated or truncated:
                        break
            per_rollout.append(
                {
                    "seed": int(rollout_seed),
                    "reward": float(total_reward),
                    "metrics": dict(self._last_metrics),
                    "terminated": bool(self._terminated),
                    "truncated": bool(self._truncated),
                }
            )

        metric_keys = sorted({key for rollout in per_rollout for key in rollout["metrics"].keys()})
        summaries = {
            key: summarize_distribution([float(rollout["metrics"].get(key, np.nan)) for rollout in per_rollout])
            for key in metric_keys
        }
        summaries["reward"] = summarize_distribution([rollout["reward"] for rollout in per_rollout])
        return {"per_rollout": per_rollout, "summary": summaries}

    def diagnostics(self) -> dict[str, Any]:
        if self._bundle is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return build_rollout_diagnostics(
            model=self._bundle.model,
            state=self._current_state,
            probe_states=self._probe_states,
            compiled=self._last_compiled,
            segment=self._last_segment,
            measurement=self._last_measurement,
            metrics=self._last_metrics,
            randomization=self._randomization_metadata,
            regime=self._bundle.regime,
            frame=self._bundle.frame,
            include_wigner=self.config.diagnostics_wigner_points > 0,
            wigner_points=int(self.config.diagnostics_wigner_points),
        )

    def render_diagnostics(self) -> dict[str, Any]:
        return self.diagnostics()

    def estimate_metrics(
        self,
        policy_or_actions: Callable[[np.ndarray, dict[str, Any]], Any] | Sequence[Any] | None = None,
        *,
        n_rollouts: int = 8,
        seeds: Sequence[int] | None = None,
        randomization_mode: str | None = None,
    ) -> dict[str, Any]:
        return self.estimate_task_metrics(
            policy_or_actions,
            n_rollouts=n_rollouts,
            seeds=seeds,
            randomization_mode=randomization_mode,
        )


__all__ = ["StepRecord", "HybridCQEDEnv"]