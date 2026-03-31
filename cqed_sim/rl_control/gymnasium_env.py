"""Gymnasium-compatible wrapper for HybridCQEDEnv.

``GymnasiumCQEDEnv`` is a thin subclass of ``gymnasium.Env`` that delegates
all physics to the existing ``HybridCQEDEnv``.  It adds the two mandatory
attributes (``action_space``, ``observation_space``) that RL frameworks such
as Stable-Baselines3 require.

``HybridCQEDEnv`` is *not* modified — all existing code continues to work.

Usage::

    from cqed_sim.rl_control import GymnasiumCQEDEnv
    env = GymnasiumCQEDEnv(config)

    # Direct SB3 training:
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50_000)

    # Or validate with the official gymnasium checker:
    from gymnasium.utils.env_checker import check_env
    check_env(env, warn=True, skip_render_check=True)
"""
from __future__ import annotations

import numpy as np

try:
    import gymnasium
    from gymnasium import spaces
    _HAS_GYMNASIUM = True
except ImportError:
    _HAS_GYMNASIUM = False
    gymnasium = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from .env import HybridCQEDEnv
from .configs import HybridEnvConfig
from .gym_spaces import action_space_to_gymnasium


if _HAS_GYMNASIUM:

    class GymnasiumCQEDEnv(gymnasium.Env):
        """Gymnasium-compatible wrapper around ``HybridCQEDEnv``.

        Parameters
        ----------
        config:
            Full environment configuration.
        obs_low, obs_high:
            Scalar bounds for the ``observation_space`` Box.  Defaults to
            ``±inf`` (unbounded), which is safe for MLP policies that
            normalise observations internally.
        """

        metadata = {"render_modes": []}

        def __init__(
            self,
            config: HybridEnvConfig,
            *,
            obs_low: float = -np.inf,
            obs_high: float = np.inf,
        ) -> None:
            super().__init__()
            self._inner = HybridCQEDEnv(config)

            # Action space — derived from the cqed_sim space's .low / .high arrays.
            self.action_space = action_space_to_gymnasium(config.action_space)

            # Observation space — inferred by doing one dry reset.
            obs, _ = self._inner.reset()
            obs_dim = int(np.asarray(obs).shape[0])
            self.observation_space = spaces.Box(
                low=np.full(obs_dim, float(obs_low), dtype=np.float32),
                high=np.full(obs_dim, float(obs_high), dtype=np.float32),
                dtype=np.float32,
            )

        # ------------------------------------------------------------------
        # gymnasium.Env interface
        # ------------------------------------------------------------------

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            obs, info = self._inner.reset(seed=seed, options=options)
            return np.asarray(obs, dtype=np.float32), info

        def step(self, action):
            obs, reward, terminated, truncated, info = self._inner.step(action)
            return (
                np.asarray(obs, dtype=np.float32),
                float(reward),
                bool(terminated),
                bool(truncated),
                info,
            )

        def render(self):
            return None

        # ------------------------------------------------------------------
        # Convenience access to the underlying environment
        # ------------------------------------------------------------------

        @property
        def inner(self) -> HybridCQEDEnv:
            """The wrapped ``HybridCQEDEnv`` instance."""
            return self._inner

        def rollout(self, actions, **kwargs):
            """Delegate to ``HybridCQEDEnv.rollout()``."""
            return self._inner.rollout(actions, **kwargs)

        def diagnostics(self):
            """Delegate to ``HybridCQEDEnv.diagnostics()``."""
            return self._inner.diagnostics()

else:  # pragma: no cover

    class GymnasiumCQEDEnv:  # type: ignore[no-redef]
        """Stub raised when gymnasium is not installed."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "GymnasiumCQEDEnv requires gymnasium.  "
                "Install with: pip install gymnasium"
            )


__all__ = ["GymnasiumCQEDEnv"]
