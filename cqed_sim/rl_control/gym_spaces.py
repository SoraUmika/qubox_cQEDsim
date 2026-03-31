"""Conversion helpers from cqed_sim action spaces to gymnasium.spaces objects.

Each cqed_sim action space (ParametricPulseActionSpace, PrimitiveActionSpace,
WaveformActionSpace) already exposes ``.low`` and ``.high`` arrays.  This
module converts those bounds into the ``gymnasium.spaces.Box`` objects that
SB3 and other RL frameworks expect.

Only imported when gymnasium is available.
"""
from __future__ import annotations

import numpy as np


def action_space_to_gymnasium(cqed_space) -> "gymnasium.spaces.Box":
    """Return a ``gymnasium.spaces.Box`` matching *cqed_space* bounds.

    Parameters
    ----------
    cqed_space:
        Any cqed_sim action space that exposes ``.low`` and ``.high`` arrays
        (``ParametricPulseActionSpace``, ``PrimitiveActionSpace``, or
        ``WaveformActionSpace``).

    Returns
    -------
    gymnasium.spaces.Box
        Continuous box space with the same shape and bounds.
    """
    import gymnasium as gym  # deferred — optional

    low = np.asarray(cqed_space.low, dtype=np.float32)
    high = np.asarray(cqed_space.high, dtype=np.float32)
    return gym.spaces.Box(low=low, high=high, dtype=np.float32)


__all__ = ["action_space_to_gymnasium"]
