from __future__ import annotations

import numpy as np

from .parameterizations import ControlSchedule
from .utils import finite_bound_scale


def zero_control_schedule(problem) -> ControlSchedule:
    return problem.parameterization.zero_schedule()


def random_control_schedule(problem, *, seed: int | None = None, scale: float = 0.15) -> ControlSchedule:
    rng = np.random.default_rng(seed)
    values = np.zeros((problem.n_controls, problem.n_slices), dtype=float)
    for index, term in enumerate(problem.control_terms):
        lower, upper = term.amplitude_bounds
        if np.isfinite(lower) and np.isfinite(upper):
            center = 0.5 * (float(lower) + float(upper))
            half_range = 0.5 * (float(upper) - float(lower))
            span = max(scale * half_range, 1.0e-12)
            values[index, :] = rng.uniform(center - span, center + span, size=problem.n_slices)
        else:
            sigma = scale * finite_bound_scale(lower, upper, fallback=1.0)
            values[index, :] = rng.normal(loc=0.0, scale=sigma, size=problem.n_slices)
    clipped = problem.parameterization.clip(values)
    return ControlSchedule(problem.parameterization, clipped)


def warm_start_schedule(result, *, noise_scale: float = 0.0, seed: int | None = None) -> ControlSchedule:
    schedule = result.schedule.copy()
    if noise_scale <= 0.0:
        return schedule
    rng = np.random.default_rng(seed)
    jitter = rng.normal(scale=float(noise_scale), size=schedule.values.shape)
    return ControlSchedule(schedule.parameterization, schedule.parameterization.clip(schedule.values + jitter))


__all__ = ["zero_control_schedule", "random_control_schedule", "warm_start_schedule"]