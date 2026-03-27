from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

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


# ---------------------------------------------------------------------------
# Pulse ansatz classes for GRAPE initialization
# ---------------------------------------------------------------------------


class PulseAnsatz(ABC):
    """Abstract base class for pulse-shape ansatz initializations for GRAPE.

    Subclasses implement :meth:`make_schedule` to produce a
    :class:`ControlSchedule` that seeds the GRAPE optimizer instead of the
    default random or zero initialization.

    Use a concrete subclass as the ``initial_guess`` argument of
    :class:`~cqed_sim.optimal_control.GrapeConfig`::

        config = GrapeConfig(initial_guess=GaussianAnsatz(sigma_fraction=0.25))
    """

    @abstractmethod
    def make_schedule(self, problem: Any) -> ControlSchedule:
        """Build an initial :class:`ControlSchedule` for *problem*.

        Args:
            problem: A :class:`~cqed_sim.optimal_control.ControlProblem`
                instance holding the parameterization, control terms, and
                time grid.

        Returns:
            A :class:`ControlSchedule` with values clipped to the control
            bounds of *problem*.
        """
        raise NotImplementedError


@dataclass
class GaussianAnsatz(PulseAnsatz):
    """Gaussian-envelope seed for GRAPE.

    Places a Gaussian waveform centered at mid-sequence on every control
    channel.  The peak amplitude is a fixed fraction of each channel's bound.

    Args:
        sigma_fraction: Gaussian width as a fraction of total sequence
            duration (default 0.25).
        amplitude_fraction: Peak amplitude as a fraction of the channel's
            amplitude bound (default 0.5).
        seed: Optional RNG seed for random sign selection per channel.
    """

    sigma_fraction: float = 0.25
    amplitude_fraction: float = 0.5
    seed: int | None = None

    def make_schedule(self, problem: Any) -> ControlSchedule:
        rng = np.random.default_rng(self.seed)
        grid = problem.time_grid
        midpoints = np.asarray(grid.midpoints_s(), dtype=float)
        t_center = float(grid.duration_s) * 0.5
        sigma = max(float(grid.duration_s) * float(self.sigma_fraction), 1.0e-30)
        envelope = np.exp(-0.5 * ((midpoints - t_center) / sigma) ** 2)

        values = np.zeros((problem.n_controls, problem.n_slices), dtype=float)
        for index, term in enumerate(problem.control_terms):
            lower, upper = term.amplitude_bounds
            if np.isfinite(lower) and np.isfinite(upper):
                peak = float(self.amplitude_fraction) * 0.5 * (float(upper) - float(lower))
            else:
                peak = float(self.amplitude_fraction) * finite_bound_scale(lower, upper, fallback=1.0)
            sign = float(rng.choice([-1.0, 1.0]))
            values[index, :] = sign * peak * envelope

        return ControlSchedule(problem.parameterization, problem.parameterization.clip(values))


@dataclass
class DRAGAnsatz(PulseAnsatz):
    """Gaussian pulse with DRAG correction as a GRAPE seed.

    The in-phase (I) component is a Gaussian; the quadrature (Q) component
    receives a DRAG correction proportional to the time derivative of the
    Gaussian, scaled by *drag_alpha*.  This is especially useful for qubit
    drives where leakage to the second excited state is relevant.

    Args:
        sigma_fraction: Gaussian width as a fraction of total duration.
        amplitude_fraction: Peak amplitude fraction.
        drag_alpha: DRAG coefficient.  Set to 0 to disable the correction.
        i_control_index: Index of the I control term (default 0).
        q_control_index: Index of the Q control term (default 1).
        seed: Optional RNG seed.
    """

    sigma_fraction: float = 0.25
    amplitude_fraction: float = 0.5
    drag_alpha: float = 0.5
    i_control_index: int = 0
    q_control_index: int = 1
    seed: int | None = None

    def make_schedule(self, problem: Any) -> ControlSchedule:
        rng = np.random.default_rng(self.seed)
        grid = problem.time_grid
        midpoints = np.asarray(grid.midpoints_s(), dtype=float)
        t_center = float(grid.duration_s) * 0.5
        sigma = max(float(grid.duration_s) * float(self.sigma_fraction), 1.0e-30)

        gauss = np.exp(-0.5 * ((midpoints - t_center) / sigma) ** 2)
        d_gauss = gauss * (-(midpoints - t_center) / (sigma ** 2))

        n = problem.n_controls
        i_idx = int(self.i_control_index) % n
        q_idx = int(self.q_control_index) % n

        i_term = problem.control_terms[i_idx]
        lower, upper = i_term.amplitude_bounds
        if np.isfinite(lower) and np.isfinite(upper):
            peak = float(self.amplitude_fraction) * 0.5 * (float(upper) - float(lower))
        else:
            peak = float(self.amplitude_fraction) * finite_bound_scale(lower, upper, fallback=1.0)

        values = np.zeros((n, problem.n_slices), dtype=float)
        sign = float(rng.choice([-1.0, 1.0]))
        values[i_idx, :] = sign * peak * gauss
        if q_idx != i_idx:
            values[q_idx, :] = float(self.drag_alpha) * peak * d_gauss

        return ControlSchedule(problem.parameterization, problem.parameterization.clip(values))


@dataclass
class MultitoneAnsatz(PulseAnsatz):
    """Sum of sinusoidal tones as a GRAPE seed.

    Generates *n_tones* sinusoidal components with random phases for each
    control channel.  The total is normalized so the peak amplitude is a
    fixed fraction of each channel's bound.

    Args:
        n_tones: Number of frequency components per channel (default 3).
        amplitude_fraction: Peak amplitude fraction (default 0.3).
        seed: Optional RNG seed.
    """

    n_tones: int = 3
    amplitude_fraction: float = 0.3
    seed: int | None = None

    def make_schedule(self, problem: Any) -> ControlSchedule:
        rng = np.random.default_rng(self.seed)
        grid = problem.time_grid
        t = np.asarray(grid.midpoints_s(), dtype=float)
        T = max(float(grid.duration_s), 1.0e-30)

        values = np.zeros((problem.n_controls, problem.n_slices), dtype=float)
        for index, term in enumerate(problem.control_terms):
            lower, upper = term.amplitude_bounds
            if np.isfinite(lower) and np.isfinite(upper):
                peak = float(self.amplitude_fraction) * 0.5 * (float(upper) - float(lower))
            else:
                peak = float(self.amplitude_fraction) * finite_bound_scale(lower, upper, fallback=1.0)
            wave = np.zeros(len(t), dtype=float)
            for k in range(1, int(self.n_tones) + 1):
                phase = float(rng.uniform(0.0, 2.0 * np.pi))
                wave += np.sin(2.0 * np.pi * k * t / T + phase)
            norm = max(float(np.max(np.abs(wave))), 1.0e-30)
            values[index, :] = peak * wave / norm

        return ControlSchedule(problem.parameterization, problem.parameterization.clip(values))


@dataclass
class CustomAnsatz(PulseAnsatz):
    """User-supplied callable ansatz.

    Delegates to a user-provided callable so that arbitrary initialization
    strategies can be plugged in without subclassing.

    Args:
        fn: Callable with signature ``fn(problem) -> ControlSchedule``.
    """

    fn: Callable[[Any], ControlSchedule] = field(repr=False)

    def make_schedule(self, problem: Any) -> ControlSchedule:
        return self.fn(problem)


def ansatz_control_schedule(problem: Any, ansatz: PulseAnsatz) -> ControlSchedule:
    """Build an initial :class:`ControlSchedule` from a :class:`PulseAnsatz`.

    This is the low-level dispatch helper used internally by
    :class:`~cqed_sim.optimal_control.GrapeSolver`.  Advanced users can call
    it directly to inspect the ansatz schedule before launching GRAPE.

    Args:
        problem: A :class:`~cqed_sim.optimal_control.ControlProblem`.
        ansatz: Any :class:`PulseAnsatz` subclass instance.

    Returns:
        A :class:`ControlSchedule` seeded by the ansatz.
    """
    return ansatz.make_schedule(problem)


__all__ = [
    "zero_control_schedule",
    "random_control_schedule",
    "warm_start_schedule",
    "PulseAnsatz",
    "GaussianAnsatz",
    "DRAGAnsatz",
    "MultitoneAnsatz",
    "CustomAnsatz",
    "ansatz_control_schedule",
]
