from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class TimeGridResult:
    mode: str
    dt: float
    raw: np.ndarray
    snapped: np.ndarray
    ticks: np.ndarray
    grid_residual: np.ndarray


@dataclass(frozen=True)
class SlewConstraintResult:
    samples_raw: np.ndarray
    samples_projected: np.ndarray
    max_slew: float
    max_violation: float
    violation_count: int
    penalty: float


@dataclass(frozen=True)
class ToneSpacingResult:
    freqs_raw: np.ndarray
    freqs_projected: np.ndarray
    max_count_violation: float
    min_spacing: float
    spacing_penalty: float
    forbidden_penalty: float
    total_penalty: float


def snap_times_to_grid(times: Iterable[float], dt: float = 1e-9, mode: str = "hard") -> TimeGridResult:
    arr = np.asarray(list(times), dtype=float)
    if arr.size == 0:
        z = np.asarray([], dtype=float)
        zi = np.asarray([], dtype=int)
        return TimeGridResult(mode=mode, dt=float(dt), raw=z, snapped=z, ticks=zi, grid_residual=z)

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if mode not in {"hard", "soft"}:
        raise ValueError("mode must be 'hard' or 'soft'.")

    ticks = np.maximum(1, np.rint(arr / dt).astype(int))
    snapped = ticks.astype(float) * dt
    residual = arr - snapped
    return TimeGridResult(mode=mode, dt=float(dt), raw=arr, snapped=snapped, ticks=ticks, grid_residual=residual)


def piecewise_constant_samples(amplitudes: Iterable[float], durations: Iterable[float], dt: float) -> np.ndarray:
    amp = np.asarray(list(amplitudes), dtype=float)
    dur = np.asarray(list(durations), dtype=float)
    if amp.size != dur.size:
        raise ValueError("amplitudes and durations must have matching length.")
    if amp.size == 0:
        return np.asarray([], dtype=float)
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    chunks: list[np.ndarray] = []
    for a, t in zip(amp, dur):
        n = max(1, int(np.rint(float(t) / dt)))
        chunks.append(np.full(n, float(a), dtype=float))
    return np.concatenate(chunks)


def enforce_slew_limit(samples: Iterable[float], dt: float, s_max: float, mode: str = "penalty") -> SlewConstraintResult:
    x = np.asarray(list(samples), dtype=float)
    if x.size < 2:
        return SlewConstraintResult(
            samples_raw=x,
            samples_projected=x.copy(),
            max_slew=0.0,
            max_violation=0.0,
            violation_count=0,
            penalty=0.0,
        )

    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    if s_max <= 0.0:
        raise ValueError("s_max must be positive.")

    dx = np.diff(x) / dt
    viol = np.maximum(0.0, np.abs(dx) - s_max)
    max_violation = float(np.max(viol)) if viol.size else 0.0
    violation_count = int(np.count_nonzero(viol > 0.0))
    max_slew = float(np.max(np.abs(dx))) if dx.size else 0.0
    penalty = float(np.sum(viol**2))

    if mode == "project":
        y = x.copy()
        limit = float(s_max * dt)
        for i in range(1, y.size):
            delta = y[i] - y[i - 1]
            if delta > limit:
                y[i] = y[i - 1] + limit
            elif delta < -limit:
                y[i] = y[i - 1] - limit
        dx_p = np.diff(y) / dt
        viol_p = np.maximum(0.0, np.abs(dx_p) - s_max)
        return SlewConstraintResult(
            samples_raw=x,
            samples_projected=y,
            max_slew=float(np.max(np.abs(dx_p))) if dx_p.size else 0.0,
            max_violation=float(np.max(viol_p)) if viol_p.size else 0.0,
            violation_count=int(np.count_nonzero(viol_p > 0.0)),
            penalty=float(np.sum(viol_p**2)),
        )

    return SlewConstraintResult(
        samples_raw=x,
        samples_projected=x.copy(),
        max_slew=max_slew,
        max_violation=max_violation,
        violation_count=violation_count,
        penalty=penalty,
    )


def _distance_to_band(freq: float, band: tuple[float, float]) -> float:
    lo, hi = float(band[0]), float(band[1])
    if lo > hi:
        lo, hi = hi, lo
    if freq < lo:
        return lo - freq
    if freq > hi:
        return freq - hi
    return -min(freq - lo, hi - freq)


def project_tone_frequencies(
    freqs: Iterable[float],
    domega_min: float,
    forbidden_bands: Iterable[tuple[float, float]] | None = None,
) -> np.ndarray:
    f = np.sort(np.asarray(list(freqs), dtype=float))
    if f.size == 0:
        return f
    out = f.copy()
    gap = float(max(0.0, domega_min))

    for i in range(1, out.size):
        if out[i] - out[i - 1] < gap:
            out[i] = out[i - 1] + gap

    bands = list(forbidden_bands or [])
    if bands:
        for i in range(out.size):
            for lo, hi in bands:
                lo_f, hi_f = (float(lo), float(hi)) if lo <= hi else (float(hi), float(lo))
                if lo_f <= out[i] <= hi_f:
                    left = lo_f - out[i]
                    right = hi_f - out[i]
                    out[i] = out[i] + left if abs(left) < abs(right) else out[i] + right
        out = np.sort(out)
        for i in range(1, out.size):
            if out[i] - out[i - 1] < gap:
                out[i] = out[i - 1] + gap

    return out


def evaluate_tone_spacing(
    freqs: Iterable[float],
    domega_min: float,
    ntones_max: int | None = None,
    forbidden_bands: Iterable[tuple[float, float]] | None = None,
    project: bool = False,
) -> ToneSpacingResult:
    raw = np.asarray(list(freqs), dtype=float)
    if raw.size == 0:
        z = np.asarray([], dtype=float)
        return ToneSpacingResult(
            freqs_raw=z,
            freqs_projected=z,
            max_count_violation=0.0,
            min_spacing=np.inf,
            spacing_penalty=0.0,
            forbidden_penalty=0.0,
            total_penalty=0.0,
        )

    projected = project_tone_frequencies(raw, domega_min, forbidden_bands) if project else raw.copy()
    sorted_freqs = np.sort(projected)

    spacing_pen = 0.0
    min_spacing = np.inf
    if sorted_freqs.size >= 2:
        diffs = np.diff(sorted_freqs)
        min_spacing = float(np.min(diffs))
        spacing_pen = float(np.sum(np.maximum(0.0, float(domega_min) - diffs) ** 2))

    forbidden_pen = 0.0
    for f in sorted_freqs:
        for band in (forbidden_bands or []):
            d = _distance_to_band(float(f), band)
            if d < 0.0:
                forbidden_pen += float((-d) ** 2)

    count_violation = 0.0
    if ntones_max is not None and sorted_freqs.size > int(ntones_max):
        count_violation = float((sorted_freqs.size - int(ntones_max)) ** 2)

    total = spacing_pen + forbidden_pen + count_violation
    return ToneSpacingResult(
        freqs_raw=raw,
        freqs_projected=sorted_freqs,
        max_count_violation=count_violation,
        min_spacing=min_spacing,
        spacing_penalty=spacing_pen,
        forbidden_penalty=forbidden_pen,
        total_penalty=total,
    )


def parallel_mean(values: Iterable[float], n_jobs: int = 1) -> float:
    arr = [float(v) for v in values]
    if len(arr) == 0:
        return 0.0
    if n_jobs <= 1:
        return float(np.mean(arr))
    with ThreadPoolExecutor(max_workers=int(n_jobs)) as ex:
        chunks = list(ex.map(float, arr))
    return float(np.mean(chunks))

