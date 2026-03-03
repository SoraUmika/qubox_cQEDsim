from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from cqed_sim.pulses.hardware import (
    HardwareConfig,
    apply_amplitude_quantization,
    apply_first_order_lowpass,
    apply_iq_distortion,
    apply_timing_quantization,
    apply_zoh,
)
from cqed_sim.pulses.pulse import Pulse


@dataclass
class CompiledChannel:
    baseband: np.ndarray
    distorted: np.ndarray
    rf: np.ndarray


@dataclass
class CompiledSequence:
    tlist: np.ndarray
    dt: float
    channels: dict[str, CompiledChannel]


class SequenceCompiler:
    def __init__(
        self,
        dt: float,
        hardware: dict[str, HardwareConfig] | None = None,
        crosstalk_matrix: dict[str, dict[str, float]] | None = None,
        enable_cache: bool = False,
    ):
        self.dt = dt
        self.hardware = hardware or {}
        self.crosstalk_matrix = crosstalk_matrix or {}
        self.enable_cache = enable_cache
        self._cache: dict[tuple, CompiledSequence] = {}

    def _pulse_key(self, p: Pulse) -> tuple:
        env_key = ("callable", id(p.envelope)) if callable(p.envelope) else ("array", hash(np.asarray(p.envelope).tobytes()))
        return (
            p.channel,
            float(p.t0),
            float(p.duration),
            env_key,
            float(p.carrier),
            float(p.phase),
            float(p.amp),
            float(p.drag),
            None if p.sample_rate is None else float(p.sample_rate),
            p.label,
        )

    def _grid(self, pulses: Iterable[Pulse], t_end: float | None = None) -> np.ndarray:
        pulses = list(pulses)
        if not pulses and t_end is None:
            raise ValueError("Need pulses or explicit t_end to build a timeline.")
        max_t = t_end if t_end is not None else max(p.t1 for p in pulses)
        n_pts = int(np.floor(max_t / self.dt + 1e-12)) + 1
        return np.arange(n_pts, dtype=float) * self.dt

    def compile(self, pulses: list[Pulse], t_end: float | None = None) -> CompiledSequence:
        if self.enable_cache:
            cache_key = (float(self.dt), t_end, tuple(self._pulse_key(p) for p in pulses), tuple(sorted(self.hardware.items(), key=lambda x: x[0])))
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        tlist = self._grid(pulses, t_end=t_end)
        by_channel: dict[str, np.ndarray] = {}
        for pulse in pulses:
            hw = self.hardware.get(pulse.channel, HardwareConfig())
            t0 = apply_timing_quantization(pulse.t0, hw.timing_quantum)
            shifted = Pulse(
                channel=pulse.channel,
                t0=t0,
                duration=pulse.duration,
                envelope=pulse.envelope,
                carrier=pulse.carrier + hw.if_freq,
                phase=pulse.phase,
                amp=pulse.amp,
                drag=pulse.drag,
                sample_rate=pulse.sample_rate,
                label=pulse.label,
            )
            by_channel.setdefault(pulse.channel, np.zeros_like(tlist, dtype=np.complex128))
            by_channel[pulse.channel] += shifted.sample(tlist)

        if self.crosstalk_matrix:
            mixed = {ch: sig.copy() for ch, sig in by_channel.items()}
            for src, dst_map in self.crosstalk_matrix.items():
                if src not in by_channel:
                    continue
                for dst, coeff in dst_map.items():
                    mixed.setdefault(dst, np.zeros_like(tlist, dtype=np.complex128))
                    mixed[dst] += coeff * by_channel[src]
            by_channel = mixed

        out: dict[str, CompiledChannel] = {}
        for channel, baseband in by_channel.items():
            hw = self.hardware.get(channel, HardwareConfig())
            bb = apply_zoh(baseband, hw.zoh_samples)
            bb = apply_first_order_lowpass(bb, self.dt, hw.lowpass_bw)
            bb = apply_amplitude_quantization(bb, hw.amplitude_bits)
            distorted, rf = apply_iq_distortion(bb, tlist, hw)
            out[channel] = CompiledChannel(baseband=bb, distorted=distorted, rf=rf)
        compiled = CompiledSequence(tlist=tlist, dt=self.dt, channels=out)
        if self.enable_cache:
            self._cache[cache_key] = compiled
        return compiled
