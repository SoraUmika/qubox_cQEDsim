from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqed_sim.core.drive_targets import TransmonTransitionDriveSpec

from .readout_chain import ReadoutResonator, _resolved_drive_steps


@dataclass(frozen=True)
class StrongReadoutMixingSpec:
    """Phenomenological strong-readout disturbance tied to occupancy and slew.

    This helper is intentionally operational rather than microscopic. It estimates the
    state-averaged readout occupancy from the linear dispersive resonator response, then
    turns on auxiliary transmon-transition drive envelopes once that occupancy becomes a
    significant fraction of ``n_crit``. The resulting envelopes can be attached to the
    standard pulse replay stack as additional coherent channels.
    """

    n_crit: float | None = None
    onset_ratio: float = 0.10
    occupancy_exponent: float = 1.20
    ge_scale: float = 0.090
    ef_scale: float = 0.050
    slew_ge_scale: float = 0.060
    slew_ef_scale: float = 0.030
    phase_lag: float = -np.pi / 3.0
    slew_phase: float = np.deg2rad(35.0)
    ge_levels: tuple[int, int] = (0, 1)
    ef_levels: tuple[int, int] = (1, 2)
    higher_ladder_scales: tuple[float, ...] = ()
    higher_ladder_start_level: int = 2
    higher_channel_prefix: str = "mix_high"


@dataclass
class StrongReadoutDisturbance:
    tlist: np.ndarray
    drive_envelope: np.ndarray
    alpha_g: np.ndarray
    alpha_e: np.ndarray
    mean_occupancy: np.ndarray
    activation: np.ndarray
    ge_envelope: np.ndarray
    ef_envelope: np.ndarray
    higher_envelopes: dict[str, np.ndarray]
    peak_mean_occupancy: float
    peak_activation: float


def infer_dispersive_coupling(
    *,
    omega_q: float,
    omega_mode: float,
    alpha: float,
    chi: float,
) -> float:
    """Infer ``g`` from the dispersive Duffing relation used by transmon studies."""

    if abs(alpha) <= 1.0e-30:
        raise ValueError("alpha must be non-zero when inferring coupling from chi.")
    delta = float(omega_q) - float(omega_mode)
    numerator = abs(float(chi)) * abs(delta) * abs(delta + float(alpha))
    return float(np.sqrt(max(numerator / abs(float(alpha)), 0.0)))


def estimate_dispersive_critical_photon_number(
    *,
    omega_q: float,
    omega_mode: float,
    alpha: float,
    chi: float | None = None,
    g: float | None = None,
) -> float:
    """Estimate the dispersive critical photon number ``n_crit = (Delta / 2g)^2``."""

    coupling = float(g) if g is not None else infer_dispersive_coupling(
        omega_q=omega_q,
        omega_mode=omega_mode,
        alpha=alpha,
        chi=0.0 if chi is None else float(chi),
    )
    if coupling <= 1.0e-30:
        return float("inf")
    delta = float(omega_q) - float(omega_mode)
    return float((delta / (2.0 * coupling)) ** 2)


def strong_readout_drive_targets(
    spec: StrongReadoutMixingSpec,
    *,
    ge_channel: str = "mix_ge",
    ef_channel: str = "mix_ef",
    max_transmon_level: int | None = None,
) -> dict[str, TransmonTransitionDriveSpec]:
    """Return structured drive targets for disturbance envelopes."""

    targets: dict[str, TransmonTransitionDriveSpec] = {}
    if abs(spec.ge_scale) > 0.0 or abs(spec.slew_ge_scale) > 0.0:
        targets[ge_channel] = TransmonTransitionDriveSpec(
            lower_level=int(spec.ge_levels[0]),
            upper_level=int(spec.ge_levels[1]),
        )
    if abs(spec.ef_scale) > 0.0 or abs(spec.slew_ef_scale) > 0.0:
        targets[ef_channel] = TransmonTransitionDriveSpec(
            lower_level=int(spec.ef_levels[0]),
            upper_level=int(spec.ef_levels[1]),
        )
    for offset, scale in enumerate(tuple(spec.higher_ladder_scales)):
        if abs(float(scale)) <= 0.0:
            continue
        lower = int(spec.higher_ladder_start_level) + int(offset)
        upper = lower + 1
        if max_transmon_level is not None and upper >= int(max_transmon_level):
            continue
        channel = f"{spec.higher_channel_prefix}_{lower}_{upper}"
        targets[channel] = TransmonTransitionDriveSpec(lower_level=lower, upper_level=upper)
    return targets


def build_strong_readout_disturbance(
    resonator: ReadoutResonator,
    drive_envelope: np.ndarray | complex | float,
    *,
    dt: float,
    spec: StrongReadoutMixingSpec | None = None,
    duration: float | None = None,
    drive_frequency: float | None = None,
    chi: float | None = None,
    initial_amplitude: complex = 0.0,
) -> StrongReadoutDisturbance:
    """Build occupancy-activated disturbance envelopes for strong readout replay."""

    spec = StrongReadoutMixingSpec() if spec is None else spec
    envelope = _resolved_drive_steps(drive_envelope, dt=float(dt), duration=duration)
    tlist, alpha_g = resonator.response_to_envelope(
        "g",
        envelope,
        dt=float(dt),
        drive_frequency=drive_frequency,
        chi=chi,
        initial_amplitude=initial_amplitude,
    )
    _, alpha_e = resonator.response_to_envelope(
        "e",
        envelope,
        dt=float(dt),
        drive_frequency=drive_frequency,
        chi=chi,
        initial_amplitude=initial_amplitude,
    )

    n_g = 0.5 * (np.abs(alpha_g[:-1]) ** 2 + np.abs(alpha_g[1:]) ** 2)
    n_e = 0.5 * (np.abs(alpha_e[:-1]) ** 2 + np.abs(alpha_e[1:]) ** 2)
    mean_occupancy = 0.5 * (n_g + n_e)

    if spec.n_crit is None or not np.isfinite(spec.n_crit) or spec.n_crit <= 0.0:
        activation = np.zeros_like(mean_occupancy, dtype=float)
    else:
        onset = max(float(spec.onset_ratio), 1.0e-12)
        ratio = mean_occupancy / float(spec.n_crit)
        activation = np.clip((ratio - onset) / onset, 0.0, 3.0) ** float(spec.occupancy_exponent)

    slew = np.concatenate([np.zeros(1, dtype=np.complex128), np.diff(envelope)])
    phase_term = np.exp(1j * float(spec.phase_lag))
    slew_phase_term = np.exp(1j * float(spec.slew_phase))
    ge_envelope = (
        float(spec.ge_scale) * activation * envelope * phase_term
        + float(spec.slew_ge_scale) * activation * slew * slew_phase_term
    ).astype(np.complex128)
    ef_envelope = (
        float(spec.ef_scale) * activation * envelope * phase_term
        + float(spec.slew_ef_scale) * activation * slew * slew_phase_term
    ).astype(np.complex128)
    higher_envelopes: dict[str, np.ndarray] = {}
    for offset, scale in enumerate(tuple(spec.higher_ladder_scales)):
        channel = f"{spec.higher_channel_prefix}_{int(spec.higher_ladder_start_level) + int(offset)}_{int(spec.higher_ladder_start_level) + int(offset) + 1}"
        higher_envelopes[channel] = (float(scale) * ef_envelope).astype(np.complex128)

    return StrongReadoutDisturbance(
        tlist=np.asarray(tlist, dtype=float),
        drive_envelope=np.asarray(envelope, dtype=np.complex128),
        alpha_g=np.asarray(alpha_g, dtype=np.complex128),
        alpha_e=np.asarray(alpha_e, dtype=np.complex128),
        mean_occupancy=np.asarray(mean_occupancy, dtype=float),
        activation=np.asarray(activation, dtype=float),
        ge_envelope=np.asarray(ge_envelope, dtype=np.complex128),
        ef_envelope=np.asarray(ef_envelope, dtype=np.complex128),
        higher_envelopes=higher_envelopes,
        peak_mean_occupancy=float(np.max(mean_occupancy) if mean_occupancy.size else 0.0),
        peak_activation=float(np.max(activation) if activation.size else 0.0),
    )


__all__ = [
    "StrongReadoutMixingSpec",
    "StrongReadoutDisturbance",
    "build_strong_readout_disturbance",
    "estimate_dispersive_critical_photon_number",
    "infer_dispersive_coupling",
    "strong_readout_drive_targets",
]
