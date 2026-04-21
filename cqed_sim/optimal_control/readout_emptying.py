from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from cqed_sim.measurement import ReadoutChain
from cqed_sim.pulses import Pulse

from .parameterizations import CallableParameterization, ControlParameterSpec, PiecewiseConstantTimeGrid
from .problems import ControlTerm


@dataclass(frozen=True)
class ReadoutEmptyingSpec:
    """Specification for piecewise-constant readout emptying synthesis.

    Convention note
    ---------------
    ``detuning_center`` is the drive frequency offset relative to the midpoint
    between the dressed ``g`` and ``e`` resonator frequencies:

    ``omega_drive = omega_mid + detuning_center``.

    Internally, the linear replay model follows the existing
    :mod:`cqed_sim.measurement.readout_chain` sign convention

    ``dot(alpha) = -(kappa/2 + i * Delta) * alpha - i * epsilon(t)``

    with branch detunings defined as ``Delta = omega_resonator - omega_drive``.
    For the default two-branch dispersive model this gives

    ``Delta_g = -detuning_center - chi/2``
    ``Delta_e = -detuning_center + chi/2``.
    """

    kappa: float
    chi: float
    tau: float
    n_segments: int
    detuning_center: float = 0.0
    segment_times: tuple[float, ...] | None = None
    allow_complex_segments: bool = True
    target_states: tuple[str, ...] = ("g", "e")
    kerr: float = 0.0
    include_kerr_phase_correction: bool = False
    kerr_correction_strategy: str = "average_branch"
    replay_dt: float | None = None

    def __post_init__(self) -> None:
        if float(self.kappa) < 0.0:
            raise ValueError("ReadoutEmptyingSpec.kappa must be nonnegative.")
        if float(self.tau) <= 0.0:
            raise ValueError("ReadoutEmptyingSpec.tau must be positive.")
        if int(self.n_segments) <= 0:
            raise ValueError("ReadoutEmptyingSpec.n_segments must be positive.")
        target_states = tuple(str(label) for label in self.target_states)
        if not target_states:
            raise ValueError("ReadoutEmptyingSpec.target_states must be nonempty.")
        object.__setattr__(self, "kappa", float(self.kappa))
        object.__setattr__(self, "chi", float(self.chi))
        object.__setattr__(self, "tau", float(self.tau))
        object.__setattr__(self, "n_segments", int(self.n_segments))
        object.__setattr__(self, "detuning_center", float(self.detuning_center))
        object.__setattr__(self, "target_states", target_states)
        object.__setattr__(self, "kerr", float(self.kerr))
        object.__setattr__(self, "kerr_correction_strategy", str(self.kerr_correction_strategy))
        if self.replay_dt is not None and float(self.replay_dt) <= 0.0:
            raise ValueError("ReadoutEmptyingSpec.replay_dt must be positive when provided.")
        if self.segment_times is not None:
            edges = tuple(float(value) for value in self.segment_times)
            if len(edges) != self.n_segments + 1:
                raise ValueError(
                    "ReadoutEmptyingSpec.segment_times must contain exactly n_segments + 1 boundaries."
                )
            if abs(edges[0]) > 1.0e-15 or abs(edges[-1] - self.tau) > 1.0e-12:
                raise ValueError("ReadoutEmptyingSpec.segment_times must start at 0 and end at tau.")
            if any(stop <= start for start, stop in zip(edges[:-1], edges[1:])):
                raise ValueError("ReadoutEmptyingSpec.segment_times must be strictly increasing.")
            object.__setattr__(self, "segment_times", edges)


@dataclass(frozen=True)
class ReadoutEmptyingConstraints:
    amplitude_max: float | None = None
    enforce_zero_start: bool = False
    enforce_zero_end: bool = False
    favor_real_waveform: bool = False
    smoothness_weight: float = 0.0
    min_average_photons: float | None = None
    preferred_solution: str = "max_separation"

    def __post_init__(self) -> None:
        if self.amplitude_max is not None and float(self.amplitude_max) <= 0.0:
            raise ValueError("ReadoutEmptyingConstraints.amplitude_max must be positive when provided.")
        if float(self.smoothness_weight) < 0.0:
            raise ValueError("ReadoutEmptyingConstraints.smoothness_weight must be nonnegative.")
        if self.min_average_photons is not None and float(self.min_average_photons) <= 0.0:
            raise ValueError("ReadoutEmptyingConstraints.min_average_photons must be positive when provided.")
        preferred = str(self.preferred_solution).lower()
        if preferred not in {"max_separation", "min_norm"}:
            raise ValueError("preferred_solution must be 'max_separation' or 'min_norm'.")
        object.__setattr__(self, "smoothness_weight", float(self.smoothness_weight))
        object.__setattr__(self, "preferred_solution", preferred)
        if self.amplitude_max is not None:
            object.__setattr__(self, "amplitude_max", float(self.amplitude_max))
        if self.min_average_photons is not None:
            object.__setattr__(self, "min_average_photons", float(self.min_average_photons))


@dataclass(frozen=True)
class ReadoutResonatorBranch:
    label: str
    detuning: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "detuning", float(self.detuning))


@dataclass
class ReadoutEmptyingReplay:
    time_grid_s: np.ndarray
    command_waveform: np.ndarray
    trajectories: dict[str, np.ndarray]
    photon_numbers: dict[str, np.ndarray]
    final_alpha: dict[str, complex]
    final_n: dict[str, float]


@dataclass
class ReadoutEmptyingResult:
    spec: ReadoutEmptyingSpec
    constraints: ReadoutEmptyingConstraints | None
    segment_amplitudes: np.ndarray
    segment_edges_s: np.ndarray
    time_grid_s: np.ndarray
    command_waveform: np.ndarray
    trajectories: dict[str, np.ndarray]
    final_alpha: dict[str, complex]
    final_n: dict[str, float]
    metrics: dict[str, float]
    diagnostics: dict[str, Any]


def _segment_edges_s(spec: ReadoutEmptyingSpec) -> np.ndarray:
    if spec.segment_times is not None:
        return np.asarray(spec.segment_times, dtype=float)
    return np.linspace(0.0, float(spec.tau), int(spec.n_segments) + 1, dtype=float)


def default_readout_emptying_branches(spec: ReadoutEmptyingSpec) -> tuple[ReadoutResonatorBranch, ...]:
    branches: list[ReadoutResonatorBranch] = []
    for label in spec.target_states:
        if label == "g":
            detuning = -float(spec.detuning_center) - 0.5 * float(spec.chi)
        elif label == "e":
            detuning = -float(spec.detuning_center) + 0.5 * float(spec.chi)
        else:
            raise ValueError(
                "Default readout-emptying branches currently support only ('g', 'e'). "
                "Pass explicit ReadoutResonatorBranch values for other targets."
            )
        branches.append(ReadoutResonatorBranch(label=label, detuning=detuning))
    return tuple(branches)


def _segment_terminal_coefficient(
    detuning: float,
    kappa: float,
    t_start: float,
    t_stop: float,
    tau: float,
) -> complex:
    """Exact finite-width segment coefficient for the terminal cavity amplitude.

    This is the exact piecewise-constant analogue of the linear cavity-reset
    response used in McClure et al., Phys. Rev. Applied 5, 011001 (2016), and
    the state-dependent dispersive-response construction described by Jerger
    et al., arXiv:2406.04891.
    """

    duration = float(t_stop) - float(t_start)
    if duration <= 0.0:
        raise ValueError("Segment duration must be positive.")
    lambda_eff = 0.5 * float(kappa) + 1j * float(detuning)
    if abs(lambda_eff) <= 1.0e-15:
        return complex(-1j * duration)
    tail = np.exp(-lambda_eff * (float(tau) - float(t_stop)))
    return complex(-1j * tail * (1.0 - np.exp(-lambda_eff * duration)) / lambda_eff)


def build_emptying_constraint_matrix(
    spec: ReadoutEmptyingSpec,
    branches: Sequence[ReadoutResonatorBranch] | None = None,
) -> np.ndarray:
    resolved_branches = default_readout_emptying_branches(spec) if branches is None else tuple(branches)
    edges = _segment_edges_s(spec)
    matrix = np.empty((len(resolved_branches), spec.n_segments), dtype=np.complex128)
    for branch_index, branch in enumerate(resolved_branches):
        for segment_index, (t_start, t_stop) in enumerate(zip(edges[:-1], edges[1:])):
            matrix[branch_index, segment_index] = _segment_terminal_coefficient(
                detuning=branch.detuning,
                kappa=spec.kappa,
                t_start=float(t_start),
                t_stop=float(t_stop),
                tau=spec.tau,
            )
    return matrix


def compute_emptying_null_space(A: np.ndarray, rtol: float = 1.0e-12) -> np.ndarray:
    matrix = np.asarray(A)
    if matrix.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix.")
    _, singular_values, vh = np.linalg.svd(matrix, full_matrices=True)
    if singular_values.size == 0:
        return np.eye(matrix.shape[1], dtype=matrix.dtype)
    tolerance = float(rtol) * max(matrix.shape) * float(singular_values[0])
    rank = int(np.sum(singular_values > tolerance))
    return np.asarray(vh[rank:, :].conj().T, dtype=matrix.dtype)


def _svd_diagnostics(A: np.ndarray, rtol: float) -> dict[str, Any]:
    _, singular_values, _ = np.linalg.svd(np.asarray(A), full_matrices=False)
    if singular_values.size == 0:
        return {
            "singular_values": [],
            "rank": 0,
            "condition_number": 0.0,
            "null_space_dimension": int(np.asarray(A).shape[1]),
        }
    tolerance = float(rtol) * max(np.asarray(A).shape) * float(singular_values[0])
    rank = int(np.sum(singular_values > tolerance))
    condition_number = float(np.inf) if singular_values[-1] <= 1.0e-15 else float(singular_values[0] / singular_values[-1])
    return {
        "singular_values": [float(value) for value in singular_values],
        "rank": rank,
        "condition_number": condition_number,
        "null_space_dimension": int(np.asarray(A).shape[1] - rank),
    }


def _free_segment_mask(spec: ReadoutEmptyingSpec, constraints: ReadoutEmptyingConstraints | None) -> np.ndarray:
    mask = np.ones(int(spec.n_segments), dtype=bool)
    if constraints is None:
        return mask
    if constraints.enforce_zero_start:
        mask[0] = False
    if constraints.enforce_zero_end:
        mask[-1] = False
    return mask


def _embed_basis(full_size: int, free_mask: np.ndarray, basis_free: np.ndarray) -> np.ndarray:
    embedded = np.zeros((int(full_size), int(basis_free.shape[1])), dtype=np.complex128)
    embedded[np.asarray(free_mask, dtype=bool), :] = np.asarray(basis_free, dtype=np.complex128)
    return embedded


def _build_replay_grid(segment_edges_s: np.ndarray, replay_dt_s: float | None = None) -> np.ndarray:
    times = [float(segment_edges_s[0])]
    durations = np.diff(np.asarray(segment_edges_s, dtype=float))
    if replay_dt_s is None:
        subdivisions = [max(16, int(np.ceil(duration / max(np.min(durations) / 8.0, 1.0e-15)))) for duration in durations]
    else:
        subdivisions = [max(1, int(np.ceil(duration / float(replay_dt_s)))) for duration in durations]
    for start, stop, substeps in zip(segment_edges_s[:-1], segment_edges_s[1:], subdivisions):
        local = np.linspace(float(start), float(stop), int(substeps) + 1, dtype=float)[1:]
        times.extend(float(value) for value in local)
    return np.asarray(times, dtype=float)


def _waveform_on_intervals(segment_edges_s: np.ndarray, segment_amplitudes: np.ndarray, time_grid_s: np.ndarray) -> np.ndarray:
    left = np.asarray(time_grid_s[:-1], dtype=float)
    right = np.asarray(time_grid_s[1:], dtype=float)
    midpoints = 0.5 * (left + right)
    indices = np.searchsorted(np.asarray(segment_edges_s[1:], dtype=float), midpoints, side="right")
    indices = np.clip(indices, 0, len(segment_amplitudes) - 1)
    return np.asarray(segment_amplitudes, dtype=np.complex128)[indices]


def _make_piecewise_envelope(
    segment_edges_s: np.ndarray,
    segment_amplitudes: np.ndarray,
    *,
    t0: float = 0.0,
) -> Callable[[np.ndarray], np.ndarray]:
    edges = np.asarray(segment_edges_s, dtype=float)
    amplitudes = np.asarray(segment_amplitudes, dtype=np.complex128)
    duration = float(edges[-1] - edges[0])
    normalized_edges = (edges - float(edges[0])) / duration

    def envelope(t_rel: np.ndarray) -> np.ndarray:
        t_rel = np.asarray(t_rel, dtype=float)
        indices = np.searchsorted(normalized_edges[1:], t_rel, side="right")
        indices = np.clip(indices, 0, amplitudes.size - 1)
        values = amplitudes[indices]
        out_of_support = (t_rel < 0.0) | (t_rel >= 1.0)
        if np.any(out_of_support):
            values = np.array(values, copy=True)
            values[out_of_support] = 0.0
        return np.asarray(values, dtype=np.complex128)

    return envelope


def replay_linear_readout_branches(
    spec: ReadoutEmptyingSpec,
    segment_amplitudes: np.ndarray,
    *,
    branches: Sequence[ReadoutResonatorBranch] | None = None,
    replay_dt: float | None = None,
    initial_alpha: dict[str, complex] | None = None,
) -> ReadoutEmptyingReplay:
    resolved_branches = default_readout_emptying_branches(spec) if branches is None else tuple(branches)
    edges = _segment_edges_s(spec)
    time_grid = _build_replay_grid(edges, replay_dt_s=spec.replay_dt if replay_dt is None else replay_dt)
    interval_values = _waveform_on_intervals(edges, segment_amplitudes, time_grid)
    trajectories: dict[str, np.ndarray] = {}
    photon_numbers: dict[str, np.ndarray] = {}
    final_alpha: dict[str, complex] = {}
    final_n: dict[str, float] = {}
    initial_alpha = {} if initial_alpha is None else {str(k): complex(v) for k, v in initial_alpha.items()}

    for branch in resolved_branches:
        lambda_eff = 0.5 * float(spec.kappa) + 1j * float(branch.detuning)
        alpha = np.empty(time_grid.size, dtype=np.complex128)
        alpha[0] = complex(initial_alpha.get(branch.label, 0.0))
        for idx, epsilon in enumerate(interval_values):
            dt = float(time_grid[idx + 1] - time_grid[idx])
            if abs(lambda_eff) <= 1.0e-15:
                alpha[idx + 1] = alpha[idx] - 1j * complex(epsilon) * dt
            else:
                alpha_ss = -1j * complex(epsilon) / lambda_eff
                decay = np.exp(-lambda_eff * dt)
                alpha[idx + 1] = alpha_ss + (alpha[idx] - alpha_ss) * decay
        trajectories[branch.label] = alpha
        photons = np.abs(alpha) ** 2
        photon_numbers[branch.label] = photons
        final_alpha[branch.label] = complex(alpha[-1])
        final_n[branch.label] = float(photons[-1])

    command_waveform = np.empty(time_grid.size, dtype=np.complex128)
    command_waveform[:-1] = interval_values
    command_waveform[-1] = interval_values[-1] if interval_values.size else 0.0
    return ReadoutEmptyingReplay(
        time_grid_s=time_grid,
        command_waveform=command_waveform,
        trajectories=trajectories,
        photon_numbers=photon_numbers,
        final_alpha=final_alpha,
        final_n=final_n,
    )


def replay_kerr_readout_branches(
    spec: ReadoutEmptyingSpec,
    segment_amplitudes: np.ndarray,
    *,
    branches: Sequence[ReadoutResonatorBranch] | None = None,
    replay_dt: float | None = None,
    initial_alpha: dict[str, complex] | None = None,
) -> ReadoutEmptyingReplay:
    resolved_branches = default_readout_emptying_branches(spec) if branches is None else tuple(branches)
    if abs(float(spec.kerr)) <= 1.0e-18:
        return replay_linear_readout_branches(
            spec,
            segment_amplitudes,
            branches=resolved_branches,
            replay_dt=replay_dt,
            initial_alpha=initial_alpha,
        )
    edges = _segment_edges_s(spec)
    time_grid = _build_replay_grid(edges, replay_dt_s=spec.replay_dt if replay_dt is None else replay_dt)
    interval_values = _waveform_on_intervals(edges, segment_amplitudes, time_grid)
    trajectories: dict[str, np.ndarray] = {}
    photon_numbers: dict[str, np.ndarray] = {}
    final_alpha: dict[str, complex] = {}
    final_n: dict[str, float] = {}
    initial_alpha = {} if initial_alpha is None else {str(k): complex(v) for k, v in initial_alpha.items()}

    def derivative(alpha: complex, *, detuning: float, epsilon: complex) -> complex:
        nonlinear_detuning = float(detuning) + float(spec.kerr) * (abs(alpha) ** 2)
        return -(0.5 * float(spec.kappa) + 1j * nonlinear_detuning) * alpha - 1j * complex(epsilon)

    for branch in resolved_branches:
        alpha = np.empty(time_grid.size, dtype=np.complex128)
        alpha[0] = complex(initial_alpha.get(branch.label, 0.0))
        for idx, epsilon in enumerate(interval_values):
            dt = float(time_grid[idx + 1] - time_grid[idx])
            current = complex(alpha[idx])
            k1 = derivative(current, detuning=branch.detuning, epsilon=epsilon)
            k2 = derivative(current + 0.5 * dt * k1, detuning=branch.detuning, epsilon=epsilon)
            k3 = derivative(current + 0.5 * dt * k2, detuning=branch.detuning, epsilon=epsilon)
            k4 = derivative(current + dt * k3, detuning=branch.detuning, epsilon=epsilon)
            alpha[idx + 1] = current + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectories[branch.label] = alpha
        photons = np.abs(alpha) ** 2
        photon_numbers[branch.label] = photons
        final_alpha[branch.label] = complex(alpha[-1])
        final_n[branch.label] = float(photons[-1])

    command_waveform = np.empty(time_grid.size, dtype=np.complex128)
    command_waveform[:-1] = interval_values
    command_waveform[-1] = interval_values[-1] if interval_values.size else 0.0
    return ReadoutEmptyingReplay(
        time_grid_s=time_grid,
        command_waveform=command_waveform,
        trajectories=trajectories,
        photon_numbers=photon_numbers,
        final_alpha=final_alpha,
        final_n=final_n,
    )


def _pairwise_branch_separation(replay: ReadoutEmptyingReplay) -> float:
    labels = tuple(replay.trajectories)
    if len(labels) < 2:
        return 0.0
    total = 0.0
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            diff = replay.trajectories[left_label] - replay.trajectories[right_label]
            total += float(np.trapezoid(np.abs(diff) ** 2, x=replay.time_grid_s))
    return total


def _waveform_l2_norm(segment_edges_s: np.ndarray, segment_amplitudes: np.ndarray) -> float:
    durations = np.diff(np.asarray(segment_edges_s, dtype=float))
    amplitudes = np.asarray(segment_amplitudes, dtype=np.complex128)
    return float(np.sqrt(np.sum(durations * np.abs(amplitudes) ** 2)))


def _metrics_from_replay(
    replay: ReadoutEmptyingReplay,
    *,
    segment_edges_s: np.ndarray,
    segment_amplitudes: np.ndarray,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "waveform_l2_norm": _waveform_l2_norm(segment_edges_s, segment_amplitudes),
        "waveform_peak_amplitude": float(np.max(np.abs(segment_amplitudes))) if len(segment_amplitudes) else 0.0,
        "integrated_branch_separation": _pairwise_branch_separation(replay),
        "max_final_residual_photons": max((float(value) for value in replay.final_n.values()), default=0.0),
    }
    duration = float(replay.time_grid_s[-1] - replay.time_grid_s[0]) if replay.time_grid_s.size >= 2 else 1.0
    for label, photons in replay.photon_numbers.items():
        metrics[f"final_residual_photons_{label}"] = float(replay.final_n[label])
        metrics[f"peak_photons_{label}"] = float(np.max(photons))
        metrics[f"avg_photons_{label}"] = float(np.trapezoid(photons, x=replay.time_grid_s) / max(duration, 1.0e-15))
    return metrics


def _real_rotation(epsilon: np.ndarray) -> np.ndarray:
    data = np.asarray(epsilon, dtype=np.complex128)
    moment = np.sum(data * data)
    if abs(moment) <= 1.0e-18:
        return data
    return np.asarray(data * np.exp(-0.5j * np.angle(moment)), dtype=np.complex128)


def _separation_gramian(
    spec: ReadoutEmptyingSpec,
    basis: np.ndarray,
    *,
    branches: Sequence[ReadoutResonatorBranch],
) -> np.ndarray:
    columns: list[np.ndarray] = []
    for column_index in range(basis.shape[1]):
        replay = replay_linear_readout_branches(
            spec,
            basis[:, column_index],
            branches=branches,
            replay_dt=spec.replay_dt,
        )
        labels = tuple(replay.trajectories)
        if len(labels) < 2:
            weighted = np.zeros(max(replay.time_grid_s.size - 1, 1), dtype=np.complex128)
        else:
            pieces: list[np.ndarray] = []
            weights = np.sqrt(np.diff(replay.time_grid_s))
            for left_index, left_label in enumerate(labels):
                for right_label in labels[left_index + 1 :]:
                    diff = replay.trajectories[left_label][:-1] - replay.trajectories[right_label][:-1]
                    pieces.append(diff * weights)
            weighted = np.concatenate(pieces) if pieces else np.zeros(max(replay.time_grid_s.size - 1, 1), dtype=np.complex128)
        columns.append(np.asarray(weighted, dtype=np.complex128))
    feature_matrix = np.column_stack(columns) if columns else np.zeros((0, 0), dtype=np.complex128)
    return np.asarray(feature_matrix.conj().T @ feature_matrix, dtype=np.complex128)


def _smoothness_gramian(basis: np.ndarray) -> np.ndarray:
    if basis.shape[0] <= 1:
        return np.zeros((basis.shape[1], basis.shape[1]), dtype=np.complex128)
    diff = np.diff(np.asarray(basis, dtype=np.complex128), axis=0)
    return np.asarray(diff.conj().T @ diff, dtype=np.complex128)


def select_min_norm_solution(
    basis: np.ndarray,
    *,
    favor_real_waveform: bool = False,
) -> np.ndarray:
    reference = np.ones(basis.shape[0], dtype=np.complex128)
    coeffs = np.asarray(basis.conj().T @ reference, dtype=np.complex128)
    candidate = np.asarray(basis @ coeffs, dtype=np.complex128)
    if np.linalg.norm(candidate) <= 1.0e-14:
        candidate = np.asarray(basis[:, 0], dtype=np.complex128)
    if favor_real_waveform:
        candidate = _real_rotation(candidate)
    norm = float(np.linalg.norm(candidate))
    return candidate if norm <= 1.0e-15 else np.asarray(candidate / norm, dtype=np.complex128)


def select_max_separation_solution(
    spec: ReadoutEmptyingSpec,
    basis: np.ndarray,
    *,
    branches: Sequence[ReadoutResonatorBranch],
    smoothness_weight: float = 0.0,
    favor_real_waveform: bool = False,
    real_only: bool = False,
) -> np.ndarray:
    gramian = _separation_gramian(spec, basis, branches=branches)
    if float(smoothness_weight) > 0.0:
        gramian = gramian - float(smoothness_weight) * _smoothness_gramian(basis)
    if real_only:
        eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(gramian.real, dtype=float))
        coords = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=float)
        candidate = np.asarray(basis @ coords, dtype=np.complex128)
    else:
        hermitian = 0.5 * (gramian + gramian.conj().T)
        eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
        coords = np.asarray(eigenvectors[:, int(np.argmax(eigenvalues))], dtype=np.complex128)
        candidate = np.asarray(basis @ coords, dtype=np.complex128)
    if favor_real_waveform:
        candidate = _real_rotation(candidate)
    norm = float(np.linalg.norm(candidate))
    return candidate if norm <= 1.0e-15 else np.asarray(candidate / norm, dtype=np.complex128)


def _resolve_scale(
    segment_amplitudes: np.ndarray,
    spec: ReadoutEmptyingSpec,
    constraints: ReadoutEmptyingConstraints | None,
    branches: Sequence[ReadoutResonatorBranch],
) -> tuple[float, dict[str, float]]:
    diagnostics: dict[str, float] = {}
    peak = float(np.max(np.abs(segment_amplitudes))) if len(segment_amplitudes) else 0.0
    scale = 1.0
    if peak > 1.0e-15:
        if constraints is not None and constraints.amplitude_max is not None:
            scale = float(constraints.amplitude_max) / peak
        else:
            scale = 1.0 / peak
    diagnostics["scale_after_peak_normalization"] = float(scale)
    if constraints is not None and constraints.min_average_photons is not None:
        replay = replay_linear_readout_branches(spec, scale * np.asarray(segment_amplitudes, dtype=np.complex128), branches=branches)
        averages = [float(np.trapezoid(photons, x=replay.time_grid_s) / max(spec.tau, 1.0e-15)) for photons in replay.photon_numbers.values()]
        current_mean = float(np.mean(averages)) if averages else 0.0
        if current_mean > 1.0e-18:
            scale *= float(np.sqrt(constraints.min_average_photons / current_mean))
    if constraints is not None and constraints.amplitude_max is not None and peak > 1.0e-15:
        scale = min(scale, float(constraints.amplitude_max) / peak)
    diagnostics["final_scale_factor"] = float(scale)
    return float(scale), diagnostics


def build_kerr_phase_correction(
    spec: ReadoutEmptyingSpec,
    replay: ReadoutEmptyingReplay,
    *,
    strategy: str | None = None,
) -> dict[str, np.ndarray | float | str]:
    resolved = str(spec.kerr_correction_strategy if strategy is None else strategy).lower()
    if resolved == "average_branch" or resolved == "weighted_average":
        photon_reference = np.mean(np.vstack([values for values in replay.photon_numbers.values()]), axis=0)
    elif resolved == "g_branch":
        photon_reference = np.asarray(replay.photon_numbers["g"], dtype=float)
    elif resolved == "e_branch":
        photon_reference = np.asarray(replay.photon_numbers["e"], dtype=float)
    else:
        raise ValueError(
            "Unsupported Kerr correction strategy. Use 'average_branch', 'weighted_average', 'g_branch', or 'e_branch'."
        )
    dt = np.diff(replay.time_grid_s)
    phase = np.zeros_like(replay.time_grid_s, dtype=float)
    if dt.size:
        # In the repository readout-envelope convention, the corrective phase
        # must oppose the nonlinear cavity phase accumulated by
        # dot(alpha) = -(kappa/2 + i(Delta + K|alpha|^2)) alpha - i epsilon(t).
        phase[1:] = -float(spec.kerr) * np.cumsum(0.5 * (photon_reference[:-1] + photon_reference[1:]) * dt)
    chirp = np.gradient(phase, replay.time_grid_s, edge_order=1) if replay.time_grid_s.size >= 2 else np.zeros_like(phase)
    return {
        "strategy": resolved,
        "phase_rad": phase,
        "instantaneous_chirp_rad_s": chirp,
        "reference_photons": np.asarray(photon_reference, dtype=float),
        "peak_phase_rad": float(np.max(np.abs(phase))) if phase.size else 0.0,
    }


def apply_phase_chirp(
    segment_amplitudes: np.ndarray,
    segment_edges_s: np.ndarray,
    phase_profile_rad: np.ndarray,
    time_grid_s: np.ndarray,
) -> np.ndarray:
    midpoints = 0.5 * (np.asarray(segment_edges_s[:-1], dtype=float) + np.asarray(segment_edges_s[1:], dtype=float))
    phase_mid = np.interp(midpoints, np.asarray(time_grid_s, dtype=float), np.asarray(phase_profile_rad, dtype=float))
    return np.asarray(segment_amplitudes, dtype=np.complex128) * np.exp(1j * phase_mid)


def synthesize_readout_emptying_pulse(
    spec: ReadoutEmptyingSpec,
    constraints: ReadoutEmptyingConstraints | None = None,
) -> ReadoutEmptyingResult:
    resolved_constraints = ReadoutEmptyingConstraints() if constraints is None else constraints
    branches = default_readout_emptying_branches(spec)
    edges = _segment_edges_s(spec)
    constraint_matrix = build_emptying_constraint_matrix(spec, branches)
    free_mask = _free_segment_mask(spec, resolved_constraints)
    if not np.any(free_mask):
        raise ValueError("Endpoint constraints removed all free segments.")

    if spec.allow_complex_segments:
        reduced_matrix = np.asarray(constraint_matrix[:, free_mask], dtype=np.complex128)
        basis_free = compute_emptying_null_space(reduced_matrix)
        svd_info = _svd_diagnostics(reduced_matrix, rtol=1.0e-12)
        real_only = False
    else:
        reduced_complex = np.asarray(constraint_matrix[:, free_mask], dtype=np.complex128)
        reduced_matrix = np.vstack([reduced_complex.real, reduced_complex.imag])
        basis_free = compute_emptying_null_space(reduced_matrix).astype(float, copy=False)
        svd_info = _svd_diagnostics(reduced_matrix, rtol=1.0e-12)
        real_only = True
    if basis_free.shape[1] <= 0:
        raise ValueError("No nontrivial readout-emptying null space was found for the requested specification.")
    basis = _embed_basis(spec.n_segments, free_mask, basis_free)

    if resolved_constraints.preferred_solution == "min_norm":
        selected = select_min_norm_solution(
            basis,
            favor_real_waveform=resolved_constraints.favor_real_waveform,
        )
    else:
        selected = select_max_separation_solution(
            spec,
            basis,
            branches=branches,
            smoothness_weight=resolved_constraints.smoothness_weight,
            favor_real_waveform=resolved_constraints.favor_real_waveform,
            real_only=real_only,
        )
    if real_only:
        selected = np.asarray(selected.real, dtype=np.complex128)

    scale, scale_diagnostics = _resolve_scale(selected, spec, resolved_constraints, branches)
    linear_segments = np.asarray(scale * selected, dtype=np.complex128)
    linear_replay = replay_linear_readout_branches(spec, linear_segments, branches=branches)

    active_segments = linear_segments
    kerr_diagnostics: dict[str, Any] = {}
    nonlinear_replay: ReadoutEmptyingReplay | None = None
    if abs(float(spec.kerr)) > 1.0e-18:
        uncorrected = replay_kerr_readout_branches(spec, linear_segments, branches=branches)
        nonlinear_replay = uncorrected
        if spec.include_kerr_phase_correction:
            phase_correction = build_kerr_phase_correction(spec, linear_replay)
            corrected_segments = apply_phase_chirp(
                linear_segments,
                edges,
                np.asarray(phase_correction["phase_rad"], dtype=float),
                linear_replay.time_grid_s,
            )
            corrected = replay_kerr_readout_branches(spec, corrected_segments, branches=branches)
            active_segments = corrected_segments
            nonlinear_replay = corrected
            kerr_diagnostics = {
                "strategy": str(phase_correction["strategy"]),
                "phase_rad": np.asarray(phase_correction["phase_rad"], dtype=float),
                "instantaneous_chirp_rad_s": np.asarray(phase_correction["instantaneous_chirp_rad_s"], dtype=float),
                "reference_photons": np.asarray(phase_correction["reference_photons"], dtype=float),
                "peak_phase_rad": float(phase_correction["peak_phase_rad"]),
                "uncorrected_final_n": {key: float(value) for key, value in uncorrected.final_n.items()},
                "corrected_final_n": {key: float(value) for key, value in corrected.final_n.items()},
                "residual_improvement": float(
                    max(uncorrected.final_n.values(), default=0.0) - max(corrected.final_n.values(), default=0.0)
                ),
            }
        else:
            kerr_diagnostics = {
                "strategy": "none",
                "uncorrected_final_n": {key: float(value) for key, value in uncorrected.final_n.items()},
            }

    active_replay = nonlinear_replay if nonlinear_replay is not None else linear_replay
    metrics = _metrics_from_replay(active_replay, segment_edges_s=edges, segment_amplitudes=active_segments)
    metrics.update(
        {
            "null_space_dimension": float(int(basis.shape[1])),
            "constraint_condition_number": float(svd_info["condition_number"]),
            "kerr_phase_peak": float(kerr_diagnostics.get("peak_phase_rad", 0.0)),
        }
    )

    diagnostics: dict[str, Any] = {
        "branches": [
            {"label": str(branch.label), "detuning": float(branch.detuning)}
            for branch in branches
        ],
        "constraint_matrix": np.asarray(constraint_matrix, dtype=np.complex128),
        "free_segment_mask": np.asarray(free_mask, dtype=bool),
        "null_space_basis": np.asarray(basis, dtype=np.complex128),
        "linear_segment_amplitudes": np.asarray(linear_segments, dtype=np.complex128),
        "selection_mode": str(resolved_constraints.preferred_solution),
        "scale_diagnostics": scale_diagnostics,
        "linear_metrics": _metrics_from_replay(linear_replay, segment_edges_s=edges, segment_amplitudes=linear_segments),
        **svd_info,
    }
    if nonlinear_replay is not None:
        diagnostics["nonlinear_metrics"] = _metrics_from_replay(
            nonlinear_replay,
            segment_edges_s=edges,
            segment_amplitudes=active_segments,
        )
    if kerr_diagnostics:
        diagnostics["kerr_correction"] = kerr_diagnostics

    return ReadoutEmptyingResult(
        spec=spec,
        constraints=resolved_constraints,
        segment_amplitudes=np.asarray(active_segments, dtype=np.complex128),
        segment_edges_s=np.asarray(edges, dtype=float),
        time_grid_s=np.asarray(active_replay.time_grid_s, dtype=float),
        command_waveform=np.asarray(active_replay.command_waveform, dtype=np.complex128),
        trajectories={key: np.asarray(value, dtype=np.complex128) for key, value in active_replay.trajectories.items()},
        final_alpha={key: complex(value) for key, value in active_replay.final_alpha.items()},
        final_n={key: float(value) for key, value in active_replay.final_n.items()},
        metrics=metrics,
        diagnostics=diagnostics,
    )


def export_readout_emptying_to_pulse(
    result: ReadoutEmptyingResult,
    *,
    channel: str = "readout",
    t0: float = 0.0,
    carrier: float | None = None,
) -> Pulse:
    envelope = _make_piecewise_envelope(result.segment_edges_s, result.segment_amplitudes)
    return Pulse(
        channel=str(channel),
        t0=float(t0),
        duration=float(result.segment_edges_s[-1] - result.segment_edges_s[0]),
        envelope=envelope,
        carrier=0.0 if carrier is None else float(carrier),
        amp=1.0,
        phase=0.0,
        label="readout_emptying",
    )


def build_readout_emptying_parameterization(
    spec: ReadoutEmptyingSpec,
    constraints: ReadoutEmptyingConstraints | None = None,
    *,
    channel: str = "readout",
    control_terms: tuple[ControlTerm, ...] | None = None,
) -> CallableParameterization:
    result = synthesize_readout_emptying_pulse(spec, constraints)
    basis = np.asarray(result.diagnostics["null_space_basis"], dtype=np.complex128)
    real_only = not bool(spec.allow_complex_segments)
    if control_terms is None:
        zero = np.zeros((1, 1), dtype=np.complex128)
        resolved_control_terms = (
            ControlTerm(
                name=f"{channel}_I",
                operator=zero,
                export_channel=str(channel),
                drive_target=str(channel),
                quadrature="I",
            ),
            ControlTerm(
                name=f"{channel}_Q",
                operator=zero,
                export_channel=str(channel),
                drive_target=str(channel),
                quadrature="Q",
            ),
        )
    else:
        resolved_control_terms = tuple(control_terms)
    if len(resolved_control_terms) != 2:
        raise ValueError("Readout-emptying parameterizations require exactly two control terms ordered as I, Q.")
    time_grid = PiecewiseConstantTimeGrid(step_durations_s=tuple(np.diff(np.asarray(result.segment_edges_s, dtype=float))))
    free_mask = np.asarray(result.diagnostics["free_segment_mask"], dtype=bool)
    free_segments = np.asarray(result.segment_amplitudes, dtype=np.complex128)[free_mask]
    basis_free = np.asarray(basis[free_mask, :], dtype=np.complex128)
    coeffs = np.asarray(basis_free.conj().T @ free_segments, dtype=np.complex128)

    if real_only:
        defaults = np.asarray(coeffs.real, dtype=float)
    else:
        defaults = np.concatenate([coeffs.real, coeffs.imag]).astype(float, copy=False)

    parameter_specs: list[ControlParameterSpec] = []
    if constraints is not None and constraints.amplitude_max is not None:
        bound = max(
            float(constraints.amplitude_max),
            1.5 * float(np.max(np.abs(defaults))) if defaults.size else 1.0,
        )
    else:
        bound = float("inf")
    if real_only:
        for index, default in enumerate(defaults):
            parameter_specs.append(
                ControlParameterSpec(
                    name=f"readout_null_{index}",
                    lower_bound=-bound,
                    upper_bound=bound,
                    default=float(default),
                    units="rad/s",
                )
            )
    else:
        n_coords = int(basis.shape[1])
        for index in range(n_coords):
            parameter_specs.append(
                ControlParameterSpec(
                    name=f"readout_null_re_{index}",
                    lower_bound=-bound,
                    upper_bound=bound,
                    default=float(defaults[index]),
                    units="rad/s",
                )
            )
        for index in range(n_coords):
            parameter_specs.append(
                ControlParameterSpec(
                    name=f"readout_null_im_{index}",
                    lower_bound=-bound,
                    upper_bound=bound,
                    default=float(defaults[n_coords + index]),
                    units="rad/s",
                )
            )

    def _coords_to_segments(values: np.ndarray) -> np.ndarray:
        data = np.asarray(values, dtype=float).reshape(-1)
        if real_only:
            coords = data.astype(np.complex128)
        else:
            half = data.size // 2
            coords = data[:half] + 1j * data[half:]
        return np.asarray(basis @ coords, dtype=np.complex128)

    def evaluator(values, _time_grid, _control_terms) -> np.ndarray:
        epsilon = _coords_to_segments(values)
        return np.vstack([epsilon.real, epsilon.imag]).astype(float, copy=False)

    def pullback(gradient_command, values, _time_grid, _control_terms, _waveform) -> np.ndarray:
        gradient = np.asarray(gradient_command, dtype=float)
        basis_local = np.asarray(basis, dtype=np.complex128)
        grad_i = gradient[0]
        grad_q = gradient[1]
        if real_only:
            return np.asarray(grad_i @ basis_local.real + grad_q @ basis_local.imag, dtype=float)
        grad_re = np.asarray(grad_i @ basis_local.real + grad_q @ basis_local.imag, dtype=float)
        grad_im = np.asarray(-grad_i @ basis_local.imag + grad_q @ basis_local.real, dtype=float)
        return np.concatenate([grad_re, grad_im]).astype(float, copy=False)

    def metrics_evaluator(values, _time_grid, _control_terms, _command_values) -> dict[str, Any]:
        epsilon = _coords_to_segments(values)
        return {
            "readout_emptying": {
                "null_space_dimension": int(basis.shape[1]),
                "segment_count": int(spec.n_segments),
                "free_segment_count": int(np.count_nonzero(free_mask)),
                "max_abs_segment_amplitude": float(np.max(np.abs(epsilon))) if epsilon.size else 0.0,
            }
        }

    return CallableParameterization(
        time_grid=time_grid,
        control_terms=resolved_control_terms,
        parameter_specs=tuple(parameter_specs),
        evaluator=evaluator,
        pullback_evaluator=pullback,
        metrics_evaluator=metrics_evaluator,
    )


def evaluate_readout_emptying_with_chain(
    result: ReadoutEmptyingResult,
    chain: ReadoutChain,
    *,
    dt: float | None = None,
    shots_per_branch: int = 128,
    seed: int | None = None,
) -> dict[str, Any]:
    envelope = _make_piecewise_envelope(result.segment_edges_s, result.segment_amplitudes)
    resolved_dt = float(chain.dt if dt is None else dt)
    drive_frequency = float(chain.resonator.omega_r + 0.5 * result.spec.chi + result.spec.detuning_center)

    noiseless_traces = {
        label: chain.simulate_waveform(
            label,
            envelope,
            dt=resolved_dt,
            duration=float(result.spec.tau),
            drive_frequency=drive_frequency,
            chi=float(result.spec.chi),
            include_noise=False,
        )
        for label in result.spec.target_states
    }
    iq_centers = {
        label: np.asarray(trace.iq_sample, dtype=float)
        for label, trace in noiseless_traces.items()
    }
    labels = tuple(iq_centers)
    iq_separation = 0.0
    if len(labels) >= 2:
        iq_separation = float(np.linalg.norm(iq_centers[labels[0]] - iq_centers[labels[1]]))

    rng = np.random.default_rng(seed)
    sampled_iq: dict[str, np.ndarray] = {}
    accuracy = float("nan")
    if shots_per_branch > 0 and len(labels) >= 2:
        for label in labels:
            samples = []
            for _ in range(int(shots_per_branch)):
                sample = chain.simulate_waveform(
                    label,
                    envelope,
                    dt=resolved_dt,
                    duration=float(result.spec.tau),
                    drive_frequency=drive_frequency,
                    chi=float(result.spec.chi),
                    include_noise=True,
                    seed=int(rng.integers(0, 2**31 - 1)),
                ).iq_sample
                samples.append(np.asarray(sample, dtype=float))
            sampled_iq[label] = np.vstack(samples) if samples else np.zeros((0, 2), dtype=float)
        correct = 0
        total = 0
        for label, samples in sampled_iq.items():
            for sample in samples:
                distances = {candidate: float(np.linalg.norm(sample - center)) for candidate, center in iq_centers.items()}
                predicted = min(distances, key=distances.get)
                correct += int(predicted == label)
                total += 1
        accuracy = float(correct / total) if total else float("nan")

    metrics = {
        "measurement_chain_separation": float(iq_separation),
        "measurement_chain_accuracy": float(accuracy),
        "drive_frequency": float(drive_frequency),
        "dt": float(resolved_dt),
    }
    return {
        "metrics": metrics,
        "traces": noiseless_traces,
        "iq_centers": iq_centers,
        "sampled_iq": sampled_iq,
    }


__all__ = [
    "ReadoutEmptyingSpec",
    "ReadoutEmptyingConstraints",
    "ReadoutResonatorBranch",
    "ReadoutEmptyingReplay",
    "ReadoutEmptyingResult",
    "default_readout_emptying_branches",
    "build_emptying_constraint_matrix",
    "compute_emptying_null_space",
    "select_min_norm_solution",
    "select_max_separation_solution",
    "replay_linear_readout_branches",
    "replay_kerr_readout_branches",
    "build_kerr_phase_correction",
    "apply_phase_chirp",
    "synthesize_readout_emptying_pulse",
    "export_readout_emptying_to_pulse",
    "build_readout_emptying_parameterization",
    "evaluate_readout_emptying_with_chain",
]
