from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .channel import HolographicChannel
from .channel_embedding import PurifiedChannelStep
from .config import BoundaryCondition, BurnInConfig, SamplingConfig
from .estimators import RunningStats
from .noise import BondNoiseChannel
from .results import BranchRecord, BurnInSummary, CorrelatorEstimate, ExactCorrelatorResult
from .schedules import ObservableSchedule
from .step_sequence import HolographicChannelSequence
from .utils import basis_vector, coerce_density_matrix, progress_wrapper, state_overlap_probability, trace_distance


def _resolve_schedule(value: ObservableSchedule | Sequence[Any]) -> ObservableSchedule:
    if isinstance(value, ObservableSchedule):
        return value
    return ObservableSchedule(value)


def _resolve_burn_in(value: BurnInConfig | int | None) -> BurnInConfig:
    if value is None:
        return BurnInConfig()
    if isinstance(value, BurnInConfig):
        return value
    return BurnInConfig(steps=int(value))


def _resolve_boundary(value: BoundaryCondition | None) -> BoundaryCondition | None:
    return value


def _resolve_sampling_config(
    shots: int | SamplingConfig,
    *,
    seed: int | None = None,
    show_progress: bool | None = None,
    store_samples: bool | None = None,
) -> SamplingConfig:
    if isinstance(shots, SamplingConfig):
        config = shots
        seed = config.seed if seed is None else seed
        show_progress = config.show_progress if show_progress is None else show_progress
        store_samples = config.store_samples if store_samples is None else store_samples
        return SamplingConfig(
            shots=config.shots,
            seed=seed,
            show_progress=False if show_progress is None else bool(show_progress),
            store_samples=False if store_samples is None else bool(store_samples),
        )
    return SamplingConfig(
        shots=int(shots),
        seed=seed,
        show_progress=False if show_progress is None else bool(show_progress),
        store_samples=False if store_samples is None else bool(store_samples),
    )


@dataclass
class HolographicSampler:
    channel: HolographicChannel | HolographicChannelSequence
    left_state: Any | None = None
    burn_in: BurnInConfig | int = BurnInConfig()
    boundary: BoundaryCondition | None = None
    bond_noise: BondNoiseChannel | None = None

    def __post_init__(self) -> None:
        self.burn_in = _resolve_burn_in(self.burn_in)
        self.boundary = _resolve_boundary(self.boundary)
        self._channel_sequence = self.channel if isinstance(self.channel, HolographicChannelSequence) else None
        self._translation_invariant_channel = self.channel if isinstance(self.channel, HolographicChannel) else None
        resolved_bond_dim = self.bond_dim
        if self.left_state is None:
            self.left_state = basis_vector(resolved_bond_dim, 0)
        self.left_state = coerce_density_matrix(self.left_state, dim=resolved_bond_dim)
        if self._channel_sequence is not None:
            if int(self.burn_in.steps) != 0:
                raise ValueError("Burn-in is only supported for translation-invariant single-channel workflows.")
            self._steps = tuple(
                PurifiedChannelStep(channel, bond_noise=self.bond_noise) for channel in self._channel_sequence.channels
            )
            self._step = None
        else:
            assert self._translation_invariant_channel is not None
            self._step = PurifiedChannelStep(self._translation_invariant_channel, bond_noise=self.bond_noise)
            self._steps = None

    @property
    def physical_dim(self) -> int:
        if self._channel_sequence is not None:
            return self._channel_sequence.physical_dim
        assert self._translation_invariant_channel is not None
        return self._translation_invariant_channel.physical_dim

    @property
    def bond_dim(self) -> int:
        if self._channel_sequence is not None:
            return self._channel_sequence.bond_dim
        assert self._translation_invariant_channel is not None
        return self._translation_invariant_channel.bond_dim

    @property
    def is_finite_sequence(self) -> bool:
        return self._channel_sequence is not None

    def _metadata_record(self, *, boundary: BoundaryCondition | None) -> dict[str, Any]:
        return {
            "channel": None if self._translation_invariant_channel is None else self._translation_invariant_channel.to_record(),
            "channel_sequence": None if self._channel_sequence is None else self._channel_sequence.to_record(),
            "bond_noise": None if self.bond_noise is None else self.bond_noise.to_record(),
            "boundary": None if boundary is None else boundary.to_record(),
        }

    def _validate_schedule_length(self, total_steps: int) -> None:
        if self._channel_sequence is None:
            return
        if int(total_steps) != self._channel_sequence.num_steps:
            raise ValueError(
                "Finite holographic channel sequences require schedule.total_steps to equal the sequence length: "
                f"got total_steps={int(total_steps)} and num_steps={self._channel_sequence.num_steps}."
            )

    def _step_for_index(self, step: int) -> PurifiedChannelStep:
        if self._steps is not None:
            return self._steps[int(step) - 1]
        assert self._step is not None
        return self._step

    @classmethod
    def from_mps_state(
        cls,
        state: Any,
        *,
        site: int = 0,
        complete: bool = True,
        chi_max: int | None = None,
        label: str | None = None,
        left_state: Any | None = None,
        burn_in: BurnInConfig | int = BurnInConfig(),
        boundary: BoundaryCondition | None = None,
        bond_noise: BondNoiseChannel | None = None,
    ) -> "HolographicSampler":
        channel = HolographicChannel.from_mps_state(
            state,
            site=site,
            complete=complete,
            chi_max=chi_max,
            label=label,
        )
        return cls(
            channel=channel,
            left_state=left_state,
            burn_in=burn_in,
            boundary=boundary,
            bond_noise=bond_noise,
        )

    @classmethod
    def from_mps_sequence(
        cls,
        state: Any,
        *,
        complete: bool = True,
        chi_max: int | None = None,
        label: str | None = None,
        left_state: Any | None = None,
        boundary: BoundaryCondition | None = None,
        bond_noise: BondNoiseChannel | None = None,
    ) -> "HolographicSampler":
        sequence = HolographicChannelSequence.from_mps_state(
            state,
            complete=complete,
            chi_max=chi_max,
            label=label,
        )
        return cls(
            channel=sequence,
            left_state=left_state,
            boundary=boundary,
            bond_noise=bond_noise,
        )

    @classmethod
    def from_unitary_sequence(
        cls,
        unitaries: Sequence[Any],
        *,
        physical_dim: int,
        bond_dim: int,
        label: str | None = None,
        left_state: Any | None = None,
        boundary: BoundaryCondition | None = None,
        bond_noise: BondNoiseChannel | None = None,
        reference_state: Any | None = None,
    ) -> "HolographicSampler":
        sequence = HolographicChannelSequence.from_unitaries(
            unitaries,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            reference_state=reference_state,
            label=label,
        )
        return cls(
            channel=sequence,
            left_state=left_state,
            boundary=boundary,
            bond_noise=bond_noise,
        )

    def summarize_burn_in(self, steps: int | None = None) -> BurnInSummary:
        num_steps = int(self.burn_in.steps if steps is None else steps)
        rho = np.array(self.left_state, copy=True)
        residuals = np.zeros(num_steps, dtype=float)
        for idx in range(num_steps):
            next_rho = self._step_for_index(1).propagate(rho)
            residuals[idx] = trace_distance(next_rho, rho)
            rho = next_rho
        return BurnInSummary(steps=num_steps, residuals=residuals, final_state=rho)

    def _initial_state_after_burn_in(self) -> np.ndarray:
        rho = np.array(self.left_state, copy=True)
        if self._channel_sequence is not None:
            return rho
        for _ in range(int(self.burn_in.steps)):
            rho = self._step_for_index(1).propagate(rho)
        return rho

    def _attempt_boundary_postselection(
        self,
        bond_state: np.ndarray,
        *,
        boundary: BoundaryCondition | None,
        rng: np.random.Generator,
    ) -> bool:
        if boundary is None or boundary.right_state is None or not boundary.postselect:
            return True
        probability = state_overlap_probability(bond_state, boundary.right_state, dim=self.bond_dim)
        probability = min(max(probability, 0.0), 1.0)
        return bool(rng.random() <= probability)

    def sample_correlator(
        self,
        schedule: ObservableSchedule | Sequence[Any],
        shots: int | SamplingConfig,
        *,
        seed: int | None = None,
        show_progress: bool | None = None,
        store_samples: bool | None = None,
        boundary: BoundaryCondition | None = None,
    ) -> CorrelatorEstimate:
        resolved_schedule = _resolve_schedule(schedule)
        self._validate_schedule_length(int(resolved_schedule.total_steps))
        resolved_boundary = self.boundary if boundary is None else boundary
        config = _resolve_sampling_config(shots, seed=seed, show_progress=show_progress, store_samples=store_samples)
        rng = np.random.default_rng(config.seed)
        stats = RunningStats(store_samples=config.store_samples)
        attempted = 0
        accepted = 0

        iterator = progress_wrapper(
            range(config.shots),
            enabled=config.show_progress,
            desc="holographic shots",
        )
        for _ in iterator:
            attempted += 1
            rho = self._initial_state_after_burn_in()
            estimator = 1.0 + 0.0j
            for step in range(1, int(resolved_schedule.total_steps) + 1):
                insertion = resolved_schedule.insertion_for_step(step)
                step_operator = self._step_for_index(step)
                if insertion is None:
                    rho = step_operator.propagate(rho)
                    continue
                outcome = step_operator.sample_measurement(rho, insertion.observable, rng=rng)
                rho = outcome.bond_state
                estimator *= complex(outcome.eigenvalue)
            if self._attempt_boundary_postselection(rho, boundary=resolved_boundary, rng=rng):
                accepted += 1
                stats.update(estimator)

        mean = 0.0 + 0.0j if accepted == 0 else complex(stats.mean)
        variance = 0.0 if accepted == 0 else float(stats.variance)
        stderr = float("nan") if accepted == 0 else float(stats.stderr)
        return CorrelatorEstimate(
            mean=mean,
            variance=variance,
            stderr=stderr,
            attempted_shots=attempted,
            accepted_shots=accepted,
            burn_in_steps=int(self.burn_in.steps),
            total_steps=int(resolved_schedule.total_steps),
            schedule_record=resolved_schedule.to_record(),
            samples=stats.samples,
            metadata=self._metadata_record(boundary=resolved_boundary),
        )

    def enumerate_correlator(
        self,
        schedule: ObservableSchedule | Sequence[Any],
        *,
        boundary: BoundaryCondition | None = None,
        atol: float = 1.0e-15,
        max_branches: int | None = None,
    ) -> ExactCorrelatorResult:
        resolved_schedule = _resolve_schedule(schedule)
        self._validate_schedule_length(int(resolved_schedule.total_steps))
        resolved_boundary = self.boundary if boundary is None else boundary
        initial_state = self._initial_state_after_burn_in()
        branches: list[dict[str, Any]] = [
            {
                "bond_state": initial_state,
                "probability": 1.0,
                "estimator": 1.0 + 0.0j,
                "eigenvalues": (),
            }
        ]

        for step in range(1, int(resolved_schedule.total_steps) + 1):
            insertion = resolved_schedule.insertion_for_step(step)
            step_operator = self._step_for_index(step)
            next_branches: list[dict[str, Any]] = []
            for branch in branches:
                if insertion is None:
                    next_branches.append(
                        {
                            "bond_state": step_operator.propagate(branch["bond_state"]),
                            "probability": float(branch["probability"]),
                            "estimator": complex(branch["estimator"]),
                            "eigenvalues": tuple(branch["eigenvalues"]),
                        }
                    )
                    continue

                outcomes = step_operator.enumerate_measurement_branches(
                    branch["bond_state"],
                    insertion.observable,
                    atol=atol,
                )
                for outcome in outcomes:
                    prob = float(branch["probability"]) * float(outcome.probability)
                    if prob <= atol:
                        continue
                    next_branches.append(
                        {
                            "bond_state": outcome.bond_state,
                            "probability": prob,
                            "estimator": complex(branch["estimator"]) * complex(outcome.eigenvalue),
                            "eigenvalues": tuple(branch["eigenvalues"]) + (complex(outcome.eigenvalue),),
                        }
                    )
            branches = next_branches
            if max_branches is not None and len(branches) > int(max_branches):
                raise ValueError(f"Exact branch enumeration exceeded max_branches={int(max_branches)}.")

        public_branches: list[BranchRecord] = []
        raw_prob_sum = float(sum(branch["probability"] for branch in branches))
        accepted_prob_sum = 0.0
        mean_numer = 0.0 + 0.0j
        second_moment = 0.0
        for branch in branches:
            accepted_probability = float(branch["probability"])
            if resolved_boundary is not None and resolved_boundary.right_state is not None and resolved_boundary.postselect:
                accepted_probability *= state_overlap_probability(
                    branch["bond_state"],
                    resolved_boundary.right_state,
                    dim=self.bond_dim,
                )
            if accepted_probability > atol:
                accepted_prob_sum += accepted_probability
                mean_numer += accepted_probability * complex(branch["estimator"])
                second_moment += accepted_probability * abs(complex(branch["estimator"])) ** 2
            public_branches.append(
                BranchRecord(
                    probability=float(branch["probability"]),
                    accepted_probability=float(accepted_probability),
                    estimator_value=complex(branch["estimator"]),
                    measurement_eigenvalues=tuple(branch["eigenvalues"]),
                )
            )

        if accepted_prob_sum <= atol:
            mean = 0.0 + 0.0j
            variance = 0.0
        else:
            mean = mean_numer / accepted_prob_sum
            variance = max(0.0, float(second_moment / accepted_prob_sum - abs(mean) ** 2))
        return ExactCorrelatorResult(
            mean=complex(mean),
            variance=float(variance),
            branch_probability_sum=float(raw_prob_sum),
            accepted_probability_sum=float(accepted_prob_sum),
            normalization_error=float(abs(raw_prob_sum - 1.0)),
            burn_in_steps=int(self.burn_in.steps),
            total_steps=int(resolved_schedule.total_steps),
            schedule_record=resolved_schedule.to_record(),
            branches=public_branches,
            metadata=self._metadata_record(boundary=resolved_boundary),
        )


@dataclass
class HolographicMPSAlgorithm:
    channel: HolographicChannel | HolographicChannelSequence
    left_state: Any | None = None
    right_boundary: BoundaryCondition | None = None
    burn_in: BurnInConfig | int = BurnInConfig()
    bond_noise: BondNoiseChannel | None = None

    @classmethod
    def from_mps_state(
        cls,
        state: Any,
        *,
        site: int = 0,
        complete: bool = True,
        chi_max: int | None = None,
        label: str | None = None,
        left_state: Any | None = None,
        right_boundary: BoundaryCondition | None = None,
        burn_in: BurnInConfig | int = BurnInConfig(),
        bond_noise: BondNoiseChannel | None = None,
    ) -> "HolographicMPSAlgorithm":
        channel = HolographicChannel.from_mps_state(
            state,
            site=site,
            complete=complete,
            chi_max=chi_max,
            label=label,
        )
        return cls(
            channel=channel,
            left_state=left_state,
            right_boundary=right_boundary,
            burn_in=burn_in,
            bond_noise=bond_noise,
        )

    @classmethod
    def from_mps_sequence(
        cls,
        state: Any,
        *,
        complete: bool = True,
        chi_max: int | None = None,
        label: str | None = None,
        left_state: Any | None = None,
        right_boundary: BoundaryCondition | None = None,
        bond_noise: BondNoiseChannel | None = None,
    ) -> "HolographicMPSAlgorithm":
        sequence = HolographicChannelSequence.from_mps_state(
            state,
            complete=complete,
            chi_max=chi_max,
            label=label,
        )
        return cls(
            channel=sequence,
            left_state=left_state,
            right_boundary=right_boundary,
            bond_noise=bond_noise,
        )

    @classmethod
    def from_unitary_sequence(
        cls,
        unitaries: Sequence[Any],
        *,
        physical_dim: int,
        bond_dim: int,
        label: str | None = None,
        left_state: Any | None = None,
        right_boundary: BoundaryCondition | None = None,
        bond_noise: BondNoiseChannel | None = None,
        reference_state: Any | None = None,
    ) -> "HolographicMPSAlgorithm":
        sequence = HolographicChannelSequence.from_unitaries(
            unitaries,
            physical_dim=physical_dim,
            bond_dim=bond_dim,
            reference_state=reference_state,
            label=label,
        )
        return cls(
            channel=sequence,
            left_state=left_state,
            right_boundary=right_boundary,
            bond_noise=bond_noise,
        )

    def _sampler(self) -> HolographicSampler:
        return HolographicSampler(
            channel=self.channel,
            left_state=self.left_state,
            boundary=self.right_boundary,
            burn_in=self.burn_in,
            bond_noise=self.bond_noise,
        )

    def estimate_observable(
        self,
        schedule: ObservableSchedule | Sequence[Any],
        *,
        shots: int | SamplingConfig = 1_000,
        exact: bool = False,
        seed: int | None = None,
        show_progress: bool = False,
    ) -> CorrelatorEstimate | ExactCorrelatorResult:
        sampler = self._sampler()
        if exact:
            return sampler.enumerate_correlator(schedule)
        return sampler.sample_correlator(schedule, shots, seed=seed, show_progress=show_progress)
