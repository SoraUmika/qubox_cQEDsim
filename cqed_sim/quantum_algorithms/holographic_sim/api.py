from __future__ import annotations

"""Stable public API for holographic quantum algorithms."""

from .channel import HolographicChannel
from .channel_embedding import PurifiedChannelStep
from .config import BoundaryCondition, BurnInConfig, SamplingConfig
from .diagnostics import (
    branch_probability_error,
    channel_diagnostics,
    compare_burn_in_states,
    compare_estimates,
    fixed_point_residual,
    validate_trace_preservation,
)
from .holo_vqe import EnergyEstimate, EnergyTerm, HoloVQEObjective
from .holoquads import HoloQUADSProgram, TimeSlice
from .models.example_channels import hadamard_reference_channel, ising_transfer_channel, partial_swap_channel
from .models.spin_models import IsingTransferSpec, transverse_field_ising_transfer_unitary
from .mps import MatrixProductState, complete_right_isometry, contract_mps, right_canonical_tensor_to_stinespring_unitary
from .noise import BondNoiseChannel
from .observables import PhysicalObservable, as_observable, identity, pauli_x, pauli_y, pauli_z
from .results import (
    BranchRecord,
    BurnInSummary,
    ChannelDiagnostics,
    CorrelatorEstimate,
    EstimatorComparison,
    ExactCorrelatorResult,
    MeasurementOutcome,
)
from .sampler import HolographicMPSAlgorithm, HolographicSampler
from .schedules import ObservableInsertion, ObservableSchedule
from .step_sequence import HolographicChannelSequence
from .step_unitary import StepUnitarySpec

__all__ = [
    "HolographicChannel",
    "HolographicChannelSequence",
    "PurifiedChannelStep",
    "PhysicalObservable",
    "as_observable",
    "identity",
    "pauli_x",
    "pauli_y",
    "pauli_z",
    "ObservableInsertion",
    "ObservableSchedule",
    "BurnInConfig",
    "BoundaryCondition",
    "SamplingConfig",
    "MeasurementOutcome",
    "BranchRecord",
    "CorrelatorEstimate",
    "ExactCorrelatorResult",
    "ChannelDiagnostics",
    "BurnInSummary",
    "EstimatorComparison",
    "HolographicSampler",
    "HolographicMPSAlgorithm",
    "StepUnitarySpec",
    "MatrixProductState",
    "complete_right_isometry",
    "contract_mps",
    "right_canonical_tensor_to_stinespring_unitary",
    "BondNoiseChannel",
    "channel_diagnostics",
    "compare_estimates",
    "branch_probability_error",
    "validate_trace_preservation",
    "fixed_point_residual",
    "compare_burn_in_states",
    "EnergyTerm",
    "EnergyEstimate",
    "HoloVQEObjective",
    "TimeSlice",
    "HoloQUADSProgram",
    "IsingTransferSpec",
    "transverse_field_ising_transfer_unitary",
    "hadamard_reference_channel",
    "partial_swap_channel",
    "ising_transfer_channel",
]
