from __future__ import annotations

from typing import Any

import numpy as np

from .channel import HolographicChannel
from .results import BurnInSummary, ChannelDiagnostics, CorrelatorEstimate, EstimatorComparison, ExactCorrelatorResult
from .utils import coerce_density_matrix, trace_distance, unitarity_error


def channel_diagnostics(channel: HolographicChannel, *, atol: float = 1.0e-10) -> ChannelDiagnostics:
    rho_test = np.eye(channel.bond_dim, dtype=np.complex128) / float(channel.bond_dim)
    evolved = channel.apply(rho_test)
    hermiticity_preserving = bool(np.linalg.norm(evolved - evolved.conj().T, ord="fro") <= atol)
    unitary_err = None if channel.joint_unitary is None else float(unitarity_error(channel.joint_unitary))
    return ChannelDiagnostics(
        physical_dim=int(channel.physical_dim),
        bond_dim=int(channel.bond_dim),
        kraus_count=int(len(channel.kraus_ops)),
        kraus_completeness_error=float(channel.kraus_completeness_error()),
        trace_preservation_error=float(abs(np.trace(evolved) - 1.0)),
        right_canonical_error=float(channel.right_canonical_error()),
        unitary_error=unitary_err,
        hermiticity_preserving=hermiticity_preserving,
    )


def compare_estimates(monte_carlo: CorrelatorEstimate, exact: ExactCorrelatorResult) -> EstimatorComparison:
    return EstimatorComparison(
        monte_carlo_mean=complex(monte_carlo.mean),
        exact_mean=complex(exact.mean),
        absolute_error=float(abs(complex(monte_carlo.mean) - complex(exact.mean))),
        monte_carlo_stderr=float(monte_carlo.stderr),
    )


def branch_probability_error(result: ExactCorrelatorResult) -> float:
    return float(result.normalization_error)


def validate_trace_preservation(channel: HolographicChannel, *, trials: int = 4, atol: float = 1.0e-10) -> float:
    rng = np.random.default_rng(1234)
    worst = 0.0
    for _ in range(int(trials)):
        mat = rng.standard_normal((channel.bond_dim, channel.bond_dim)) + 1j * rng.standard_normal((channel.bond_dim, channel.bond_dim))
        rho = coerce_density_matrix(mat @ mat.conj().T)
        evolved = channel.apply(rho)
        worst = max(worst, float(abs(np.trace(evolved) - 1.0)))
    return worst


def fixed_point_residual(summary: BurnInSummary) -> float:
    return float(summary.max_residual)


def compare_burn_in_states(left: Any, right: Any) -> float:
    return float(trace_distance(left, right))
