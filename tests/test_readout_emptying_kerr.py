from __future__ import annotations

import numpy as np

from cqed_sim.optimal_control.readout_emptying import (
    ReadoutEmptyingConstraints,
    ReadoutEmptyingSpec,
    synthesize_readout_emptying_pulse,
)


def test_kerr_degrades_linear_emptying_and_phase_correction_recovers_it() -> None:
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    linear_spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=320e-9,
        n_segments=4,
        kerr=2.0 * np.pi * 0.08e6,
        include_kerr_phase_correction=False,
    )
    corrected_spec = ReadoutEmptyingSpec(
        kappa=linear_spec.kappa,
        chi=linear_spec.chi,
        tau=linear_spec.tau,
        n_segments=linear_spec.n_segments,
        kerr=linear_spec.kerr,
        include_kerr_phase_correction=True,
        kerr_correction_strategy="average_branch",
    )

    linear_result = synthesize_readout_emptying_pulse(linear_spec, constraints)
    corrected_result = synthesize_readout_emptying_pulse(corrected_spec, constraints)

    linear_replay_residual = float(linear_result.diagnostics["linear_metrics"]["max_final_residual_photons"])
    kerr_residual = float(linear_result.metrics["max_final_residual_photons"])
    corrected_residual = float(corrected_result.metrics["max_final_residual_photons"])

    assert linear_replay_residual < 1.0e-12
    assert kerr_residual > 1.0e-4
    assert corrected_residual < 5.0e-3
    assert corrected_residual < 0.05 * kerr_residual
    assert corrected_result.diagnostics["kerr_correction"]["strategy"] == "average_branch"


def test_kerr_correction_strategy_diagnostics_cover_shared_and_branch_specific_choices() -> None:
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    strategies = ("average_branch", "g_branch", "e_branch")
    residuals: dict[str, float] = {}
    for strategy in strategies:
        spec = ReadoutEmptyingSpec(
            kappa=2.0 * np.pi * 2.0e6,
            chi=2.0 * np.pi * 1.0e6,
            tau=320e-9,
            n_segments=4,
            kerr=2.0 * np.pi * 0.08e6,
            include_kerr_phase_correction=True,
            kerr_correction_strategy=strategy,
        )
        result = synthesize_readout_emptying_pulse(spec, constraints)
        residuals[strategy] = float(result.metrics["max_final_residual_photons"])
        assert result.diagnostics["kerr_correction"]["strategy"] == strategy

    assert residuals["average_branch"] < 1.0e-2
    assert residuals["g_branch"] > 0.0
    assert residuals["e_branch"] > 0.0
