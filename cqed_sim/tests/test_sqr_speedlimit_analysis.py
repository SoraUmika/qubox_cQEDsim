from __future__ import annotations

from dataclasses import replace

import numpy as np

from cqed_sim.analysis import SQRSpeedLimitConfig, TargetCase, run_speedlimit_sweep_point
from cqed_sim.calibration import RandomSQRTarget


def _speedlimit_case() -> TargetCase:
    target = RandomSQRTarget(
        target_id="parallel_case",
        target_class="test",
        logical_n=2,
        guard_levels=1,
        theta=(np.pi / 2.0, np.pi / 4.0, 0.0),
        phi=(0.0, np.pi / 3.0, 0.0),
    )
    return TargetCase(name="parallel_case", description="parallel regression", target=target, phase="test")


def _study_template() -> SQRSpeedLimitConfig:
    return SQRSpeedLimitConfig(
        seed=17,
        n_match=1,
        guard_levels=1,
        durations_ns=(250,),
        sigma_fractions=(0.16, 0.22),
        multistart=2,
        dt_s=5.0e-9,
        max_step_s=5.0e-9,
        optimizer_maxiter_stage1=1,
        optimizer_maxiter_stage2=1,
        progress_every=1000,
    )


def test_speedlimit_sweep_point_parallel_matches_serial() -> None:
    case = _speedlimit_case()
    serial = run_speedlimit_sweep_point(
        case=case,
        duration_s=250.0e-9,
        study=_study_template(),
    )
    parallel_study = replace(_study_template(), parallel_enabled=True, parallel_n_jobs=2)
    parallel = run_speedlimit_sweep_point(
        case=case,
        duration_s=250.0e-9,
        study=parallel_study,
    )

    assert np.isclose(serial["subspace_fidelity"], parallel["subspace_fidelity"], atol=1.0e-10)
    assert np.isclose(serial["logical_fidelity_weighted"], parallel["logical_fidelity_weighted"], atol=1.0e-10)
    assert np.isclose(serial["guard_selectivity_error"], parallel["guard_selectivity_error"], atol=1.0e-12)
    assert np.isclose(serial["sigma_fraction"], parallel["sigma_fraction"], atol=0.0)
    assert serial["selected_run_id"] == parallel["selected_run_id"]
    assert parallel["parallel_used"] is True