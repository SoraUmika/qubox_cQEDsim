from __future__ import annotations

import numpy as np

import examples.studies.sqr_multitone_study as sms


def test_resolve_active_support_explicit_weights():
    support = sms.ActiveSupportParams(mode="explicit", active_levels=(0, 2, 4), active_weights=(1.0, 2.0, 1.0))
    levels, weights, inactive = sms.resolve_active_support(n_levels=6, support=support)
    assert levels == [0, 2, 4]
    assert np.isclose(weights[0], 0.25)
    assert np.isclose(weights[2], 0.5)
    assert np.isclose(weights[4], 0.25)
    assert inactive == [1, 3, 5]


def test_support_objective_loss_terms_floor_penalties():
    metrics = {
        "active_weighted_mean_infidelity": 0.08,
        "active_theta_rms_rad": 0.3,
        "active_phase_axis_rms_rad": 0.2,
        "active_residual_pre_z_rms_rad": 0.1,
        "active_residual_post_z_rms_rad": 0.12,
        "support_state_fidelity_mean": 0.94,
        "support_state_fidelity_min": 0.90,
        "support_phase_superposition_rms_rad": 0.15,
        "support_state_leakage_mean": 0.01,
        "support_state_leakage_max": 0.03,
        "support_weighted_leakage_mean": 0.02,
        "support_weighted_leakage_max": 0.05,
        "active_min_process_fidelity": 0.91,
        "inactive_mean_infidelity": 0.12,
    }
    terms = sms.support_objective_loss_terms(
        support_metrics=metrics,
        support_weights=sms.SupportObjectiveWeights(active_block_fidelity_floor=0.97, active_state_fidelity_floor=0.96),
        support=sms.ActiveSupportParams(inactive_weight=0.03),
        reg=0.01,
    )
    assert terms["worst_block"] > 0.0
    assert terms["active_state_min"] > 0.0
    assert np.isclose(terms["regularization"], 0.01)


def test_optimize_case_support_aware_wires_support_metrics():
    params = sms.StudyParams(
        system=sms.SystemParams(n_max=1, chi_nominal_hz=-2.84e6, chi_easy_hz=-2.84e6, chi_hard_hz=-2.84e6),
        optimization=sms.OptimizationParams(
            objective_scope="support_aware",
            maxiter_stage1_basic=1,
            maxiter_stage2_basic=1,
            maxiter_stage1_extended=1,
            maxiter_stage2_extended=1,
            maxiter_stage1_chirp=1,
            maxiter_stage2_chirp=1,
        ),
        active_support=sms.ActiveSupportParams(mode="explicit", active_levels=(0, 1)),
    )
    profile = sms.TargetProfile(
        name="tiny_manual",
        mode="manual",
        theta=np.array([np.pi / 2.0, np.pi / 2.0], dtype=float),
        phi=np.array([0.0, np.pi / 2.0], dtype=float),
        seed=0,
    )
    model, frame = sms.build_model_and_frame(params.system, chi_hz=float(params.system.chi_nominal_hz))
    reference_states = sms.build_reference_states(model, n_max=params.system.n_max, coherent_alpha=params.coherent_alpha)
    ideal = sms.apply_unitary_to_states(sms.build_target_unitary(profile), reference_states)
    controls = sms.build_controls_from_target(
        profile=profile,
        model=model,
        frame=frame,
        duration_s=float(params.pulse.duration_nominal_s),
        theta_cutoff=float(params.pulse.theta_cutoff),
    )
    case = sms.optimize_case(
        mode=sms.MODE_BASIC,
        base_controls=controls,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        opt_params=params.optimization,
        duration_s=float(params.pulse.duration_nominal_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal,
        support_config=params.active_support,
    )
    assert case.summary["objective_scope"] == "support_aware"
    assert "support_metrics" in case.summary
    assert "loss_terms" in case.summary
    assert "active_infidelity" in case.summary["loss_terms"]
