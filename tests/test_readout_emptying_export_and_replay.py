from __future__ import annotations

import numpy as np

from cqed_sim import DispersiveReadoutTransmonStorageModel, FrameSpec, SequenceCompiler
from cqed_sim.measurement import AmplifierChain, ReadoutChain, ReadoutResonator
from cqed_sim.optimal_control import FirstOrderLowPassHardwareMap, HardwareModel
from cqed_sim.optimal_control.readout_emptying import (
    ReadoutEmptyingConstraints,
    ReadoutEmptyingSpec,
    build_readout_emptying_parameterization,
    evaluate_readout_emptying_with_chain,
    export_readout_emptying_to_pulse,
    synthesize_readout_emptying_pulse,
)
from cqed_sim.optimal_control.readout_emptying_eval import (
    ReadoutEmptyingRefinementConfig,
    ReadoutEmptyingVerificationConfig,
    refine_readout_emptying_pulse,
    verify_readout_emptying_pulse,
)
from cqed_sim.sim import NoiseSpec


def test_exported_pulse_and_parameterization_match_segment_waveform() -> None:
    spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=300e-9,
        n_segments=4,
    )
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    result = synthesize_readout_emptying_pulse(spec, constraints)
    pulse = export_readout_emptying_to_pulse(result)
    parameterization = build_readout_emptying_parameterization(spec, constraints)

    segment_midpoints = 0.5 * (result.segment_edges_s[:-1] + result.segment_edges_s[1:])
    sampled = pulse.sample(segment_midpoints)
    command_values = parameterization.command_values(parameterization.zero_array())

    assert np.allclose(sampled, result.segment_amplitudes, atol=1.0e-12)
    assert np.allclose(command_values[0], result.segment_amplitudes.real, atol=1.0e-12)
    assert np.allclose(command_values[1], result.segment_amplitudes.imag, atol=1.0e-12)


def test_exported_pulse_compiles_back_to_the_same_baseband() -> None:
    spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=300e-9,
        n_segments=4,
    )
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    result = synthesize_readout_emptying_pulse(spec, constraints)
    pulse = export_readout_emptying_to_pulse(result)

    compiler = SequenceCompiler(dt=spec.tau / 200.0)
    compiled = compiler.compile([pulse], t_end=spec.tau)
    midpoints = 0.5 * (result.segment_edges_s[:-1] + result.segment_edges_s[1:])
    sample_indices = [int(np.argmin(np.abs(compiled.tlist - time))) for time in midpoints]

    assert np.allclose(compiled.channels["readout"].baseband[sample_indices], result.segment_amplitudes, atol=1.0e-10)


def test_measurement_chain_evaluation_reports_separation_and_accuracy() -> None:
    spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=300e-9,
        n_segments=4,
    )
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 8.0e6)
    result = synthesize_readout_emptying_pulse(spec, constraints)
    chain = ReadoutChain(
        ReadoutResonator(
            omega_r=2.0 * np.pi * 7.0e9,
            kappa=spec.kappa,
            g=2.0 * np.pi * 80e6,
            epsilon=1.0,
            chi=spec.chi,
        ),
        amplifier=AmplifierChain(noise_temperature=6.0),
        integration_time=spec.tau,
        dt=2e-9,
    )

    evaluation = evaluate_readout_emptying_with_chain(result, chain, shots_per_branch=32, seed=7)
    metrics = evaluation["metrics"]

    assert metrics["measurement_chain_separation"] > 0.0
    assert metrics["measurement_chain_snr"] > 0.0
    assert np.isfinite(metrics["measurement_chain_gaussian_overlap_error"])
    assert metrics["measurement_chain_noise_std"] > 0.0
    assert set(evaluation["iq_centers"]) == {"g", "e"}


def _verification_fixture():
    spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=220e-9,
        n_segments=4,
        kerr=2.0 * np.pi * 0.06e6,
        include_kerr_phase_correction=True,
    )
    constraints = ReadoutEmptyingConstraints(amplitude_max=2.0 * np.pi * 7.0e6)
    result = synthesize_readout_emptying_pulse(spec, constraints)
    chain = ReadoutChain(
        ReadoutResonator(
            omega_r=2.0 * np.pi * 7.0e9,
            kappa=spec.kappa,
            g=2.0 * np.pi * 80.0e6,
            epsilon=1.0,
            chi=spec.chi,
        ),
        integration_time=spec.tau,
        dt=4.0e-9,
    )
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=2.0 * np.pi * 5.0e9,
        omega_r=2.0 * np.pi * 7.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=2.0 * np.pi * (-220.0e6),
        chi_s=0.0,
        chi_r=spec.chi,
        chi_sr=0.0,
        kerr_s=0.0,
        kerr_r=spec.kerr,
        n_storage=1,
        n_readout=6,
        n_tr=3,
    )
    noise = NoiseSpec(kappa_readout=spec.kappa, t1=20e-6, tphi=30e-6)
    nominal_hardware = HardwareModel(
        maps=(FirstOrderLowPassHardwareMap(cutoff_hz=200e6, export_channels=("readout",)),)
    )
    variant_hardware = {
        "narrower_lp": HardwareModel(
            maps=(FirstOrderLowPassHardwareMap(cutoff_hz=120e6, export_channels=("readout",)),)
        )
    }
    return result, chain, model, noise, nominal_hardware, variant_hardware


def test_verification_report_covers_square_corrected_and_hardware_replay() -> None:
    result, chain, model, noise, nominal_hardware, variant_hardware = _verification_fixture()
    report = verify_readout_emptying_pulse(
        result,
        ReadoutEmptyingVerificationConfig(
            measurement_chain=chain,
            hardware_model=nominal_hardware,
            readout_model=model,
            frame=FrameSpec(omega_q_frame=model.omega_q),
            noise=noise,
            compiler_dt_s=6.0e-9,
            shots_per_branch=4,
            seed=11,
            hardware_variants=variant_hardware,
        ),
    )

    assert {"square", "analytic_seed", "kerr_corrected"} <= set(report.comparison_table)
    assert report.comparison_table["square"]["max_final_residual_photons"] > report.comparison_table["analytic_seed"]["max_final_residual_photons"]
    assert report.comparison_table["kerr_corrected"]["measurement_chain_separation"] > 0.0
    assert report.comparison_table["square"]["measurement_chain_gaussian_overlap_error"] > 0.0
    assert report.comparison_table["square"]["measurement_chain_gaussian_overlap_error"] != report.comparison_table["analytic_seed"]["measurement_chain_gaussian_overlap_error"]
    assert report.comparison_table["square"]["background_relaxation_total"] == report.comparison_table["square"]["non_qnd_total"]
    assert report.comparison_table["square"]["strong_readout_disturbance_proxy"] != report.comparison_table["analytic_seed"]["strong_readout_disturbance_proxy"]
    assert report.comparison_table["square"]["ringdown_time_to_threshold"] > report.comparison_table["kerr_corrected"]["ringdown_time_to_threshold"]
    assert report.comparison_table["kerr_corrected"]["lindblad_output_separation"] > 0.0
    assert report.hardware_metrics["kerr_corrected"]["command_physical_rms_delta"] > 0.0
    assert "narrower_lp" in report.robustness["kerr_corrected"]["hardware_variants"]


def test_refinement_harness_returns_a_nonworse_nominal_solution() -> None:
    result, chain, model, noise, nominal_hardware, variant_hardware = _verification_fixture()
    refined = refine_readout_emptying_pulse(
        result,
        ReadoutEmptyingRefinementConfig(
            measurement_chain=chain,
            hardware_model=nominal_hardware,
            readout_model=model,
            frame=FrameSpec(omega_q_frame=model.omega_q),
            noise=noise,
            compiler_dt_s=6.0e-9,
            shots_per_branch=4,
            maxiter=3,
            hardware_variants=variant_hardware,
            build_verification_report=False,
        ),
    )

    assert refined.metrics["objective_improvement"] >= -1.0e-9
    assert refined.refined_result.metrics["integrated_branch_separation"] > 0.0
    assert refined.refined_result.metrics["max_final_residual_photons"] >= 0.0
    assert np.isfinite(refined.metrics["final_measurement_error"])
    assert np.isfinite(refined.metrics["final_leakage"])


def test_summary_benchmark_artifact_metrics_are_not_flat() -> None:
    from examples.studies.readout_emptying import summary_benchmark
    from examples.studies.readout_emptying.common import comparison_payload, hardware_models, nonlinear_spec, refinement_config

    spec, constraints = nonlinear_spec(include_kerr_phase_correction=True)
    seed_result = synthesize_readout_emptying_pulse(spec, constraints)
    hardware, variants = hardware_models()
    refined = refine_readout_emptying_pulse(
        seed_result,
        refinement_config(spec, hardware=hardware, hardware_variants=variants, shots_per_branch=8, maxiter=2),
    )
    report = refined.verification_report
    assert report is not None

    payload = comparison_payload(report, refined=refined)
    labels = ("square", "analytic_seed", "kerr_corrected", "refined")
    overlap_values = [payload["comparison_table"][label]["measurement_chain_gaussian_overlap_error"] for label in labels]
    ringdown_values = [payload["comparison_table"][label]["ringdown_time_to_threshold"] for label in labels]
    disturbance_values = [payload["comparison_table"][label]["strong_readout_disturbance_proxy"] for label in labels]

    assert len({round(value, 12) for value in overlap_values}) > 1
    assert len({round(value, 12) for value in ringdown_values}) > 1
    assert len({round(value, 12) for value in disturbance_values}) > 1

    summary_benchmark._benchmark_bars  # smoke-check that the updated artifact helper exists
    summary_benchmark._tradeoff_frontier
