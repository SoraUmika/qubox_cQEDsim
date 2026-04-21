from __future__ import annotations

import numpy as np

from cqed_sim import DispersiveReadoutTransmonStorageModel, FrameSpec, SequenceCompiler
from cqed_sim.measurement import ReadoutChain, ReadoutResonator
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
        integration_time=spec.tau,
        dt=2e-9,
    )

    evaluation = evaluate_readout_emptying_with_chain(result, chain, shots_per_branch=32, seed=7)
    metrics = evaluation["metrics"]

    assert metrics["measurement_chain_separation"] > 0.0
    assert metrics["measurement_chain_accuracy"] >= 0.95
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
    assert report.comparison_table["kerr_corrected"]["measurement_chain_separation"] > 0.0
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
