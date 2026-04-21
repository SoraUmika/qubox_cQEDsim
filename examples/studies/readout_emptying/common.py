from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec
from cqed_sim.measurement import ReadoutChain, ReadoutResonator
from cqed_sim.optimal_control import (
    FirstOrderLowPassHardwareMap,
    HardwareModel,
    ReadoutEmptyingConstraints,
    ReadoutEmptyingRefinementConfig,
    ReadoutEmptyingSpec,
    ReadoutEmptyingVerificationConfig,
    refine_readout_emptying_pulse,
    synthesize_readout_emptying_pulse,
    verify_readout_emptying_pulse,
)
from cqed_sim.sim import NoiseSpec


OUTPUT_ROOT = Path("outputs") / "readout_emptying_qualification"
DOC_ASSET_ROOT = Path("documentations") / "assets" / "images" / "tutorials" / "readout_emptying"


def study_output_dir(stage: str) -> Path:
    path = OUTPUT_ROOT / str(stage)
    path.mkdir(parents=True, exist_ok=True)
    return path


def doc_asset_dir() -> Path:
    DOC_ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    return DOC_ASSET_ROOT


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def base_constraints(amplitude_max: float = 2.0 * np.pi * 7.0e6) -> ReadoutEmptyingConstraints:
    return ReadoutEmptyingConstraints(amplitude_max=float(amplitude_max))


def linear_spec(*, n_segments: int = 4) -> ReadoutEmptyingSpec:
    return ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=220e-9,
        n_segments=int(n_segments),
        kerr=0.0,
        include_kerr_phase_correction=False,
    )


def nonlinear_spec(
    *,
    include_kerr_phase_correction: bool = True,
    kerr_correction_strategy: str = "average_branch",
    amplitude_max: float | None = None,
) -> tuple[ReadoutEmptyingSpec, ReadoutEmptyingConstraints]:
    spec = ReadoutEmptyingSpec(
        kappa=2.0 * np.pi * 2.0e6,
        chi=2.0 * np.pi * 1.0e6,
        tau=220e-9,
        n_segments=4,
        kerr=2.0 * np.pi * 0.06e6,
        include_kerr_phase_correction=bool(include_kerr_phase_correction),
        kerr_correction_strategy=str(kerr_correction_strategy),
    )
    constraints = base_constraints(2.0 * np.pi * 7.0e6 if amplitude_max is None else amplitude_max)
    return spec, constraints


def readout_chain(spec: ReadoutEmptyingSpec) -> ReadoutChain:
    return ReadoutChain(
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


def readout_model(spec: ReadoutEmptyingSpec) -> DispersiveReadoutTransmonStorageModel:
    return DispersiveReadoutTransmonStorageModel(
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
        n_readout=8,
        n_tr=3,
    )


def noise_spec(spec: ReadoutEmptyingSpec) -> NoiseSpec:
    return NoiseSpec(kappa_readout=spec.kappa, t1=20.0e-6, tphi=30.0e-6)


def hardware_models() -> tuple[HardwareModel, dict[str, HardwareModel]]:
    nominal = HardwareModel(
        maps=(FirstOrderLowPassHardwareMap(cutoff_hz=200e6, export_channels=("readout",)),)
    )
    variants = {
        "narrower_lp": HardwareModel(
            maps=(FirstOrderLowPassHardwareMap(cutoff_hz=120e6, export_channels=("readout",)),)
        ),
        "wider_lp": HardwareModel(
            maps=(FirstOrderLowPassHardwareMap(cutoff_hz=280e6, export_channels=("readout",)),)
        ),
    }
    return nominal, variants


def verification_config(
    spec: ReadoutEmptyingSpec,
    *,
    hardware: HardwareModel | None = None,
    hardware_variants: dict[str, HardwareModel] | None = None,
    shots_per_branch: int = 64,
) -> ReadoutEmptyingVerificationConfig:
    model = readout_model(spec)
    return ReadoutEmptyingVerificationConfig(
        measurement_chain=readout_chain(spec),
        hardware_model=hardware,
        readout_model=model,
        frame=FrameSpec(omega_q_frame=model.omega_q),
        noise=noise_spec(spec),
        compiler_dt_s=4.0e-9,
        shots_per_branch=int(shots_per_branch),
        seed=7,
        measurement_noise_mode="calibrated_target_error",
        measurement_target_square_error=0.10,
        hardware_variants={} if hardware_variants is None else dict(hardware_variants),
    )


def refinement_config(
    spec: ReadoutEmptyingSpec,
    *,
    hardware: HardwareModel | None = None,
    hardware_variants: dict[str, HardwareModel] | None = None,
    shots_per_branch: int = 32,
    maxiter: int = 10,
    build_verification_report: bool = True,
) -> ReadoutEmptyingRefinementConfig:
    model = readout_model(spec)
    return ReadoutEmptyingRefinementConfig(
        measurement_chain=readout_chain(spec),
        hardware_model=hardware,
        readout_model=model,
        frame=FrameSpec(omega_q_frame=model.omega_q),
        noise=noise_spec(spec),
        compiler_dt_s=4.0e-9,
        shots_per_branch=int(shots_per_branch),
        measurement_noise_mode="calibrated_target_error",
        measurement_target_square_error=0.10,
        maxiter=int(maxiter),
        hardware_variants={} if hardware_variants is None else dict(hardware_variants),
        build_verification_report=bool(build_verification_report),
    )


def nominal_report(
    *,
    include_refinement: bool = True,
    shots_per_branch: int = 64,
) -> tuple[Any, Any | None]:
    spec, constraints = nonlinear_spec(include_kerr_phase_correction=True)
    result = synthesize_readout_emptying_pulse(spec, constraints)
    hardware, variants = hardware_models()
    report = verify_readout_emptying_pulse(
        result,
        verification_config(
            spec,
            hardware=hardware,
            hardware_variants=variants,
            shots_per_branch=shots_per_branch,
        ),
    )
    refined = None
    if include_refinement:
        refined = refine_readout_emptying_pulse(
            result,
            refinement_config(
                spec,
                hardware=hardware,
                hardware_variants=variants,
                shots_per_branch=max(8, shots_per_branch // 2),
                maxiter=8,
            ),
        )
    return report, refined


def comparison_payload(report, refined=None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "comparison_table": report.comparison_table,
        "measurement_metrics": report.measurement_metrics,
        "disturbance_metrics": report.disturbance_metrics,
        "ringdown_metrics": report.ringdown_metrics,
        "lindblad_metrics": report.lindblad_metrics,
        "hardware_metrics": report.hardware_metrics,
    }
    if refined is not None:
        payload["refinement"] = {
            "metrics": refined.metrics,
            "objective_value": refined.objective_value,
            "initial_objective_value": refined.initial_objective_value,
        }
    return payload
