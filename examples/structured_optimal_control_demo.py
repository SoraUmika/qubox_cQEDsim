from __future__ import annotations

from pathlib import Path

import numpy as np

from cqed_sim import (
    DelayHardwareMap,
    DispersiveTransmonCavityModel,
    FirstOrderLowPassHardwareMap,
    FourierSeriesPulseFamily,
    FrameSpec,
    GainHardwareMap,
    GaussianDragPulseFamily,
    HardwareModel,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    StructuredControlChannel,
    StructuredControlConfig,
    build_structured_control_problem_from_model,
    save_structured_control_artifacts,
    solve_structured_control,
    state_preparation_objective,
)


def _qubit_only_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=1,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return model, frame


def _channel_specs() -> tuple[ModelControlChannelSpec, ...]:
    return (
        ModelControlChannelSpec(
            name="qubit",
            target="qubit",
            quadratures=("I", "Q"),
            amplitude_bounds=(-8.0e7, 8.0e7),
            export_channel="qubit",
        ),
    )


def _hardware_model() -> HardwareModel:
    return HardwareModel(
        maps=(
            GainHardwareMap(gain=0.93, export_channels=("qubit",)),
            DelayHardwareMap(delay_samples=1, export_channels=("qubit",)),
            FirstOrderLowPassHardwareMap(cutoff_hz=28.0e6, export_channels=("qubit",)),
        )
    )


def _run_family(label: str, family, output_dir: Path) -> None:
    model, frame = _qubit_only_model()
    problem = build_structured_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=32, dt_s=4.0e-9),
        channel_specs=_channel_specs(),
        structured_channels=(
            StructuredControlChannel(
                name=label,
                pulse_family=family,
                export_channel="qubit",
            ),
        ),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
        hardware_model=_hardware_model(),
        metadata={"example": "structured_optimal_control_demo", "family": label},
    )
    result = solve_structured_control(
        problem,
        config=StructuredControlConfig(maxiter=60, seed=7, initial_guess="random", random_scale=0.2),
    )
    artifacts = save_structured_control_artifacts(problem, result, output_dir / label)

    print(f"Structured optimal-control demo: {label}")
    print(f"  success: {result.success}")
    print(f"  objective: {result.objective_value:.6e}")
    print(f"  nominal command fidelity: {result.metrics.get('nominal_command_fidelity', float('nan')):.6f}")
    print(f"  nominal physical fidelity: {result.metrics.get('nominal_physical_fidelity', float('nan')):.6f}")
    print(f"  backend: {result.backend}")
    print(f"  parameter values: {np.asarray(result.schedule.values, dtype=float).tolist()}")
    print(f"  artifacts: {artifacts.directory}")


def main() -> None:
    output_dir = Path("outputs") / "structured_optimal_control_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    gaussian_family = GaussianDragPulseFamily(
        amplitude_bounds=(0.0, 7.0e7),
        sigma_fraction_bounds=(0.1, 0.24),
        center_fraction_bounds=(0.42, 0.58),
        phase_bounds=(-np.pi, np.pi),
        drag_bounds=(-0.3, 0.3),
        default_phase=-0.5 * np.pi,
    )
    fourier_family = FourierSeriesPulseFamily(n_modes=3, coefficient_bound=4.0e7)

    _run_family("gaussian_drag", gaussian_family, output_dir)
    _run_family("fourier_basis", fourier_family, output_dir)


if __name__ == "__main__":
    main()