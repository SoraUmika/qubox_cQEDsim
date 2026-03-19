from __future__ import annotations

import numpy as np

from cqed_sim import (
    BoundaryWindowHardwareMap,
    DispersiveTransmonCavityModel,
    FirstOrderLowPassHardwareMap,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    HardwareModel,
    HeldSampleParameterization,
    ModelControlChannelSpec,
    PiecewiseConstantTimeGrid,
    SmoothIQRadiusLimitHardwareMap,
    build_control_problem_from_model,
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


def _base_channel_spec() -> tuple[ModelControlChannelSpec, ...]:
    return (
        ModelControlChannelSpec(
            name="qubit",
            target="qubit",
            quadratures=("I", "Q"),
            amplitude_bounds=(-8.0e7, 8.0e7),
            export_channel="qubit",
        ),
    )


def _build_unconstrained_problem(model, frame):
    return build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=20.0e-9),
        channel_specs=_base_channel_spec(),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
    )


def _build_hardware_aware_problem(model, frame):
    return build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=20.0e-9),
        channel_specs=_base_channel_spec(),
        objectives=(state_preparation_objective(model.basis_state(0, 0), model.basis_state(1, 0)),),
        parameterization_cls=HeldSampleParameterization,
        parameterization_kwargs={"sample_period_s": 40.0e-9},
        hardware_model=HardwareModel(
            maps=(
                FirstOrderLowPassHardwareMap(cutoff_hz=25.0e6, export_channels=("qubit",)),
                SmoothIQRadiusLimitHardwareMap(amplitude_max=6.0e7, export_channels=("qubit",)),
                BoundaryWindowHardwareMap(ramp_slices=1, export_channels=("qubit",)),
            )
        ),
    )


def _summarize(label: str, result) -> None:
    command_radius = np.sqrt(np.square(result.command_values[0, :]) + np.square(result.command_values[1, :]))
    physical_radius = np.sqrt(np.square(result.physical_values[0, :]) + np.square(result.physical_values[1, :]))
    print(label)
    print(f"  success: {result.success}")
    print(f"  objective: {result.objective_value:.6e}")
    print(f"  nominal command fidelity: {result.metrics.get('nominal_command_fidelity', float('nan')):.6f}")
    print(f"  nominal physical fidelity: {result.metrics.get('nominal_physical_fidelity', float('nan')):.6f}")
    print(f"  parameter slices: {result.schedule.values.shape[1]}")
    print(f"  time slices: {result.command_values.shape[1]}")
    print(f"  max command radius: {np.max(command_radius):.6e}")
    print(f"  max physical radius: {np.max(physical_radius):.6e}")
    print(f"  max command slew: {result.hardware_metrics.get('command_max_slew', float('nan')):.6e}")
    print(f"  max physical slew: {result.hardware_metrics.get('physical_max_slew', float('nan')):.6e}")
    print(f"  first physical sample: {result.physical_values[:, 0]}")
    print(f"  last physical sample: {result.physical_values[:, -1]}")


def main() -> None:
    model, frame = _qubit_only_model()
    unconstrained_problem = _build_unconstrained_problem(model, frame)
    hardware_problem = _build_hardware_aware_problem(model, frame)

    solver = GrapeSolver(GrapeConfig(maxiter=80, seed=7, random_scale=0.15, report_command_reference=True))
    unconstrained_result = solver.solve(
        unconstrained_problem,
        initial_schedule=np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0e7, 3.0e7, 3.0e7, 3.0e7, 3.0e7, 3.0e7],
            ],
            dtype=float,
        ),
    )
    hardware_result = solver.solve(
        hardware_problem,
        initial_schedule=np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0e7, 3.0e7, 3.0e7],
            ],
            dtype=float,
        ),
    )

    print("Hardware-Constrained GRAPE Comparison")
    print("===================================")
    _summarize("Unconstrained GRAPE", unconstrained_result)
    _summarize("Hardware-aware GRAPE", hardware_result)

    command_replay = hardware_result.evaluate_with_simulator(
        hardware_problem,
        model=model,
        frame=frame,
        compiler_dt_s=1.0e-9,
        waveform_mode="command",
    )
    physical_replay = hardware_result.evaluate_with_simulator(
        hardware_problem,
        model=model,
        frame=frame,
        compiler_dt_s=1.0e-9,
        waveform_mode="physical",
    )
    print("Replay fidelity comparison for the hardware-aware result")
    print(f"  command replay fidelity: {command_replay.metrics['aggregate_fidelity']:.6f}")
    print(f"  physical replay fidelity: {physical_replay.metrics['aggregate_fidelity']:.6f}")


if __name__ == "__main__":
    main()