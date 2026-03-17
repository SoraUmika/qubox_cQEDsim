from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim import (
    ControlEvaluationCase,
    DispersiveTransmonCavityModel,
    FrameSpec,
    GrapeConfig,
    GrapeSolver,
    ModelControlChannelSpec,
    NoiseSpec,
    PiecewiseConstantTimeGrid,
    SequenceCompiler,
    SimulationConfig,
    UnitaryObjective,
    build_control_problem_from_model,
    simulate_sequence,
)
from cqed_sim.unitary_synthesis import Subspace


def rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def main() -> None:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=2,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    storage_logical = Subspace.custom(
        full_dim=4,
        indices=(0, 1),
        labels=("|g,0>", "|g,1>"),
    )

    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(steps=1, dt_s=40.0e-9),
        channel_specs=(
            ModelControlChannelSpec(
                name="storage",
                target="storage",
                quadratures=("Q",),
                amplitude_bounds=(-1.0e8, 1.0e8),
                export_channel="storage",
            ),
        ),
        objectives=(
            UnitaryObjective(
                target_operator=rotation_y(np.pi / 2.0),
                subspace=storage_logical,
                ignore_global_phase=True,
                name="storage_y90",
            ),
        ),
        metadata={"example": "storage_subspace_gate_demo"},
    )

    result = GrapeSolver(GrapeConfig(maxiter=80, seed=7)).solve(
        problem,
        initial_schedule=np.array([[8.0e6]], dtype=float),
    )

    pulses, drive_ops, pulse_meta = result.to_pulses()
    compiler = SequenceCompiler(dt=1.0e-9)
    compiled = compiler.compile(pulses, t_end=problem.time_grid.duration_s)

    initial_state = model.basis_state(0, 0)
    runtime = simulate_sequence(
        model,
        compiled,
        initial_state,
        drive_ops,
        config=SimulationConfig(frame=frame),
    )

    target_state = qt.Qobj(
        np.array([1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0), 0.0, 0.0], dtype=np.complex128),
        dims=initial_state.dims,
    )
    runtime_fidelity = float(qt.metrics.fidelity(runtime.final_state, target_state))
    noisy_replay = result.evaluate_with_simulator(
        problem,
        cases=(
            ControlEvaluationCase(
                model=model,
                frame=frame,
                noise=NoiseSpec(kappa=2.0e5),
                label="kappa_replay",
            ),
        ),
    )

    print("GRAPE storage-subspace Y/2 demo")
    print("Success:", result.success)
    print("Optimizer message:", result.message)
    print("Nominal objective:", f"{result.objective_value:.6e}")
    print("Nominal fidelity:", f"{result.metrics['nominal_fidelity']:.6f}")
    print("Optimized controls (rad/s):", result.schedule.values.tolist())
    print("Exported pulse count:", len(pulses))
    print("Pulse export summary:", pulse_meta["channels"])
    print("Runtime replay fidelity on |g,0>:", f"{runtime_fidelity:.6f}")
    print("Noisy replay aggregate fidelity:", f"{noisy_replay.metrics['aggregate_fidelity']:.6f}")


if __name__ == "__main__":
    main()