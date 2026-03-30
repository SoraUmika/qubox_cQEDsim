"""End-to-end synthesis → GRAPE → validate pipeline.

Demonstrates the full gate design workflow:
    1. Synthesize a target unitary via QuantumMapSynthesizer (high-level sequence)
  2. Refine the resulting control schedule with GRAPE (pulse-level optimisation)
  3. Validate the refined pulses through the full cqed_sim.sim runtime

The target is a pi/2 Y-rotation on the storage-mode logical subspace
{|g,0>, |g,1>} of a dispersive qubit-cavity system.
"""

from __future__ import annotations

import json
from pathlib import Path

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
from cqed_sim.map_synthesis import (
    CQEDSystemAdapter,
    QuantumMapSynthesizer,
    Subspace,
    make_run_report,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"


# ── Target unitary ────────────────────────────────────────────────
def rotation_y(theta: float) -> np.ndarray:
    """Single-mode Y-rotation matrix."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def main() -> None:
    # ── 1.  Model and subspace ────────────────────────────────────
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=2.0 * np.pi * (-220.0e6),
        chi=2.0 * np.pi * (-2.0e6),
        kerr=2.0 * np.pi * (-5.0e3),
        n_cav=4,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

    storage_logical = Subspace.custom(
        full_dim=model.n_tr * model.n_cav,
        indices=(0, 1),
        labels=("|g,0>", "|g,1>"),
    )
    target_matrix = rotation_y(np.pi / 2.0)

    # ── 2.  Map synthesis (high-level sequence search) ────────────
    print("=" * 60)
    print("Step 1: Map synthesis")
    print("=" * 60)

    system = CQEDSystemAdapter(model=model)
    synth = QuantumMapSynthesizer(
        subspace=storage_logical,
        backend="pulse",
        gateset=["QubitRotation", "SQR", "Displacement"],
        optimize_times=True,
        time_bounds={"default": (20.0e-9, 500.0e-9)},
        leakage_weight=5.0,
        time_reg_weight=1.0e-3,
        seed=42,
    )

    from cqed_sim.map_synthesis import TargetUnitary

    target = TargetUnitary(
        matrix=target_matrix,
        ignore_global_phase=True,
    )
    synth_result = synth.fit(
        target=target,
        system=system,
        init_guess="heuristic",
        multistart=4,
        maxiter=100,
    )
    report = make_run_report(synth_result.report, synth_result.simulation.subspace_operator)

    print(f"  Synthesis success : {synth_result.success}")
    print(f"  Objective         : {synth_result.objective:.4e}")
    print(f"  Subspace fidelity : {report['metrics']['fidelity']:.6f}")
    print(f"  Worst leakage     : {report['metrics']['leakage_worst']:.4e}")
    print(f"  Gate count        : {len(synth_result.sequence.gates)}")
    print()

    # ── 3.  GRAPE refinement (pulse-level optimisation) ───────────
    print("=" * 60)
    print("Step 2: GRAPE refinement")
    print("=" * 60)

    gate_duration = 40.0e-9
    n_steps = 20

    problem = build_control_problem_from_model(
        model,
        frame=frame,
        time_grid=PiecewiseConstantTimeGrid.uniform(
            steps=n_steps,
            dt_s=gate_duration / n_steps,
        ),
        channel_specs=(
            ModelControlChannelSpec(
                name="storage_I",
                target="storage",
                quadratures=("I",),
                amplitude_bounds=(-2.0e8, 2.0e8),
                export_channel="storage",
            ),
            ModelControlChannelSpec(
                name="storage_Q",
                target="storage",
                quadratures=("Q",),
                amplitude_bounds=(-2.0e8, 2.0e8),
                export_channel="storage",
            ),
        ),
        objectives=(
            UnitaryObjective(
                target_operator=target_matrix,
                subspace=storage_logical,
                ignore_global_phase=True,
                name="storage_y90",
            ),
        ),
        metadata={"pipeline": "synthesis_grape_validate"},
    )

    grape_result = GrapeSolver(
        GrapeConfig(maxiter=120, seed=7),
    ).solve(problem)

    print(f"  GRAPE success     : {grape_result.success}")
    print(f"  GRAPE message     : {grape_result.message}")
    print(f"  Nominal objective : {grape_result.objective_value:.4e}")
    print(f"  Nominal fidelity  : {grape_result.metrics.get('nominal_fidelity', 'N/A')}")
    print()

    # ── 4.  Validate with full simulator ──────────────────────────
    print("=" * 60)
    print("Step 3: Full-simulator validation")
    print("=" * 60)

    # 4a. Closed-system replay
    pulses, drive_ops, pulse_meta = grape_result.to_pulses()
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

    # Expected: (|g,0> + |g,1>) / sqrt(2)
    target_state = qt.Qobj(
        (model.basis_state(0, 0) + model.basis_state(0, 1)).unit().full().ravel(),
        dims=initial_state.dims,
    )
    closed_fidelity = float(qt.metrics.fidelity(runtime.final_state, target_state))
    print(f"  Closed-system fidelity on |g,0> : {closed_fidelity:.6f}")

    # 4b. Noisy replay via evaluate_with_simulator
    noisy_eval = grape_result.evaluate_with_simulator(
        problem,
        cases=(
            ControlEvaluationCase(
                model=model,
                frame=frame,
                noise=NoiseSpec(kappa=1.0e5, t1=50.0e-6),
                label="kappa_and_t1",
            ),
        ),
    )
    print(f"  Noisy aggregate fidelity        : {noisy_eval.metrics['aggregate_fidelity']:.6f}")
    print(f"  Noisy aggregate leakage         : {noisy_eval.metrics.get('aggregate_leakage', 'N/A')}")
    print()

    # ── 5.  Summary ───────────────────────────────────────────────
    print("=" * 60)
    print("Pipeline summary")
    print("=" * 60)
    summary = {
        "synthesis": {
            "success": synth_result.success,
            "objective": float(synth_result.objective),
            "subspace_fidelity": float(report["metrics"]["fidelity"]),
            "gate_count": len(synth_result.sequence.gates),
        },
        "grape": {
            "success": grape_result.success,
            "objective": float(grape_result.objective_value),
            "nominal_fidelity": float(grape_result.metrics.get("nominal_fidelity", 0.0)),
            "n_steps": n_steps,
            "gate_duration_s": gate_duration,
        },
        "validation": {
            "closed_system_fidelity": closed_fidelity,
            "noisy_aggregate_fidelity": float(noisy_eval.metrics["aggregate_fidelity"]),
        },
    }
    print(json.dumps(summary, indent=2))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "synthesis_grape_validate_summary.json"
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to {output_path}")


if __name__ == "__main__":
    main()
