from __future__ import annotations

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer, make_target
from cqed_sim.unitary_synthesis.reporting import make_run_report


if __name__ == "__main__":
    subspace = Subspace.qubit_cavity_block(n_match=3)

    target_name = "cluster"  # try "ghz" or "easy"
    u_target = make_target(target_name, n_match=3, variant="mps")

    synth = UnitarySynthesizer(
        subspace=subspace,
        backend="pulse",
        gateset=["QubitRotation", "SQR", "SNAP", "Displacement"],
        optimize_times=True,
        time_bounds={"default": (20e-9, 2000e-9)},
        leakage_weight=10.0,
        time_reg_weight=1e-2,
        seed=1234,
    )

    result = synth.fit(target=u_target, init_guess="heuristic", multistart=8, maxiter=220)
    report = make_run_report(result.report, result.simulation.subspace_operator)

    print("Final fidelity:", report["metrics"]["fidelity"])
    print("Final worst leakage:", report["metrics"]["leakage_worst"])
    print("Final durations (s):", report["parameters"]["durations"])
    print("Objective breakdown:", report["objective"])
