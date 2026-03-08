from __future__ import annotations

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer, make_target
from cqed_sim.unitary_synthesis.reporting import make_run_report


def test_g1_report_schema() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target = make_target("easy", n_match=2)
    synth = UnitarySynthesizer(subspace=sub, backend="ideal", gateset=["SQR", "SNAP"], optimize_times=True, seed=123)
    result = synth.fit(target=target, multistart=1, maxiter=120)
    report = make_run_report(result.report, result.simulation.subspace_operator)

    assert "subspace" in report
    assert "backend" in report
    assert "objective" in report
    assert "metrics" in report
    assert "parameters" in report
    assert "durations" in report["parameters"]
    assert "per_fock_blocks" in report


def test_g2_determinism_fixed_seed() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target = make_target("easy", n_match=2)

    synth1 = UnitarySynthesizer(subspace=sub, backend="ideal", gateset=["SQR", "SNAP"], optimize_times=True, seed=999)
    synth2 = UnitarySynthesizer(subspace=sub, backend="ideal", gateset=["SQR", "SNAP"], optimize_times=True, seed=999)

    r1 = synth1.fit(target=target, multistart=2, maxiter=120)
    r2 = synth2.fit(target=target, multistart=2, maxiter=120)

    f1 = r1.report["metrics"]["fidelity"]
    f2 = r2.report["metrics"]["fidelity"]
    assert abs(f1 - f2) < 1e-9
