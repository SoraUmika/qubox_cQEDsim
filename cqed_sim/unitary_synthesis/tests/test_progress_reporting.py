from __future__ import annotations

import csv
import json

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer, make_target


def _make_synth(seed: int = 123) -> tuple[Subspace, UnitarySynthesizer]:
    subspace = Subspace.qubit_cavity_block(n_match=2)
    synth = UnitarySynthesizer(
        subspace=subspace,
        backend="ideal",
        gateset=["SQR", "SNAP"],
        optimize_times=True,
        seed=seed,
        progress={"enabled": False, "every": 1, "live": False, "print_every": 10},
    )
    return subspace, synth


def test_f1_history_is_recorded() -> None:
    subspace, synth = _make_synth(seed=41)
    target = make_target("easy", n_match=2)

    result = synth.fit(target=target, multistart=1, maxiter=6)

    assert result.history
    assert list(result.history_by_run) == ["run_000"]

    required_keys = {
        "progress_schema_version",
        "run_id",
        "iteration",
        "timestamp",
        "objective_total",
        "objective_terms",
        "metrics",
        "best_so_far",
        "params_summary",
        "backend",
        "solver_stats",
    }
    assert required_keys.issubset(result.history[0].keys())
    assert result.progress_schema_version == 1

    run_events = result.history_by_run["run_000"]
    iterations = [int(event["iteration"]) for event in run_events]
    assert iterations == sorted(iterations)
    assert all(event["backend"] == "ideal" for event in run_events)
    assert all("fidelity_subspace" in event["metrics"] for event in run_events)
    assert subspace.dim > 0


def test_f2_multistart_aggregation() -> None:
    _, synth = _make_synth(seed=77)
    target = make_target("easy", n_match=2)

    result = synth.fit(target=target, init_guess="random", multistart=3, maxiter=6)

    assert len(result.history_by_run) == 3
    assert set(result.history_by_run) == {"run_000", "run_001", "run_002"}

    best_objective = min(float(event["best_so_far"]["objective_total"]) for event in result.history)
    report_best = float(result.report["optimizer"]["progress"]["global_best"]["objective_total"])
    assert abs(best_objective - report_best) < 1e-9


def test_f3_history_serialization(tmp_path) -> None:
    _, synth = _make_synth(seed=88)
    target = make_target("easy", n_match=2)

    result = synth.fit(target=target, multistart=2, maxiter=5)

    json_path = tmp_path / "history.json"
    csv_path = tmp_path / "history.csv"
    result.save_history(json_path)
    result.save_history_csv(csv_path)

    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(loaded_json, list)
    assert len(loaded_json) == len(result.history)

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert len(rows) == len(result.history)
    assert "objective_total" in rows[0]
    assert "metrics.fidelity_subspace" in rows[0]
