from __future__ import annotations

from pathlib import Path

from cqed_sim.snap_prl133.reproduce import ReproduceConfig, run_reproduction


def test_reproduce_baseline(tmp_path: Path):
    cfg = ReproduceConfig(
        n_cav=6,
        n_tr=2,
        n_max=2,
        durations=(70.0, 120.0, 220.0),
        iteration_cap=24,
        epsilon_target=1e-5,
        local_refine_maxiter=0,
    )
    summary = run_reproduction(config=cfg, output_root=tmp_path)
    eps_v = summary["curves"]["epsilon_vanilla"]
    eps_o = summary["curves"]["epsilon_optimized"]
    f_o = summary["curves"]["fidelity_optimized"]
    assert len(eps_v) == len(cfg.durations)
    assert len(eps_o) == len(cfg.durations)
    assert len(f_o) == len(cfg.durations)
    assert all(0.0 <= x <= 1.0 for x in f_o)
    assert all(0.0 <= x <= 1.0 for x in eps_v)
    assert all(0.0 <= x <= 1.0 for x in eps_o)
    assert eps_o[-1] <= eps_v[-1]
    assert summary["metric_bounds_ok"] is True
    te = summary["threshold_evidence"]
    assert te["hit_above"] == (te["epsilon_above"] < cfg.epsilon_target)
    assert te["hit_below"] == (te["epsilon_below"] < cfg.epsilon_target)
    if any(e < cfg.epsilon_target for e in eps_o):
        assert te["hit_above"] is True
        assert te["hit_below"] is False
    assert len(summary["per_manifold_at_topt"]) == cfg.n_max + 1
    assert (tmp_path / "report.md").exists()
