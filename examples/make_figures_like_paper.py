from __future__ import annotations

from pathlib import Path

import numpy as np

from cqed_sim.snap_opt import SnapModelConfig, SnapRunConfig, optimize_snap_parameters, target_difficulty_metric


def _plot_or_dump(path: Path, x, ys, labels, title, ylabel):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        np.savez(path.with_suffix(".npz"), x=np.asarray(x), ys=np.asarray(ys), labels=np.asarray(labels, dtype=object))
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for y, lab in zip(ys, labels):
        ax.plot(x, y, marker="o", label=lab)
    ax.set_title(title)
    ax.set_xlabel("Gate time T")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    out = Path("examples") / "outputs_snap_opt"
    out.mkdir(parents=True, exist_ok=True)

    model = SnapModelConfig(n_cav=7, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    targets = [
        np.array([0.0, 0.4, -0.2, 0.1], dtype=float),
        np.array([0.0, 1.0, -0.8, 0.7], dtype=float),
        np.array([0.0, -1.2, 1.0, -0.9], dtype=float),
    ]
    durations = [70.0, 90.0, 120.0, 150.0, 180.0]
    vanilla_errors = {i: [] for i in range(len(targets))}
    opt_errors = {i: [] for i in range(len(targets))}
    topts = []
    diffs = []

    for i, target in enumerate(targets):
        found_t = None
        for t in durations:
            cfg = SnapRunConfig(duration=t, dt=0.25, base_amp=0.010)
            r = optimize_snap_parameters(model, target, cfg, max_iter=30, learning_rate=0.3, threshold=1e-2)
            vanilla_errors[i].append(r.history_error[0])
            opt_errors[i].append(r.history_error[-1])
            if found_t is None and r.history_error[-1] < 1e-2:
                found_t = t
        topts.append(found_t if found_t is not None else max(durations))
        diffs.append(target_difficulty_metric(target))
        _plot_or_dump(
            out / f"error_vs_time_target_{i}.png",
            durations,
            [vanilla_errors[i], opt_errors[i]],
            ["vanilla", "optimized"],
            f"Target {i}: error scaling",
            "coherent error",
        )

    _plot_or_dump(out / "topt_vs_difficulty.png", diffs, [topts], ["Topt"], "Optimization limit vs target difficulty", "Topt")
    print("Figure data written to", out)


if __name__ == "__main__":
    main()

