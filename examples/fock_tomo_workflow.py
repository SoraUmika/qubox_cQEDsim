from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.tomo.protocol import (
    QubitPulseCal,
    autocalibrate_all_xy,
    calibrate_leakage_matrix,
    run_all_xy,
    run_fock_resolved_tomo,
    true_fock_resolved_vectors,
)


def _save_plot(path: Path, x, ys, labels, title, ylabel):
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        np.savez(path.with_suffix(".npz"), x=np.asarray(x), ys=np.asarray(ys), labels=np.asarray(labels, dtype=object))
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    for y, lbl in zip(ys, labels):
        ax.plot(x, y, marker="o", label=lbl)
    ax.set_title(title)
    ax.set_xlabel("Index")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    out_dir = Path("examples") / "outputs_fock_tomo"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=2 * np.pi * 0.020, chi_higher=(), kerr=0.0, n_cav=9, n_tr=2
    )
    bad = QubitPulseCal(amp90=QubitPulseCal.nominal().amp90 * 0.8, y_phase=np.pi / 2 + 0.3, drag=0.0)
    allxy_before = run_all_xy(model, bad, dt_ns=0.2)
    cal, allxy_after = autocalibrate_all_xy(model, bad, dt_ns=0.2, max_iter=10, target_rms=0.08)

    n_max = 3
    alphas = [0.4, 0.9, 1.3]
    bloch_cal = [np.array([0, 0, 1]), np.array([0, 0, -1]), np.array([1, 0, 0]), np.array([0, 1, 0])]
    w, b, cond = calibrate_leakage_matrix(model, n_max, alphas, bloch_cal, cal, tag_duration_ns=1200.0, tag_amp=0.0014, dt_ns=1.0)

    probs = np.array([0.45, 0.25, 0.20, 0.10], dtype=float)
    blochs = [np.array([0, 0, 1]), np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, -1])]
    rho = 0
    for n in range(n_max + 1):
        rho_q = 0.5 * (qt.qeye(2) + blochs[n][0] * qt.sigmax() + blochs[n][1] * qt.sigmay() + blochs[n][2] * qt.sigmaz())
        rho += probs[n] * qt.tensor(qt.basis(model.n_cav, n).proj(), rho_q)
    base = run_fock_resolved_tomo(
        model=model, state_prep=lambda: rho, n_max=n_max, cal=cal, ideal_tag=False, tag_duration_ns=1200.0, tag_amp=0.0014, dt_ns=1.0
    )
    corr = run_fock_resolved_tomo(
        model=model,
        state_prep=lambda: rho,
        n_max=n_max,
        cal=cal,
        ideal_tag=False,
        tag_duration_ns=1200.0,
        tag_amp=0.0014,
        dt_ns=1.0,
        leakage_cal=(w, b),
    )
    true_v = true_fock_resolved_vectors(rho, n_max)
    true_p = np.array([np.real((rho * qt.tensor(qt.basis(model.n_cav, n).proj(), qt.qeye(2))).tr()) for n in range(n_max + 1)], dtype=float)

    idx = np.arange(len(allxy_before["measured_z"]))
    _save_plot(out_dir / "allxy.png", idx, [allxy_before["measured_z"], allxy_after["measured_z"], allxy_after["expected_z"]], ["before", "after", "ideal"], "ALL_XY", "Z")
    _save_plot(out_dir / "pn_compare.png", np.arange(n_max + 1), [base.p_n, true_p], ["estimated", "true"], "P(n)", "Probability")
    _save_plot(
        out_dir / "bloch_x_compare.png",
        np.arange(n_max + 1),
        [base.v_hat["x"], corr.v_rec["x"], true_v["x"]],  # type: ignore[index]
        ["raw", "unmixed", "true"],
        "Fock-resolved X",
        "X_n * P(n)",
    )
    np.save(out_dir / "leakage_matrix.npy", w)

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "allxy_rms_before": float(allxy_before["rms_error"]),
                "allxy_rms_after": float(allxy_after["rms_error"]),
                "leakage_condition_number": float(cond),
            },
            f,
            indent=2,
        )
    print("Outputs written to", out_dir)


if __name__ == "__main__":
    main()

