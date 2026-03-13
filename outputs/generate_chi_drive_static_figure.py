from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.calibration import rotation_gaussian_amplitude
from cqed_sim.pulses.envelopes import normalized_gaussian
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence
from cqed_sim.sim.extractors import bloch_xyz_from_joint


def simulate_gaussian_drive(model, initial_state, g_state, e_state, sigma_fraction, duration_s):
    pulse = Pulse(
        channel="qubit",
        t0=0.0,
        duration=duration_s,
        envelope=lambda t_rel: normalized_gaussian(t_rel, sigma_fraction=sigma_fraction),
        amp=rotation_gaussian_amplitude(np.pi, duration_s),
        phase=0.0,
        label="gaussian_x_pi",
    )
    compiled = SequenceCompiler(dt=float(duration_s / 319.0)).compile([pulse], t_end=duration_s)
    result = simulate_sequence(
        model,
        compiled,
        initial_state,
        {"qubit": "qubit"},
        config=SimulationConfig(frame=FrameSpec(), store_states=True),
    )
    states = result.states
    if states is None:
        raise RuntimeError("Expected stored states from Gaussian drive simulation.")
    xyz = np.array([bloch_xyz_from_joint(state) for state in states], dtype=float)
    rho_q_final = qt.ptrace(states[-1], 0)
    z_final = float(np.real((rho_q_final * qt.sigmaz()).tr()))
    p_e = float(np.real((rho_q_final * (e_state * e_state.dag())).tr()))
    return {
        "duration_s": duration_s,
        "compiled": compiled,
        "states": states,
        "xyz": xyz,
        "z_final": z_final,
        "p_e": p_e,
    }


def main():
    n_cav = 20
    g = qt.basis(2, 0)
    e = qt.basis(2, 1)
    alpha_drive = 0.35
    psi_c_drive = qt.coherent(n_cav, alpha_drive)
    psi0_drive = qt.tensor(g, psi_c_drive)

    chi_hz = -2.84e6
    chi2_hz = -21912.0
    kerr_hz = -28844.0
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2 * np.pi * chi_hz,
        chi_higher=(2 * np.pi * chi2_hz,),
        kerr=2 * np.pi * kerr_hz,
        n_cav=n_cav,
        n_tr=2,
    )

    sigma_fraction = 0.16
    duration_candidates = np.linspace(300e-9, 1200e-9, 24)
    candidates = [
        simulate_gaussian_drive(model, psi0_drive, g, e, sigma_fraction, duration_s)
        for duration_s in duration_candidates
    ]
    best = min(candidates, key=lambda item: abs(item["z_final"] + 1.0))

    compiled = best["compiled"]
    xyz = best["xyz"]
    tlist_drive = compiled.tlist
    x_drive = xyz[:, 0]
    y_drive = xyz[:, 1]
    z_drive = xyz[:, 2]
    t_drive = best["duration_s"]

    drive_rel_t = np.clip(tlist_drive / t_drive, 0.0, 1.0)
    env_drive = normalized_gaussian(drive_rel_t, sigma_fraction=sigma_fraction)
    dtheta = env_drive * (tlist_drive[1] - tlist_drive[0])
    theta_ref = np.pi * np.cumsum(dtheta) / np.sum(dtheta)
    theta_ref[0] = 0.0
    theta_ref[-1] = np.pi

    x_ref = np.zeros_like(theta_ref)
    y_ref = -np.sin(theta_ref)
    z_ref = np.cos(theta_ref)
    traj_deviation = np.sqrt((x_drive - x_ref) ** 2 + (y_drive - y_ref) ** 2 + (z_drive - z_ref) ** 2)
    traj_deviation = np.asarray(np.real_if_close(traj_deviation), dtype=float)
    max_dev_idx = int(np.argmax(traj_deviation))

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "lines.linewidth": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=(12.8, 5.1), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.18)

    ax_sphere = fig.add_subplot(gs[0, 0], projection="3d")
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 36)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))
    ax_sphere.plot_surface(sphere_x, sphere_y, sphere_z, color="#dbeafe", edgecolor="none", alpha=0.08, shade=False)
    ax_sphere.plot_wireframe(sphere_x, sphere_y, sphere_z, rstride=8, cstride=8, color="0.78", linewidth=0.4, alpha=0.28)
    ax_sphere.plot([-1, 1], [0, 0], [0, 0], color="0.55", lw=0.8)
    ax_sphere.plot([0, 0], [-1, 1], [0, 0], color="0.55", lw=0.8)
    ax_sphere.plot([0, 0], [0, 0], [-1, 1], color="0.55", lw=0.8)
    ax_sphere.plot(x_ref, y_ref, z_ref, linestyle=(0, (4, 3)), color="#64748b", lw=1.8, label="ideal great-circle path")
    ax_sphere.plot(x_drive, y_drive, z_drive, color="#b91c1c", lw=2.8, label="simulated path with chi")
    ax_sphere.scatter([x_drive[0]], [y_drive[0]], [z_drive[0]], s=44, color="#1d4ed8", depthshade=False)
    ax_sphere.scatter([x_drive[-1]], [y_drive[-1]], [z_drive[-1]], s=52, color="#ea580c", depthshade=False)
    ax_sphere.scatter([x_drive[max_dev_idx]], [y_drive[max_dev_idx]], [z_drive[max_dev_idx]], s=50, color="#111827", depthshade=False)
    ax_sphere.plot(
        [x_ref[max_dev_idx], x_drive[max_dev_idx]],
        [y_ref[max_dev_idx], y_drive[max_dev_idx]],
        [z_ref[max_dev_idx], z_drive[max_dev_idx]],
        color="#111827",
        lw=1.2,
    )
    ax_sphere.text(1.11, 0.0, 0.0, "X", color="0.25")
    ax_sphere.text(0.0, 1.11, 0.0, "Y", color="0.25")
    ax_sphere.text(0.0, 0.0, 1.11, "Z", color="0.25")
    ax_sphere.set_title("Bloch-sphere trajectory during the longer Gaussian drive", pad=12)
    ax_sphere.set_xlim(-1.05, 1.05)
    ax_sphere.set_ylim(-1.05, 1.05)
    ax_sphere.set_zlim(-1.05, 1.05)
    ax_sphere.set_box_aspect((1, 1, 1))
    ax_sphere.set_xticks([-1, 0, 1])
    ax_sphere.set_yticks([-1, 0, 1])
    ax_sphere.set_zticks([-1, 0, 1])
    ax_sphere.set_xlabel("X", labelpad=2)
    ax_sphere.set_ylabel("Y", labelpad=2)
    ax_sphere.set_zlabel("Z", labelpad=2)
    ax_sphere.view_init(elev=20, azim=36)
    ax_sphere.xaxis.pane.set_alpha(0.0)
    ax_sphere.yaxis.pane.set_alpha(0.0)
    ax_sphere.zaxis.pane.set_alpha(0.0)
    ax_sphere.grid(False)
    ax_sphere.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.00), frameon=False)
    ax_sphere.text2D(0.03, 0.95, "a", transform=ax_sphere.transAxes, fontsize=13, fontweight="bold")
    ax_sphere.text2D(
        0.03,
        0.03,
        f"max bend = {traj_deviation[max_dev_idx]:.3f}\nfinal z = {best['z_final']:.3f}",
        transform=ax_sphere.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.30", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.96},
    )

    sub = gs[0, 1].subgridspec(2, 1, hspace=0.18)
    ax_comp = fig.add_subplot(sub[0, 0])
    ax_z = fig.add_subplot(sub[1, 0], sharex=ax_comp)
    time_ns = tlist_drive * 1e9
    ax_comp.plot(time_ns, x_drive, color="#2563eb", label="X")
    ax_comp.plot(time_ns, y_drive, color="#dc2626", label="Y")
    ax_comp.plot(time_ns, z_drive, color="#16a34a", label="Z")
    ax_comp.plot(time_ns, z_ref, linestyle=(0, (4, 3)), color="#64748b", lw=1.7, label="ideal Z")
    ax_comp.axhline(0.0, color="0.55", lw=0.8)
    ax_comp.set_ylabel("Bloch component")
    ax_comp.set_ylim(-1.05, 1.05)
    ax_comp.set_title("The driven path does not stay on the ideal meridian", pad=10)
    ax_comp.grid(alpha=0.18)
    ax_comp.legend(loc="lower left", ncol=4, frameon=False)
    ax_comp.text(-0.09, 1.04, "b", transform=ax_comp.transAxes, fontsize=13, fontweight="bold")

    ax_z.plot(duration_candidates * 1e9, [item["z_final"] for item in candidates], color="#111827", lw=2.1)
    ax_z.scatter(t_drive * 1e9, best["z_final"], color="#ea580c", s=30, zorder=5)
    ax_z.axhline(-1.0, color="#64748b", linestyle="--", lw=1.0)
    ax_z.set_xlabel("Pulse duration (ns)")
    ax_z.set_ylabel("Final Z")
    ax_z.set_title("Duration tuning to approach |e> (Z = -1)", pad=10)
    ax_z.grid(alpha=0.18)

    fig.suptitle("Chi evolution bends the Gaussian-drive trajectory while a longer pulse is tuned toward |e>", y=0.99, fontsize=13, fontweight="semibold")
    fig.subplots_adjust(left=0.05, right=0.98, bottom=0.10, top=0.88)

    output_path = REPO_ROOT / "outputs" / "chi_drive_static_figure.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved figure to: {output_path}")
    print(f"Selected pulse duration = {t_drive * 1e9:.2f} ns")
    print(f"Final excited-state population = {best['p_e']:.4f}")
    print(f"Final Z = {best['z_final']:.4f}")
    print(f"Maximum deviation from the ideal great-circle path = {traj_deviation[max_dev_idx]:.4f}")


if __name__ == "__main__":
    main()