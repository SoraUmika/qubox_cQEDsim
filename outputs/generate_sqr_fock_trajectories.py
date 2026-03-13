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
from cqed_sim.io.gates import SQRGate
from cqed_sim.pulses.builders import build_sqr_multitone_pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim import SimulationConfig, simulate_sequence
from cqed_sim.sim.extractors import conditioned_qubit_state


def conditioned_xyz_trace(states, n_level):
    x_vals = []
    y_vals = []
    z_vals = []
    p_vals = []
    for state in states:
        rho_q_n, p_n, valid = conditioned_qubit_state(state, n=n_level, fallback="nan")
        p_vals.append(float(p_n))
        if valid:
            x_vals.append(float(np.real((rho_q_n * qt.sigmax()).tr())))
            y_vals.append(float(np.real((rho_q_n * qt.sigmay()).tr())))
            z_vals.append(float(np.real((rho_q_n * qt.sigmaz()).tr())))
        else:
            x_vals.append(np.nan)
            y_vals.append(np.nan)
            z_vals.append(np.nan)
    return np.asarray(x_vals), np.asarray(y_vals), np.asarray(z_vals), np.asarray(p_vals)


def coherent_fock_population(dim, alpha, n):
    state = qt.coherent(dim, alpha)
    return float(np.abs(state.overlap(qt.basis(dim, n))) ** 2)


def main():
    slide_size_inches = (13.333, 7.5)
    chi_sqr_hz = -2.84e6
    alpha_sqr = 1.0
    n_cav = 20
    n_levels_plot = (0, 1, 2)
    theta_targets = {
        0: 0.5 * np.pi,
        1: 0.33 * np.pi,
        2: 0.25 * np.pi,
    }
    phi_targets = {
        0: 0.0,
        1: 0.5 * np.pi,
        2: -0.33 * np.pi,
    }

    g = qt.basis(2, 0)

    model_sqr = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=2 * np.pi * chi_sqr_hz,
        n_cav=n_cav,
        n_tr=2,
    )
    frame_sqr = FrameSpec(omega_c_frame=0.0, omega_q_frame=0.0)

    psi0_sqr = qt.tensor(g, qt.coherent(n_cav, alpha_sqr))
    pop_init = [coherent_fock_population(n_cav, alpha_sqr, n_level) for n_level in n_levels_plot]

    theta_values = [0.0] * n_cav
    phi_values = [0.0] * n_cav
    for n_level in n_levels_plot:
        theta_values[n_level] = float(theta_targets[n_level])
        phi_values[n_level] = float(phi_targets[n_level])

    gate_sqr = SQRGate(
        index=0,
        name="sqr_n012_custom_angles",
        theta=tuple(theta_values),
        phi=tuple(phi_values),
    )

    pulses_sqr, drive_ops_sqr, meta_sqr = build_sqr_multitone_pulse(
        gate_sqr,
        model_sqr,
        {
            "duration_sqr_s": 1.0e-6,
            "sqr_sigma_fraction": 1.0 / 6.0,
            "sqr_theta_cutoff": 1.0e-10,
            "use_rotating_frame": True,
        },
    )
    compiled_sqr = SequenceCompiler(dt=2.0e-9).compile(
        pulses_sqr,
        t_end=max(p.t1 for p in pulses_sqr) + 2.0e-9,
    )
    result_sqr = simulate_sequence(
        model_sqr,
        compiled_sqr,
        psi0_sqr,
        drive_ops_sqr,
        config=SimulationConfig(frame=frame_sqr, store_states=True),
    )
    states_sqr = result_sqr.states
    if states_sqr is None:
        raise RuntimeError("SQR simulation did not return states; expected store_states=True.")

    sqr_traces = {n_level: conditioned_xyz_trace(states_sqr, n_level) for n_level in n_levels_plot}
    time_us_sqr = compiled_sqr.tlist * 1e6
    colors = {0: "#2563eb", 1: "#dc2626", 2: "#16a34a"}

    plt.rcParams.update(
        {
            "font.size": 15,
            "axes.labelsize": 15,
            "axes.titlesize": 17,
            "legend.fontsize": 12,
            "lines.linewidth": 2.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig = plt.figure(figsize=slide_size_inches, dpi=220)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.06)

    u = np.linspace(0, 2 * np.pi, 56)
    v = np.linspace(0, np.pi, 28)
    sphere_x = np.outer(np.cos(u), np.sin(v))
    sphere_y = np.outer(np.sin(u), np.sin(v))
    sphere_z = np.outer(np.ones_like(u), np.cos(v))

    for col, n_level in enumerate(n_levels_plot):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        x_n, y_n, z_n, _p_n = sqr_traces[n_level]
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color="#e2e8f0", edgecolor="none", alpha=0.08, shade=False)
        ax.plot_wireframe(sphere_x, sphere_y, sphere_z, rstride=7, cstride=7, color="0.78", linewidth=0.35, alpha=0.25)
        ax.plot([-1, 1], [0, 0], [0, 0], color="0.6", lw=0.7)
        ax.plot([0, 0], [-1, 1], [0, 0], color="0.6", lw=0.7)
        ax.plot([0, 0], [0, 0], [-1, 1], color="0.6", lw=0.7)
        ax.plot(x_n, y_n, z_n, color=colors[n_level], lw=2.8, label=f"n={n_level} trajectory")
        ax.scatter([x_n[0]], [y_n[0]], [z_n[0]], color="#0f172a", s=20, depthshade=False)
        ax.scatter([x_n[-1]], [y_n[-1]], [z_n[-1]], color=colors[n_level], s=34, depthshade=False)
        ax.text2D(
            0.04,
            0.06,
            f"P(n) ≈ {pop_init[n_levels_plot.index(n_level)]:.3f}\nfinal Z = {z_n[-1]:.3f}",
            transform=ax.transAxes,
            fontsize=12,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.95},
        )
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_zlim(-1.05, 1.05)
        ax.set_box_aspect((1, 1, 1))
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.set_zticks([-1, 0, 1])
        ax.tick_params(axis="x", labelsize=11.5, pad=-1)
        ax.tick_params(axis="y", labelsize=11.5, pad=-1)
        ax.tick_params(axis="z", labelsize=11.5, pad=-1)
        ax.set_xlabel("X", labelpad=-2)
        ax.set_ylabel("Y", labelpad=-2)
        ax.set_zlabel("Z", labelpad=-2)
        ax.view_init(elev=20, azim=36)
        ax.xaxis.pane.set_alpha(0.0)
        ax.yaxis.pane.set_alpha(0.0)
        ax.zaxis.pane.set_alpha(0.0)
        ax.grid(False)
        ax.legend(loc="upper right", fontsize=23, frameon=False, handlelength=2.4)

    fig.subplots_adjust(left=0.02, right=0.992, bottom=0.06, top=0.97)

    output_path = REPO_ROOT / "outputs" / "sqr_fock_trajectories_n012_custom_angles.png"
    ppt_output_path = REPO_ROOT / "outputs" / "sqr_fock_trajectories_n012_custom_angles_ppt.png"
    fig.savefig(output_path, dpi=300, facecolor="white")
    fig.savefig(ppt_output_path, dpi=300, facecolor="white")

    print(f"Saved figure to: {output_path}")
    print(f"Saved PPT-scaled figure to: {ppt_output_path}")
    print(f"Number of active SQR tones = {len(meta_sqr['active_tones'])}")
    print(f"Coherent-state populations P(0), P(1), P(2) = {[round(val, 4) for val in pop_init]}")
    print(f"Theta targets / pi = {[round(theta_targets[n] / np.pi, 4) for n in n_levels_plot]}")
    print(f"Phi targets / pi = {[round(phi_targets[n] / np.pi, 4) for n in n_levels_plot]}")
    for n_level in n_levels_plot:
        x_n, y_n, z_n, p_n = sqr_traces[n_level]
        print(
            f"n={n_level}: final Bloch = ({x_n[-1]:.4f}, {y_n[-1]:.4f}, {z_n[-1]:.4f}), "
            f"conditioned population = {p_n[-1]:.4f}"
        )


if __name__ == "__main__":
    main()