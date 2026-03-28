"""Generate tutorial plots for the documentation website.

Each function generates one or more publication-quality figures and saves them
to documentations/assets/images/tutorials/.

Run from the repository root:
    python tools/generate_tutorial_plots.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is on path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

OUT_DIR = REPO / "documentations" / "assets" / "images" / "tutorials"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------
STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
}


def _apply_style():
    plt.rcParams.update(STYLE)


# ---------------------------------------------------------------------------
# 1. Displacement + Qubit Spectroscopy
# ---------------------------------------------------------------------------
def plot_displacement_spectroscopy():
    """Number-splitting spectrum: P(e) vs qubit-drive detuning."""
    _apply_style()
    print("[1/4] Displacement spectroscopy …")
    from examples.displacement_qubit_spectroscopy import (
        run_displacement_then_qubit_spectroscopy,
        save_artifacts,
    )

    result = run_displacement_then_qubit_spectroscopy()
    save_artifacts(result)
    print("    ✓ displacement_spectroscopy.png")


# ---------------------------------------------------------------------------
# 2. Kerr Free Evolution ‒ Wigner snapshots
# ---------------------------------------------------------------------------
def plot_kerr_free_evolution():
    """Wigner-function snapshots during Kerr-only free evolution."""
    _apply_style()
    print("[2/4] Kerr free evolution Wigner snapshots …")
    from cqed_sim.core import coherent_state
    from examples.workflows.kerr_free_evolution import (
        plot_kerr_wigner_snapshots,
        run_kerr_free_evolution,
    )

    kerr_hz = -2.0e3
    t_kerr_s = 1.0 / abs(kerr_hz)
    snapshot_times_s = [0.0, 0.25 * t_kerr_s, 0.5 * t_kerr_s, 0.75 * t_kerr_s]
    result = run_kerr_free_evolution(
        snapshot_times_s,
        cavity_state=coherent_state(2.0),
        parameter_set="phase_evolution",
        n_cav=25,
        wigner_times_s=snapshot_times_s,
        wigner_n_points=121,
        wigner_extent=4.6,
        wigner_coordinate="alpha",
    )
    fig = plot_kerr_wigner_snapshots(result, max_cols=4, show_colorbar=True, coordinate="alpha")
    symbolic_labels = [r"$t = 0$", r"$t = T_K/4$", r"$t = T_K/2$", r"$t = 3T_K/4$"]
    for axis, label in zip(fig.axes[:4], symbolic_labels, strict=True):
        axis.set_title(label)
    fig.suptitle(
        r"Kerr Free Evolution -- Wigner Function Snapshots  ($|\alpha|=2$, $K/2\pi=-2$ kHz)",
        fontsize=13,
        y=0.96,
    )
    fig.savefig(OUT_DIR / "kerr_free_evolution_wigner.png", dpi=170)
    plt.close(fig)
    print("    ✓ kerr_free_evolution_wigner.png")


# ---------------------------------------------------------------------------
# 3. Sideband Swap — Population dynamics
# ---------------------------------------------------------------------------
def plot_sideband_swap():
    """Transmon ↔ cavity population exchange during a red-sideband drive."""
    _apply_style()
    print("[3/4] Sideband swap dynamics …")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel,
        FrameSpec,
        SidebandDriveSpec,
        StatePreparationSpec,
        qubit_state,
        fock_state,
        prepare_state,
        carrier_for_transition_frequency,
    )
    from cqed_sim.pulses import build_sideband_pulse
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import (
        SimulationConfig,
        simulate_sequence,
        reduced_qubit_state,
        reduced_cavity_state,
    )

    model = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 5.0e9,
        omega_q=2 * np.pi * 6.0e9,
        alpha=2 * np.pi * (-200e6),
        chi=2 * np.pi * (-2.84e6),
        n_cav=8,
        n_tr=3,
    )
    frame = FrameSpec(
        omega_c_frame=model.omega_c,
        omega_q_frame=model.omega_q,
    )

    target = SidebandDriveSpec(
        mode="storage",
        lower_level=0,
        upper_level=1,
        sideband="red",
    )
    omega_sb = model.sideband_transition_frequency(
        cavity_level=0, lower_level=0, upper_level=1,
        sideband="red", frame=frame,
    )

    # Sweep pulse duration to see Rabi oscillation
    durations_ns = np.linspace(10, 1000, 60)
    pe_vals = []
    n_cav_vals = []

    initial = prepare_state(
        model,
        StatePreparationSpec(qubit=qubit_state("e"), storage=fock_state(0)),
    )

    for dur_ns in durations_ns:
        dur_s = dur_ns * 1e-9
        pulses, drive_ops, _ = build_sideband_pulse(
            target,
            duration_s=dur_s,
            amplitude_rad_s=2 * np.pi * 1e6,
            channel="sideband",
            carrier=carrier_for_transition_frequency(omega_sb),
        )
        compiled = SequenceCompiler(dt=2e-9).compile(pulses, t_end=dur_s + 50e-9)
        result = simulate_sequence(
            model, compiled, initial, drive_ops,
            config=SimulationConfig(frame=frame),
        )
        rho_q = reduced_qubit_state(result.final_state)
        rho_c = reduced_cavity_state(result.final_state)
        pe_vals.append(float(np.real(rho_q[1, 1])))
        rho_c_arr = np.array(rho_c) if not hasattr(rho_c, "full") else rho_c.full()
        n_hat = np.diag(np.arange(rho_c_arr.shape[0], dtype=float))
        n_cav_vals.append(float(np.real(np.trace(n_hat @ rho_c_arr))))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.plot(durations_ns, pe_vals, color="#d62728", linewidth=1.5)
    ax1.set_ylabel("$P(e)$")
    ax1.set_title("Red-Sideband Swap: Transmon ↔ Cavity Population Exchange")

    ax2.plot(durations_ns, n_cav_vals, color="#2ca02c", linewidth=1.5)
    ax2.set_ylabel("Cavity $\\langle n \\rangle$")
    ax2.set_xlabel("Sideband pulse duration (ns)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "sideband_swap_dynamics.png")
    plt.close(fig)
    print("    ✓ sideband_swap_dynamics.png")


# ---------------------------------------------------------------------------
# 4. GRAPE Optimal Control — Convergence + Waveform
# ---------------------------------------------------------------------------
def plot_grape_optimal_control():
    """GRAPE convergence curve and optimized waveform for a qubit π-pulse."""
    _apply_style()
    print("[4/4] GRAPE optimal control …")
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
    from cqed_sim.optimal_control import (
        PiecewiseConstantTimeGrid,
        ModelControlChannelSpec,
        build_control_problem_from_model,
        state_preparation_objective,
        GrapeSolver,
        GrapeConfig,
    )

    model = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 5e9,
        omega_q=2 * np.pi * 6e9,
        alpha=0.0, chi=0.0, kerr=0.0,
        n_cav=1, n_tr=2,
    )
    frame = FrameSpec(
        omega_c_frame=model.omega_c,
        omega_q_frame=model.omega_q,
    )

    grid = PiecewiseConstantTimeGrid.uniform(steps=6, dt_s=20e-9)

    problem = build_control_problem_from_model(
        model, frame=frame,
        time_grid=grid,
        channel_specs=(ModelControlChannelSpec(
            name="qubit", target="qubit", quadratures=("I", "Q"),
            amplitude_bounds=(-8e7, 8e7), export_channel="qubit",
        ),),
        objectives=(state_preparation_objective(
            model.basis_state(0, 0),
            model.basis_state(1, 0),
        ),),
    )

    solver = GrapeSolver(GrapeConfig(maxiter=80, seed=42, random_scale=0.15))
    result = solver.solve(problem)

    # Extract convergence history (list of GrapeIterationRecord)
    history = result.history

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Convergence
    ax1 = axes[0]
    if history and len(history) > 1:
        obj_vals = [rec.objective for rec in history]
        ax1.semilogy(range(1, len(obj_vals) + 1), obj_vals,
                     color="#1f77b4", linewidth=1.5)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Infidelity (objective)")
        ax1.set_title("GRAPE Convergence")
    else:
        ax1.text(0.5, 0.5, f"Final objective: {result.objective_value:.4e}",
                 ha="center", va="center", transform=ax1.transAxes, fontsize=14)
        ax1.set_title("GRAPE Result")

    # Optimized waveform
    ax2 = axes[1]
    controls = result.command_values  # shape (n_quadratures, n_steps)
    tb = np.array(result.time_boundaries_s)
    t_centers = 0.5 * (tb[:-1] + tb[1:]) * 1e9  # ns

    quad_labels = ["I", "Q"]
    for i, row in enumerate(controls):
        label = quad_labels[i] if i < len(quad_labels) else f"Ch {i}"
        ax2.step(t_centers, row / 1e6, where="mid",
                 label=label, linewidth=1.5)

    ax2.set_xlabel("Time (ns)")
    ax2.set_ylabel("Amplitude (MHz)")
    ax2.set_title("Optimized Control Waveform")
    ax2.legend(loc="best", fontsize=9)

    fig.suptitle(f"GRAPE $|g,0\\rangle \\to |e,0\\rangle$ — Final infidelity: {result.objective_value:.2e}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUT_DIR / "grape_optimal_control.png")
    plt.close(fig)
    print("    ✓ grape_optimal_control.png")


# ---------------------------------------------------------------------------
# 5. Cross-Kerr Conditional Phase Accumulation
# ---------------------------------------------------------------------------
def plot_cross_kerr_phase():
    """Relative phase vs free-evolution time for storage-readout cross-Kerr."""
    _apply_style()
    print("[5/6] Cross-Kerr phase accumulation …")
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec, prepare_state
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence

    try:
        from cqed_sim.core import DispersiveReadoutTransmonStorageModel
        model = DispersiveReadoutTransmonStorageModel(
            omega_s=2 * np.pi * 5.0e9,
            omega_r=2 * np.pi * 7.5e9,
            omega_q=2 * np.pi * 6.0e9,
            chi_sr=2 * np.pi * 1.5e6,
            chi_s=0.0,
            chi_r=0.0,
            n_storage=4,
            n_readout=4,
            n_tr=2,
        )
        from cqed_sim.core import FrameSpec as FS
        frame = FS(
            omega_s_frame=model.omega_s,
            omega_r_frame=model.omega_r,
            omega_q_frame=model.omega_q,
        )

        s0r1 = model.basis_state(0, 1, 0)
        s1r1 = model.basis_state(1, 1, 0)
        initial_state = (s0r1 + s1r1).unit()

        chi_sr_hz = 1.5e6
        times_ns = np.linspace(0, 700, 50)
        phases = []

        for t_ns in times_ns:
            t_s = t_ns * 1e-9
            if t_s == 0:
                phases.append(0.0)
                continue
            compiled = SequenceCompiler(dt=2e-9).compile([], t_end=t_s)
            result = simulate_sequence(
                model, compiled, initial_state, {},
                config=SimulationConfig(frame=frame),
            )
            state = result.final_state
            amp_ref = s0r1.overlap(state)
            amp_shifted = s1r1.overlap(state)
            phases.append(float(np.angle(amp_shifted / amp_ref)))

        theory_times_ns = np.linspace(0, 700, 300)
        theory_phases = 2 * np.pi * chi_sr_hz * (theory_times_ns * 1e-9)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times_ns, phases, "o", ms=4, color="#1f77b4", label="Simulation")
        ax.plot(theory_times_ns, (theory_phases + np.pi) % (2 * np.pi) - np.pi,
                "--", linewidth=1.5, color="#ff7f0e",
                label=r"Theory: $\chi_{sr} \cdot t$")
        ax.set_xlabel("Free evolution time (ns)")
        ax.set_ylabel("Relative phase (rad)")
        ax.set_title(
            r"Cross-Kerr Conditional Phase  ($\chi_{sr}/2\pi = 1.5$ MHz)"
        )
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cross_kerr_phase.png", dpi=150)
        plt.close(fig)
        print("    ✓ cross_kerr_phase.png")
    except Exception as exc:
        print(f"    ✗ cross_kerr_phase skipped: {exc}")


# ---------------------------------------------------------------------------
# 6. Floquet Quasienergy Scan
# ---------------------------------------------------------------------------
def plot_floquet_quasienergy_scan():
    """Quasienergy branches and avoided-crossing gap for a sideband drive sweep."""
    _apply_style()
    print("[6/6] Floquet quasienergy scan …")

    try:
        from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
        from cqed_sim.floquet import (
            FloquetProblem,
            FloquetConfig,
            run_floquet_sweep,
            build_target_drive_term,
            SidebandDriveSpec,
        )

        model = DispersiveTransmonCavityModel(
            omega_c=2 * np.pi * 5.05e9,
            omega_q=2 * np.pi * 6.25e9,
            alpha=2 * np.pi * (-250e6),
            chi=2 * np.pi * (-15.0e6),
            kerr=0.0,
            n_cav=4,
            n_tr=3,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

        sideband = SidebandDriveSpec(
            mode="storage", lower_level=0, upper_level=2, sideband="red"
        )
        omega_sb0 = model.sideband_transition_frequency(
            cavity_level=0, lower_level=0, upper_level=2,
            sideband="red", frame=frame,
        )

        scan_detunings_mhz = np.linspace(-0.25, 0.25, 25)
        problems = []
        for det_mhz in scan_detunings_mhz:
            freq_hz = omega_sb0 / (2 * np.pi) + det_mhz * 1e6
            drive = build_target_drive_term(
                model, sideband,
                amplitude=2 * np.pi * 0.03e6,
                frequency=2 * np.pi * freq_hz,
                waveform="cos",
            )
            problems.append(FloquetProblem(
                model=model, frame=frame,
                periodic_terms=(drive,),
                period=1.0 / freq_hz,
                label="sideband_scan",
            ))

        sweep = run_floquet_sweep(
            problems,
            parameter_values=scan_detunings_mhz,
            config=FloquetConfig(n_time_samples=64),
        )

        tracked_mhz = sweep.tracked_quasienergies / (2 * np.pi * 1e6)
        gap_mhz = np.min(np.abs(np.diff(tracked_mhz, axis=1)), axis=1)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for i in range(tracked_mhz.shape[1]):
            ax1.plot(scan_detunings_mhz, tracked_mhz[:, i], linewidth=1.5)
        ax1.set_ylabel("Quasienergy (MHz)")
        ax1.set_title("Floquet Quasienergy Branches — Sideband Drive Sweep")

        ax2.plot(scan_detunings_mhz, gap_mhz, color="#d62728", linewidth=1.5)
        ax2.axvline(
            scan_detunings_mhz[np.argmin(gap_mhz)],
            ls="--", color="gray", alpha=0.6, label="Min gap (resonance)",
        )
        ax2.set_ylabel("Min quasienergy gap (MHz)")
        ax2.set_xlabel("Drive detuning (MHz)")
        ax2.set_title("Avoided-Crossing Gap Diagnostic")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR / "floquet_quasienergy_scan.png", dpi=150)
        plt.close(fig)
        print("    ✓ floquet_quasienergy_scan.png")
    except Exception as exc:
        print(f"    ✗ floquet_quasienergy_scan skipped: {exc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    plot_displacement_spectroscopy()
    plot_kerr_free_evolution()
    plot_sideband_swap()
    plot_grape_optimal_control()
    plot_cross_kerr_phase()
    plot_floquet_quasienergy_scan()
    print(f"\nAll plots saved to {OUT_DIR}")
