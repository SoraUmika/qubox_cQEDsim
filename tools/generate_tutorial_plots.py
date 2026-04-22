"""Generate tutorial plots for the documentation website.

Each function generates one or more publication-quality figures and saves them
to documentations/assets/images/tutorials/.

Run from the repository root:
    python tools/generate_tutorial_plots.py
"""

from __future__ import annotations

import json
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
VALIDATION_DIR = OUT_DIR / "validation"
VALIDATION_DIR.mkdir(parents=True, exist_ok=True)


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


def _write_validation_summary(filename: str, payload: dict) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    path = VALIDATION_DIR / filename
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _relative_branch_phase(states, reference_state, shifted_state) -> np.ndarray:
    phases: list[float] = []
    for state in states:
        amp_reference = complex(reference_state.overlap(state))
        amp_shifted = complex(shifted_state.overlap(state))
        if abs(amp_reference) < 1.0e-12 or abs(amp_shifted) < 1.0e-12:
            raise ValueError("Relative phase extraction encountered a vanishing branch amplitude.")
        phases.append(float(np.angle(amp_shifted / amp_reference)))
    return np.unwrap(np.asarray(phases, dtype=float))


def compute_cross_kerr_phase_data() -> dict[str, object]:
    """Return the conditional storage-phase trace induced by readout occupancy."""
    from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence

    model = DispersiveReadoutTransmonStorageModel(
        omega_s=2 * np.pi * 5.0e9,
        omega_r=2 * np.pi * 7.5e9,
        omega_q=2 * np.pi * 6.0e9,
        alpha=2 * np.pi * (-200e6),
        chi_sr=2 * np.pi * 1.5e6,
        chi_s=0.0,
        chi_r=0.0,
        n_storage=4,
        n_readout=4,
        n_tr=2,
    )
    frame = FrameSpec(
        omega_c_frame=model.omega_s,
        omega_q_frame=model.omega_q,
        omega_r_frame=model.omega_r,
    )

    duration_s = 700.0e-9
    dt_s = 2.0e-9
    compiled = SequenceCompiler(dt=dt_s).compile([], t_end=duration_s)

    empty_readout_branch = (model.basis_state(0, 0, 0) + model.basis_state(0, 1, 0)).unit()
    occupied_readout_branch = (model.basis_state(0, 0, 1) + model.basis_state(0, 1, 1)).unit()

    result_empty = simulate_sequence(
        model,
        compiled,
        empty_readout_branch,
        {},
        config=SimulationConfig(frame=frame, store_states=True, max_step=dt_s),
    )
    result_occupied = simulate_sequence(
        model,
        compiled,
        occupied_readout_branch,
        {},
        config=SimulationConfig(frame=frame, store_states=True, max_step=dt_s),
    )

    times_s = np.asarray(compiled.tlist, dtype=float)
    empty_phase = _relative_branch_phase(
        result_empty.states,
        model.basis_state(0, 0, 0),
        model.basis_state(0, 1, 0),
    )
    occupied_phase = _relative_branch_phase(
        result_occupied.states,
        model.basis_state(0, 0, 1),
        model.basis_state(0, 1, 1),
    )
    conditional_phase = occupied_phase - empty_phase
    theory_phase = -float(model.chi_sr) * times_s
    fitted_slope_rad_s, fitted_offset_rad = np.polyfit(times_s, conditional_phase, deg=1)

    return {
        "times_ns": (times_s * 1.0e9).tolist(),
        "conditional_phase_rad": conditional_phase.tolist(),
        "theory_phase_rad": theory_phase.tolist(),
        "chi_sr_hz": float(model.chi_sr / (2.0 * np.pi)),
        "fitted_slope_hz": float(fitted_slope_rad_s / (2.0 * np.pi)),
        "fitted_offset_rad": float(fitted_offset_rad),
        "max_abs_phase_error_rad": float(np.max(np.abs(conditional_phase - theory_phase))),
    }


def compute_floquet_quasienergy_scan_data() -> dict[str, object]:
    """Return a visibly hybridized Floquet branch pair and its avoided-crossing gap."""
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec, SidebandDriveSpec
    from cqed_sim.floquet import FloquetProblem, FloquetConfig, build_target_drive_term, run_floquet_sweep

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

    sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2, sideband="red")
    omega_sb0 = model.sideband_transition_frequency(
        cavity_level=0,
        lower_level=0,
        upper_level=2,
        sideband="red",
        frame=frame,
    )

    drive_amplitude_mhz = 0.03
    scan_detunings_mhz = np.linspace(-0.30, 0.30, 25)
    problems = []
    for detuning_mhz in scan_detunings_mhz:
        drive_frequency_hz = omega_sb0 / (2.0 * np.pi) + detuning_mhz * 1.0e6
        drive = build_target_drive_term(
            model,
            sideband,
            amplitude=2 * np.pi * drive_amplitude_mhz * 1.0e6,
            frequency=2 * np.pi * drive_frequency_hz,
            waveform="cos",
        )
        problems.append(
            FloquetProblem(
                model=model,
                frame=frame,
                periodic_terms=(drive,),
                period=1.0 / abs(drive_frequency_hz),
                label="sideband_scan",
            )
        )

    sweep = run_floquet_sweep(
        problems,
        parameter_values=scan_detunings_mhz,
        config=FloquetConfig(n_time_samples=96),
    )

    tracked_quasienergies_mhz = np.asarray(sweep.tracked_quasienergies, dtype=float) / (2.0 * np.pi * 1.0e6)
    center_index = len(scan_detunings_mhz) // 2
    center_result = sweep.results[center_index]
    center_max_overlaps = np.max(center_result.bare_state_overlaps, axis=1)
    highlighted_pair = np.argsort(center_max_overlaps)[:2]
    highlighted_pair = highlighted_pair[np.argsort(tracked_quasienergies_mhz[center_index, highlighted_pair])]

    highlighted_branches_mhz = tracked_quasienergies_mhz[:, highlighted_pair]
    highlighted_gap_mhz = np.abs(highlighted_branches_mhz[:, 1] - highlighted_branches_mhz[:, 0])
    resonance_index = int(np.argmin(highlighted_gap_mhz))

    return {
        "scan_detunings_mhz": scan_detunings_mhz.tolist(),
        "tracked_quasienergies_mhz": tracked_quasienergies_mhz.tolist(),
        "highlighted_pair_indices": [int(index) for index in highlighted_pair],
        "highlighted_branches_mhz": highlighted_branches_mhz.tolist(),
        "highlighted_gap_mhz": highlighted_gap_mhz.tolist(),
        "center_pair_max_overlaps": center_max_overlaps[highlighted_pair].tolist(),
        "drive_amplitude_mhz": drive_amplitude_mhz,
        "resonance_detuning_mhz": float(scan_detunings_mhz[resonance_index]),
        "min_gap_mhz": float(highlighted_gap_mhz[resonance_index]),
    }


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

    try:
        data = compute_cross_kerr_phase_data()
        times_ns = np.asarray(data["times_ns"], dtype=float)
        conditional_phase = np.asarray(data["conditional_phase_rad"], dtype=float)
        theory_phase = np.asarray(data["theory_phase_rad"], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(times_ns, conditional_phase, "o", ms=3.5, color="#1f77b4", label="Simulation")
        ax.plot(times_ns, theory_phase, "--", linewidth=1.5, color="#ff7f0e", label=r"Theory: $-\chi_{sr} \cdot t$")
        ax.set_xlabel("Free evolution time (ns)")
        ax.set_ylabel("Conditional relative phase (rad)")
        ax.set_title(r"Cross-Kerr Conditional Phase  ($\Delta\phi_{r=1} - \Delta\phi_{r=0} = -\chi_{sr} t$)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "cross_kerr_phase.png", dpi=150)
        plt.close(fig)
        _write_validation_summary("cross_kerr_phase.json", data)
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
        data = compute_floquet_quasienergy_scan_data()
        scan_detunings_mhz = np.asarray(data["scan_detunings_mhz"], dtype=float)
        tracked_mhz = np.asarray(data["tracked_quasienergies_mhz"], dtype=float)
        highlighted_branches_mhz = np.asarray(data["highlighted_branches_mhz"], dtype=float)
        highlighted_gap_mhz = np.asarray(data["highlighted_gap_mhz"], dtype=float)
        resonance_detuning_mhz = float(data["resonance_detuning_mhz"])
        local_window_mhz = float(1.15 * max(np.max(np.abs(highlighted_branches_mhz)), 0.15))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        for i in range(tracked_mhz.shape[1]):
            ax1.plot(scan_detunings_mhz, tracked_mhz[:, i], linewidth=1.0, color="0.78", zorder=1)
        ax1.plot(scan_detunings_mhz, highlighted_branches_mhz[:, 0], linewidth=2.2, color="#1f77b4", label="Highlighted branch A", zorder=3)
        ax1.plot(scan_detunings_mhz, highlighted_branches_mhz[:, 1], linewidth=2.2, color="#ff7f0e", label="Highlighted branch B", zorder=3)
        ax1.set_ylabel("Quasienergy window (MHz)")
        ax1.set_ylim(-local_window_mhz, local_window_mhz)
        ax1.set_title("Floquet quasienergies near the resonant avoided crossing")
        ax1.legend(loc="lower left", fontsize=9)

        ax2.plot(scan_detunings_mhz, highlighted_gap_mhz, color="#d62728", linewidth=1.8)
        ax2.axvline(
            resonance_detuning_mhz,
            ls="--", color="gray", alpha=0.6, label="Min gap (resonance)",
        )
        ax2.set_ylabel("Highlighted-pair gap (MHz)")
        ax2.set_xlabel("Drive detuning (MHz)")
        ax2.set_title("Avoided-crossing gap for the resonant Floquet pair")
        ax2.legend()

        fig.tight_layout()
        fig.savefig(OUT_DIR / "floquet_quasienergy_scan.png", dpi=150)
        plt.close(fig)
        _write_validation_summary("floquet_quasienergy_scan.json", data)
        print("    ✓ floquet_quasienergy_scan.png")
    except Exception as exc:
        print(f"    ✗ floquet_quasienergy_scan skipped: {exc}")


# ---------------------------------------------------------------------------
# 7. Fock State Wigner Functions
# ---------------------------------------------------------------------------
def plot_fock_state_wigners():
    """4-panel Wigner function grid for Fock states |0⟩ – |3⟩."""
    _apply_style()
    print("[7/8] Fock state Wigner functions …")
    try:
        import qutip as qt

        N = 25
        xvec = np.linspace(-4, 4, 120)
        labels = [r"$|0\rangle$ (vacuum)", r"$|1\rangle$", r"$|2\rangle$", r"$|3\rangle$"]

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        for n, (ax, label) in enumerate(zip(axes, labels)):
            state = qt.fock_dm(N, n)
            W = qt.wigner(state, xvec, xvec)
            vmax = max(abs(W.min()), abs(W.max()))
            pcm = ax.pcolormesh(
                xvec, xvec, W,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
            )
            plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlabel(r"Re($\alpha$)")
            ax.set_ylabel(r"Im($\alpha$)")
            ax.set_title(label)
            ax.set_aspect("equal")

        fig.suptitle("Fock State Wigner Functions", fontsize=14)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fock_state_wigners.png", dpi=150)
        plt.close(fig)
        print("    [ok] fock_state_wigners.png")
    except Exception as exc:
        print(f"    [skip] fock_state_wigners: {exc}")


# ---------------------------------------------------------------------------
# 8. DSD Fock State Preparation — Ideal Target Wigner Functions
# ---------------------------------------------------------------------------
def plot_dsd_fock_preparation():
    """Ideal Wigner functions for |1⟩, |2⟩, |3⟩ DSD preparation targets."""
    _apply_style()
    print("[8/8] DSD Fock preparation target Wigner functions …")
    try:
        import qutip as qt

        N = 25
        xvec = np.linspace(-4, 4, 120)
        targets = [1, 2, 3]
        labels = [r"Target $|1\rangle$", r"Target $|2\rangle$", r"Target $|3\rangle$"]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for n, (ax, label) in zip(targets, zip(axes, labels)):
            state = qt.fock_dm(N, n)
            W = qt.wigner(state, xvec, xvec)
            vmax = max(abs(W.min()), abs(W.max()))
            pcm = ax.pcolormesh(
                xvec, xvec, W,
                cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
            )
            plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
            # Annotate Wigner negativity at origin
            w_origin = qt.wigner(state, np.array([0.0]), np.array([0.0]))
            ax.plot(0, 0, "k+", ms=8, mew=2)
            ax.set_xlabel(r"Re($\alpha$)")
            ax.set_ylabel(r"Im($\alpha$)")
            ax.set_title(f"{label}\n$W(0)={float(w_origin[0,0]):.3f}$")
            ax.set_aspect("equal")

        fig.suptitle(
            "DSD Fock State Preparation — Ideal Target States",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / "dsd_fock_preparation.png", dpi=150)
        plt.close(fig)
        print("    [ok] dsd_fock_preparation.png")
    except Exception as exc:
        print(f"    [skip] dsd_fock_preparation: {exc}")


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
    plot_fock_state_wigners()
    plot_dsd_fock_preparation()
    print(f"\nAll plots saved to {OUT_DIR}")
