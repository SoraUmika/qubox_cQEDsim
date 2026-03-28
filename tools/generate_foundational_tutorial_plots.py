"""Generate real simulation plots for the foundational-curriculum tutorial pages.

Complements tools/generate_tutorial_plots.py which covers the workflow tutorials.
Run from the repository root:

    python tools/generate_foundational_tutorial_plots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = REPO / "documentations" / "assets" / "images" / "tutorials"
OUT.mkdir(parents=True, exist_ok=True)

STYLE = {
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
}
plt.rcParams.update(STYLE)


def _angular(f_hz: float) -> float:
    return 2.0 * np.pi * f_hz


def _save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [ok] {name}")


# ---------------------------------------------------------------------------
# 1. Minimal dispersive model — dressed energy levels + omega_ge(n)
# ---------------------------------------------------------------------------
def plot_minimal_model():
    print("[1] Minimal dispersive model ...")
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
    from cqed_sim.core import compute_energy_spectrum, manifold_transition_frequency

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5.15e9), omega_q=_angular(6.35e9),
        alpha=_angular(-220e6), chi=_angular(-2.4e6),
        kerr=_angular(-2e3), n_cav=8, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    spec = compute_energy_spectrum(model, frame=frame, levels=12)
    energies_mhz = np.array(spec.energies) / (2 * np.pi * 1e6)

    ns = list(range(6))
    wge = [manifold_transition_frequency(model, n=n, frame=frame) / (2 * np.pi * 1e6) for n in ns]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(range(len(energies_mhz)), energies_mhz, color="steelblue", width=0.6)
    axes[0].set_xlabel("Eigenstate index")
    axes[0].set_ylabel("Energy (MHz, rotating frame)")
    axes[0].set_title("Dressed Energy Levels")
    axes[1].plot(ns, wge, "o-", color="crimson", markersize=7, linewidth=2)
    axes[1].set_xlabel("Cavity photon number $n$")
    axes[1].set_ylabel(r"$\omega_{ge}(n)$ (MHz, rotating frame)")
    axes[1].set_title(r"Qubit Transition vs Photon Number ($\chi < 0$)")
    fig.tight_layout()
    _save(fig, "minimal_dispersive_model.png")


# ---------------------------------------------------------------------------
# 2. Units, frames & conventions — chi sign comparison
# ---------------------------------------------------------------------------
def plot_units_frames():
    print("[2] Units / frames / conventions ...")
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec, manifold_transition_frequency

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for chi_mhz, label, color in [(-2.5, r"$\chi/2\pi = -2.5$ MHz", "steelblue"),
                                    (2.5, r"$\chi/2\pi = +2.5$ MHz", "darkorange")]:
        model = DispersiveTransmonCavityModel(
            omega_c=_angular(5e9), omega_q=_angular(6e9),
            alpha=_angular(-220e6), chi=_angular(chi_mhz * 1e6),
            kerr=_angular(-2e3), n_cav=8, n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        ns = list(range(6))
        wge = [manifold_transition_frequency(model, n=n, frame=frame) / (2 * np.pi * 1e6) for n in ns]
        ax.plot(ns, wge, "o-", label=label, color=color, markersize=7, linewidth=2)

    ax.set_xlabel("Cavity photon number $n$")
    ax.set_ylabel(r"$\omega_{ge}(n)$ (MHz, rotating frame)")
    ax.set_title(r"Effect of $\chi$ Sign on Qubit Transition")
    ax.legend()
    fig.tight_layout()
    _save(fig, "units_frames_chi_sign.png")


# ---------------------------------------------------------------------------
# 3. Qubit drive — Rabi oscillation
# ---------------------------------------------------------------------------
def plot_qubit_drive():
    print("[3] Qubit drive / Rabi oscillation ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, fock_state, prepare_state,
    )
    from cqed_sim.pulses import Pulse, square_envelope
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=4, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    psi0 = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(0)))

    omega_rabi = _angular(12e6)
    t_end = 120e-9
    pulse = Pulse("qubit", 0.0, t_end, square_envelope, carrier=0.0, amp=omega_rabi)
    compiled = SequenceCompiler(dt=0.5e-9).compile([pulse], t_end=t_end)
    result = simulate_sequence(
        model, compiled, psi0, {"qubit": "qubit"},
        config=SimulationConfig(frame=frame, store_states=True),
    )

    times_s = np.array(result.solver_result.times) if hasattr(result.solver_result, 'times') else np.linspace(0, t_end, len(result.expectations["P_e"]))
    times_ns = times_s * 1e9
    pe = np.array(result.expectations["P_e"])
    pg = np.array(result.expectations["P_g"])
    theory = np.sin(omega_rabi * times_s) ** 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(times_ns, pg, "b-", linewidth=1.5, label=r"$P_g$ (simulated)")
    ax.plot(times_ns, pe, "r-", linewidth=1.5, label=r"$P_e$ (simulated)")
    ax.plot(times_ns, theory, "k--", linewidth=1, alpha=0.6, label=r"$\sin^2(\Omega t)$ theory")
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Population")
    ax.set_title(r"Resonant Rabi Oscillation ($\Omega/2\pi = 12$ MHz)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, "qubit_drive_rabi.png")


# ---------------------------------------------------------------------------
# 4. Power Rabi
# ---------------------------------------------------------------------------
def plot_power_rabi():
    print("[4] Power Rabi ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, fock_state, prepare_state,
    )
    from cqed_sim.pulses import Pulse, square_envelope
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence, reduced_qubit_state

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=4, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    psi0 = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(0)))

    duration = 40e-9
    amps_mhz = np.linspace(0.5, 25, 31)
    pe_sim = []
    for amp_mhz in amps_mhz:
        omega = _angular(amp_mhz * 1e6)
        pulse = Pulse("qubit", 0.0, duration, square_envelope, carrier=0.0, amp=omega)
        compiled = SequenceCompiler(dt=0.5e-9).compile([pulse], t_end=duration)
        result = simulate_sequence(
            model, compiled, psi0, {"qubit": "qubit"},
            config=SimulationConfig(frame=frame),
        )
        rho_q = reduced_qubit_state(result.final_state)
        pe_sim.append(float(np.real(rho_q[1, 1])))

    theory = np.sin(_angular(amps_mhz * 1e6) * duration) ** 2

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(amps_mhz, pe_sim, "ro", markersize=4, label="Simulated")
    ax.plot(amps_mhz, theory, "k-", linewidth=1, alpha=0.6, label=r"$\sin^2(\Omega T)$")
    ax.set_xlabel(r"Drive amplitude $\Omega/2\pi$ (MHz)")
    ax.set_ylabel(r"$P_e$")
    ax.set_title(r"Power Rabi ($T = 40$ ns)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "power_rabi.png")


# ---------------------------------------------------------------------------
# 5. Observables — Bloch + Wigner
# ---------------------------------------------------------------------------
def plot_observables():
    print("[5] Observables / Bloch + Wigner ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, coherent_state, prepare_state,
    )
    from cqed_sim.io import RotationGate
    from cqed_sim.pulses import build_rotation_pulse
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import (
        SimulationConfig, simulate_sequence,
        reduced_qubit_state, reduced_cavity_state, cavity_wigner,
    )
    import qutip as qt

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=10, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    psi0 = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=coherent_state(1.5)))

    gate = RotationGate(index=0, name="x90", theta=np.pi / 2, phi=0.0)
    pulses, drive_ops, _ = build_rotation_pulse(gate, {"duration_rotation_s": 64e-9, "rotation_sigma_fraction": 0.18})
    compiled = SequenceCompiler(dt=1e-9).compile(pulses, t_end=70e-9)
    result = simulate_sequence(model, compiled, psi0, drive_ops, config=SimulationConfig(frame=frame))

    rho_q = reduced_qubit_state(result.final_state)
    rho_q_qt = qt.Qobj(rho_q, dims=[[2], [2]])
    sx = float(np.real(qt.expect(qt.sigmax(), rho_q_qt)))
    sy = float(np.real(qt.expect(qt.sigmay(), rho_q_qt)))
    sz = float(np.real(qt.expect(qt.sigmaz(), rho_q_qt)))

    rho_c = reduced_cavity_state(result.final_state)
    xvec, yvec, W = cavity_wigner(rho_c, coordinate="alpha")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [r"$\langle\sigma_x\rangle$", r"$\langle\sigma_y\rangle$", r"$\langle\sigma_z\rangle$"]
    vals = [sx, sy, sz]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    axes[0].bar(labels, vals, color=colors, width=0.5)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].set_ylabel("Expectation value")
    axes[0].set_title("Qubit Bloch Vector after X90")
    axes[0].axhline(0, color="gray", linewidth=0.5)

    cax = axes[1].pcolormesh(xvec, yvec, W.T, cmap="RdBu_r", shading="auto")
    axes[1].set_xlabel(r"Re($\alpha$)")
    axes[1].set_ylabel(r"Im($\alpha$)")
    axes[1].set_title(r"Cavity Wigner ($\alpha = 1.5$)")
    axes[1].set_aspect("equal")
    fig.colorbar(cax, ax=axes[1], label=r"$W(\alpha)$")
    fig.tight_layout()
    _save(fig, "observables_bloch_wigner.png")


# ---------------------------------------------------------------------------
# 6. Dispersive shift and dressed frequencies
# ---------------------------------------------------------------------------
def plot_dispersive():
    print("[6] Dispersive shift / dressed frequencies ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        compute_energy_spectrum, manifold_transition_frequency,
    )

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5.15e9), omega_q=_angular(6.35e9),
        alpha=_angular(-220e6), chi=_angular(-2.4e6),
        kerr=_angular(-2e3), n_cav=10, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

    ns = list(range(8))
    wge = [manifold_transition_frequency(model, n=n, frame=frame) / (2 * np.pi * 1e6) for n in ns]
    linear = [wge[0] + model.chi / (2 * np.pi * 1e6) * n for n in ns]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, wge, "o-", color="crimson", markersize=7, linewidth=2, label=r"Exact $\omega_{ge}(n)$")
    ax.plot(ns, linear, "--", color="gray", linewidth=1, label=r"Linear $\omega_{ge}(0) + n\chi$")
    ax.set_xlabel("Cavity photon number $n$")
    ax.set_ylabel(r"$\omega_{ge}(n)$ (MHz, rotating frame)")
    ax.set_title("Dispersive Shift: Exact vs Linear Approximation")
    ax.legend()
    fig.tight_layout()
    _save(fig, "dispersive_dressed.png")


# ---------------------------------------------------------------------------
# 7. Open-system dynamics: T1, Ramsey, Echo
# ---------------------------------------------------------------------------
def plot_open_system():
    print("[7] Open-system: T1, Ramsey, Echo ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, fock_state, prepare_state,
    )
    from cqed_sim.sim import NoiseSpec
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence, reduced_qubit_state
    from tutorials.tutorial_support import (
        gaussian_quasistatic_ramsey_excited_population,
        gaussian_quasistatic_echo_excited_population,
    )

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=4, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

    # T1
    t1_val = 18e-6
    noise_t1 = NoiseSpec(t1=t1_val)
    psi_e = prepare_state(model, StatePreparationSpec(qubit=qubit_state("e"), storage=fock_state(0)))
    delays_t1 = np.linspace(0, 40e-6, 17)
    pe_t1 = []
    for d in delays_t1:
        compiled = SequenceCompiler(dt=2e-9).compile([], t_end=max(d, 4e-9))
        result = simulate_sequence(model, compiled, psi_e, {},
                                    config=SimulationConfig(frame=frame),
                                    noise=noise_t1)
        rho_q = reduced_qubit_state(result.final_state)
        pe_t1.append(float(np.real(rho_q[1, 1])))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(delays_t1 * 1e6, pe_t1, "ro", markersize=5, label="Simulated")
    axes[0].plot(delays_t1 * 1e6, np.exp(-delays_t1 / t1_val), "k-", linewidth=1.5, alpha=0.6, label=r"$e^{-t/T_1}$")
    axes[0].set_xlabel(r"Delay ($\mu$s)")
    axes[0].set_ylabel(r"$P_e$")
    axes[0].set_title(r"$T_1$ Relaxation (18 $\mu$s)")
    axes[0].legend(fontsize=9)

    # Ramsey + Echo (theory comparison for visual clarity)
    sigma_det = _angular(0.3e6)
    echo_delays = np.linspace(0, 8e-6, 80)
    axes[1].plot(echo_delays * 1e6,
                  gaussian_quasistatic_ramsey_excited_population(echo_delays, sigma_det),
                  "b-", linewidth=2, label="Ramsey")
    axes[1].fill_between(echo_delays * 1e6, 0.5,
                          gaussian_quasistatic_ramsey_excited_population(echo_delays, sigma_det),
                          alpha=0.15, color="blue")
    axes[1].set_xlabel(r"Delay ($\mu$s)")
    axes[1].set_ylabel(r"$\langle P_e \rangle$")
    axes[1].set_title(r"Ramsey: Gaussian Collapse ($\sigma_\Delta/2\pi = 0.3$ MHz)")
    axes[1].legend(fontsize=9)

    axes[2].plot(echo_delays * 1e6,
                  gaussian_quasistatic_ramsey_excited_population(echo_delays, sigma_det),
                  "b--", linewidth=1.5, alpha=0.5, label="Ramsey")
    axes[2].plot(echo_delays * 1e6,
                  gaussian_quasistatic_echo_excited_population(echo_delays),
                  "r-", linewidth=2, label="Hahn Echo")
    axes[2].set_xlabel(r"Total delay ($\mu$s)")
    axes[2].set_ylabel(r"$\langle P_e \rangle$")
    axes[2].set_title("Echo Refocuses Static Dephasing")
    axes[2].legend(fontsize=9)

    fig.tight_layout()
    _save(fig, "open_system_t1_ramsey_echo.png")


# ---------------------------------------------------------------------------
# 8. Storage cavity coherent-state decay
# ---------------------------------------------------------------------------
def plot_storage_decay():
    print("[8] Storage cavity decay ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, coherent_state, prepare_state,
    )
    from cqed_sim.sim import NoiseSpec
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence, storage_photon_number

    kappa = 1.0 / 50e-6
    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=12, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    noise = NoiseSpec(kappa=kappa)
    psi0 = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=coherent_state(2.0)))

    times_us = np.linspace(0, 100, 21)
    n_vals = []
    for t_us in times_us:
        t_s = t_us * 1e-6
        compiled = SequenceCompiler(dt=2e-9).compile([], t_end=max(t_s, 4e-9))
        result = simulate_sequence(model, compiled, psi0, {},
                                    config=SimulationConfig(frame=frame),
                                    noise=noise)
        n_vals.append(float(storage_photon_number(result.final_state)))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(times_us, n_vals, "ro", markersize=5, label="Simulated")
    ax.plot(times_us, 4.0 * np.exp(-kappa * times_us * 1e-6), "k-", linewidth=1.5, alpha=0.6,
            label=r"$|\alpha|^2 e^{-\kappa t}$")
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"$\langle n \rangle$")
    ax.set_title(r"Storage Cavity Decay ($\kappa^{-1}=50\,\mu$s, $|\alpha|^2=4$)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "storage_cavity_decay.png")


# ---------------------------------------------------------------------------
# 9. Multilevel transmon spectrum
# ---------------------------------------------------------------------------
def plot_multilevel():
    print("[9] Multilevel transmon ...")
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec, compute_energy_spectrum

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    for idx, (ntr, color, title) in enumerate([
        (2, "steelblue", "Two-Level ($n_{tr}=2$)"),
        (3, "darkorange", "Three-Level ($n_{tr}=3$)"),
    ]):
        model = DispersiveTransmonCavityModel(
            omega_c=_angular(5e9), omega_q=_angular(6e9),
            alpha=_angular(-220e6), chi=_angular(-2.5e6),
            kerr=_angular(-2e3), n_cav=6, n_tr=ntr,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        spec = compute_energy_spectrum(model, frame=frame, levels=min(12 + 4 * (ntr - 2), 18))
        e_mhz = np.array(spec.energies) / (2 * np.pi * 1e6)
        axes[idx].bar(range(len(e_mhz)), e_mhz, color=color, width=0.6)
        axes[idx].set_xlabel("Eigenstate index")
        axes[idx].set_ylabel("Energy (MHz, rotating frame)")
        axes[idx].set_title(title)
    fig.suptitle("Multilevel Transmon Effects", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "multilevel_transmon_spectrum.png")


# ---------------------------------------------------------------------------
# 10. Truncation convergence
# ---------------------------------------------------------------------------
def plot_truncation():
    print("[10] Truncation convergence ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, coherent_state, prepare_state,
    )
    from cqed_sim.sim import storage_photon_number

    alpha = 2.0
    expected_n = abs(alpha) ** 2
    n_cavs = list(range(4, 26))
    n_vals = []
    for nc in n_cavs:
        model = DispersiveTransmonCavityModel(
            omega_c=_angular(5e9), omega_q=_angular(6e9),
            alpha=_angular(-220e6), chi=_angular(-2.5e6),
            kerr=_angular(-2e3), n_cav=nc, n_tr=2,
        )
        psi0 = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=coherent_state(alpha)))
        n_vals.append(float(storage_photon_number(psi0)))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(n_cavs, n_vals, "o-", color="steelblue", markersize=5, linewidth=1.5)
    ax.axhline(expected_n, color="crimson", linestyle="--", linewidth=1.5, label=rf"$|\alpha|^2 = {expected_n:.1f}$")
    ax.set_xlabel("Cavity truncation $n_{cav}$")
    ax.set_ylabel(r"$\langle n \rangle$")
    ax.set_title(r"Truncation Convergence ($\alpha = 2.0$)")
    ax.legend()
    fig.tight_layout()
    _save(fig, "truncation_convergence.png")


# ---------------------------------------------------------------------------
# 11. Sequence building — compiled waveforms
# ---------------------------------------------------------------------------
def plot_sequences():
    print("[11] Sequence building ...")
    from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
    from cqed_sim.io import DisplacementGate, RotationGate
    from cqed_sim.pulses import Pulse, build_displacement_pulse, build_rotation_pulse
    from cqed_sim.sequence import SequenceCompiler

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=6, n_tr=2,
    )

    disp_gate = DisplacementGate(index=0, name="displace", re=1.5, im=0.0)
    disp_pulses, disp_ops, _ = build_displacement_pulse(disp_gate, {"duration_displacement_s": 100e-9})

    rot_gate = RotationGate(index=0, name="x90", theta=np.pi / 2, phi=0.0)
    rot_pulses, rot_ops, _ = build_rotation_pulse(rot_gate, {"duration_rotation_s": 64e-9, "rotation_sigma_fraction": 0.18})
    # Shift rotation pulses to start after displacement (frozen dataclass, must recreate)
    shifted_rot_pulses = [
        Pulse(
            channel=p.channel, t0=120e-9, duration=p.duration,
            envelope=p.envelope, carrier=p.carrier, amp=p.amp,
            phase=p.phase if hasattr(p, 'phase') else 0.0,
        )
        for p in rot_pulses
    ]

    all_pulses = list(disp_pulses) + shifted_rot_pulses
    compiled = SequenceCompiler(dt=1e-9).compile(all_pulses, t_end=200e-9)

    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for ch_name, ax, color in [("storage", axes[0], "steelblue"), ("qubit", axes[1], "crimson")]:
        if ch_name in compiled.channels:
            ch = compiled.channels[ch_name]
            wf = ch.baseband
            ts = np.arange(len(wf)) * compiled.dt * 1e9
            ax.plot(ts, np.real(wf) / (2 * np.pi * 1e6), color=color, linewidth=1.2)
        ax.set_ylabel(f"{ch_name.capitalize()}\n(MHz)")
    axes[-1].set_xlabel("Time (ns)")
    axes[0].set_title("Compiled Multi-Channel Waveforms: Displacement + X90")
    fig.tight_layout()
    _save(fig, "sequence_waveforms.png")


# ---------------------------------------------------------------------------
# 12. Calibration summary (2x2)
# ---------------------------------------------------------------------------
def plot_calibration_summary():
    print("[12] Calibration summary ...")
    from cqed_sim.core import (
        DispersiveTransmonCavityModel, FrameSpec,
        StatePreparationSpec, qubit_state, fock_state, prepare_state,
        carrier_for_transition_frequency,
    )
    from cqed_sim.sim import NoiseSpec
    from cqed_sim.pulses import Pulse, square_envelope
    from cqed_sim.pulses.envelopes import gaussian_envelope
    from cqed_sim.sequence import SequenceCompiler
    from cqed_sim.sim import SimulationConfig, simulate_sequence, reduced_qubit_state
    from tutorials.tutorial_support import (
        gaussian_quasistatic_ramsey_excited_population,
        gaussian_quasistatic_echo_excited_population,
    )
    from functools import partial

    model = DispersiveTransmonCavityModel(
        omega_c=_angular(5e9), omega_q=_angular(6e9),
        alpha=_angular(-220e6), chi=_angular(-2.5e6),
        kerr=_angular(-2e3), n_cav=4, n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    psi_g = prepare_state(model, StatePreparationSpec(qubit=qubit_state("g"), storage=fock_state(0)))
    psi_e = prepare_state(model, StatePreparationSpec(qubit=qubit_state("e"), storage=fock_state(0)))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Spectroscopy
    dets_mhz = np.linspace(-3, 3, 25)
    pe_spec = []
    for d in dets_mhz:
        carrier = carrier_for_transition_frequency(_angular(d * 1e6))
        probe = Pulse("qubit", 0.0, 1e-6, partial(gaussian_envelope, sigma=0.18),
                       carrier=carrier, amp=_angular(0.08e6))
        compiled = SequenceCompiler(dt=2e-9).compile([probe], t_end=1.05e-6)
        result = simulate_sequence(model, compiled, psi_g, {"qubit": "qubit"},
                                    config=SimulationConfig(frame=frame))
        rho_q = reduced_qubit_state(result.final_state)
        pe_spec.append(float(np.real(rho_q[1, 1])))

    axes[0, 0].plot(dets_mhz, pe_spec, "b-", linewidth=1.5)
    axes[0, 0].set_xlabel("Detuning (MHz)")
    axes[0, 0].set_ylabel(r"$P_e$")
    axes[0, 0].set_title("Qubit Spectroscopy")

    # Power Rabi
    amps = np.linspace(0.5, 20, 21)
    pe_rabi = []
    for amp in amps:
        pulse = Pulse("qubit", 0.0, 40e-9, square_envelope, carrier=0.0, amp=_angular(amp * 1e6))
        compiled = SequenceCompiler(dt=0.5e-9).compile([pulse], t_end=40e-9)
        result = simulate_sequence(model, compiled, psi_g, {"qubit": "qubit"},
                                    config=SimulationConfig(frame=frame))
        rho_q = reduced_qubit_state(result.final_state)
        pe_rabi.append(float(np.real(rho_q[1, 1])))

    axes[0, 1].plot(amps, pe_rabi, "r-", linewidth=1.5)
    axes[0, 1].set_xlabel(r"$\Omega/2\pi$ (MHz)")
    axes[0, 1].set_ylabel(r"$P_e$")
    axes[0, 1].set_title("Power Rabi")

    # T1
    t1_val = 18e-6
    delays_t1 = np.linspace(0, 40e-6, 15)
    pe_decay = []
    for d in delays_t1:
        compiled = SequenceCompiler(dt=2e-9).compile([], t_end=max(d, 4e-9))
        result = simulate_sequence(model, compiled, psi_e, {},
                                    config=SimulationConfig(frame=frame),
                                    noise=NoiseSpec(t1=t1_val))
        rho_q = reduced_qubit_state(result.final_state)
        pe_decay.append(float(np.real(rho_q[1, 1])))

    axes[1, 0].plot(delays_t1 * 1e6, pe_decay, "go", markersize=5, label="Simulated")
    axes[1, 0].plot(delays_t1 * 1e6, np.exp(-delays_t1 / t1_val), "k--", linewidth=1, alpha=0.6, label=r"$e^{-t/T_1}$")
    axes[1, 0].set_xlabel(r"Delay ($\mu$s)")
    axes[1, 0].set_ylabel(r"$P_e$")
    axes[1, 0].set_title(r"$T_1$ Relaxation")
    axes[1, 0].legend(fontsize=9)

    # Ramsey vs Echo
    sigma_det = _angular(0.3e6)
    echo_delays = np.linspace(0, 8e-6, 60)
    axes[1, 1].plot(echo_delays * 1e6,
                     gaussian_quasistatic_ramsey_excited_population(echo_delays, sigma_det),
                     "b-", linewidth=2, label="Ramsey")
    axes[1, 1].plot(echo_delays * 1e6,
                     gaussian_quasistatic_echo_excited_population(echo_delays),
                     "r-", linewidth=2, label="Echo")
    axes[1, 1].set_xlabel(r"Total delay ($\mu$s)")
    axes[1, 1].set_ylabel(r"$\langle P_e \rangle$")
    axes[1, 1].set_title("Ramsey vs Echo")
    axes[1, 1].legend(fontsize=9)

    fig.suptitle("End-to-End Calibration Workflow", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    _save(fig, "calibration_summary.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Output: {OUT}\n")
    for gen in [
        plot_minimal_model,
        plot_units_frames,
        plot_qubit_drive,
        plot_power_rabi,
        plot_observables,
        plot_dispersive,
        plot_open_system,
        plot_storage_decay,
        plot_multilevel,
        plot_truncation,
        plot_sequences,
        plot_calibration_summary,
    ]:
        try:
            gen()
        except Exception as exc:
            print(f"  [FAIL] {gen.__name__}: {exc}")
    print("\nDone.")
