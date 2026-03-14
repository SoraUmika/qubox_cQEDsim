from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

COMMON_IMPORTS = """
from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

REPO_ROOT = next(
    (
        candidate
        for candidate in (Path.cwd(), *Path.cwd().parents)
        if (candidate / "pyproject.toml").exists() and (candidate / "cqed_sim").is_dir()
    ),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("Could not resolve the repository root from the notebook working directory.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from cqed_sim import (
    AmplifierChain,
    BosonicModeSpec,
    DispersiveCouplingSpec,
    DispersiveReadoutTransmonStorageModel,
    DispersiveTransmonCavityModel,
    DisplacementGate,
    FrameSpec,
    NoiseSpec,
    Pulse,
    PurcellFilter,
    QubitMeasurementSpec,
    ReadoutChain,
    ReadoutResonator,
    RotationGate,
    SidebandDriveSpec,
    SequenceCompiler,
    SimulationConfig,
    StatePreparationSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
    build_displacement_pulse,
    build_rotation_pulse,
    build_sideband_pulse,
    carrier_for_transition_frequency,
    coherent_state,
    compute_energy_spectrum,
    fock_state,
    manifold_transition_frequency,
    measure_qubit,
    prepare_simulation,
    prepare_state,
    pure_dephasing_time_from_t1_t2,
    qubit_state,
    run_rabi,
    run_ramsey,
    run_spectroscopy,
    run_t1,
    run_t2_echo,
    sideband_transition_frequency,
    simulate_batch,
    simulate_sequence,
)
from cqed_sim.plotting import plot_energy_levels
from cqed_sim.pulses import gaussian_envelope, square_envelope
from cqed_sim.sim import (
    cavity_wigner,
    conditioned_bloch_xyz,
    mode_moments,
    qubit_conditioned_mode_moments,
    readout_response_by_qubit_state,
    reduced_cavity_state,
    reduced_qubit_state,
    reduced_storage_state,
    storage_photon_number,
    subsystem_level_population,
    transmon_level_populations,
)
from tutorials.tutorial_support import (
    GHz,
    MHz,
    angular_to_ghz,
    angular_to_hz,
    angular_to_mhz,
    final_expectation,
    fit_echo_signal,
    fit_exponential_decay,
    fit_lorentzian_peak,
    fit_rabi_vs_amplitude,
    fit_rabi_vs_duration,
    fit_ramsey_signal,
    ns,
    us,
)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (7.0, 4.2)
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
"""


def normalize(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": normalize(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalize(text),
    }


def title_cell(
    number: int,
    title: str,
    summary: str,
    prereqs: str | None = None,
    status: str | None = None,
) -> dict:
    parts = [f"# Tutorial {number:02d} -- {title}", "", summary]
    if prereqs:
        parts.extend(["", f"**Prerequisites.** {prereqs}"])
    if status:
        parts.extend(["", f"**Scope note.** {status}"])
    return md("\n".join(parts))


def write_notebook(path: Path, cells: list[dict]) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def generate_batch_1() -> None:
    write_notebook(
        TUTORIALS / "00_tutorial_index.ipynb",
        [
            md(
                """
        # Tutorial 00 -- cqed_sim Tutorial Index

        This notebook is the landing page for the structured `cqed_sim` curriculum under the top-level `tutorials/` folder.

        `cqed_sim` is a pulse-level circuit-QED simulator for dispersive transmon-storage and transmon-storage-readout systems. The tutorials below are written for users who want to learn the actual current public API, understand the repository's physics conventions, and build realistic notebook-sized workflows without relying on private helper code.
        """
            ),
            md(
                """
        ## 1. Who These Tutorials Are For

        - New users who want a clean first path through the `cqed_sim` API.
        - Experimentalists who want notebook-scale calibrations, spectroscopy, and readout examples.
        - Theory and simulation users who want explicit control over frames, Hamiltonians, truncation, and observables.
        - Developers who need examples of the intended model -> frame -> pulse -> compile -> simulate -> analyze workflow.
        """
            ),
            md(
                """
        ## 2. How To Read The Curriculum

        The tutorials are numbered in the recommended reading order:

        - Tier 0 starts with orientation, units, and conventions.
        - Tier 1 focuses on basic operations and observables.
        - Tier 2 builds canonical cQED signatures such as spectroscopy and number splitting.
        - Tier 3 turns those ingredients into calibration-style workflows.
        - Higher tiers cover cavity physics, realism, workflow composition, and advanced interaction models.

        If you are brand new, start with Tutorials 01, 02, 03, and 06 before moving into the calibration notebooks.
        """
            ),
            md(
                """
        ## 3. Suggested Learning Paths

        ### Beginner / first-time user

        1. Tutorial 01 -- minimal dispersive model
        2. Tutorial 02 -- units, frames, and conventions
        3. Tutorial 03 -- cavity displacement basics
        4. Tutorial 04 -- qubit drive basics
        5. Tutorial 06 -- qubit spectroscopy

        ### Experimentalist workflow path

        1. Tutorials 06, 07, and 08 for spectroscopy and dispersive interpretation
        2. Tutorials 09, 10, 11, 12, and 13 for Rabi, T1, Ramsey, and echo
        3. Tutorial 17 for readout resonator response
        4. Tutorial 25 for a compact end-to-end calibration notebook

        ### Simulation / theory path

        1. Tutorials 01, 02, and 08 for Hamiltonians and dressed frequencies
        2. Tutorials 14, 15, and 16 for Kerr and bosonic dynamics
        3. Tutorials 18, 19, and 20 for multilevel effects and truncation care
        4. Tutorial 24 for effective sideband-style interactions

        ### Developer / API path

        1. Tutorials 01, 03, 04, and 21 for core object construction
        2. Tutorial 22 for prepared sessions and batch execution
        3. Tutorial 23 for result objects, fits, and convenience calibration targets
        4. Tutorial 26 for common failure modes and frame sanity checks
        """
            ),
            md(
                """
        ## 4. Conventions Warning

        Before trusting any spectroscopy axis, frame-dependent transition frequency, or Kerr interpretation, read the conventions notes:

        - `tutorials/conventions_quick_reference.md`
        - `physics_and_conventions/physics_conventions_report.tex`

        Important reminders used throughout the curriculum:

        - internal frequencies are in `rad/s`
        - time is in `s`
        - tensor ordering is qubit first, then bosonic modes
        - `Pulse.carrier` is the negative of the rotating-frame transition frequency it addresses
        - negative runtime `chi` lowers the qubit transition frequency with photon number
        """
            ),
            md(
                """
        ## 5. Common Prerequisites

        Most notebooks expect:

        - NumPy
        - Matplotlib
        - QuTiP
        - the repository root on the Python path, or an editable install of `cqed_sim`

        Each notebook includes a small bootstrap cell so it can usually be run directly from the `tutorials/` folder without extra path surgery.
        """
            ),
            md(
                """
        ## 6. Tutorial Categories At A Glance

        - Orientation: Tutorials 00--02
        - Basic operations and observables: Tutorials 03--05
        - Canonical cQED signatures: Tutorials 06--08
        - Calibration workflows: Tutorials 09--13 and 25
        - Bosonic / cavity physics: Tutorials 14--17
        - Realism and numerical care: Tutorials 18--20 and 26
        - Workflow composition and advanced interactions: Tutorials 21--24
        """
            ),
            md(
                """
        ## 7. `tutorials/` Versus `tests/`

        - `tutorials/` is for guided learning material, narrative walkthroughs, and professionally labeled notebook examples.
        - `tests/` is for automated correctness checks, regression coverage, and validation.

        A tutorial may mention a numerical expectation or a fit, but if the repository needs a lasting correctness guarantee, that belongs in `tests/` instead.
        """
            ),
        ],
    )

    write_notebook(
        TUTORIALS / "01_getting_started_minimal_dispersive_model.ipynb",
        [
            title_cell(
                1,
                "Getting Started with a Minimal Dispersive Model",
                "Build the smallest useful transmon-storage model, inspect its dressed energies, and verify how the qubit transition changes with cavity photon number in the rotating frame.",
                "Comfort with Python and the idea of a dispersive qubit-cavity Hamiltonian is enough.",
            ),
            md(
                """
        ## 1. Goal

        We will construct a minimal two-mode dispersive model, define a rotating frame, inspect the first few dressed energy levels, and compare the `|g,n> <-> |e,n>` transition for `n = 0` and `n = 3`.
        """
            ),
            md(
                """
        ## 2. Physical Background

        In the dispersive regime the qubit and cavity are not exchanging excitations resonantly, but the qubit transition still depends on cavity occupancy. In `cqed_sim` the runtime two-mode Hamiltonian includes a conditional term proportional to `chi * n_c * n_q`, so negative `chi` shifts the qubit transition downward with photon number.
        """
            ),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        omega_c_hz = 5.15e9
        omega_q_hz = 6.35e9
        alpha_hz = -220.0e6
        chi_hz = -2.4e6
        kerr_hz = -3.0e3
        n_cav = 8
        n_tr = 2
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(omega_c_hz / 1.0e9),
            omega_q=GHz(omega_q_hz / 1.0e9),
            alpha=MHz(alpha_hz / 1.0e6),
            chi=MHz(chi_hz / 1.0e6),
            kerr=MHz(kerr_hz / 1.0e6),
            n_cav=n_cav,
            n_tr=n_tr,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

        transition_n0 = manifold_transition_frequency(model, 0, frame=frame)
        transition_n3 = manifold_transition_frequency(model, 3, frame=frame)
        lab_spectrum = compute_energy_spectrum(model, frame=FrameSpec(), levels=10)

        print("Subsystem dimensions:", model.subsystem_dims)
        print("Qubit-first basis state |e,2> dims:", model.basis_state(1, 2).dims)
        print(f"omega_ge(n=0) / 2pi = {angular_to_mhz(transition_n0):+.3f} MHz in the chosen frame")
        print(f"omega_ge(n=3) / 2pi = {angular_to_mhz(transition_n3):+.3f} MHz in the chosen frame")
        print(f"difference / 2pi = {angular_to_mhz(transition_n3 - transition_n0):+.3f} MHz")
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        idle_duration = 80.0 * ns
        compiled_idle = SequenceCompiler(dt=2.0 * ns).compile([], t_end=idle_duration)
        initial_state = (model.basis_state(0, 0) + model.basis_state(1, 0)).unit()
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        matched_frame_result = simulate_sequence(
            model,
            compiled_idle,
            initial_state,
            {},
            config=SimulationConfig(frame=frame, store_states=True),
        )
        lab_frame_result = simulate_sequence(
            model,
            compiled_idle,
            initial_state,
            {},
            config=SimulationConfig(frame=FrameSpec(), store_states=True),
        )

        matched_pe = final_expectation(matched_frame_result, "P_e")
        lab_pe = final_expectation(lab_frame_result, "P_e")
        print(f"Final P_e in the matched rotating frame: {matched_pe:.3f}")
        print(f"Final P_e in the lab frame: {lab_pe:.3f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig = plot_energy_levels(
            lab_spectrum,
            max_levels=10,
            energy_scale=1.0 / (2.0 * np.pi * 1.0e6),
            energy_unit_label="MHz",
            title="Minimal dispersive model: low-lying dressed energies",
        )
        plt.show()

        n_values = np.arange(6)
        transition_mhz = [angular_to_mhz(manifold_transition_frequency(model, int(n), frame=frame)) for n in n_values]
        fig, ax = plt.subplots()
        ax.plot(n_values, transition_mhz, "o-")
        ax.set_xlabel("Cavity photon number n")
        ax.set_ylabel(r"$\\omega_{ge}(n) / 2\\pi$ in the rotating frame [MHz]")
        ax.set_title("Photon-number dependence of the qubit transition")
        plt.show()
        """
            ),
            md(
                """
        ## 9. Physical Interpretation

        The matched rotating frame removes the large bare cavity and qubit oscillations, so the remaining transition frequencies are the small residual detunings that matter for spectroscopy and pulse design. Because `chi` is negative here, the `n = 3` manifold sits below the `n = 0` manifold by roughly `3 * chi`.
        """
            ),
            md(
                """
        ## 10. Exercises / Next Steps

        - Change `chi_hz` from negative to positive and re-run the transition-versus-`n` plot.
        - Increase `n_tr` to `3` and compare the low-lying spectrum to see where transmon anharmonicity starts to matter.
        - Continue to Tutorial 02 for a focused walkthrough of units, frames, and carrier-sign conventions.
        """
            ),
        ],
    )

    write_notebook(
        TUTORIALS / "02_units_frames_and_conventions.ipynb",
        [
            title_cell(
                2,
                "Units, Frames, and Conventions",
                "Check the conventions that matter most for cQED users: SI-style internal units, rotating-frame interpretation, `Pulse.carrier` sign, and the meaning of the dispersive `chi` sign.",
                "Tutorial 01 is recommended first.",
            ),
            md(
                """
        ## 1. Goal

        This tutorial turns the most important convention questions into explicit notebook checks so you can verify them before running large parameter sweeps.
        """
            ),
            md(
                """
        ## 2. Physical Background

        Three convention choices matter constantly in `cqed_sim`:

        1. internal angular frequencies are in `rad/s`, not in `Hz`
        2. rotating-frame frequencies are specified through `FrameSpec`
        3. `Pulse.carrier` uses the waveform convention `exp(+i (omega t + phase))`, so the resonant carrier is the negative of the rotating-frame transition frequency

        A fourth convention is the repository-wide interpretation of runtime `chi`: negative `chi` lowers the qubit transition as bosonic occupation increases.
        """
            ),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        chi_values_hz = (-2.5e6, +2.5e6)
        n_values = np.arange(6)
        transition_detuning_demo_mhz = np.array([-3.0, -1.5, 0.0, 1.5, 3.0])
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        models = {
            f"chi = {chi_hz / 1.0e6:+.1f} MHz": DispersiveTransmonCavityModel(
                omega_c=GHz(5.0),
                omega_q=GHz(6.2),
                alpha=MHz(-220.0),
                chi=MHz(chi_hz / 1.0e6),
                kerr=0.0,
                n_cav=8,
                n_tr=2,
            )
            for chi_hz in chi_values_hz
        }
        frame = FrameSpec(omega_c_frame=GHz(5.0), omega_q_frame=GHz(6.2))

        carrier_demo = carrier_for_transition_frequency(MHz(transition_detuning_demo_mhz[1]))
        round_trip_demo = angular_to_mhz(-carrier_demo)
        print(f"Example transition detuning = {transition_detuning_demo_mhz[1]:+.3f} MHz")
        print(f"carrier_for_transition_frequency(...) returns {angular_to_mhz(carrier_demo):+.3f} MHz as a raw carrier")
        print(f"Negating that carrier returns the original transition detuning: {round_trip_demo:+.3f} MHz")
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        transition_curves = {
            label: [angular_to_mhz(manifold_transition_frequency(model, int(n), frame=frame)) for n in n_values]
            for label, model in models.items()
        }
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        reference_model = models["chi = -2.5 MHz"]
        lab_spectrum = compute_energy_spectrum(reference_model, frame=FrameSpec(), levels=8)
        rot_spectrum = compute_energy_spectrum(reference_model, frame=frame, levels=8)
        print("Vacuum-referenced lab-frame energies [MHz]:")
        print(np.round(lab_spectrum.energies[:6] / (2.0 * np.pi * 1.0e6), 4))
        print("Vacuum-referenced rotating-frame energies [MHz]:")
        print(np.round(rot_spectrum.energies[:6] / (2.0 * np.pi * 1.0e6), 4))
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        for label, values in transition_curves.items():
            ax.plot(n_values, values, "o-", label=label)
        ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.5)
        ax.set_xlabel("Bosonic occupation n")
        ax.set_ylabel(r"$\\omega_{ge}(n) / 2\\pi$ in the chosen rotating frame [MHz]")
        ax.set_title("The sign of runtime chi controls the slope of the qubit line")
        ax.legend()
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
        plot_energy_levels(lab_spectrum, max_levels=8, energy_scale=1.0 / (2.0 * np.pi * 1.0e6), energy_unit_label="MHz", title="Lab frame", ax=axes[0])
        plot_energy_levels(rot_spectrum, max_levels=8, energy_scale=1.0 / (2.0 * np.pi * 1.0e6), energy_unit_label="MHz", title="Matched rotating frame", ax=axes[1])
        plt.show()
        """
            ),
            md(
                """
        ## 9. Physical Interpretation

        A spectroscopy axis should be labeled by physical transition detuning, not by the raw carrier. The helper `carrier_for_transition_frequency(...)` is the safe way to map between those two languages. The frame comparison also shows why rotating-frame energies can cluster near zero even when the lab-frame spectrum still sits near several gigahertz.
        """
            ),
            md(
                """
        ## 10. Exercises / Next Steps

        - Change the rotating frame to be intentionally off-resonant and see how every dressed level shifts.
        - Open `tutorials/conventions_quick_reference.md` beside this notebook and confirm that each rule is reflected in the code above.
        - Continue to Tutorial 06 before doing any spectroscopy work.
        """
            ),
        ],
    )

    write_notebook(
        TUTORIALS / "03_cavity_displacement_basics.ipynb",
        [
            title_cell(
                3,
                "Cavity Displacement Basics",
                "Create coherent states with `build_displacement_pulse(...)`, verify the expected cavity amplitude, and visualize the final storage state in phase space.",
                "Tutorials 01 and 02 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will start from vacuum, apply calibrated displacement pulses, and compare the simulated cavity moments to the requested coherent-state amplitudes."),
            md("## 2. Physical Background\n\nA displacement pulse moves the cavity state in phase space. In the rotating frame, the ideal target is a coherent state `|alpha>`, whose mean field satisfies `<a> = alpha` and whose mean photon number is `|alpha|^2`."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        alpha_targets = [0.5 + 0.0j, 1.0 + 0.0j, 1.5 + 0.0j]
        displacement_duration = 120.0 * ns
        dt = 2.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.1),
            omega_q=GHz(6.2),
            alpha=MHz(-220.0),
            chi=0.0,
            kerr=0.0,
            n_cav=20,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        initial_state = model.basis_state(0, 0)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        displacement_jobs = []
        for alpha in alpha_targets:
            pulses, drive_ops, meta = build_displacement_pulse(
                DisplacementGate(index=0, name=f"D({alpha.real:+.2f}{alpha.imag:+.2f}i)", re=float(np.real(alpha)), im=float(np.imag(alpha))),
                {"duration_displacement_s": displacement_duration},
            )
            displacement_jobs.append((alpha, pulses, drive_ops, meta))
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        results = []
        for alpha, pulses, drive_ops, meta in displacement_jobs:
            compiled = SequenceCompiler(dt=dt).compile(pulses, t_end=displacement_duration + dt)
            result = simulate_sequence(
                model,
                compiled,
                initial_state,
                drive_ops,
                config=SimulationConfig(frame=frame),
            )
            moments = mode_moments(result.final_state, "storage")
            results.append(
                {
                    "target_alpha": alpha,
                    "simulated_alpha": moments["a"],
                    "simulated_n": moments["n"],
                    "result": result,
                }
            )

        for row in results:
            print(
                f"target={row['target_alpha']:+.2f}, simulated <a>={row['simulated_alpha']:+.3f}, <n>={row['simulated_n']:.3f}"
            )
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        axes[0].plot([np.real(row["target_alpha"]) for row in results], [np.real(row["simulated_alpha"]) for row in results], "o-")
        axes[0].plot([0.0, 1.6], [0.0, 1.6], "--", color="black", alpha=0.6)
        axes[0].set_xlabel("Target Re(alpha)")
        axes[0].set_ylabel("Simulated Re(<a>)")
        axes[0].set_title("Displacement calibration check")

        chosen_state = reduced_cavity_state(results[-1]["result"].final_state)
        xvec, yvec, w = cavity_wigner(chosen_state, n_points=81, extent=4.0)
        image = axes[1].imshow(w, origin="lower", extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], cmap="RdBu_r", aspect="equal")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("p")
        axes[1].set_title("Wigner function after the largest displacement")
        fig.colorbar(image, ax=axes[1], shrink=0.86)
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe near-identity line between target `alpha` and simulated `<a>` confirms that the public displacement builder is already calibrated in the rotating frame. The Wigner function stays Gaussian because the model here has negligible Kerr during the short pulse."),
            md("## 10. Exercises / Next Steps\n\n- Repeat the scan with a complex displacement such as `alpha = 1.0 + 0.8j`.\n- Add a small nonzero Kerr and make the pulse much longer to see where the ideal coherent-state picture starts to bend.\n- Continue to Tutorial 05 for more state and observable extraction tools."),
        ],
    )

    write_notebook(
        TUTORIALS / "04_qubit_drive_and_basic_population_dynamics.ipynb",
        [
            title_cell(
                4,
                "Qubit Drive and Basic Population Dynamics",
                "Drive a resonant qubit pulse in the matched rotating frame and inspect the time-domain populations of `|g>` and `|e>`.",
                "Tutorials 01 and 02 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will apply a resonant square pulse, store the trajectory, and verify that a calibrated pulse produces the expected population transfer."),
            md("## 2. Physical Background\n\nIn a matched rotating frame, a resonant qubit drive produces Rabi oscillations between `|g>` and `|e>`. For a two-level qubit with a square envelope, a pulse of duration `t_pi = pi / Omega` produces an ideal `pi` rotation."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        omega_rabi = 2.0 * np.pi * 8.0e6
        pulse_duration = np.pi / omega_rabi
        dt = pulse_duration / 200.0
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.1),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        initial_state = model.basis_state(0, 0)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        pulse = Pulse(
            "q",
            t0=0.0,
            duration=pulse_duration,
            envelope=square_envelope,
            amp=omega_rabi,
            carrier=0.0,
            label="resonant_pi",
        )
        compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=pulse_duration + dt)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result = simulate_sequence(
            model,
            compiled,
            initial_state,
            {"q": "qubit"},
            config=SimulationConfig(frame=frame, store_states=True, max_step=dt),
        )
        trajectory_t_ns = compiled.tlist * 1.0e9
        p_e = np.asarray(result.expectations["P_e"], dtype=float)
        p_g = np.asarray(result.expectations["P_g"], dtype=float)
        print(f"Final excited-state population: {p_e[-1]:.4f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(trajectory_t_ns, p_g, label=r"$P_g$")
        ax.plot(trajectory_t_ns, p_e, label=r"$P_e$")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Population")
        ax.set_title("Resonant qubit drive in the matched rotating frame")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nBecause the frame is matched and the drive is resonant, the dynamics are clean population exchange between `|g>` and `|e>`. Later tutorials reuse the same ingredients for power-Rabi and time-Rabi sweeps."),
            md("## 10. Exercises / Next Steps\n\n- Halve the pulse duration and confirm that the final state is close to an `x90` rotation instead of a `pi` rotation.\n- Add a small detuning through `carrier_for_transition_frequency(...)` and observe the loss of full inversion.\n- Continue to Tutorials 09 and 10 for calibration-style Rabi sweeps."),
        ],
    )

    write_notebook(
        TUTORIALS / "05_observables_states_and_visualization.ipynb",
        [
            title_cell(
                5,
                "Observables, States, and Visualization",
                "Extract reduced qubit and cavity states, compute conditioned observables, and visualize a displaced cavity state in phase space.",
                "Tutorials 03 and 04 are useful prerequisites.",
            ),
            md("## 1. Goal\n\nWe will prepare a simple joint state, extract subsystem objects from the simulation result, and visualize both qubit and cavity observables."),
            md("## 2. Physical Background\n\n`cqed_sim` separates trajectory generation from observable extraction. After a simulation you can reduce onto the qubit or storage subsystem, compute moments, and condition on a selected Fock level."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        rotation_duration = 40.0 * ns
        displacement_duration = 120.0 * ns
        dt = 2.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.05),
            omega_q=GHz(6.15),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=16,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        initial_state = model.basis_state(0, 0)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        disp_pulses, _, _ = build_displacement_pulse(
            DisplacementGate(index=0, name="D(alpha)", re=1.2, im=0.0),
            {"duration_displacement_s": displacement_duration},
        )
        rot_pulses, _, _ = build_rotation_pulse(
            RotationGate(index=1, name="x90", theta=np.pi / 2.0, phi=0.0),
            {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": 0.18},
        )
        qubit_pulse = rot_pulses[0]
        shifted_qubit_pulse = Pulse(
            channel=qubit_pulse.channel,
            t0=displacement_duration + 10.0 * ns,
            duration=qubit_pulse.duration,
            envelope=qubit_pulse.envelope,
            amp=qubit_pulse.amp,
            carrier=qubit_pulse.carrier,
            phase=qubit_pulse.phase,
            label=qubit_pulse.label,
        )
        all_pulses = disp_pulses + [shifted_qubit_pulse]
        drive_ops = {"storage": "cavity", "qubit": "qubit"}
        compiled = SequenceCompiler(dt=dt).compile(all_pulses, t_end=displacement_duration + rotation_duration + 20.0 * ns)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result = simulate_sequence(
            model,
            compiled,
            initial_state,
            drive_ops,
            config=SimulationConfig(frame=frame),
        )

        rho_q = reduced_qubit_state(result.final_state)
        rho_c = reduced_cavity_state(result.final_state)
        moments = mode_moments(result.final_state, "storage")
        conditioned_n0 = conditioned_bloch_xyz(result.final_state, n=0, fallback="zero")
        conditioned_n1 = conditioned_bloch_xyz(result.final_state, n=1, fallback="zero")
        print("Reduced qubit state:")
        print(rho_q)
        print(f"Storage <a> = {moments['a']:+.3f}")
        print(f"Storage <n> = {moments['n']:.3f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
        qubit_xyz = np.array([
            float(np.real((rho_q * qt.sigmax()).tr())),
            float(np.real((rho_q * qt.sigmay()).tr())),
            float(np.real((rho_q * qt.sigmaz()).tr())),
        ])
        axes[0].bar(["X", "Y", "Z"], qubit_xyz, color=["tab:blue", "tab:orange", "tab:green"])
        axes[0].set_ylim(-1.05, 1.05)
        axes[0].set_title("Bloch-vector components of the reduced qubit state")

        xvec, yvec, w = cavity_wigner(rho_c, n_points=81, extent=4.0)
        image = axes[1].imshow(w, origin="lower", extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], cmap="RdBu_r", aspect="equal")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("p")
        axes[1].set_title("Reduced cavity Wigner function")
        fig.colorbar(image, ax=axes[1], shrink=0.86)
        plt.show()

        print("Conditioned Bloch vector for n = 0:", conditioned_n0)
        print("Conditioned Bloch vector for n = 1:", conditioned_n1)
        """
            ),
            md("## 9. Physical Interpretation\n\nThe reduced qubit state captures the spin-like degrees of freedom, while the Wigner function keeps the cavity's phase-space picture. The conditioned Bloch vectors show how you can ask qubit questions inside a selected Fock sector."),
            md("## 10. Exercises / Next Steps\n\n- Change the displacement amplitude and re-run the conditioned Bloch diagnostics.\n- Add a nonzero `chi` and see how the qubit and cavity observables become correlated even for short sequences.\n- Continue to Tutorial 07 for a direct use of Fock-conditioned spectroscopy."),
        ],
    )

    write_notebook(
        TUTORIALS / "06_qubit_spectroscopy.ipynb",
        [
            title_cell(
                6,
                "Qubit Spectroscopy",
                "Perform a weak-drive qubit spectroscopy sweep in the matched rotating frame and fit the resonance with a Lorentzian model.",
                "Tutorials 01, 02, and 04 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will sweep a weak qubit probe around the rotating-frame resonance, read out the final excited-state population, and fit the peak position."),
            md("## 2. Physical Background\n\nA long, weak drive is the simplest spectroscopy experiment. In the matched rotating frame the bare qubit line appears near zero detuning, and the probe frequency is specified as a physical transition detuning before being converted to an internal waveform carrier."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        detuning_mhz = np.linspace(-3.0, 3.0, 61)
        probe_duration = 1.0 * us
        probe_amplitude = MHz(0.08)
        dt = 4.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        spectrum_levels = min(6, int(np.prod(model.subsystem_dims)))
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        responses = []
        for point_mhz in detuning_mhz:
            probe = Pulse(
                "q",
                t0=0.0,
                duration=probe_duration,
                envelope=square_envelope,
                carrier=carrier_for_transition_frequency(MHz(point_mhz)),
                amp=probe_amplitude,
                label="spectroscopy_probe",
            )
            compiled = SequenceCompiler(dt=dt).compile([probe], t_end=probe_duration + dt)
            result = simulate_sequence(
                model,
                compiled,
                model.basis_state(0, 0),
                {"q": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
            )
            responses.append(final_expectation(result, "P_e"))
        responses = np.asarray(responses, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        fit = fit_lorentzian_peak(detuning_mhz, responses)
        print("Fitted spectroscopy center [MHz]:", fit.parameters["center"])
        print("Fitted width [MHz]:", fit.parameters["width"])
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(detuning_mhz, responses, "o", label="simulation")
        ax.plot(detuning_mhz, fit.model_y, "-", label="Lorentzian fit")
        ax.axvline(fit.parameters["center"], color="tab:red", linestyle="--", label="fit center")
        ax.set_xlabel("Transition detuning relative to the qubit frame [MHz]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Weak-drive qubit spectroscopy")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe resonance appears near zero because the frame matches the bare qubit frequency. The key convention detail is that the x-axis is a physical transition detuning, while the simulator receives the internal raw carrier through `carrier_for_transition_frequency(...)`."),
            md("## 10. Exercises / Next Steps\n\n- Repeat the scan with a deliberate frame offset and see how the fitted center moves.\n- Increase the probe amplitude until power broadening becomes obvious.\n- Continue to Tutorial 07 for photon-number-resolved spectroscopy."),
        ],
    )

    write_notebook(
        TUTORIALS / "07_cavity_conditioned_qubit_spectroscopy_number_splitting.ipynb",
        [
            title_cell(
                7,
                "Cavity-Conditioned Qubit Spectroscopy (Number Splitting)",
                "Resolve the qubit line for several fixed cavity Fock states and compare the simulated peaks to the dispersive prediction from `manifold_transition_frequency(...)`.",
                "Tutorials 02, 03, and 06 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will run the same weak qubit spectroscopy scan for `n = 0`, `1`, and `2` cavity Fock states and overlay the predicted line positions."),
            md(
                """
        ## 2. Physical Background

        In the dispersive regime each cavity photon shifts the qubit line. In the repository's current convention, negative `chi` moves the `n`-resolved qubit lines to lower transition detuning. That makes number splitting a sharp diagnostic of both the physics and the sign conventions.
        """
            ),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        fock_levels = [0, 1, 2]
        detuning_mhz = np.linspace(-8.0, 2.0, 81)
        probe_duration = 1.0 * us
        probe_amplitude = MHz(0.08)
        dt = 4.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=MHz(-2.4),
            kerr=0.0,
            n_cav=10,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        predicted_lines_mhz = [angular_to_mhz(manifold_transition_frequency(model, n, frame=frame)) for n in fock_levels]
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        responses = {}
        for n in fock_levels:
            trace = []
            for detuning_point_mhz in detuning_mhz:
                probe = Pulse(
                    "q",
                    t0=0.0,
                    duration=probe_duration,
                    envelope=square_envelope,
                    carrier=carrier_for_transition_frequency(MHz(detuning_point_mhz)),
                    amp=probe_amplitude,
                    label=f"probe_n{n}",
                )
                compiled = SequenceCompiler(dt=dt).compile([probe], t_end=probe_duration + dt)
                result = simulate_sequence(
                    model,
                    compiled,
                    model.basis_state(0, n),
                    {"q": "qubit"},
                    config=SimulationConfig(frame=frame, max_step=dt),
                )
                trace.append(final_expectation(result, "P_e"))
            responses[n] = np.asarray(trace, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        peak_locations = {n: float(detuning_mhz[int(np.argmax(response))]) for n, response in responses.items()}
        print("Predicted peak positions [MHz]:", dict(zip(fock_levels, np.round(predicted_lines_mhz, 4), strict=True)))
        print("Observed peak positions [MHz]:", peak_locations)
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        for n in fock_levels:
            ax.plot(detuning_mhz, responses[n], label=fr"$n = {n}$")
        for n, line_mhz in zip(fock_levels, predicted_lines_mhz, strict=True):
            ax.axvline(line_mhz, linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("Transition detuning relative to the qubit frame [MHz]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Number splitting from fixed cavity Fock states")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe line ordering is the physics check: with negative `chi`, the `n = 2` peak lies to the left of the `n = 1` peak, which lies to the left of the `n = 0` peak. Using `manifold_transition_frequency(...)` keeps the overlay tied to the same sign convention as the Hamiltonian itself."),
            md("## 10. Exercises / Next Steps\n\n- Replace the fixed Fock states with a displaced cavity state and think about how the Poisson weights would combine these lines.\n- Add a small cavity Kerr to see how higher-`n` manifolds drift away from perfect equal spacing.\n- Continue to Tutorial 08 for dressed energies and dispersive frequency bookkeeping."),
        ],
    )

    write_notebook(
        TUTORIALS / "08_dispersive_shift_and_dressed_frequencies.ipynb",
        [
            title_cell(
                8,
                "Dispersive Shift and Dressed Frequencies",
                "Connect the low-level Hamiltonian, the dressed energy spectrum, and the photon-number-dependent qubit transition frequencies in one place.",
                "Tutorials 01, 02, and 07 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will compare dressed energies from exact diagonalization with the manifold transition helpers exposed by the public API."),
            md("## 2. Physical Background\n\nThe dispersive approximation is easiest to trust when the dressed energy picture and the transition helper functions tell the same story. This notebook keeps both views side by side."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code("max_n = 6"),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.05),
            omega_q=GHz(6.25),
            alpha=MHz(-220.0),
            chi=MHz(-2.6),
            kerr=MHz(-0.002),
            n_cav=10,
            n_tr=3,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        lab_spectrum = compute_energy_spectrum(model, frame=FrameSpec(), levels=14)
        rot_spectrum = compute_energy_spectrum(model, frame=frame, levels=14)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        n_values = np.arange(max_n + 1)
        transition_mhz = np.array([angular_to_mhz(manifold_transition_frequency(model, int(n), frame=frame)) for n in n_values])
        line_spacing_mhz = np.diff(transition_mhz)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        print("Transition frequencies [MHz] in the matched rotating frame:")
        for n, value in zip(n_values, transition_mhz, strict=True):
            print(f"  n = {n}: {value:+.4f} MHz")
        print("Adjacent line spacing [MHz]:", np.round(line_spacing_mhz, 4))
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
        plot_energy_levels(lab_spectrum, max_levels=12, energy_scale=1.0 / (2.0 * np.pi * 1.0e6), energy_unit_label="MHz", title="Lab-frame dressed energies", ax=axes[0])
        axes[1].plot(n_values, transition_mhz, "o-")
        axes[1].set_xlabel("Photon number n")
        axes[1].set_ylabel(r"$\\omega_{ge}(n) / 2\\pi$ [MHz]")
        axes[1].set_title("Dressed qubit transition versus bosonic occupation")
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe energy-level plot is the dressed-spectrum view, while the `omega_ge(n)` curve is the spectroscopy view. Both are generated from the same public model object, so agreement between them is a strong sanity check before moving into calibration notebooks."),
            md("## 10. Exercises / Next Steps\n\n- Increase the cavity Kerr and see when the transition-versus-`n` curve bends away from a strictly linear `chi` shift.\n- Compare the rotating-frame and lab-frame spectra directly by changing `frame` in the `compute_energy_spectrum(...)` calls.\n- Continue to Tutorials 09 and 10 for Rabi calibrations."),
        ],
    )

    write_notebook(
        TUTORIALS / "09_power_rabi.ipynb",
        [
            title_cell(
                9,
                "Power Rabi",
                "Sweep the drive amplitude at fixed pulse duration, fit the oscillation, and estimate the `pi`-pulse amplitude.",
                "Tutorials 04 and 06 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will perform a fixed-duration amplitude sweep and fit the resulting oscillation to estimate the drive scale needed for a `pi` pulse."),
            md("## 2. Physical Background\n\nAt fixed duration, the excited-state population oscillates as the drive area changes. The `pi`-pulse amplitude is the amplitude that produces one half-cycle of the Rabi oscillation."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        duration = 40.0 * ns
        amplitudes_mhz = np.linspace(0.0, 25.0, 51)
        dt = 2.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        spectrum_levels = min(6, int(np.prod(model.subsystem_dims)))
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        responses = []
        amplitudes_rad_s = np.array([MHz(value) for value in amplitudes_mhz], dtype=float)
        for amplitude in amplitudes_rad_s:
            pulse = Pulse("q", 0.0, duration, square_envelope, amp=float(amplitude), carrier=0.0, label="power_rabi")
            compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=duration + dt)
            result = simulate_sequence(
                model,
                compiled,
                model.basis_state(0, 0),
                {"q": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
            )
            responses.append(final_expectation(result, "P_e"))
        responses = np.asarray(responses, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        fit = fit_rabi_vs_amplitude(amplitudes_rad_s, responses, duration=duration)
        print(f"Estimated pi amplitude / 2pi = {angular_to_mhz(fit.parameters['pi_amplitude']):.3f} MHz")
        print(f"Estimated pi/2 amplitude / 2pi = {angular_to_mhz(fit.parameters['pi_over_two_amplitude']):.3f} MHz")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(amplitudes_mhz, responses, "o", label="simulation")
        ax.plot(amplitudes_mhz, fit.model_y, "-", label="fit")
        ax.axvline(angular_to_mhz(fit.parameters["pi_amplitude"]), color="tab:red", linestyle="--", label="pi amplitude")
        ax.axvline(angular_to_mhz(fit.parameters["pi_over_two_amplitude"]), color="tab:green", linestyle="--", label="pi/2 amplitude")
        ax.set_xlabel("Drive amplitude / 2pi [MHz]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Power-Rabi calibration sweep")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe fitted `pi` and `pi/2` amplitudes are useful calibration targets for later sequence notebooks. This is also a good place to see the difference between a waveform amplitude and a physical transition frequency: here the sweep variable is the drive strength, not the carrier."),
            md("## 10. Exercises / Next Steps\n\n- Repeat the sweep with a Gaussian envelope instead of a square envelope.\n- Increase the amplitude range and watch for fit breakdown if the pulse is no longer cleanly two-level.\n- Continue to Tutorial 10 for a time-domain Rabi calibration."),
        ],
    )


def generate_batch_2() -> None:
    write_notebook(
        TUTORIALS / "10_time_rabi.ipynb",
        [
            title_cell(
                10,
                "Time Rabi",
                "Sweep the pulse duration at fixed amplitude, fit the oscillation period, and estimate the `pi` and `pi/2` durations.",
                "Tutorials 04 and 09 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will keep the pulse amplitude fixed, vary the pulse duration, and fit the resulting oscillation to recover the Rabi rate."),
            md("## 2. Physical Background\n\nA time-Rabi experiment keeps the drive strength fixed and measures how long it takes to reach the desired rotation angle. This is the duration-space complement of the power-Rabi notebook."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        omega_rabi = 2.0 * np.pi * 12.0e6
        durations_ns = np.linspace(0.0, 120.0, 49)
        dt = 2.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        spectrum_levels = min(6, int(np.prod(model.subsystem_dims)))
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        durations_s = durations_ns * ns
        responses = []
        for duration_s in durations_s:
            pulse = Pulse("q", 0.0, float(max(duration_s, dt)), square_envelope, amp=omega_rabi, carrier=0.0, label="time_rabi")
            compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=float(max(duration_s, dt)) + dt)
            result = simulate_sequence(
                model,
                compiled,
                model.basis_state(0, 0),
                {"q": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
            )
            responses.append(final_expectation(result, "P_e"))
        responses = np.asarray(responses, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        fit = fit_rabi_vs_duration(durations_s, responses)
        print(f"Estimated pi time = {fit.parameters['pi_time_s'] / ns:.3f} ns")
        print(f"Estimated pi/2 time = {fit.parameters['pi_over_two_time_s'] / ns:.3f} ns")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(durations_ns, responses, "o", label="simulation")
        ax.plot(durations_ns, fit.model_y, "-", label="fit")
        ax.axvline(fit.parameters["pi_time_s"] / ns, color="tab:red", linestyle="--", label="pi time")
        ax.axvline(fit.parameters["pi_over_two_time_s"] / ns, color="tab:green", linestyle="--", label="pi/2 time")
        ax.set_xlabel("Pulse duration [ns]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Time-Rabi calibration sweep")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nTime-Rabi and power-Rabi are two views of the same control scale. In practice one is often easier to scan than the other depending on how the hardware exposes pulse settings."),
            md("## 10. Exercises / Next Steps\n\n- Repeat the fit with a different fixed amplitude and verify that the `pi` time rescales as expected.\n- Change the frame or add a detuning to see how off-resonant driving distorts the simple sinusoidal picture.\n- Continue to Tutorial 25 for a compact workflow that combines spectroscopy and Rabi information."),
        ],
    )

    write_notebook(
        TUTORIALS / "11_qubit_T1_relaxation.ipynb",
        [
            title_cell(
                11,
                "Qubit T1 Relaxation",
                "Simulate energy relaxation with `NoiseSpec(t1=...)`, fit the decay, and recover the relaxation time from the final excited-state population.",
                "Tutorial 04 is a helpful precursor.",
            ),
            md("## 1. Goal\n\nWe will start in `|e>` and measure how the excited-state population decays under a Lindblad `T1` model."),
            md("## 2. Physical Background\n\n`T1` is the energy-relaxation time. In a simple two-level model with no drive and no thermal repopulation, the excited-state population decays exponentially as `exp(-t / T1)`."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        t1_true = 18.0 * us
        delays_us = np.linspace(0.0, 40.0, 33)
        dt = 20.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        initial_state = model.basis_state(1, 0)
        noise = NoiseSpec(t1=t1_true)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        delays_s = delays_us * us
        responses = []
        for delay_s in delays_s:
            compiled = SequenceCompiler(dt=dt).compile([], t_end=float(max(delay_s, dt)))
            result = simulate_sequence(
                model,
                compiled,
                initial_state,
                {},
                config=SimulationConfig(frame=frame, max_step=dt),
                noise=noise,
            )
            responses.append(final_expectation(result, "P_e"))
        responses = np.asarray(responses, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        fit = fit_exponential_decay(delays_s, responses, parameter_name="t1")
        print(f"True T1 = {t1_true / us:.3f} us")
        print(f"Fitted T1 = {fit.parameters['t1'] / us:.3f} us")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(delays_us, responses, "o", label="simulation")
        ax.plot(delays_us, fit.model_y, "-", label="fit")
        ax.set_xlabel("Delay [us]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Qubit T1 relaxation")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThis is the cleanest calibration-style notebook in the suite because the model prediction is directly exponential. The fitted `T1` becomes part of the open-system parameter set used in later Ramsey, echo, and readout examples."),
            md("## 10. Exercises / Next Steps\n\n- Add a nonzero cavity mode with finite `chi` and verify that the relaxation fit is still dominated by the same `T1` model when the cavity stays in vacuum.\n- Explore what happens when `t1` becomes comparable to the pulse durations used in Tutorials 09 and 10.\n- Continue to Tutorial 12 for dephasing-sensitive Ramsey fringes."),
        ],
    )

    write_notebook(
        TUTORIALS / "12_qubit_ramsey_T2star.ipynb",
        [
            title_cell(
                12,
                "Qubit Ramsey T2*",
                "Build a Ramsey sequence from two `x90` pulses, add dephasing noise, and fit both the detuning and `T2*` from the fringe envelope.",
                "Tutorials 04, 10, and 11 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will implement a textbook Ramsey sequence and extract the detuning and dephasing time from the simulated fringe pattern."),
            md("## 2. Physical Background\n\nA Ramsey experiment uses two `pi/2` pulses separated by free evolution. The fringe frequency encodes the residual detuning, while the envelope decays on the `T2*` timescale."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        t1_true = 30.0 * us
        t2_star_true = 8.0 * us
        detuning = 2.0 * np.pi * 0.6e6
        delays_us = np.linspace(0.0, 16.0, 41)
        rotation_duration = 30.0 * ns
        rotation_sigma_fraction = 0.18
        dt = 4.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        tphi_true = pure_dephasing_time_from_t1_t2(t1_s=t1_true, t2_s=t2_star_true)
        noise = NoiseSpec(t1=t1_true, tphi=tphi_true)
        base_pulses, _, _ = build_rotation_pulse(
            RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
            {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": rotation_sigma_fraction},
        )
        base_x90 = base_pulses[0]
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        delays_s = delays_us * us
        responses = []
        for delay_s in delays_s:
            first_pulse = Pulse(
                channel=base_x90.channel,
                t0=0.0,
                duration=base_x90.duration,
                envelope=base_x90.envelope,
                amp=base_x90.amp,
                carrier=base_x90.carrier,
                phase=0.0,
                label="ramsey_x90_a",
            )
            second_pulse = Pulse(
                channel=base_x90.channel,
                t0=rotation_duration + delay_s,
                duration=base_x90.duration,
                envelope=base_x90.envelope,
                amp=base_x90.amp,
                carrier=base_x90.carrier,
                phase=float(detuning * delay_s),
                label="ramsey_x90_b",
            )
            t_end = 2.0 * rotation_duration + delay_s + dt
            compiled = SequenceCompiler(dt=dt).compile([first_pulse, second_pulse], t_end=t_end)
            result = simulate_sequence(
                model,
                compiled,
                model.basis_state(0, 0),
                {"qubit": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
                noise=noise,
            )
            responses.append(final_expectation(result, "P_e"))
        responses = np.asarray(responses, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        fit = fit_ramsey_signal(delays_s, responses, p0=(detuning, t2_star_true, 0.5, 0.5, 0.0))
        print(f"True detuning / 2pi = {detuning / (2.0 * np.pi * 1.0e6):.3f} MHz")
        print(f"Fitted detuning / 2pi = {fit.parameters['detuning'] / (2.0 * np.pi * 1.0e6):.3f} MHz")
        print(f"True T2* = {t2_star_true / us:.3f} us")
        print(f"Fitted T2* = {fit.parameters['t2_star'] / us:.3f} us")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(delays_us, responses, "o", label="simulation")
        ax.plot(delays_us, fit.model_y, "-", label="fit")
        ax.set_xlabel("Free-evolution delay [us]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Ramsey fringe with dephasing")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe oscillation frequency reflects the residual phase advance between the two `x90` pulses, while the envelope is set by the combination of energy relaxation and pure dephasing. This is why Ramsey is the natural place to estimate `T2*`."),
            md("## 10. Exercises / Next Steps\n\n- Set `detuning = 0` and confirm that the fringes collapse into a pure decay envelope.\n- Increase the dephasing rate while keeping `T1` fixed to see how `T2*` changes.\n- Continue to Tutorial 13 for a spin-echo sequence that mitigates static phase accumulation."),
        ],
    )

    write_notebook(
        TUTORIALS / "13_spin_echo_and_dephasing_mitigation.ipynb",
        [
            title_cell(
                13,
                "Spin Echo and Dephasing Mitigation",
                "Compare a Ramsey-style free-evolution experiment to a Hahn-echo sequence under the same dephasing model and fit the echo envelope.",
                "Tutorials 11 and 12 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will show that a spin-echo sequence suppresses static phase accumulation and extends the visible coherence envelope relative to a Ramsey-style experiment."),
            md("## 2. Physical Background\n\nA Hahn echo inserts a `pi` pulse halfway through the free evolution. That pulse reverses the sign of static phase accumulation, which makes the sequence less sensitive to low-frequency detuning offsets than Ramsey."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        t1_true = 40.0 * us
        t2_star_true = 10.0 * us
        t2_echo_target = 18.0 * us
        delays_us = np.linspace(0.0, 24.0, 37)
        rotation_duration = 30.0 * ns
        rotation_sigma_fraction = 0.18
        dt = 4.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        tphi_true = pure_dephasing_time_from_t1_t2(t1_s=t1_true, t2_s=t2_star_true)
        noise = NoiseSpec(t1=t1_true, tphi=tphi_true)
        x90_pulses, _, _ = build_rotation_pulse(
            RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0),
            {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": rotation_sigma_fraction},
        )
        x180_pulses, _, _ = build_rotation_pulse(
            RotationGate(index=1, name="x180", theta=np.pi, phi=0.0),
            {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": rotation_sigma_fraction},
        )
        x90 = x90_pulses[0]
        x180 = x180_pulses[0]
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        delays_s = delays_us * us
        ramsey_like = []
        echo = []
        for delay_s in delays_s:
            ramsey_pulses = [
                Pulse(x90.channel, 0.0, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=0.0, label="ramsey_a"),
                Pulse(x90.channel, x90.duration + delay_s, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=0.0, label="ramsey_b"),
            ]
            ramsey_t_end = 2.0 * x90.duration + delay_s + dt
            ramsey_compiled = SequenceCompiler(dt=dt).compile(ramsey_pulses, t_end=ramsey_t_end)
            ramsey_result = simulate_sequence(
                model,
                ramsey_compiled,
                model.basis_state(0, 0),
                {"qubit": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
                noise=noise,
            )
            ramsey_like.append(final_expectation(ramsey_result, "P_e"))

            echo_pulses = [
                Pulse(x90.channel, 0.0, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=0.0, label="echo_a"),
                Pulse(x180.channel, x90.duration + 0.5 * delay_s, x180.duration, x180.envelope, amp=x180.amp, carrier=x180.carrier, phase=0.0, label="echo_pi"),
                Pulse(x90.channel, x90.duration + delay_s + x180.duration, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=0.0, label="echo_b"),
            ]
            echo_t_end = 2.0 * x90.duration + x180.duration + delay_s + dt
            echo_compiled = SequenceCompiler(dt=dt).compile(echo_pulses, t_end=echo_t_end)
            echo_result = simulate_sequence(
                model,
                echo_compiled,
                model.basis_state(0, 0),
                {"qubit": "qubit"},
                config=SimulationConfig(frame=frame, max_step=dt),
                noise=noise,
            )
            echo.append(final_expectation(echo_result, "P_e"))
        ramsey_like = np.asarray(ramsey_like, dtype=float)
        echo = np.asarray(echo, dtype=float)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        fit = fit_echo_signal(delays_s, echo, p0=(t2_echo_target, 0.5, 0.5))
        print(f"Representative echo-envelope fit T2_echo = {fit.parameters['t2_echo'] / us:.3f} us")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(delays_us, ramsey_like, "o-", label="Ramsey-style sequence")
        ax.plot(delays_us, echo, "o-", label="Spin echo")
        ax.plot(delays_us, fit.model_y, "--", label="Echo fit")
        ax.set_xlabel("Total delay [us]")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Spin echo mitigates low-frequency dephasing")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe echo trace decays more slowly because the middle `pi` pulse refocuses static or slowly varying phase errors. That is why echo is often used to distinguish reversible dephasing from irreversible coherence loss."),
            md("## 10. Exercises / Next Steps\n\n- Add a deliberate detuning offset and verify that the Ramsey-like sequence is much more sensitive to it than the echo sequence.\n- Compare the fitted echo envelope to the direct `run_t2_echo(...)` calibration-target helper introduced later in Tutorial 23.\n- Continue to Tutorial 14 for bosonic Kerr dynamics."),
        ],
    )

    write_notebook(
        TUTORIALS / "14_kerr_free_evolution.ipynb",
        [
            title_cell(
                14,
                "Kerr Free Evolution",
                "Prepare a coherent cavity state, let it evolve under self-Kerr in the matched rotating frame, and inspect the resulting phase-space dynamics.",
                "Tutorials 03, 05, and 08 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will isolate self-Kerr evolution in the storage mode and visualize how the cavity moments and Wigner function change over time."),
            md("## 2. Physical Background\n\nIn the matched rotating frame, a cavity with self-Kerr no longer undergoes a large bare rotation, but it still accumulates number-dependent phase. That phase bends the coherent-state trajectory away from rigid harmonic motion."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        total_time = 20.0 * us
        dt = 0.2 * us
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.05),
            omega_q=GHz(6.25),
            alpha=MHz(-220.0),
            chi=0.0,
            kerr=MHz(-0.080),
            n_cav=28,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        initial_state = prepare_state(
            model,
            StatePreparationSpec(qubit=qubit_state("g"), storage=coherent_state(1.8)),
        )
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        compiled = SequenceCompiler(dt=dt).compile([], t_end=total_time)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result = simulate_sequence(
            model,
            compiled,
            initial_state,
            {},
            config=SimulationConfig(frame=frame, store_states=True, max_step=dt),
        )
        cavity_means = np.array([mode_moments(state, "storage")["a"] for state in result.states], dtype=np.complex128)
        photon_numbers = np.array([mode_moments(state, "storage")["n"] for state in result.states], dtype=float)
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        axes[0].plot(np.real(cavity_means), np.imag(cavity_means), "o-")
        axes[0].set_xlabel(r"Re$\\langle a \\rangle$")
        axes[0].set_ylabel(r"Im$\\langle a \\rangle$")
        axes[0].set_title("Coherent-state trajectory under self-Kerr")

        axes[1].plot(compiled.tlist / us, photon_numbers)
        axes[1].set_xlabel("Time [us]")
        axes[1].set_ylabel(r"$\\langle n \\rangle$")
        axes[1].set_title("Mean cavity occupation during Kerr evolution")
        plt.show()

        final_rho_c = reduced_cavity_state(result.final_state)
        xvec, yvec, w = cavity_wigner(final_rho_c, n_points=81, extent=4.5)
        fig, ax = plt.subplots()
        image = ax.imshow(w, origin="lower", extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]], cmap="RdBu_r", aspect="equal")
        ax.set_xlabel("x")
        ax.set_ylabel("p")
        ax.set_title("Final cavity Wigner function after Kerr evolution")
        fig.colorbar(image, ax=ax, shrink=0.86)
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe cavity photon number stays nearly constant because there is no loss term here. The interesting physics is the nonlinear phase accumulation, which shears the phase-space distribution and bends the coherent-state trajectory."),
            md("## 10. Exercises / Next Steps\n\n- Reverse the sign of the Kerr coefficient and compare the direction of the phase-space bending.\n- Add a small cavity decay rate and observe how loss and Kerr compete.\n- Continue to Tutorials 15 and 16 for cross-Kerr and lossy bosonic dynamics."),
        ],
    )

    write_notebook(
        TUTORIALS / "15_cross_kerr_and_conditional_phase_accumulation.ipynb",
        [
            title_cell(
                15,
                "Cross-Kerr and Conditional Phase Accumulation",
                "Use the three-mode storage-readout model to show how a storage superposition accumulates a conditional phase when the readout mode is occupied.",
                "Tutorials 08 and 14 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will compare free evolution with and without a readout photon and measure the extra phase accumulated by a storage-mode superposition."),
            md("## 2. Physical Background\n\nA cross-Kerr term shifts one bosonic mode depending on the occupation of another. In a storage-readout setting, that means the storage phase evolution can depend on the readout photon number even when the transmon stays in `|g>`."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        total_time = 40.0 * us
        dt = 0.5 * us
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveReadoutTransmonStorageModel(
            omega_s=GHz(5.1),
            omega_r=GHz(7.3),
            omega_q=GHz(6.2),
            alpha=MHz(-220.0),
            chi_s=0.0,
            chi_r=0.0,
            chi_sr=MHz(0.030),
            kerr_s=0.0,
            kerr_r=0.0,
            n_storage=4,
            n_readout=4,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_s, omega_q_frame=model.omega_q, omega_r_frame=model.omega_r)
        initial_r0 = (model.basis_state(0, 0, 0) + model.basis_state(0, 1, 0)).unit()
        initial_r1 = (model.basis_state(0, 0, 1) + model.basis_state(0, 1, 1)).unit()
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        compiled = SequenceCompiler(dt=dt).compile([], t_end=total_time)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result_r0 = simulate_sequence(model, compiled, initial_r0, {}, config=SimulationConfig(frame=frame, store_states=True, max_step=dt))
        result_r1 = simulate_sequence(model, compiled, initial_r1, {}, config=SimulationConfig(frame=frame, store_states=True, max_step=dt))

        def relative_phase(states, readout_level):
            phases = []
            ref = model.basis_state(0, 0, readout_level)
            shifted = model.basis_state(0, 1, readout_level)
            for state in states:
                amp_ref = complex(ref.overlap(state))
                amp_shift = complex(shifted.overlap(state))
                phases.append(np.angle(amp_shift / amp_ref))
            return np.unwrap(np.asarray(phases, dtype=float))

        phase_r0 = relative_phase(result_r0.states, 0)
        phase_r1 = relative_phase(result_r1.states, 1)
        conditional_phase = phase_r1 - phase_r0
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(compiled.tlist / us, conditional_phase, label="simulated conditional phase")
        ax.set_xlabel("Time [us]")
        ax.set_ylabel("Extra phase [rad]")
        ax.set_title("Storage phase conditioned on readout occupancy")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe difference between the two phase traces is the operational meaning of the storage-readout cross-Kerr term. This is the bosonic analogue of a conditional frequency pull: one mode's phase evolution knows about the other's occupation."),
            md("## 10. Exercises / Next Steps\n\n- Increase `chi_sr` and confirm that the phase slope scales linearly.\n- Add storage or readout self-Kerr and separate the conditional effect from the single-mode nonlinearity.\n- Continue to Tutorial 17 for readout-chain modeling on top of the three-mode picture."),
        ],
    )

    write_notebook(
        TUTORIALS / "16_storage_cavity_coherent_state_dynamics.ipynb",
        [
            title_cell(
                16,
                "Storage-Cavity Coherent-State Dynamics",
                "Track a coherent storage state under cavity loss and extract both the mean field and the photon-number decay.",
                "Tutorials 03, 05, and 14 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will prepare a coherent state, add cavity loss, and observe how both `|<a>|` and `<n>` decay in time."),
            md("## 2. Physical Background\n\nWithout Kerr, a damped coherent state stays coherent and shrinks toward vacuum. That makes coherent-state ringdown a clean way to connect the dynamical solver, the bosonic moments, and the open-system parameters."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        total_time = 25.0 * us
        dt = 0.25 * us
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.05),
            omega_q=GHz(6.25),
            alpha=MHz(-220.0),
            chi=0.0,
            kerr=0.0,
            n_cav=24,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        initial_state = prepare_state(
            model,
            StatePreparationSpec(qubit=qubit_state("g"), storage=coherent_state(1.6)),
        )
        noise = NoiseSpec(kappa=1.0 / (18.0 * us))
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        compiled = SequenceCompiler(dt=dt).compile([], t_end=total_time)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result = simulate_sequence(
            model,
            compiled,
            initial_state,
            {},
            config=SimulationConfig(frame=frame, store_states=True, max_step=dt),
            noise=noise,
        )
        alpha_t = np.array([mode_moments(state, "storage")["a"] for state in result.states], dtype=np.complex128)
        n_t = np.array([mode_moments(state, "storage")["n"] for state in result.states], dtype=float)
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        axes[0].plot(compiled.tlist / us, np.abs(alpha_t))
        axes[0].set_xlabel("Time [us]")
        axes[0].set_ylabel(r"$|\\langle a \\rangle|$")
        axes[0].set_title("Decay of the coherent-state amplitude")

        axes[1].plot(compiled.tlist / us, n_t)
        axes[1].set_xlabel("Time [us]")
        axes[1].set_ylabel(r"$\\langle n \\rangle$")
        axes[1].set_title("Photon-number ringdown")
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nBecause the Hamiltonian is linear in this notebook, the coherent state stays Gaussian while its amplitude decays. This is the lossy counterpart to the Kerr-only evolution in Tutorial 14."),
            md("## 10. Exercises / Next Steps\n\n- Compare the decay of `|<a>|` to the decay of `<n>` and relate the two.\n- Add a small Kerr term and see when the state stops looking like a simple damped coherent state.\n- Continue to Tutorial 20 for truncation checks on bosonic simulations."),
        ],
    )

    write_notebook(
        TUTORIALS / "17_readout_resonator_response.ipynb",
        [
            title_cell(
                17,
                "Readout Resonator Response",
                "Use the measurement-layer readout-chain model to compare the resonator response for `|g>` and `|e>`, inspect the resulting IQ clusters, and read off the measurement-induced dephasing rate.",
                "Tutorials 02 and 11 are useful prerequisites.",
                "This notebook uses the repository's effective readout-chain model rather than embedding a full driven resonator inside the QuTiP solver.",
            ),
            md("## 1. Goal\n\nWe will simulate the readout-chain response directly from the measurement layer and inspect both time traces and IQ discrimination geometry."),
            md("## 2. Physical Background\n\nThe measurement API models the resonator, any Purcell filter, and the amplifier chain. That gives a fast path to readout-conditioned responses, Purcell-limited `T1` estimates, and synthetic IQ samples."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        readout_chain = ReadoutChain(
            resonator=ReadoutResonator(
                omega_r=GHz(7.1),
                kappa=MHz(8.0),
                g=MHz(90.0),
                epsilon=MHz(0.7),
                chi=MHz(1.5),
            ),
            purcell_filter=PurcellFilter(bandwidth=MHz(35.0)),
            amplifier=AmplifierChain(noise_temperature=4.0, gain=12.0),
            integration_time=300.0 * ns,
            dt=5.0 * ns,
        )
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        trace_g = readout_chain.simulate_trace("g", include_noise=False)
        trace_e = readout_chain.simulate_trace("e", include_noise=False)
        iq_samples = readout_chain.sample_iq([0] * 200 + [1] * 200, seed=7)
        classified = readout_chain.classify_iq(iq_samples)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        gamma_meas = readout_chain.gamma_meas()
        purcell_limited_t1 = readout_chain.purcell_limited_t1(GHz(6.2))
        print(f"gamma_meas / 2pi = {gamma_meas / (2.0 * np.pi * 1.0e6):.3f} MHz")
        print(f"Purcell-limited T1 = {purcell_limited_t1 / us:.3f} us")
        print(f"IQ classification accuracy = {np.mean(classified[:200] == 0) * 0.5 + np.mean(classified[200:] == 1) * 0.5:.3f}")
        """
            ),
            md("## 7. Running the Simulation"),
            code("pass"),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        axes[0].plot(trace_g.tlist / ns, np.real(trace_g.cavity_field), label="Re alpha_g")
        axes[0].plot(trace_e.tlist / ns, np.real(trace_e.cavity_field), label="Re alpha_e")
        axes[0].set_xlabel("Time [ns]")
        axes[0].set_ylabel("Cavity field quadrature")
        axes[0].set_title("Readout resonator response")
        axes[0].legend()

        axes[1].scatter(iq_samples[:200, 0], iq_samples[:200, 1], s=10, alpha=0.5, label="prepared g")
        axes[1].scatter(iq_samples[200:, 0], iq_samples[200:, 1], s=10, alpha=0.5, label="prepared e")
        axes[1].set_xlabel("I")
        axes[1].set_ylabel("Q")
        axes[1].set_title("Synthetic IQ clusters")
        axes[1].legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe separation between the `|g>` and `|e>` trajectories sets the measurement rate, while the amplifier noise broadens those ideal centers into finite IQ clusters. This is the notebook-scale bridge between Hamiltonian simulation and experiment-style readout language."),
            md("## 10. Exercises / Next Steps\n\n- Turn the Purcell filter off and compare the resulting linewidth and Purcell-limited `T1`.\n- Increase the integration time and watch the IQ clusters tighten.\n- Continue to Tutorial 25 to see how spectroscopy, Rabi, and relaxation estimates can be collected into one compact workflow."),
        ],
    )

    write_notebook(
        TUTORIALS / "18_multilevel_transmon_effects.ipynb",
        [
            title_cell(
                18,
                "Multilevel Transmon Effects",
                "Use `UniversalCQEDModel` with a multilevel transmon to compare the `g-e`, `e-f`, and `f-h` transition frequencies and inspect the dressed spectrum.",
                "Tutorials 01 and 08 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will move beyond the two-level qubit approximation and look at the next transmon transitions in a multilevel model."),
            md("## 2. Physical Background\n\nA transmon is only approximately a qubit. Its finite anharmonicity means the `g-e` and `e-f` transitions are close, which matters for strong driving, leakage, and sideband-style control."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code("max_levels = 12"),
            md("## 5. Model Construction"),
            code(
                """
        model = UniversalCQEDModel(
            transmon=TransmonModeSpec(omega=GHz(6.2), dim=5, alpha=MHz(-220.0), label="qubit", aliases=("qubit", "transmon")),
            bosonic_modes=(BosonicModeSpec(label="storage", omega=GHz(5.0), dim=8, kerr=0.0, aliases=("storage", "cavity")),),
            dispersive_couplings=(DispersiveCouplingSpec(mode="storage", chi=MHz(-2.2)),),
        )
        frame = FrameSpec(omega_c_frame=GHz(5.0), omega_q_frame=GHz(6.2))
        spectrum = compute_energy_spectrum(model, frame=FrameSpec(), levels=max_levels)
        ge = model.transmon_transition_frequency(mode_levels={"storage": 0}, lower_level=0, upper_level=1, frame=frame)
        ef = model.transmon_transition_frequency(mode_levels={"storage": 0}, lower_level=1, upper_level=2, frame=frame)
        fh = model.transmon_transition_frequency(mode_levels={"storage": 0}, lower_level=2, upper_level=3, frame=frame)
        print(f"g-e / 2pi = {angular_to_mhz(ge):+.3f} MHz in the matched frame")
        print(f"e-f / 2pi = {angular_to_mhz(ef):+.3f} MHz in the matched frame")
        print(f"f-h / 2pi = {angular_to_mhz(fh):+.3f} MHz in the matched frame")
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code("transition_labels = ['g-e', 'e-f', 'f-h']\ntransition_values = [ge, ef, fh]"),
            md("## 7. Running the Simulation"),
            code("pass"),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        plot_energy_levels(spectrum, max_levels=max_levels, energy_scale=1.0 / (2.0 * np.pi * 1.0e6), energy_unit_label="MHz", title="Multilevel dressed spectrum", ax=axes[0])
        axes[1].bar(transition_labels, [angular_to_mhz(value) for value in transition_values], color=["tab:blue", "tab:orange", "tab:green"])
        axes[1].set_ylabel(r"Transition frequency / 2pi [MHz]")
        axes[1].set_title("Nearest transmon transitions in the matched frame")
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe transitions are not equally spaced: the difference between `g-e` and `e-f` is the anharmonicity. That is the physical reason strong resonant `g-e` driving can leak population into `|f>` and higher levels."),
            md("## 10. Exercises / Next Steps\n\n- Increase the transmon dimension and compare how the higher ladder continues.\n- Change the anharmonicity and see how the `e-f` spacing moves relative to `g-e`.\n- Continue to Tutorial 19 for a direct leakage example under stronger drive."),
        ],
    )


def generate_batch_3() -> None:
    write_notebook(
        TUTORIALS / "19_anharmonicity_and_leakage_under_strong_drive.ipynb",
        [
            title_cell(
                19,
                "Anharmonicity and Leakage Under Strong Drive",
                "Compare weak and strong resonant driving in a three-level transmon and track the population that leaks into `|f>`.",
                "Tutorial 18 is recommended first.",
            ),
            md("## 1. Goal\n\nWe will demonstrate how stronger resonant `g-e` driving in a multilevel model starts to populate `|f>` even when the carrier is aimed at the lowest transition."),
            md("## 2. Physical Background\n\nA real transmon is not an ideal two-level system. When the Rabi rate becomes too large compared with the anharmonicity, the drive can no longer cleanly isolate the `g-e` manifold."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        weak_drive = 2.0 * np.pi * 8.0e6
        strong_drive = 2.0 * np.pi * 45.0e6
        pulse_duration = 80.0 * ns
        dt = 1.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=0.0,
            omega_q=GHz(6.2),
            alpha=MHz(-220.0),
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=3,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        def run_drive(amplitude):
            pulse = Pulse("q", 0.0, pulse_duration, square_envelope, amp=amplitude, carrier=0.0, label="strong_drive")
            compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=pulse_duration + dt)
            return simulate_sequence(
                model,
                compiled,
                model.basis_state(0, 0),
                {"q": "qubit"},
                config=SimulationConfig(frame=frame, store_states=True, max_step=dt),
            )

        weak_result = run_drive(weak_drive)
        strong_result = run_drive(strong_drive)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        weak_pf = np.asarray(weak_result.expectations["P_f"], dtype=float)
        strong_pf = np.asarray(strong_result.expectations["P_f"], dtype=float)
        print(f"Weak-drive maximum P_f = {np.max(weak_pf):.4f}")
        print(f"Strong-drive maximum P_f = {np.max(strong_pf):.4f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        time_ns = np.asarray(weak_result.solver_result.times, dtype=float) * 1.0e9
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), sharey=True)
        axes[0].plot(time_ns, weak_result.expectations["P_e"], label=r"$P_e$")
        axes[0].plot(time_ns, weak_result.expectations["P_f"], label=r"$P_f$")
        axes[0].set_title("Weak drive")
        axes[0].set_xlabel("Time [ns]")
        axes[0].set_ylabel("Population")
        axes[0].legend()

        axes[1].plot(time_ns, strong_result.expectations["P_e"], label=r"$P_e$")
        axes[1].plot(time_ns, strong_result.expectations["P_f"], label=r"$P_f$")
        axes[1].set_title("Strong drive")
        axes[1].set_xlabel("Time [ns]")
        axes[1].legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe strong-drive trace leaks more noticeably into `|f>` because the drive is no longer perturbatively small compared with the transmon anharmonicity. That is the multilevel reason to be careful with large pulse amplitudes."),
            md("## 10. Exercises / Next Steps\n\n- Change the anharmonicity and repeat the comparison.\n- Try a Gaussian envelope instead of a square pulse to see how the spectral width changes the leakage behavior.\n- Continue to Tutorial 20 for another kind of numerical care: truncation convergence."),
        ],
    )

    write_notebook(
        TUTORIALS / "20_truncation_convergence_checks.ipynb",
        [
            title_cell(
                20,
                "Truncation Convergence Checks",
                "Sweep the cavity truncation, repeat the same displacement experiment, and quantify when the chosen Hilbert-space cutoff is no longer trustworthy.",
                "Tutorials 03 and 16 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will use a simple displacement experiment to see how the cavity cutoff `n_cav` affects a basic observable."),
            md("## 2. Physical Background\n\nBosonic truncation is a numerical approximation, not a physical assumption. A coherent state with `|alpha|^2 = 4` needs enough Fock levels to represent its Poisson tail cleanly."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        cutoff_list = [4, 6, 8, 10, 14, 20]
        target_alpha = 2.0 + 0.0j
        displacement_duration = 120.0 * ns
        dt = 2.0 * ns
        expected_n = abs(target_alpha) ** 2
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        results = []
        for n_cav in cutoff_list:
            model = DispersiveTransmonCavityModel(
                omega_c=GHz(5.0),
                omega_q=GHz(6.2),
                alpha=0.0,
                chi=0.0,
                kerr=0.0,
                n_cav=n_cav,
                n_tr=2,
            )
            frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
            pulses, drive_ops, _ = build_displacement_pulse(
                DisplacementGate(index=0, name="D(alpha)", re=float(np.real(target_alpha)), im=float(np.imag(target_alpha))),
                {"duration_displacement_s": displacement_duration},
            )
            compiled = SequenceCompiler(dt=dt).compile(pulses, t_end=displacement_duration + dt)
            result = simulate_sequence(
                model,
                compiled,
                model.basis_state(0, 0),
                drive_ops,
                config=SimulationConfig(frame=frame),
            )
            results.append((n_cav, mode_moments(result.final_state, "storage")["n"]))
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code("cutoffs = np.array([row[0] for row in results], dtype=int)\nobserved_n = np.array([row[1] for row in results], dtype=float)"),
            md("## 7. Running the Simulation"),
            code(
                """
        errors = observed_n - expected_n
        for cutoff, n_obs, err in zip(cutoffs, observed_n, errors, strict=True):
            print(f"n_cav = {cutoff:2d} -> <n> = {n_obs:.4f} (error = {err:+.4f})")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(cutoffs, observed_n, "o-", label="simulated")
        ax.axhline(expected_n, color="tab:red", linestyle="--", label="expected |alpha|^2")
        ax.set_xlabel("Cavity truncation n_cav")
        ax.set_ylabel(r"Final $\\langle n \\rangle$")
        ax.set_title("Truncation convergence for a displacement experiment")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe low-cutoff models underestimate the displaced state's photon number because too much of the coherent-state tail is chopped away. This is why convergence checks should be tied to the specific observable and state family you actually care about."),
            md("## 10. Exercises / Next Steps\n\n- Repeat the study with a larger target displacement and see how the required cutoff increases.\n- Add cavity loss and compare whether convergence becomes easier or harder for the same experiment duration.\n- Continue to Tutorial 26 for more frame and bookkeeping sanity checks."),
        ],
    )

    write_notebook(
        TUTORIALS / "21_building_sequences_from_gates_and_pulses.ipynb",
        [
            title_cell(
                21,
                "Building Sequences from Gates and Pulses",
                "Combine public pulse builders with `SequenceCompiler` to create a short multi-channel schedule and inspect the resulting sampled control waveforms.",
                "Tutorials 03, 04, and 09 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will build a displacement followed by two qubit rotations, compile the schedule onto a common time grid, and inspect the resulting channel data."),
            md("## 2. Physical Background\n\nThe public gate classes (`DisplacementGate`, `RotationGate`, and friends) are small API objects. The pulse builders turn them into control waveforms, and `SequenceCompiler` places everything onto one global timeline."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        displacement_duration = 120.0 * ns
        rotation_duration = 40.0 * ns
        dt = 2.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=10,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        disp_pulses, disp_drive_ops, _ = build_displacement_pulse(
            DisplacementGate(index=0, name="D(alpha)", re=0.8, im=0.0),
            {"duration_displacement_s": displacement_duration},
        )
        x90_pulses, x90_drive_ops, _ = build_rotation_pulse(
            RotationGate(index=1, name="x90", theta=np.pi / 2.0, phi=0.0),
            {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": 0.18},
        )
        y90_pulses, _, _ = build_rotation_pulse(
            RotationGate(index=2, name="y90", theta=np.pi / 2.0, phi=np.pi / 2.0),
            {"duration_rotation_s": rotation_duration, "rotation_sigma_fraction": 0.18},
        )
        x90 = x90_pulses[0]
        y90 = y90_pulses[0]
        pulse_sequence = [
            disp_pulses[0],
            Pulse(x90.channel, displacement_duration + 10.0 * ns, x90.duration, x90.envelope, amp=x90.amp, carrier=x90.carrier, phase=x90.phase, label=x90.label),
            Pulse(y90.channel, displacement_duration + rotation_duration + 20.0 * ns, y90.duration, y90.envelope, amp=y90.amp, carrier=y90.carrier, phase=y90.phase, label=y90.label),
        ]
        drive_ops = {"storage": "cavity", "qubit": "qubit"}
        t_end = displacement_duration + 2.0 * rotation_duration + 30.0 * ns
        compiled = SequenceCompiler(dt=dt).compile(pulse_sequence, t_end=t_end)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 0),
            drive_ops,
            config=SimulationConfig(frame=frame, max_step=dt),
        )
        print(f"Final qubit excitation = {final_expectation(result, 'P_e'):.4f}")
        print(f"Final cavity photon number = {mode_moments(result.final_state, 'storage')['n']:.4f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.8), sharex=True)
        axes[0].plot(compiled.tlist / ns, np.real(compiled.channels["storage"].distorted), label="storage channel")
        axes[0].set_ylabel("Re[drive]")
        axes[0].legend()
        axes[0].set_title("Compiled sequence waveforms")

        axes[1].plot(compiled.tlist / ns, np.real(compiled.channels["qubit"].distorted), label="qubit channel")
        axes[1].set_xlabel("Time [ns]")
        axes[1].set_ylabel("Re[drive]")
        axes[1].legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe gate objects are just the front door. The actual solver sees sampled complex envelopes on named channels, and the compiler is the layer that makes multi-channel timing explicit."),
            md("## 10. Exercises / Next Steps\n\n- Insert an explicit idle gap and compare the compiled waveforms.\n- Replace the manual pulse shifts with your own small scheduling helper if you need a repeated pattern.\n- Continue to Tutorial 22 to see how prepared sessions help with repeated execution."),
        ],
    )

    write_notebook(
        TUTORIALS / "22_parameter_sweeps_and_batch_simulation.ipynb",
        [
            title_cell(
                22,
                "Parameter Sweeps and Batch Simulation",
                "Use `prepare_simulation(...)` and `simulate_batch(...)` to reuse one compiled schedule across multiple initial states and compare the resulting observables.",
                "Tutorials 07 and 21 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will prepare one spectroscopy-style session and run it across several initial cavity Fock states without rebuilding the Hamiltonian or timeline each time."),
            md("## 2. Physical Background\n\nThe prepared-session path is useful when the model, drive mapping, and compiled schedule stay fixed while the initial state changes. This is common in scan-style workloads and small inner loops."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        initial_levels = [0, 1, 2, 3]
        dt = 4.0 * ns
        probe_duration = 1.0 * us
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=MHz(-2.2),
            kerr=0.0,
            n_cav=8,
            n_tr=2,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        target_detuning = manifold_transition_frequency(model, 0, frame=frame)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        probe = Pulse(
            "q",
            0.0,
            probe_duration,
            square_envelope,
            amp=MHz(0.08),
            carrier=carrier_for_transition_frequency(target_detuning),
            label="batch_probe",
        )
        compiled = SequenceCompiler(dt=dt).compile([probe], t_end=probe_duration + dt)
        session = prepare_simulation(
            model,
            compiled,
            {"q": "qubit"},
            config=SimulationConfig(frame=frame, max_step=dt),
        )
        initial_states = [model.basis_state(0, n) for n in initial_levels]
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        batch_results = simulate_batch(session, initial_states, max_workers=1)
        final_populations = [final_expectation(result, "P_e") for result in batch_results]
        for n, p_e in zip(initial_levels, final_populations, strict=True):
            print(f"initial cavity level n = {n}: final P_e = {p_e:.4f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.bar([str(n) for n in initial_levels], final_populations, color="tab:blue")
        ax.set_xlabel("Initial cavity Fock level")
        ax.set_ylabel(r"Final $P_e$")
        ax.set_title("Batch execution with one prepared spectroscopy session")
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nBecause the probe is resonant with the `n = 0` qubit line, the higher-Fock initial states respond less strongly. The key workflow point is that the simulation session is reused while only the initial state changes."),
            md("## 10. Exercises / Next Steps\n\n- Wrap the prepared-session pattern in a Python loop over several detunings to build a larger sweep efficiently.\n- Re-run the batch in a three-mode model and compare storage versus readout conditioning.\n- Continue to Tutorial 23 for fitting and convenience calibration-target helpers."),
        ],
    )

    write_notebook(
        TUTORIALS / "23_analysis_fitting_and_result_extraction.ipynb",
        [
            title_cell(
                23,
                "Analysis, Fitting, and Result Extraction",
                "Use the `cqed_sim.calibration_targets` helpers to generate fitted calibration summaries and inspect the returned `CalibrationResult` objects.",
                "Tutorials 09, 11, and 12 are recommended first.",
                "The `calibration_targets` module is a lightweight convenience layer that returns fitted curves and metadata; it is not a full pulse-level experimental loop.",
            ),
            md("## 1. Goal\n\nWe will call the public calibration-target helpers, inspect the fitted parameters and uncertainties, and visualize the returned raw data."),
            md("## 2. Physical Background\n\nA common workflow is to turn simulated or measured traces into a small set of fit parameters. `cqed_sim` exposes a convenience API for that pattern through `CalibrationResult` objects."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=MHz(-220.0),
            chi=MHz(-2.2),
            kerr=0.0,
            n_cav=8,
            n_tr=3,
        )

        spectroscopy_frequencies = np.linspace(model.omega_q - MHz(12.0), model.omega_q + MHz(12.0), 201)
        rabi_amplitudes = np.linspace(0.0, 1.4, 81)
        delays = np.linspace(0.0, 25.0 * us, 101)
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        spectroscopy = run_spectroscopy(model, spectroscopy_frequencies)
        rabi = run_rabi(model, rabi_amplitudes, duration=40.0 * ns, omega_scale=2.0 * np.pi * 12.0e6)
        t1 = run_t1(model, delays, t1=18.0 * us)
        ramsey = run_ramsey(model, delays, detuning=2.0 * np.pi * 0.6e6, t2_star=8.0 * us)
        echo = run_t2_echo(model, delays, t2_echo=14.0 * us)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        pi_amplitude = np.pi / (rabi.fitted_parameters["omega_scale"] * rabi.fitted_parameters["duration"])
        summary = {
            "omega_01_mhz": angular_to_mhz(spectroscopy.fitted_parameters["omega_01"]),
            "omega_12_mhz": angular_to_mhz(spectroscopy.fitted_parameters["omega_12"]),
            "omega_scale": rabi.fitted_parameters["omega_scale"],
            "pi_amplitude": pi_amplitude,
            "t1_us": t1.fitted_parameters["t1"] / us,
            "delta_omega_mhz": angular_to_mhz(ramsey.fitted_parameters["delta_omega"]),
            "t2_star_us": ramsey.fitted_parameters["t2_star"] / us,
            "t2_echo_us": echo.fitted_parameters["t2_echo"] / us,
        }
        summary
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        print("Spectroscopy fit:", spectroscopy.fitted_parameters)
        print("Rabi fit:", rabi.fitted_parameters)
        print("T1 fit:", t1.fitted_parameters)
        print("Ramsey fit:", ramsey.fitted_parameters)
        print("Echo fit:", echo.fitted_parameters)
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        axes[0].plot(
            (spectroscopy.raw_data["drive_frequencies"] - model.omega_q) / (2.0 * np.pi * 1.0e6),
            spectroscopy.raw_data["response"],
        )
        axes[0].set_xlabel("Drive frequency relative to bare omega_q [MHz]")
        axes[0].set_ylabel("Spectroscopy response")
        axes[0].set_title("Calibration-target spectroscopy trace")

        axes[1].plot(rabi.raw_data["amplitudes"], rabi.raw_data["excited_population"])
        axes[1].set_xlabel("Normalized drive amplitude")
        axes[1].set_ylabel(r"Excited population")
        axes[1].set_title("Calibration-target Rabi trace")
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe convenience fit objects are useful when you want a fast notebook summary or a compact developer-facing API. They are especially handy in workflow notebooks where the goal is to collect a few calibration targets instead of replaying every low-level pulse step. The spectroscopy helper reports absolute angular transition frequencies, while the Ramsey helper reports the fitted angular detuning under the explicit key `delta_omega`; the plot above shows the spectroscopy sweep re-expressed relative to the bare `omega_q` for readability."),
            md("## 10. Exercises / Next Steps\n\n- Compare these fitted outputs to the explicit pulse-level notebooks from Tutorials 09 through 13.\n- Use the returned `summary` dictionary as input to your own higher-level workflow code.\n- Continue to Tutorial 25 for a compact end-to-end calibration example."),
        ],
    )

    write_notebook(
        TUTORIALS / "24_sideband_like_interactions.ipynb",
        [
            title_cell(
                24,
                "Sideband-Like Interactions",
                "Use `SidebandDriveSpec` and `build_sideband_pulse(...)` to simulate an effective `|f,0> <-> |g,1>` exchange in a multilevel transmon-storage model.",
                "Tutorials 18 and 21 are recommended first.",
                "This notebook uses the repository's effective sideband-drive interface rather than a microscopic coupler model.",
            ),
            md("## 1. Goal\n\nWe will drive an effective red sideband and watch population transfer between `|f,0>` and `|g,1>`."),
            md("## 2. Physical Background\n\nThe structured `SidebandDriveSpec` API lets the runtime know which transmon transition and which bosonic mode participate in the effective sideband interaction. That is the stable public entry point for notebook-scale sideband studies."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        g_sb = 2.0 * np.pi * 8.0e6
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=MHz(-220.0),
            chi=MHz(-0.8),
            kerr=0.0,
            n_cav=4,
            n_tr=3,
        )
        frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
        target = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2, sideband="red")
        omega_sb = sideband_transition_frequency(model, cavity_level=0, lower_level=0, upper_level=2, frame=frame)
        t_swap = np.pi / (2.0 * g_sb)
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        pulses, drive_ops, meta = build_sideband_pulse(
            target,
            duration_s=t_swap,
            amplitude_rad_s=g_sb,
            channel="sb",
            carrier=carrier_for_transition_frequency(omega_sb),
            label="gf_red_sideband",
        )
        compiled = SequenceCompiler(dt=t_swap / 400.0).compile(pulses, t_end=t_swap)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(2, 0),
            drive_ops,
            config=SimulationConfig(frame=frame, store_states=True, max_step=t_swap / 400.0),
        )
        target_state = model.basis_state(0, 1)
        source_state = model.basis_state(2, 0)
        p_g1 = np.array([abs(target_state.overlap(state)) ** 2 for state in result.states], dtype=float)
        p_f0 = np.array([abs(source_state.overlap(state)) ** 2 for state in result.states], dtype=float)
        print(f"Maximum transfer to |g,1> = {np.max(p_g1):.4f}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, ax = plt.subplots()
        ax.plot(compiled.tlist / ns, p_f0, label=r"$P_{|f,0\\rangle}$")
        ax.plot(compiled.tlist / ns, p_g1, label=r"$P_{|g,1\\rangle}$")
        ax.axvline(t_swap / ns, color="tab:red", linestyle="--", label=r"$t_{swap}$")
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Population")
        ax.set_title("Effective sideband-driven excitation exchange")
        ax.legend()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe sideband drive does not look like a simple qubit-only rotation or a simple cavity displacement. Instead it moves amplitude between a multilevel transmon state and a bosonic Fock state according to the structured drive target."),
            md("## 10. Exercises / Next Steps\n\n- Change `lower_level` and `upper_level` in the `SidebandDriveSpec` and see how the addressed transition changes.\n- Add noise to study how open-system effects degrade the swap.\n- Continue to Tutorial 26 for common bookkeeping mistakes around frames and carriers."),
        ],
    )

    write_notebook(
        TUTORIALS / "25_small_calibration_workflow_end_to_end.ipynb",
        [
            title_cell(
                25,
                "Small Calibration Workflow End to End",
                "Collect a compact set of spectroscopy, Rabi, relaxation, and coherence estimates and assemble them into one notebook-scale calibration summary.",
                "Tutorials 09 through 13 and 23 are recommended first.",
                "This is a deliberately small, deterministic workflow. It is meant to show how the public helpers fit together, not to mimic a full lab automation stack.",
            ),
            md("## 1. Goal\n\nWe will build one compact calibration notebook that outputs a small parameter table a user could hand to a follow-up pulse-design or open-system simulation step."),
            md("## 2. Physical Background\n\nA realistic calibration loop touches multiple observables: line positions, control amplitudes, relaxation, and coherence. The workflow here keeps those pieces small enough to run comfortably inside one notebook."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.15),
            alpha=MHz(-220.0),
            chi=MHz(-2.0),
            kerr=0.0,
            n_cav=8,
            n_tr=3,
        )
        spectroscopy_frequencies = np.linspace(model.omega_q - MHz(12.0), model.omega_q + MHz(12.0), 201)
        rabi_amplitudes = np.linspace(0.0, 1.4, 81)
        delays = np.linspace(0.0, 25.0 * us, 101)
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        spectroscopy = run_spectroscopy(model, spectroscopy_frequencies)
        rabi = run_rabi(model, rabi_amplitudes, duration=40.0 * ns, omega_scale=2.0 * np.pi * 11.0e6)
        t1 = run_t1(model, delays, t1=19.0 * us)
        ramsey = run_ramsey(model, delays, detuning=2.0 * np.pi * 0.5e6, t2_star=8.5 * us)
        echo = run_t2_echo(model, delays, t2_echo=15.0 * us)
        pi_amplitude = np.pi / (rabi.fitted_parameters["omega_scale"] * rabi.fitted_parameters["duration"])
        calibration_summary = {
            "omega_01_hz": angular_to_hz(spectroscopy.fitted_parameters["omega_01"]),
            "omega_12_hz": angular_to_hz(spectroscopy.fitted_parameters["omega_12"]),
            "pi_amplitude": float(pi_amplitude),
            "pi_over_two_amplitude": float(0.5 * pi_amplitude),
            "t1_s": float(t1.fitted_parameters["t1"]),
            "t2_star_s": float(ramsey.fitted_parameters["t2_star"]),
            "t2_echo_s": float(echo.fitted_parameters["t2_echo"]),
            "ramsey_delta_omega_hz": angular_to_hz(ramsey.fitted_parameters["delta_omega"]),
        }
        calibration_summary
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code("report_rows = list(calibration_summary.items())"),
            md("## 7. Running the Simulation"),
            code(
                """
        for key, value in report_rows:
            print(f"{key:>22}: {value}")
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0))
        axes[0, 0].plot((spectroscopy.raw_data["drive_frequencies"] - model.omega_q) / (2.0 * np.pi * 1.0e6), spectroscopy.raw_data["response"])
        axes[0, 0].set_title("Spectroscopy")
        axes[0, 0].set_xlabel("Drive frequency relative to bare omega_q [MHz]")

        axes[0, 1].plot(rabi.raw_data["amplitudes"], rabi.raw_data["excited_population"])
        axes[0, 1].set_title("Power Rabi")
        axes[0, 1].set_xlabel("Normalized amplitude")

        axes[1, 0].plot(np.asarray(t1.raw_data["delays"], dtype=float) / us, t1.raw_data["excited_population"])
        axes[1, 0].set_title("T1")
        axes[1, 0].set_xlabel("Delay [us]")

        axes[1, 1].plot(np.asarray(ramsey.raw_data["delays"], dtype=float) / us, ramsey.raw_data["excited_population"], label="Ramsey")
        axes[1, 1].plot(np.asarray(echo.raw_data["delays"], dtype=float) / us, echo.raw_data["excited_population"], label="Echo")
        axes[1, 1].set_title("Coherence targets")
        axes[1, 1].set_xlabel("Delay [us]")
        axes[1, 1].legend()
        plt.tight_layout()
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThis notebook is intentionally small, but it already produces the ingredients needed for a practical simulation model: transition frequencies, pulse scale estimates, and open-system times. In a larger workflow, this dictionary would feed later pulse-construction or readout notebooks. The summary keeps the spectroscopy outputs as ordinary frequencies in hertz and converts the Ramsey fit's angular detuning `delta_omega` into hertz under the explicit key `ramsey_delta_omega_hz`."),
            md("## 10. Exercises / Next Steps\n\n- Export `calibration_summary` as JSON and reuse it in a follow-up simulation notebook.\n- Swap the convenience calibration-target helpers for explicit pulse-level notebooks when you want more physical detail.\n- Continue to Tutorial 26 for sanity checks before trusting the resulting numbers."),
        ],
    )

    write_notebook(
        TUTORIALS / "26_frame_sanity_checks_and_common_failure_modes.ipynb",
        [
            title_cell(
                26,
                "Frame Sanity Checks and Common Failure Modes",
                "Demonstrate a few of the easiest ways to misread `cqed_sim` results: using the raw carrier instead of the physical transition detuning, comparing lab and rotating frames without noticing, and trusting too-small cutoffs.",
                "Tutorials 02, 06, and 20 are recommended first.",
            ),
            md("## 1. Goal\n\nWe will compare correct and incorrect spectroscopy setup choices and summarize the kinds of mistakes that show up most often when debugging notebook results."),
            md("## 2. Physical Background\n\nMost cQED notebook mistakes are bookkeeping mistakes, not solver bugs. In this repository the two biggest ones are mislabeling the spectroscopy axis and forgetting which frame a transition frequency lives in."),
            md("## 3. Imports"),
            code(COMMON_IMPORTS),
            md("## 4. Simulation Parameters"),
            code(
                """
        detuning_mhz = np.linspace(-4.0, 4.0, 61)
        probe_duration = 0.8 * us
        probe_amplitude = MHz(0.08)
        dt = 4.0 * ns
        """
            ),
            md("## 5. Model Construction"),
            code(
                """
        model = DispersiveTransmonCavityModel(
            omega_c=GHz(5.0),
            omega_q=GHz(6.2),
            alpha=0.0,
            chi=0.0,
            kerr=0.0,
            n_cav=1,
            n_tr=2,
        )
        frame = FrameSpec(omega_q_frame=model.omega_q)
        spectrum_levels = min(6, int(np.prod(model.subsystem_dims)))
        """
            ),
            md("## 6. Pulse / Sequence Construction"),
            code(
                """
        correct_response = []
        wrong_response = []
        for point_mhz in detuning_mhz:
            correct_probe = Pulse("q", 0.0, probe_duration, square_envelope, amp=probe_amplitude, carrier=carrier_for_transition_frequency(MHz(point_mhz)), label="correct")
            wrong_probe = Pulse("q", 0.0, probe_duration, square_envelope, amp=probe_amplitude, carrier=MHz(point_mhz), label="wrong")
            correct_compiled = SequenceCompiler(dt=dt).compile([correct_probe], t_end=probe_duration + dt)
            wrong_compiled = SequenceCompiler(dt=dt).compile([wrong_probe], t_end=probe_duration + dt)
            correct_result = simulate_sequence(model, correct_compiled, model.basis_state(0, 0), {"q": "qubit"}, config=SimulationConfig(frame=frame, max_step=dt))
            wrong_result = simulate_sequence(model, wrong_compiled, model.basis_state(0, 0), {"q": "qubit"}, config=SimulationConfig(frame=frame, max_step=dt))
            correct_response.append(final_expectation(correct_result, "P_e"))
            wrong_response.append(final_expectation(wrong_result, "P_e"))
        correct_response = np.asarray(correct_response, dtype=float)
        wrong_response = np.asarray(wrong_response, dtype=float)
        matched_spectrum = compute_energy_spectrum(model, frame=frame, levels=spectrum_levels)
        lab_spectrum = compute_energy_spectrum(model, frame=FrameSpec(), levels=spectrum_levels)
        """
            ),
            md("## 7. Running the Simulation"),
            code(
                """
        correct_peak = float(detuning_mhz[int(np.argmax(correct_response))])
        wrong_peak = float(detuning_mhz[int(np.argmax(wrong_response))])
        print(f"Correct carrier mapping peak detuning [MHz]: {correct_peak:+.3f}")
        print(f"Wrong raw-carrier scan peak detuning [MHz]: {wrong_peak:+.3f}")
        print("Matched-frame low energies [MHz]:", np.round(matched_spectrum.energies[:4] / (2.0 * np.pi * 1.0e6), 4))
        print("Lab-frame low energies [MHz]:", np.round(lab_spectrum.energies[:4] / (2.0 * np.pi * 1.0e6), 4))
        """
            ),
            md("## 8. Visualizing the Results"),
            code(
                """
        fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))
        axes[0].plot(detuning_mhz, correct_response, label="correct transition-detuning scan")
        axes[0].plot(detuning_mhz, wrong_response, label="wrong raw-carrier scan")
        axes[0].set_xlabel("Intended detuning label [MHz]")
        axes[0].set_ylabel(r"Final $P_e$")
        axes[0].set_title("Carrier-sign mistake in spectroscopy")
        axes[0].legend()

        plot_energy_levels(lab_spectrum, max_levels=spectrum_levels, energy_scale=1.0 / (2.0 * np.pi * 1.0e6), energy_unit_label="MHz", title="Lab frame", ax=axes[1])
        plt.show()
        """
            ),
            md("## 9. Physical Interpretation\n\nThe raw-carrier scan is not just mislabeled; it moves the resonance to the wrong side of the axis because the waveform convention flips the sign. The frame-energy comparison is the other routine footgun: a matched rotating frame can make the same physical system look numerically tiny compared with the lab frame."),
            md("## 10. Exercises / Next Steps\n\n- Repeat the carrier-sign comparison in Tutorial 07 and watch the number-splitting lines mirror incorrectly.\n- Pair this notebook with Tutorial 20 whenever a result looks suspiciously truncation-dependent.\n- Use this notebook as a quick debugging checklist whenever a new workflow seems to disagree with intuition."),
        ],
    )


def main() -> None:
    generate_batch_1()
    generate_batch_2()
    generate_batch_3()
    print("Generated tutorials")


if __name__ == "__main__":
    main()
