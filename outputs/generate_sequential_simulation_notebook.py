import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sequential_simulation.ipynb"


def normalize(source: str) -> list[str]:
    text = textwrap.dedent(source).strip("\n")
    return [line + "\n" for line in text.splitlines()]


def md_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": normalize(source)}


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalize(source),
    }


cells: list[dict] = [
    md_cell(
        """
        ## Section 0: Overview and Usage

        This notebook implements three gate-by-gate simulations for the same validated JSON gate list:

        - **Case A**: instantaneous ideal gates.
        - **Case B**: pulse-level QuTiP dynamics with no dissipation.
        - **Case C**: pulse-level QuTiP dynamics with dissipation from `T1`, `T2`, and optional cavity loss.

        Usage:

        1. Edit the configuration in **Section 2**.
        2. Run the notebook top-to-bottom.
        3. Compare Bloch trajectories, compact Wigner snapshots, relative phases, and weakness metrics across A/B/C.

        Pulse mapping used in this notebook:

        - `Displacement` -> square cavity drive.
        - `Rotation` -> Gaussian qubit drive.
        - `SQR` -> simplified multitone Gaussian qubit drive using `cqed_sim` dispersive manifold frequencies.
        """
    ),
    md_cell(
        """
        ## Section 1: Imports and Environment Checks
        """
    ),
    code_cell(
        """
        from __future__ import annotations

        import copy
        import importlib
        import importlib.metadata

        REQUIRED_HINTS = {
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "qutip": "pip install qutip",
            "cqed_sim": "pip install -e .",
        }
        OPTIONAL_MODULES = ["ipywidgets"]

        missing = []
        versions = {}
        for module_name, hint in REQUIRED_HINTS.items():
            try:
                module = importlib.import_module(module_name)
                versions[module_name] = getattr(module, "__version__", "unknown")
            except ModuleNotFoundError as exc:
                missing.append((module_name, hint, exc))

        if missing:
            lines = ["Missing required notebook dependencies:"]
            for module_name, hint, exc in missing:
                lines.append(f"  - {module_name}: {exc}. Minimal install hint: {hint}")
            raise ModuleNotFoundError("\\n".join(lines))

        import matplotlib.pyplot as plt
        import numpy as np
        import qutip as qt
        from IPython.display import Markdown, display

        import cqed_sim
        from cqed_sim.io.gates import load_gate_sequence, render_gate_table
        from cqed_sim.observables.weakness import attach_weakness_metrics, comparison_metrics
        from cqed_sim.plotting.bloch_plots import plot_bloch_track
        from cqed_sim.plotting.phase_plots import plot_relative_phase_track
        from cqed_sim.plotting.weakness_plots import (
            plot_cavity_population_comparison,
            plot_component_comparison,
            plot_weakness,
            print_mapping_rows,
        )
        from cqed_sim.plotting.wigner_grids import plot_wigner_grid
        from cqed_sim.simulators.common import final_case_summary
        from cqed_sim.simulators.ideal import run_case_a
        from cqed_sim.simulators.pulse_open import run_case_c
        from cqed_sim.simulators.pulse_unitary import run_case_b
        from cqed_sim.tests.test_sanity import baseline_vs_refactor_sanity, run_notebook_sanity_suite

        def package_version(dist_name: str, default: str = "editable/local") -> str:
            try:
                return importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                return default

        optional_versions = {}
        for module_name in OPTIONAL_MODULES:
            try:
                module = importlib.import_module(module_name)
                optional_versions[module_name] = getattr(module, "__version__", "available")
            except ModuleNotFoundError:
                optional_versions[module_name] = "not installed"

        np.set_printoptions(precision=4, suppress=True)
        print("Required versions:")
        print(f"  numpy     : {versions['numpy']}")
        print(f"  matplotlib: {versions['matplotlib']}")
        print(f"  qutip     : {versions['qutip']}")
        print(f"  cqed_sim  : {package_version('cqed-sim')}")
        print("Optional:")
        for name, version in optional_versions.items():
            print(f"  {name:<10}: {version}")
        """
    ),
    md_cell(
        """
        ## Section 2: User Configuration
        """
    ),
    code_cell(
        r"""
        CONFIG = {
            "json_path": r"C:\Users\jl82323\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\JJL_Experiments\decomposition\cluster_U_T_1-1e+03ns-3_sqr-no_phases.josn",
            "cavity_fock_cutoff": 24,
            "initial_qubit": "g",
            "initial_cavity_kind": "fock",
            "initial_cavity_fock": 0,
            "initial_cavity_alpha": {"re": 0.0, "im": 0.0},
            "initial_cavity_amplitudes": None,
            "wigner_every_gate": True,
            "wigner_stride": 1,
            "wigner_points": 81,
            "wigner_extent": 4.0,
            "wigner_max_cols": 5,
            "top_axis_label_stride": 1,
            "summary_max_rows": 20,
            "phase_track_max_n": 2,
            "phase_reference_threshold": 1.0e-8,
            "phase_unwrap": False,
            "dt_s": 1.0e-9,
            "max_step_s": 1.0e-9,
            "duration_displacement_s": 32.0e-9,
            "duration_rotation_s": 64.0e-9,
            "duration_sqr_s": 1.0e-6,
            "rotation_sigma_fraction": 0.18,
            "sqr_sigma_fraction": 0.18,
            "sqr_theta_cutoff": 1.0e-10,
            "use_rotating_frame": True,
            "omega_c_hz": 0.0,
            "omega_q_hz": 0.0,
            "qubit_alpha_hz": 0.0,
            "st_chi_hz": -2840421.354241756,
            "st_chi2_hz": -21912.638362342423,
            "st_chi3_hz": -327.37857577643325,
            "st_K_hz": -28844.0,
            "st_K2_hz": 1406.0,
            "cavity_kappa_1_per_s": 0.0,
            "qb_T1_relax_ns": 9812.873848245112,
            "qb_T2_ramsey_ns": 6324.73112712837,
            "qb_T2_echo_ns": 8070.0,
            "t2_source": "ramsey",
        }
        CONFIG["n_cav_dim"] = int(CONFIG["cavity_fock_cutoff"]) + 1

        display(Markdown("Configured parameters:"))
        for key in sorted(CONFIG):
            print(f"{key:>24}: {CONFIG[key]}")
        """
    ),
    md_cell(
        """
        ## Section 3: Load and Validate JSON Gate List
        """
    ),
    code_cell(
        """
        GATE_PATH, GATES = load_gate_sequence(CONFIG["json_path"])
        print(f"Loaded {len(GATES)} gates from:\\n  {GATE_PATH}")
        render_gate_table(GATES, max_rows=int(CONFIG["summary_max_rows"]))
        """
    ),
    md_cell(
        """
        ## Section 4: Shared Operator Builders and Utilities
        """
    ),
    code_cell(
        """
        SHARED_API = [
            "run_case_a",
            "run_case_b",
            "run_case_c",
            "plot_bloch_track",
            "plot_wigner_grid",
            "plot_relative_phase_track",
            "plot_component_comparison",
            "plot_weakness",
            "run_notebook_sanity_suite",
        ]

        print("Notebook orchestration uses cqed_sim package helpers:")
        for name in SHARED_API:
            print(f"  - {name}")
        """
    ),
    md_cell(
        """
        ## Section 5: Case A --- Ideal Gate Simulation
        """
    ),
    code_cell(
        """
        CASE_A = run_case_a(GATES, CONFIG, case_label="Case A")

        print("Case A diagnostics:")
        print(
            {
                "solver": CASE_A["metadata"]["solver"],
                "final_x": CASE_A["x"][-1],
                "final_y": CASE_A["y"][-1],
                "final_z": CASE_A["z"][-1],
                "final_n": CASE_A["n"][-1],
            }
        )

        plot_bloch_track(
            CASE_A,
            title="Case A: ideal Bloch trajectory",
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()

        plot_wigner_grid(
            CASE_A,
            title="Case A: Wigner tomography by gate index",
            stride=int(CONFIG["wigner_stride"]),
            max_cols=int(CONFIG["wigner_max_cols"]),
        )
        plt.show()

        plot_relative_phase_track(
            CASE_A,
            max_n=int(CONFIG["phase_track_max_n"]),
            threshold=float(CONFIG["phase_reference_threshold"]),
            unwrap=bool(CONFIG["phase_unwrap"]),
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()
        """
    ),
    md_cell(
        """
        ## Section 6: Case B --- Pulse-level Simulation (No Dissipation)

        SQR note:

        `cqed_sim` provides dispersive manifold-frequency helpers, but not a hardware-calibrated selective-SQR compiler. This notebook therefore uses a simplified multitone Gaussian rotating-wave model:

        - one Gaussian-windowed tone per active Fock manifold,
        - tone frequencies from `cqed_sim.snap_opt.model.manifold_transition_frequency(...)`,
        - per-tone area calibration `theta_n ≈ 2 * ∫ Ω_n(t) dt`.
        """
    ),
    code_cell(
        """
        CASE_B = run_case_b(GATES, CONFIG, case_label="Case B")

        print("Case B diagnostics:")
        print(
            {
                "solver": CASE_B["metadata"]["solver"],
                "final_x": CASE_B["x"][-1],
                "final_y": CASE_B["y"][-1],
                "final_z": CASE_B["z"][-1],
                "final_n": CASE_B["n"][-1],
            }
        )
        print("Case B gate-to-pulse mapping:")
        print_mapping_rows(CASE_B)

        plot_bloch_track(
            CASE_B,
            title="Case B: pulse-level Bloch trajectory (no dissipation)",
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()

        plot_wigner_grid(
            CASE_B,
            title="Case B: pulse-level Wigner tomography by gate index",
            stride=int(CONFIG["wigner_stride"]),
            max_cols=int(CONFIG["wigner_max_cols"]),
        )
        plt.show()

        plot_relative_phase_track(
            CASE_B,
            max_n=int(CONFIG["phase_track_max_n"]),
            threshold=float(CONFIG["phase_reference_threshold"]),
            unwrap=bool(CONFIG["phase_unwrap"]),
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()
        """
    ),
    md_cell(
        """
        ## Section 7: Case C --- Pulse-level Simulation (With Dissipation)
        """
    ),
    code_cell(
        """
        CASE_C = run_case_c(GATES, CONFIG, case_label="Case C")

        print("Case C diagnostics:")
        print(
            {
                "solver": CASE_C["metadata"]["solver"],
                "final_x": CASE_C["x"][-1],
                "final_y": CASE_C["y"][-1],
                "final_z": CASE_C["z"][-1],
                "final_n": CASE_C["n"][-1],
            }
        )
        noise = CASE_C["metadata"]["noise"]
        print(
            {
                "t1_s": None if noise is None else noise.t1,
                "tphi_s": None if noise is None else noise.tphi,
                "kappa_1_per_s": None if noise is None else noise.kappa,
                "gamma1_1_per_s": None if noise is None else noise.gamma1,
                "gamma_phi_prefactor_1_per_s": None if noise is None else noise.gamma_phi,
            }
        )
        print("Case C gate-to-pulse mapping:")
        print_mapping_rows(CASE_C)

        plot_bloch_track(
            CASE_C,
            title="Case C: pulse-level Bloch trajectory (with dissipation)",
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()

        plot_wigner_grid(
            CASE_C,
            title="Case C: pulse-level Wigner tomography by gate index",
            stride=int(CONFIG["wigner_stride"]),
            max_cols=int(CONFIG["wigner_max_cols"]),
        )
        plt.show()

        plot_relative_phase_track(
            CASE_C,
            max_n=int(CONFIG["phase_track_max_n"]),
            threshold=float(CONFIG["phase_reference_threshold"]),
            unwrap=bool(CONFIG["phase_unwrap"]),
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()
        """
    ),
    md_cell(
        """
        ## Section 8: Weakness Metrics and Comparison Plots

        Definitions used here:

        - **Wigner negativity**: `0.5 * (integral |W| dx dp - 1)`.
        - **Fidelity-based weakness**: `1 - F(rho_case, rho_A)` referenced to Case A at the same gate index.
        """
    ),
    code_cell(
        """
        CASE_A = attach_weakness_metrics(CASE_A, CASE_A)
        CASE_B = attach_weakness_metrics(CASE_A, CASE_B)
        CASE_C = attach_weakness_metrics(CASE_A, CASE_C)

        plot_component_comparison(
            CASE_A,
            CASE_B,
            CASE_C,
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()

        plot_cavity_population_comparison(
            CASE_A,
            CASE_B,
            CASE_C,
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()

        plot_weakness(
            CASE_B,
            CASE_C,
            reference_track=CASE_A,
            label_stride=int(CONFIG["top_axis_label_stride"]),
        )
        plt.show()

        COMPARISON_AB = comparison_metrics(CASE_A, CASE_B)
        COMPARISON_AC = comparison_metrics(CASE_A, CASE_C)
        print("A vs B:", COMPARISON_AB)
        print("A vs C:", COMPARISON_AC)
        """
    ),
    md_cell(
        """
        ## Section 9: Test Suite
        """
    ),
    code_cell(
        """
        BASELINE_VS_REFACTOR = baseline_vs_refactor_sanity(CONFIG)
        print("Baseline vs refactor sanity:", BASELINE_VS_REFACTOR)

        TEST_RESULTS = run_notebook_sanity_suite(CONFIG)
        for row in TEST_RESULTS:
            print(f"{row['status']}: {row['label']}")

        TEST_RESULTS
        """
    ),
    md_cell(
        """
        ## Section 10: Final Summary Tables and Notes
        """
    ),
    code_cell(
        """
        FINAL_SUMMARIES = [final_case_summary(track) for track in (CASE_A, CASE_B, CASE_C)]
        print("Final case summaries:")
        for row in FINAL_SUMMARIES:
            print(row)

        print("\\nNotes:")
        print("- Case A is the instantaneous-unitary baseline.")
        print("- Case B and Case C use the same pulse shapes; Case C adds Lindblad dissipation.")
        print("- SQR in Cases B/C uses a simplified multitone Gaussian selective-drive approximation tied to cqed_sim dispersive manifold frequencies.")
        print("- Bloch plots intentionally keep gate-type labels only on the top x-axis to avoid duplication.")
        print("- The baseline-vs-refactor check compares the refactored Case A path against an independent direct-unitary reference.")
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


with open(DEST, "w", encoding="utf-8", newline="\n") as handle:
    handle.write(json.dumps(notebook, indent=1))
print(DEST)
