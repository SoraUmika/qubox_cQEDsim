import json
import textwrap
from pathlib import Path


DEST = Path(__file__).resolve().with_name("sqr_calibration_workflow.ipynb")


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
        # SQR Calibration Workflow

        ## Section 0: Overview

        This notebook numerically calibrates one selected SQR gate from a decomposition JSON using a qubit-only conditional simulation per Fock manifold.

        The workflow is:

        1. Load and validate the decomposition JSON.
        2. Select one SQR gate by index or by name.
        3. Evaluate the uncalibrated per-manifold conditional process fidelity.
        4. Optimize per-manifold corrections `(d_lambda, d_alpha, d_omega)` with a two-stage optimizer.
        5. Plot the fitted corrections and export the result as JSON for reuse in Case D of `sequential_simulation.ipynb`.
        6. Benchmark guard-aware SQR calibration on seeded random targets across a sweep of pulse durations.
        """
    ),
    md_cell("## Section 1: Imports and Environment Checks"),
    code_cell(
        """
        from __future__ import annotations

        import importlib
        import importlib.metadata
        import json
        from pathlib import Path

        REQUIRED_HINTS = {
            "numpy": "pip install numpy",
            "matplotlib": "pip install matplotlib",
            "scipy": "pip install scipy",
            "qutip": "pip install qutip",
            "cqed_sim": "pip install -e .",
        }

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
        import scipy
        from IPython.display import Markdown, display

        from cqed_sim.calibration.sqr import (
            benchmark_random_sqr_targets_vs_duration,
            benchmark_results_table,
            calibration_cache_path,
            calibrate_sqr_gate,
            conditional_loss,
            evaluate_guarded_sqr_target,
            evaluate_sqr_gate_levels,
            export_calibration_result,
            extract_effective_qubit_unitary,
            extract_sqr_gates,
            generate_random_sqr_targets,
            load_or_calibrate_sqr_gate,
            select_sqr_gate,
            summarize_duration_benchmark,
            target_qubit_unitary,
        )
        from cqed_sim.io.gates import load_gate_sequence, render_gate_table
        from cqed_sim.plotting.calibration_plots import plot_sqr_calibration_result
        from cqed_sim.tests.test_sqr_calibration import run_sqr_calibration_sanity_suite

        def package_version(dist_name: str, default: str = "editable/local") -> str:
            try:
                return importlib.metadata.version(dist_name)
            except importlib.metadata.PackageNotFoundError:
                return default

        np.set_printoptions(precision=4, suppress=True)
        print("Versions:")
        print(f"  numpy   : {versions['numpy']}")
        print(f"  scipy   : {versions['scipy']}")
        print(f"  qutip   : {versions['qutip']}")
        print(f"  cqed_sim: {package_version('cqed-sim')}")
        """
    ),
    md_cell("## Section 2: Configuration"),
    code_cell(
        r"""
        CONFIG = {
          "json_path": r"C:\Users\jl82323\Box\Shyam Shankar Quantum Circuits Group\Users\Users_JianJun\JJL_Experiments\decomposition\cluster_U_T_1-1e+03ns-3_sqr-no_phases.josn",
          "json_fallback_path": "examples/sequences/sequential_demo.json",
          "cavity_fock_cutoff": 24,
          "initial_qubit": "g",
          "initial_cavity_kind": "fock",
          "initial_cavity_fock": 0,
          "initial_cavity_alpha": {"re": 0.0, "im": 0.0},
          "initial_cavity_amplitudes": None,

          "dt_s": 1.0e-9,
          "max_step_s": 1.0e-9,

          "duration_displacement_s": 48.0e-9,
          "duration_rotation_s": 16.0e-9,
          "duration_sqr_s": 1.0e-6,

          "rotation_sigma_fraction": 1.0 / 6.0,
          "sqr_sigma_fraction": 1.0 / 6.0,
          "sqr_theta_cutoff": 1.0e-10,

          "use_rotating_frame": True,

          "omega_c_hz": 0.0,
          "omega_q_hz": 0.0,
          "qubit_alpha_hz": 0.0,

          "st_chi_hz": -2840421.354241756,
          "st_chi2_hz": 0.0,
          "st_chi3_hz": 0.0,
          "st_K_hz": 0.0,
          "st_K2_hz": 0.0,

          "cavity_kappa_1_per_s": 0.0,

          "qb_T1_relax_ns": 98120.873848245112,
          "qb_T2_ramsey_ns": 63240.73112712837,
          "qb_T2_echo_ns": 80700.0,
          "t2_source": "ramsey",

          "max_n_cal": 8,
          "selected_sqr_index": 0,
          "selected_sqr_name": None,
          "optimizer_method_stage1": "Powell",
          "optimizer_method_stage2": "L-BFGS-B",
          "optimizer_maxiter_stage1": 40,
          "optimizer_maxiter_stage2": 60,
          "d_lambda_bounds": (-0.5, 0.5),
          "d_alpha_bounds": (-np.pi, np.pi),
          "d_omega_hz_bounds": (-2.0e6, 2.0e6),
          "regularization_lambda": 1.0e-6,
          "regularization_alpha": 1.0e-6,
          "regularization_omega": 1.0e-18,
          "calibration_cache_dir": "calibrations",
          "calibration_force_recompute": False,
          "qutip_nsteps_sqr_calibration": 100000,

          "benchmark_random_seed": 7,
          "benchmark_logical_n": 3,
          "benchmark_guard_levels": 1,
          "benchmark_targets_per_class": 1,
          "benchmark_target_classes": ("iid", "smooth", "sparse"),
          "benchmark_theta_max_rad": float(np.pi),
          "benchmark_duration_list_s": (2.5e-7, 7.5e-7, 1.0e-6),
          "benchmark_lambda_guard": 0.15,
          "benchmark_weight_mode": "uniform",
          "benchmark_poisson_alpha": None,
          "benchmark_fidelity_threshold": 0.99,
          "benchmark_guard_threshold": 1.0e-2,
          "benchmark_optimizer_maxiter_stage1": 5,
          "benchmark_optimizer_maxiter_stage2": 7,
          "benchmark_representative_target_index": 0,
          "benchmark_representative_duration_indices": (0, 1, 2),
          "benchmark_output_dir": "outputs/figures",
          "benchmark_export_path": "outputs/sqr_guard_benchmark_results.json",
        }
        CONFIG["n_cav_dim"] = int(CONFIG["cavity_fock_cutoff"]) + 1

        display(Markdown("Configured parameters:"))
        for key in sorted(CONFIG):
            print(f"{key:>32}: {CONFIG[key]}")
        """
    ),
    md_cell("## Section 3: Load Decomposition JSON and Select SQR Gate"),
    code_cell(
        """
        requested_path = Path(CONFIG["json_path"])
        fallback_path = Path(CONFIG["json_fallback_path"])
        selected_path = CONFIG["json_path"] if requested_path.exists() else CONFIG["json_fallback_path"]
        if not requested_path.exists():
            print("Configured JSON path was not found.")
            print(f"Requested: {requested_path}")
            print(f"Falling back to repo-local demo sequence: {fallback_path.resolve()}")

        GATE_PATH, GATES = load_gate_sequence(selected_path)
        print(f"Loaded {len(GATES)} gates from:\\n  {GATE_PATH}")
        render_gate_table(GATES, max_rows=20)

        SQR_GATES = extract_sqr_gates(GATES)
        if not SQR_GATES:
            raise RuntimeError("No SQR gates were found in the decomposition JSON.")

        SELECTED_SQR_GATE = select_sqr_gate(
            SQR_GATES,
            index=int(CONFIG["selected_sqr_index"]),
            name=CONFIG["selected_sqr_name"],
        )
        print("Selected SQR gate:", SELECTED_SQR_GATE)
        """
    ),
    md_cell("## Section 4: Physics Model and Pulse Model for SQR"),
    code_cell(
        """
        sigma_sqr_s = float(CONFIG["sqr_sigma_fraction"]) * float(CONFIG["duration_sqr_s"])
        max_n_target = min(
            int(CONFIG["max_n_cal"]),
            int(CONFIG["cavity_fock_cutoff"]),
            len(SELECTED_SQR_GATE.theta) - 1,
            len(SELECTED_SQR_GATE.phi) - 1,
        )

        print("SQR conditional model summary:")
        print(f"  selected gate      : {SELECTED_SQR_GATE.name}")
        print(f"  duration_sqr_s     : {CONFIG['duration_sqr_s']}")
        print(f"  sigma_sqr_s        : {sigma_sqr_s}")
        print(f"  calibrated max_n   : {max_n_target}")
        print("  Hamiltonian model  : H^(n) = 0.5 * Delta(n) * sigma_z + 0.5 * Omega(t) * [cos(phi) sigma_x + sin(phi) sigma_y] + 0.5 * d_omega * sigma_z")
        print("  Detuning convention: Delta(n) = 2*pi*(chi*n + chi2*n^2 + chi3*n^3)")
        print("  Benchmark model    : multitone off-resonant Gaussian drive evaluated per manifold with logical + guard levels.")
        """
    ),
    md_cell("## Section 5: Extract Effective Conditional Qubit Unitary per Fock Level"),
    code_cell(
        """
        INITIAL_LEVEL_EVAL = evaluate_sqr_gate_levels(SELECTED_SQR_GATE, CONFIG, corrections=None)
        print("Uncalibrated conditional process data:")
        for row in INITIAL_LEVEL_EVAL:
            print(
                {
                    "n": row["n"],
                    "theta_target": row["theta_target"],
                    "phi_target": row["phi_target"],
                    "process_fidelity": row["process_fidelity"],
                    "loss": row["loss"],
                    "base_amp_rad_s": row["base_amp_rad_s"],
                    "detuning_rad_s": row["detuning_rad_s"],
                }
            )
        """
    ),
    md_cell("## Section 6: Objective Function (match target $\\theta_n,\\phi_n$)"),
    code_cell(
        """
        active_levels = [row for row in INITIAL_LEVEL_EVAL if abs(row["theta_target"]) >= float(CONFIG["sqr_theta_cutoff"])]
        OBJECTIVE_PREVIEW_N = 0 if not active_levels else int(active_levels[0]["n"])
        OBJECTIVE_PREVIEW = conditional_loss(
            np.zeros(3, dtype=float),
            n=OBJECTIVE_PREVIEW_N,
            theta_target=float(INITIAL_LEVEL_EVAL[OBJECTIVE_PREVIEW_N]["theta_target"]),
            phi_target=float(INITIAL_LEVEL_EVAL[OBJECTIVE_PREVIEW_N]["phi_target"]),
            config=CONFIG,
        )
        print(
            {
                "preview_n": OBJECTIVE_PREVIEW_N,
                "initial_loss": OBJECTIVE_PREVIEW,
                "bounds": {
                    "d_lambda": CONFIG["d_lambda_bounds"],
                    "d_alpha": CONFIG["d_alpha_bounds"],
                    "d_omega_hz": CONFIG["d_omega_hz_bounds"],
                },
                "stage1": CONFIG["optimizer_method_stage1"],
                "stage2": CONFIG["optimizer_method_stage2"],
            }
        )
        print("Benchmark aggregate objective: L = (1 - F_logical) + lambda_guard * epsilon_guard")
        print("Per-manifold metric used in this notebook: unitary process fidelity |Tr(U_tgt^dagger U_sim)|^2 / 4.")
        """
    ),
    md_cell("## Section 7: Optimization Loop per Fock Level"),
    code_cell(
        """
        CALIBRATION_RESULT = load_or_calibrate_sqr_gate(
            SELECTED_SQR_GATE,
            CONFIG,
            cache_dir=CONFIG["calibration_cache_dir"],
        )
        print("Calibration summary:")
        print(CALIBRATION_RESULT.improvement_summary())
        for level in CALIBRATION_RESULT.levels:
            print(
                {
                    "n": level.n,
                    "skipped": level.skipped,
                    "initial_loss": level.initial_loss,
                    "optimized_loss": level.optimized_loss,
                    "optimized_params": level.optimized_params,
                }
            )
        """
    ),
    md_cell("## Section 8: Results Summary and Plots"),
    code_cell(
        """
        plot_sqr_calibration_result(CALIBRATION_RESULT)
        plt.show()

        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        n_values = np.arange(CALIBRATION_RESULT.max_n + 1, dtype=int)
        ax.semilogy(n_values, np.maximum(CALIBRATION_RESULT.initial_loss, 1.0e-16), "o--", label="Initial loss")
        ax.semilogy(n_values, np.maximum(CALIBRATION_RESULT.optimized_loss, 1.0e-16), "o-", label="Optimized loss")
        ax.set_xlabel("Fock level n")
        ax.set_ylabel("Process infidelity")
        ax.set_title(f"SQR calibration loss improvement for {CALIBRATION_RESULT.sqr_name}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        plt.show()
        """
    ),
    md_cell("## Section 9: Export Calibration Result"),
    code_cell(
        """
        EXPORT_PATH = export_calibration_result(CALIBRATION_RESULT, Path("sqr_calibration_result.json"), config=CONFIG)
        CACHE_PATH = calibration_cache_path(SELECTED_SQR_GATE, CONFIG, cache_dir=CONFIG["calibration_cache_dir"])
        print(f"Exported calibration result to: {EXPORT_PATH}")
        print(f"Cache file path: {CACHE_PATH}")
        """
    ),
    md_cell("## Section 10: Sanity Tests"),
    code_cell(
        """
        CAL_TEST_RESULTS = run_sqr_calibration_sanity_suite(CONFIG)
        for row in CAL_TEST_RESULTS:
            print(f"{row['status']}: {row['label']}")
        CAL_TEST_RESULTS
        """
    ),
    md_cell(
        """
        ## Benchmark: Random SQR Targets + Guard Levels + Fidelity vs Duration

        This section generates seeded random SQR targets on a logical subspace, appends identity guard levels, and benchmarks the same optimizer family across a duration sweep.

        Reported metrics:

        - per-manifold unitary process fidelity `F_proc^(n) = |Tr(U_tgt^\\dagger U_sim)|^2 / 4`
        - aggregate logical fidelity `F_logical`
        - guard leakage summary `epsilon_guard = max_n sqrt(X_n^2 + Y_n^2)` on guard levels
        - combined objective `L = (1 - F_logical) + lambda_guard * epsilon_guard`
        """
    ),
    code_cell(
        """
        BENCHMARK_CONFIG = dict(CONFIG)
        BENCHMARK_CONFIG["max_n_cal"] = min(
            int(CONFIG["benchmark_logical_n"]) + int(CONFIG["benchmark_guard_levels"]) - 1,
            int(CONFIG["cavity_fock_cutoff"]),
        )
        BENCHMARK_CONFIG["optimizer_maxiter_stage1"] = int(CONFIG["benchmark_optimizer_maxiter_stage1"])
        BENCHMARK_CONFIG["optimizer_maxiter_stage2"] = int(CONFIG["benchmark_optimizer_maxiter_stage2"])

        BENCHMARK_TARGETS = generate_random_sqr_targets(
            logical_n=int(CONFIG["benchmark_logical_n"]),
            guard_levels=int(CONFIG["benchmark_guard_levels"]),
            n_targets_per_class=int(CONFIG["benchmark_targets_per_class"]),
            seed=int(CONFIG["benchmark_random_seed"]),
            target_classes=CONFIG["benchmark_target_classes"],
            theta_max=float(CONFIG["benchmark_theta_max_rad"]),
        )
        BENCHMARK_RESULTS = benchmark_random_sqr_targets_vs_duration(
            BENCHMARK_CONFIG,
            CONFIG["benchmark_duration_list_s"],
            BENCHMARK_TARGETS,
            lambda_guard=float(CONFIG["benchmark_lambda_guard"]),
            weight_mode=str(CONFIG["benchmark_weight_mode"]),
            poisson_alpha=CONFIG["benchmark_poisson_alpha"],
            fidelity_threshold=float(CONFIG["benchmark_fidelity_threshold"]),
            guard_threshold=float(CONFIG["benchmark_guard_threshold"]),
        )
        BENCHMARK_TABLE = benchmark_results_table(BENCHMARK_RESULTS)
        BENCHMARK_SUMMARY = summarize_duration_benchmark(BENCHMARK_RESULTS)

        export_path = Path(CONFIG["benchmark_export_path"])
        export_path.parent.mkdir(parents=True, exist_ok=True)
        export_path.write_text(
            json.dumps(
                {
                    "config": {
                        "seed": int(CONFIG["benchmark_random_seed"]),
                        "logical_n": int(CONFIG["benchmark_logical_n"]),
                        "guard_levels": int(CONFIG["benchmark_guard_levels"]),
                        "duration_list_s": [float(x) for x in CONFIG["benchmark_duration_list_s"]],
                        "lambda_guard": float(CONFIG["benchmark_lambda_guard"]),
                        "weight_mode": str(CONFIG["benchmark_weight_mode"]),
                    },
                    "targets": [
                        {
                            "target_id": target.target_id,
                            "target_class": target.target_class,
                            "theta": list(target.theta),
                            "phi": list(target.phi),
                        }
                        for target in BENCHMARK_TARGETS
                    ],
                    "rows": BENCHMARK_TABLE,
                    "summary": BENCHMARK_SUMMARY,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Generated {len(BENCHMARK_TARGETS)} random targets.")
        for target in BENCHMARK_TARGETS:
            print(
                {
                    "target_id": target.target_id,
                    "class": target.target_class,
                    "logical_n": target.logical_n,
                    "guard_levels": target.guard_levels,
                    "theta": [round(value, 4) for value in target.theta],
                    "phi": [round(value, 4) for value in target.phi],
                }
            )
        print("\\nFirst benchmark rows:")
        for row in BENCHMARK_TABLE[: min(10, len(BENCHMARK_TABLE))]:
            print(row)

        CLASS_SUCCESS = {}
        for target_class in CONFIG["benchmark_target_classes"]:
            class_rows = [row for row in BENCHMARK_TABLE if row["target_class"] == target_class]
            CLASS_SUCCESS[str(target_class)] = {
                "n_trials": len(class_rows),
                "success_rate": float(np.mean([1.0 if row["success"] else 0.0 for row in class_rows])) if class_rows else float("nan"),
                "mean_logical_fidelity": float(np.mean([row["F_logical"] for row in class_rows])) if class_rows else float("nan"),
                "mean_guard": float(np.mean([row["epsilon_guard"] for row in class_rows])) if class_rows else float("nan"),
            }
        print("\\nSuccess summary by target class:")
        for key, value in CLASS_SUCCESS.items():
            print(key, value)
        print(f"Saved benchmark table to: {export_path}")
        """
    ),
    code_cell(
        """
        fig_dir = Path(CONFIG["benchmark_output_dir"])
        fig_dir.mkdir(parents=True, exist_ok=True)

        durations_us = np.asarray([row["T"] for row in BENCHMARK_SUMMARY], dtype=float) * 1.0e6
        f_median = np.asarray([row["f_median"] for row in BENCHMARK_SUMMARY], dtype=float)
        f_q25 = np.asarray([row["f_q25"] for row in BENCHMARK_SUMMARY], dtype=float)
        f_q75 = np.asarray([row["f_q75"] for row in BENCHMARK_SUMMARY], dtype=float)
        f_min = np.asarray([row["f_min"] for row in BENCHMARK_SUMMARY], dtype=float)
        f_max = np.asarray([row["f_max"] for row in BENCHMARK_SUMMARY], dtype=float)
        g_median = np.asarray([row["guard_median"] for row in BENCHMARK_SUMMARY], dtype=float)
        g_q25 = np.asarray([row["guard_q25"] for row in BENCHMARK_SUMMARY], dtype=float)
        g_q75 = np.asarray([row["guard_q75"] for row in BENCHMARK_SUMMARY], dtype=float)
        success = np.asarray([row["success_rate"] for row in BENCHMARK_SUMMARY], dtype=float)

        fig, axes = plt.subplots(3, 1, figsize=(10.5, 10.5), sharex=True)
        axes[0].plot(durations_us, f_median, "o-", label="Median")
        axes[0].fill_between(durations_us, f_q25, f_q75, alpha=0.25, label="25-75% band")
        axes[0].plot(durations_us, f_min, "--", color="tab:red", label="Worst")
        axes[0].plot(durations_us, f_max, "--", color="tab:green", label="Best")
        axes[0].set_ylabel(r"$F_{logical}$")
        axes[0].set_ylim(0.0, 1.02)
        axes[0].set_title(r"Logical fidelity vs SQR duration with guard-band optimization")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(durations_us, g_median, "o-", label="Median")
        axes[1].fill_between(durations_us, g_q25, g_q75, alpha=0.25, label="25-75% band")
        axes[1].set_ylabel(r"$\\epsilon_{guard}$")
        axes[1].set_title("Guard leakage summary vs SQR duration")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="best")

        axes[2].plot(durations_us, success, "o-", color="tab:purple")
        axes[2].set_xlabel("SQR duration [us]")
        axes[2].set_ylabel("Success rate")
        axes[2].set_ylim(-0.02, 1.02)
        axes[2].set_title(
            rf"Joint success probability vs duration ($F_{{logical}} \\geq {CONFIG['benchmark_fidelity_threshold']}$, "
            rf"$\\epsilon_{{guard}} \\leq {CONFIG['benchmark_guard_threshold']}$)"
        )
        axes[2].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "sqr_benchmark_duration_summary.png", dpi=160, bbox_inches="tight")
        plt.show()

        class_names = list(CLASS_SUCCESS)
        class_success = np.asarray([CLASS_SUCCESS[name]["success_rate"] for name in class_names], dtype=float)
        fig, ax = plt.subplots(figsize=(8.5, 4.4))
        ax.bar(class_names, class_success, color="tab:cyan")
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel("Success rate")
        ax.set_title("Success rate by random target class")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "sqr_benchmark_success_by_class.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    code_cell(
        """
        representative_target = BENCHMARK_TARGETS[int(CONFIG["benchmark_representative_target_index"])]
        duration_indices = [int(idx) for idx in CONFIG["benchmark_representative_duration_indices"]]
        representative_durations = [CONFIG["benchmark_duration_list_s"][idx] for idx in duration_indices if idx < len(CONFIG["benchmark_duration_list_s"])]
        representative_results = [
            result
            for result in BENCHMARK_RESULTS
            if result.target_id == representative_target.target_id and any(np.isclose(result.duration_s, duration) for duration in representative_durations)
        ]
        representative_results = sorted(representative_results, key=lambda row: row.duration_s)
        if not representative_results:
            raise RuntimeError("No representative benchmark results were found.")

        logical_n = int(representative_target.logical_n)
        total_n = int(representative_target.total_levels)
        n_logical = np.arange(logical_n, dtype=int)
        n_all = np.arange(total_n, dtype=int)

        fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True)
        axes[0].plot(n_logical, np.asarray(representative_target.theta[:logical_n], dtype=float), "k--", linewidth=2.0, label="Target")
        for result in representative_results:
            achieved_theta = [row["achieved_theta"] for row in result.per_manifold[:logical_n]]
            axes[0].plot(n_logical, achieved_theta, "o-", label=rf"{result.duration_s * 1e6:.2f} us")
        axes[0].set_ylabel(r"Achieved $\theta_n$")
        axes[0].set_title(f"Representative target {representative_target.target_id}: logical rotation angle vs duration")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="best")

        axes[1].plot(n_logical, np.asarray(representative_target.phi[:logical_n], dtype=float), "k--", linewidth=2.0, label="Target")
        for result in representative_results:
            achieved_phi = [row["achieved_phi"] for row in result.per_manifold[:logical_n]]
            axes[1].plot(n_logical, achieved_phi, "o-", label=rf"{result.duration_s * 1e6:.2f} us")
        axes[1].set_xlabel("Logical Fock level n")
        axes[1].set_ylabel(r"Achieved $\\phi_n$ [rad]")
        axes[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "sqr_benchmark_representative_angles.png", dpi=160, bbox_inches="tight")
        plt.show()

        fig, ax = plt.subplots(figsize=(10.5, 4.6))
        for result in representative_results:
            guard_xy = [
                row.get("guard_xy", 0.0)
                for row in result.per_manifold[logical_n:]
            ]
            ax.plot(n_all[logical_n:], guard_xy, "o-", label=rf"{result.duration_s * 1e6:.2f} us")
        ax.set_xlabel("Guard Fock level n")
        ax.set_ylabel(r"$\\epsilon^{(n)}_{guard,XY}$")
        ax.set_title(f"Representative target {representative_target.target_id}: guard leakage by duration")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / "sqr_benchmark_guard_leakage.png", dpi=160, bbox_inches="tight")
        plt.show()

        fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
        for result in representative_results:
            trace = result.convergence_trace
            iterations = np.asarray([row["iteration"] for row in trace], dtype=float)
            axes[0].plot(iterations, [row["best_logical_fidelity"] for row in trace], label=rf"{result.duration_s * 1e6:.2f} us")
            axes[1].plot(iterations, [row["best_epsilon_guard"] for row in trace], label=rf"{result.duration_s * 1e6:.2f} us")
            axes[2].plot(iterations, [row["best_loss_total"] for row in trace], label=rf"{result.duration_s * 1e6:.2f} us")
        axes[0].set_ylabel(r"$F_{logical}$")
        axes[1].set_ylabel(r"$\\epsilon_{guard}$")
        axes[2].set_ylabel(r"$\\mathcal{L}$")
        axes[2].set_xlabel("Objective evaluation index")
        axes[0].set_title(f"Representative target {representative_target.target_id}: convergence traces")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / "sqr_benchmark_convergence.png", dpi=160, bbox_inches="tight")
        plt.show()

        infidelity_matrix = []
        duration_labels = []
        for result in representative_results:
            infidelity_matrix.append([1.0 - row["process_fidelity"] for row in result.per_manifold])
            duration_labels.append(f"{result.duration_s * 1e6:.2f}")
        infidelity_matrix = np.asarray(infidelity_matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(10.5, 4.8))
        image = ax.imshow(infidelity_matrix, origin="lower", aspect="auto", cmap="magma")
        ax.set_xticks(np.arange(total_n, dtype=int))
        ax.set_xticklabels([str(n) for n in n_all])
        ax.set_yticks(np.arange(len(duration_labels), dtype=int))
        ax.set_yticklabels(duration_labels)
        ax.set_xlabel("Fock level n")
        ax.set_ylabel("Duration [us]")
        ax.set_title(f"Representative target {representative_target.target_id}: per-manifold infidelity heatmap")
        fig.colorbar(image, ax=ax, label=r"$1 - F^{(n)}_{proc}$")
        fig.tight_layout()
        fig.savefig(fig_dir / "sqr_benchmark_infidelity_heatmap.png", dpi=160, bbox_inches="tight")
        plt.show()
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
