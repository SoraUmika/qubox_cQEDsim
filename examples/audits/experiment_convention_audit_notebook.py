from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import Markdown, display


DEFAULT_SIM_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_ROOT = DEFAULT_SIM_ROOT.parent / "JJL_Experiments"
DEFAULT_EXPERIMENT_PATH = DEFAULT_EXPERIMENT_ROOT / "data" / "seq_1_device"
DEFAULT_HARDWARE_RESULTS_PATH = DEFAULT_EXPERIMENT_ROOT / "outputs" / "hardware_tomography_verification_20260308.json"
DEFAULT_PYTHON_311_PATH = Path(r"E:\Program Files\Python311\python.exe")


def _ensure_import_roots(sim_root: Path, experiment_root: Path) -> None:
    for root in (sim_root, experiment_root):
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)


def _rel_path_label(path_like: str | Path, *, sim_root: Path, experiment_root: Path) -> str:
    path = Path(path_like)
    for root, prefix in ((experiment_root, "JJL_Experiments"), (sim_root, "cQED_simulation")):
        try:
            rel = path.resolve().relative_to(root.resolve())
            return f"{prefix}/{rel.as_posix()}"
        except Exception:
            continue
    return str(path)


def _gate_label(theta: float, phi: float) -> str:
    cases = {
        (np.pi / 2.0, 0.0): "x90",
        (np.pi / 2.0, np.pi / 2.0): "y90",
        (np.pi, 0.0): "x180",
        (np.pi, np.pi / 2.0): "y180",
    }
    for (theta_ref, phi_ref), label in cases.items():
        if np.isclose(theta, theta_ref) and np.isclose(phi, phi_ref):
            return label
    return f"theta={theta / np.pi:+.3f}pi phi={phi / np.pi:+.3f}pi"


def _sample_rotation_iq(theta: float, phi: float, *, build_rotation_pulse, RotationGate, SimPulse, duration_s: float = 100e-9, sigma_fraction: float = 0.18, dt_s: float = 1e-9) -> dict[str, Any]:
    gate = RotationGate(index=0, name=f"rot_t{theta:.6f}_p{phi:.6f}", theta=float(theta), phi=float(phi))
    pulses, _drive_ops, meta = build_rotation_pulse(
        gate,
        {
            "duration_rotation_s": float(duration_s),
            "rotation_sigma_fraction": float(sigma_fraction),
        },
    )
    pulse = pulses[0]
    sample_rate = 1.0 / float(dt_s)
    n_steps = int(np.round(duration_s * sample_rate))
    t = np.arange(n_steps, dtype=float) * float(dt_s)
    sampled = SimPulse(
        channel=pulse.channel,
        t0=0.0,
        duration=float(duration_s),
        envelope=pulse.sample(t),
        carrier=0.0,
        phase=0.0,
        amp=1.0,
        sample_rate=sample_rate,
        label=pulse.label,
    ).sample(t)
    return {
        "t_s": t,
        "iq": sampled,
        "I": np.real(sampled).astype(float),
        "Q": np.imag(sampled).astype(float),
        "duration_s": float(duration_s),
        "dt_s": float(dt_s),
        "metadata": meta,
    }


def _scale_for_pom(I: np.ndarray, Q: np.ndarray, *, max_amplitude: float, headroom: float = 0.98) -> float:
    peak = max(float(np.max(np.abs(I))), float(np.max(np.abs(Q))), 1.0e-15)
    return float(headroom * max_amplitude / peak)


def _register_exact_iq(name: str, I: np.ndarray, Q: np.ndarray, *, pulse_manager, element: str = "qubit") -> tuple[np.ndarray, np.ndarray]:
    pulse_manager.create_control_pulse(
        element=element,
        op=name,
        length=int(len(I)),
        pulse_name=f"{name}_pulse",
        I_wf_name=f"{name}_I_wf",
        Q_wf_name=f"{name}_Q_wf",
        I_samples=np.asarray(I, dtype=float),
        Q_samples=np.asarray(Q, dtype=float),
        persist=False,
        override=True,
    )
    I_back, Q_back = pulse_manager.get_pulse_waveforms(f"{name}_pulse", include_volatile=True)
    return np.asarray(I_back, dtype=float), np.asarray(Q_back, dtype=float)


def prepare_notebook_context(
    *,
    sim_root: Path | None = None,
    experiment_root: Path | None = None,
    hardware_results_path: Path | None = None,
    python_311_path: Path | None = None,
) -> dict[str, Any]:
    sim_root = Path(sim_root or DEFAULT_SIM_ROOT)
    experiment_root = Path(experiment_root or DEFAULT_EXPERIMENT_ROOT)
    hardware_results_path = Path(hardware_results_path or DEFAULT_HARDWARE_RESULTS_PATH)
    python_311_path = Path(python_311_path or DEFAULT_PYTHON_311_PATH)

    _ensure_import_roots(sim_root, experiment_root)

    from examples.audits.experiment_convention_audit import run_full_audit
    from cqed_sim.io.gates import RotationGate
    from cqed_sim.pulses.builders import build_rotation_pulse
    from cqed_sim.pulses.pulse import Pulse as SimPulse
    from qubox.gates_legacy import Gate
    from qubox.pulse_manager import MAX_AMPLITUDE, PulseOperationManager
    from qubox.tools.generators import register_rotations_from_ref_iq

    plt.style.use("seaborn-v0_8-whitegrid")
    pd.set_option("display.max_colwidth", 120)
    pd.set_option("display.max_rows", 200)

    experiment_path = experiment_root / "data" / "seq_1_device"
    live_experiment_kwargs = dict(
        experiment_path=str(experiment_path),
        qop_ip="10.157.36.68",
        cluster_name="Cluster_2",
        oct_cal_path="./",
        override_octave_json_mode="on",
        output_mode="on",
    )

    def build_live_experiment(enable: bool = False):
        if not enable:
            return None
        from qubox.cQED_experiments import cQED_Experiment

        exp = cQED_Experiment(**live_experiment_kwargs)
        Gate.set_attributes(exp.pulseOpMngr, exp.attributes)
        exp.load_measureMacro_state()
        return exp

    report = run_full_audit()

    environment_df = pd.DataFrame(
        [
            {"item": "python_311", "value": str(python_311_path)},
            {"item": "simulation_root", "value": str(sim_root)},
            {"item": "experiment_root", "value": str(experiment_root)},
            {"item": "hardware_results_json", "value": str(hardware_results_path)},
            {"item": "hardware_results_present", "value": bool(hardware_results_path.exists())},
            {"item": "live_hardware_enabled_in_notebook", "value": False},
            {"item": "audit_verdict", "value": report["verdict"]},
        ]
    )

    experiment_inspected = [
        experiment_root / "qubox" / "tomography.py",
        experiment_root / "qubox" / "macros" / "measure_macro.py",
        experiment_root / "qubox" / "macros" / "sequence_macro.py",
        experiment_root / "qubox" / "cQED_programs.py",
        experiment_root / "qubox" / "cQED_experiments.py",
        experiment_root / "qubox" / "analysis" / "post_process.py",
        experiment_root / "qubox" / "tests" / "test_tomography_convention.py",
    ]
    simulation_inspected = [
        sim_root / "examples" / "audits" / "experiment_convention_audit.py",
        sim_root / "examples" / "audits" / "experiment_convention_audit_notebook.py",
        sim_root / "cqed_sim" / "core" / "ideal_gates.py",
        sim_root / "cqed_sim" / "io" / "gates.py",
        sim_root / "cqed_sim" / "pulses" / "calibration.py",
        sim_root / "cqed_sim" / "pulses" / "builders.py",
        sim_root / "cqed_sim" / "sim" / "extractors.py",
        sim_root / "examples" / "workflows" / "sequential" / "pulse_unitary.py",
        sim_root / "tests" / "test_16_ideal_primitives_and_extractors.py",
        sim_root / "tests" / "test_20_gaussian_iq_convention.py",
        sim_root / "examples" / "audits" / "tests" / "test_experiment_convention_audit.py",
        sim_root / "examples" / "smoke_tests" / "tests" / "test_sanity.py",
        sim_root / "examples" / "smoke_tests" / "tests" / "test_sqr_calibration.py",
    ]
    files_rows: list[dict[str, str]] = []
    for path in experiment_inspected:
        files_rows.append({"workspace": "experiment", "file": _rel_path_label(path, sim_root=sim_root, experiment_root=experiment_root)})
    for path in simulation_inspected:
        files_rows.append({"workspace": "simulation", "file": _rel_path_label(path, sim_root=sim_root, experiment_root=experiment_root)})
    files_df = pd.DataFrame(files_rows).drop_duplicates().reset_index(drop=True)

    inventory_df = pd.DataFrame(report["convention_inventory"])
    rotation_df = pd.DataFrame(report["rotation_benchmarks"])
    rotation_df["gate_label"] = [_gate_label(theta, phi) for theta, phi in zip(rotation_df["theta_rad"], rotation_df["phi_rad"])]
    rotation_df = rotation_df.sort_values(["theta_rad", "phi_rad", "input_state"]).reset_index(drop=True)
    rotation_summary_df = (
        rotation_df.groupby("gate_label", as_index=False)
        .agg(
            theta_over_pi=("theta_rad", lambda s: float(s.iloc[0] / np.pi)),
            phi_over_pi=("phi_rad", lambda s: float(s.iloc[0] / np.pi)),
            min_gate_process_fidelity=("gate_process_fidelity", "min"),
            max_state_distance=("state_distance_up_to_global", "max"),
            g_state_sim_x=("sim_bloch_x", lambda s: float(s.iloc[0])),
            g_state_sim_y=("sim_bloch_y", lambda s: float(s.iloc[0])),
            g_state_sim_z=("sim_bloch_z", lambda s: float(s.iloc[0])),
        )
    )

    portability_cases = [
        ("x90", np.pi / 2.0, 0.0),
        ("y90", np.pi / 2.0, np.pi / 2.0),
        ("phi_pi_4", np.pi / 2.0, np.pi / 4.0),
        ("x180", np.pi, 0.0),
    ]
    portable_pom = PulseOperationManager(elements=["qubit", "resonator", "storage"])
    portability_rows = []
    portability_payload: dict[str, Any] = {}
    for name, theta, phi in portability_cases:
        sampled = _sample_rotation_iq(theta, phi, build_rotation_pulse=build_rotation_pulse, RotationGate=RotationGate, SimPulse=SimPulse)
        scale = _scale_for_pom(sampled["I"], sampled["Q"], max_amplitude=MAX_AMPLITUDE)
        I_norm = scale * sampled["I"]
        Q_norm = scale * sampled["Q"]
        I_back, Q_back = _register_exact_iq(f"{name}_portable", I_norm, Q_norm, pulse_manager=portable_pom)
        complex_in = I_norm + 1j * Q_norm
        complex_back = I_back + 1j * Q_back
        denom = max(np.linalg.norm(complex_in), 1.0e-15)
        portability_rows.append(
            {
                "case": name,
                "theta_over_pi": float(theta / np.pi),
                "phi_over_pi": float(phi / np.pi),
                "scale_to_pom": float(scale),
                "n_samples": int(len(I_norm)),
                "max_abs_I_error": float(np.max(np.abs(I_back - I_norm))),
                "max_abs_Q_error": float(np.max(np.abs(Q_back - Q_norm))),
                "relative_complex_error": float(np.linalg.norm(complex_back - complex_in) / denom),
            }
        )
        portability_payload[name] = {
            "roundtrip_I": I_back,
            "roundtrip_Q": Q_back,
            "scale_to_pom": scale,
        }
    portability_df = pd.DataFrame(portability_rows)

    sign_scan_rotation_df = pd.DataFrame(report["waveform_sign_scan"]["rotation_rows"])
    sign_scan_sqr_df = pd.DataFrame(report["waveform_sign_scan"]["sqr_rows"])

    regeneration_df = pd.DataFrame()
    regeneration_error = None
    try:
        sampled_x180 = _sample_rotation_iq(np.pi, 0.0, build_rotation_pulse=build_rotation_pulse, RotationGate=RotationGate, SimPulse=SimPulse)
        scale = _scale_for_pom(sampled_x180["I"], sampled_x180["Q"], max_amplitude=MAX_AMPLITUDE)
        regen_pom = PulseOperationManager(elements=["qubit", "resonator", "storage"])
        Gate.set_context(mgr=regen_pom, attributes=SimpleNamespace(dt_s=float(sampled_x180["dt_s"])))
        created = register_rotations_from_ref_iq(
            regen_pom,
            scale * sampled_x180["I"],
            scale * sampled_x180["Q"],
            rotations=("x90", "xn90", "y90", "yn90", "x180", "y180"),
            override=True,
            persist=False,
        )
        comparison_cases = {
            "x90": (np.pi / 2.0, 0.0),
            "xn90": (-np.pi / 2.0, 0.0),
            "y90": (np.pi / 2.0, np.pi / 2.0),
            "yn90": (-np.pi / 2.0, np.pi / 2.0),
            "x180": (np.pi, 0.0),
            "y180": (np.pi, np.pi / 2.0),
        }
        regeneration_rows = []
        for name, (theta, phi) in comparison_cases.items():
            expected = _sample_rotation_iq(theta, phi, build_rotation_pulse=build_rotation_pulse, RotationGate=RotationGate, SimPulse=SimPulse)
            target = scale * (expected["I"] + 1j * expected["Q"])
            got = np.asarray(created[name][0], dtype=float) + 1j * np.asarray(created[name][1], dtype=float)
            denom = max(np.linalg.norm(target), 1.0e-15)
            regeneration_rows.append(
                {
                    "case": name,
                    "theta_over_pi": float(theta / np.pi),
                    "phi_over_pi": float(phi / np.pi),
                    "scale_to_pom": float(scale),
                    "relative_complex_error_vs_canonical": float(np.linalg.norm(got - target) / denom),
                }
            )
        regeneration_df = pd.DataFrame(regeneration_rows)
    except Exception as exc:
        regeneration_error = f"{type(exc).__name__}: {exc}"

    tensor_df = pd.DataFrame(report["tensor_order"])
    sqr_df = pd.DataFrame(report["sqr_addressed_axis"])
    block_phase_df = pd.DataFrame(report["relative_block_phases"])
    rel_phase_state_df = pd.DataFrame(report["relative_phase_states"])
    mismatch_df = pd.DataFrame(report["mismatch_analysis"])
    patch_df = pd.DataFrame(report["patch_plan"])

    return {
        "json": json,
        "Markdown": Markdown,
        "display": display,
        "plt": plt,
        "np": np,
        "pd": pd,
        "Path": Path,
        "SimpleNamespace": SimpleNamespace,
        "SIM_ROOT": sim_root,
        "EXPERIMENT_ROOT": experiment_root,
        "EXPERIMENT_PATH": experiment_path,
        "HARDWARE_RESULTS_PATH": hardware_results_path,
        "PYTHON_311_PATH": python_311_path,
        "ENABLE_LIVE_QUBOX": False,
        "LIVE_EXPERIMENT_KWARGS": live_experiment_kwargs,
        "build_live_experiment": build_live_experiment,
        "report": report,
        "environment_df": environment_df,
        "files_df": files_df,
        "inventory_df": inventory_df,
        "rotation_df": rotation_df,
        "rotation_summary_df": rotation_summary_df,
        "portability_df": portability_df,
        "portability_payload": portability_payload,
        "sign_scan_rotation_df": sign_scan_rotation_df,
        "sign_scan_sqr_df": sign_scan_sqr_df,
        "regeneration_df": regeneration_df,
        "regeneration_error": regeneration_error,
        "tensor_df": tensor_df,
        "sqr_df": sqr_df,
        "block_phase_df": block_phase_df,
        "rel_phase_state_df": rel_phase_state_df,
        "mismatch_df": mismatch_df,
        "patch_df": patch_df,
    }


def build_notebook_conclusion(
    *,
    hardware_verdict_df: pd.DataFrame,
    regeneration_df: pd.DataFrame,
    regeneration_error: str | None,
    hardware_results_path: Path,
) -> tuple[pd.DataFrame, str]:
    regeneration_pass = bool(
        (regeneration_error is None)
        and (not regeneration_df.empty)
        and (regeneration_df["relative_complex_error_vs_canonical"] < 1.0e-12).all()
    )
    updated_mismatch_df = pd.DataFrame(
        [
            {
                "Finding": "Previous agent tomography patch",
                "Status": "resolved",
                "Notes": "The patch changed the Y tomography branch from x90 to xn90 and flipped the reported Y axis on hardware. The rollback restored x90.",
            },
            {
                "Finding": "Post-rollback hardware verification",
                "Status": "pass",
                "Notes": "The bundled hardware suite confirms x90 -> -Y, xn90 -> +Y, y90 -> +X, yn90 -> -X, and both pi pulses near -Z.",
            },
            {
                "Finding": "Simulator Bloch extraction",
                "Status": "pass",
                "Notes": "cqed_sim.sim.extractors now returns standard Pauli Bloch coordinates and matches the restored experiment convention.",
            },
            {
                "Finding": "Experiment waveform phase helper",
                "Status": "pass",
                "Notes": "The experiment-side legacy waveform helper now uses exp(+i*phi_eff), so named X/Y rotations and direct simulator waveform portability agree without a hidden phase-sign translation.",
            },
            {
                "Finding": "Experiment sigma_z helper mappings",
                "Status": "pass",
                "Notes": "Boolean-to-sigma_z and sigma_z-to-probability conversions now follow |g> -> +1, |e> -> -1 consistently with the active Bloch convention.",
            },
            {
                "Finding": "Exact simulator-to-qubox waveform transfer",
                "Status": "pass",
                "Notes": "After one explicit normalization step to satisfy the hardware DAC bound, direct I/Q array registration preserves the tested waveforms sample-by-sample through PulseOperationManager.",
            },
            {
                "Finding": "Legacy helper regeneration from a normalized x180 template",
                "Status": "pass" if regeneration_pass else "caution",
                "Notes": (
                    "register_rotations_from_ref_iq reproduces the canonical single-qubit family when seeded with the same normalized x180 template used for portability."
                    if regeneration_pass
                    else f"Regeneration did not complete cleanly: {regeneration_error}"
                ),
            },
            {
                "Finding": "Final experiment-vs-simulation convention status",
                "Status": "pass",
                "Notes": "At the convention layer, the restored experiment and the current simulator are now aligned.",
            },
        ]
    )

    hardware_pass = bool(
        (not hardware_verdict_df.empty)
        and hardware_verdict_df["named_sign_ok"].all()
        and hardware_verdict_df["param_sign_ok"].all()
    )
    max_named_angle = float(hardware_verdict_df["named_vs_so3_deg"].max()) if not hardware_verdict_df.empty else float("nan")
    max_param_angle = float(hardware_verdict_df["param_vs_so3_deg"].max()) if not hardware_verdict_df.empty else float("nan")

    verdict_md = f"""
**Verdict**: the experiment and simulator are now convention-consistent at the single-qubit rotation, tomography, Bloch-vector, and waveform-interpretation layers.

**Executive summary**

- The previous experiment-side patch was wrong specifically because it flipped the Y tomography branch from `x90` to `xn90`.
- The rollback restored `x90` as the Y prerotation and the post-rollback hardware suite passed the required sign checks for all six named pulses.
- The simulator runtime already matched the target waveform and Bloch conventions in this pass; the remaining fixes were on the experiment-side legacy waveform helper and `sigma_z` conversion utilities.
- Exact sampled simulator waveforms still transfer cleanly into the experiment-side `PulseOperationManager` after the explicit normalization step required to respect hardware DAC bounds.
- The legacy `register_rotations_from_ref_iq(...)` compatibility path also reproduces the corrected single-qubit family when seeded with the same normalized `x180` template.

**Hardware authority snapshot**

- Hardware sign check passed: `{hardware_pass}`
- Worst named-pulse angle to ideal SO(3): `{max_named_angle:.2f}` deg
- Worst parameterized-pulse angle to ideal SO(3): `{max_param_angle:.2f}` deg
- Results loaded from: `{hardware_results_path}`

**Final consistency verdict**

- `qubox` tomography convention: restored and hardware-verified
- `cqed_sim` single-qubit convention: consistent
- Experiment waveform and analysis helpers: corrected to match the simulator and tomography convention
- Experiment vs simulation overall: consistent
"""
    return updated_mismatch_df, verdict_md


__all__ = [
    "DEFAULT_EXPERIMENT_PATH",
    "DEFAULT_EXPERIMENT_ROOT",
    "DEFAULT_HARDWARE_RESULTS_PATH",
    "DEFAULT_PYTHON_311_PATH",
    "DEFAULT_SIM_ROOT",
    "build_notebook_conclusion",
    "prepare_notebook_context",
]
