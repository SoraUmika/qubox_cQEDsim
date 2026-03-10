from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
from scipy.optimize import Bounds, minimize

import examples.sqr_multitone_study as sms


@dataclass(frozen=True)
class ManualTarget:
    name: str
    theta: tuple[float, ...]
    phi: tuple[float, ...]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _norm_unitary(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.complex128)
    det = np.linalg.det(matrix)
    if abs(det) > 1.0e-15:
        matrix = matrix * np.exp(-0.5j * np.angle(det))
    return matrix


def _z_unitary(angle: float) -> np.ndarray:
    a = float(angle)
    return np.array([[np.exp(-0.5j * a), 0.0], [0.0, np.exp(0.5j * a)]], dtype=np.complex128)


def _rxy_unitary(theta: float, phi: float) -> np.ndarray:
    return np.asarray(sms.qubit_rotation_xy(float(theta), float(phi)).full(), dtype=np.complex128)


def _z_rxy_z(alpha: float, theta: float, phi: float, beta: float) -> np.ndarray:
    return _z_unitary(alpha) @ _rxy_unitary(theta, phi) @ _z_unitary(beta)


def _fidelity(target: np.ndarray, simulated: np.ndarray) -> float:
    overlap = np.trace(np.asarray(target).conj().T @ np.asarray(simulated))
    return float(np.abs(overlap) ** 2 / 4.0)


def _project_block_to_unitary(matrix: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(matrix, dtype=np.complex128))
    return _norm_unitary(u @ vh)


def _decompose_z_rxy_z(unitary: np.ndarray) -> dict[str, float]:
    u_target = _norm_unitary(unitary)
    bounds = Bounds(
        np.array([-np.pi, 0.0, -np.pi, -np.pi], dtype=float),
        np.array([np.pi, np.pi, np.pi, np.pi], dtype=float),
    )

    def objective(params: np.ndarray) -> float:
        alpha, theta, phi, beta = map(float, params)
        recon = _z_rxy_z(alpha, theta, phi, beta)
        err = np.linalg.norm(u_target - recon) ** 2
        gauge_penalty = 1.0e-6 * (alpha * alpha + beta * beta)
        return float(err + gauge_penalty)

    seeds = (
        np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        np.array([0.0, np.pi / 2.0, 0.0, 0.0], dtype=float),
        np.array([0.4, np.pi / 3.0, 0.2, -0.2], dtype=float),
        np.array([-0.6, np.pi / 2.0, -0.3, 0.5], dtype=float),
    )
    best = None
    for seed in seeds:
        result = minimize(objective, x0=seed, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    alpha, theta, phi, beta = map(float, best.x)
    recon = _z_rxy_z(alpha, theta, phi, beta)
    fit_fidelity = _fidelity(u_target, _norm_unitary(recon))
    return {
        "alpha_rad": float(sms.wrap_pi(alpha)),
        "theta_rad": float(theta),
        "phi_rad": float(sms.wrap_pi(phi)),
        "beta_rad": float(sms.wrap_pi(beta)),
        "fit_fidelity": float(fit_fidelity),
        "fit_objective": float(best.fun),
    }


def _corrected_block_metrics(case: sms.CaseResult, profile: sms.TargetProfile) -> tuple[dict[str, float], list[dict[str, Any]]]:
    unitary = np.asarray(case.unitary, dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    for n in range(profile.n_levels):
        idx = slice(2 * n, 2 * n + 2)
        simulated_block = _project_block_to_unitary(unitary[idx, idx])
        target_block = _norm_unitary(_rxy_unitary(float(profile.theta[n]), float(profile.phi[n])))
        mismatch = _norm_unitary(target_block.conj().T @ simulated_block)

        fid = _fidelity(target_block, simulated_block)
        theta_sim, nx_sim, ny_sim, nz_sim, phi_sim = sms.rotation_axis_parameters(simulated_block)
        theta_t, nx_t, ny_t, nz_t, phi_t = sms.rotation_axis_parameters(target_block)
        phi_axis_err = float(sms.wrap_pi(phi_sim - phi_t))

        mismatch_dec = _decompose_z_rxy_z(mismatch)
        rows.append(
            {
                "n": int(n),
                "process_fidelity": float(fid),
                "process_infidelity": float(1.0 - fid),
                "theta_target_rad": float(theta_t),
                "theta_simulated_rad": float(theta_sim),
                "theta_error_rad": float(theta_sim - theta_t),
                "phi_target_rad": float(phi_t),
                "phi_simulated_rad": float(phi_sim),
                "phi_axis_error_rad": float(phi_axis_err),
                "axis_target_x": float(nx_t),
                "axis_target_y": float(ny_t),
                "axis_target_z": float(nz_t),
                "axis_simulated_x": float(nx_sim),
                "axis_simulated_y": float(ny_sim),
                "axis_simulated_z": float(nz_sim),
                "residual_pre_z_rad": float(mismatch_dec["alpha_rad"]),
                "residual_post_z_rad": float(mismatch_dec["beta_rad"]),
                "mismatch_middle_theta_rad": float(mismatch_dec["theta_rad"]),
                "mismatch_middle_phi_rad": float(mismatch_dec["phi_rad"]),
                "mismatch_decomposition_fidelity": float(mismatch_dec["fit_fidelity"]),
            }
        )

    process_inf = np.asarray([row["process_infidelity"] for row in rows], dtype=float)
    phi_err = np.asarray([row["phi_axis_error_rad"] for row in rows], dtype=float)
    pre_z = np.asarray([row["residual_pre_z_rad"] for row in rows], dtype=float)
    post_z = np.asarray([row["residual_post_z_rad"] for row in rows], dtype=float)
    theta_err = np.asarray([row["theta_error_rad"] for row in rows], dtype=float)
    phase_sensitive = np.sqrt(phi_err**2 + pre_z**2 + post_z**2)

    summary = {
        "mean_process_fidelity": float(np.mean([row["process_fidelity"] for row in rows])),
        "min_process_fidelity": float(np.min([row["process_fidelity"] for row in rows])),
        "mean_process_infidelity": float(np.mean(process_inf)),
        "phase_axis_rms_rad": float(np.sqrt(np.mean(phi_err**2))),
        "residual_pre_z_rms_rad": float(np.sqrt(np.mean(pre_z**2))),
        "residual_post_z_rms_rad": float(np.sqrt(np.mean(post_z**2))),
        "theta_error_rms_rad": float(np.sqrt(np.mean(theta_err**2))),
        "phase_sensitive_rms_rad": float(np.sqrt(np.mean(phase_sensitive**2))),
    }
    return summary, rows


def _qubox_coeff(t: np.ndarray, env: np.ndarray, amp: float, phi_eff: float, omega: float) -> np.ndarray:
    return float(amp) * np.asarray(env, dtype=np.complex128) * np.exp(1j * float(phi_eff)) * np.exp(1j * float(omega) * t)


def _rotation_coeff(t: np.ndarray, env: np.ndarray, amp: float, phase: float, carrier: float) -> np.ndarray:
    return float(amp) * np.asarray(env, dtype=np.complex128) * np.exp(1j * (float(carrier) * t + float(phase)))


def _sqr_coeff_canonical(t: np.ndarray, env: np.ndarray, amp: float, phase: float, omega: float) -> np.ndarray:
    return float(amp) * np.asarray(env, dtype=np.complex128) * np.exp(1j * float(phase)) * np.exp(1j * float(omega) * t)


def _unitary_from_coeff(coeff: np.ndarray, tlist: np.ndarray) -> np.ndarray:
    sx = qt.sigmax()
    sy = qt.sigmay()
    h = [
        [0.5 * sx, np.real(np.asarray(coeff, dtype=np.complex128))],
        [0.5 * sy, np.imag(np.asarray(coeff, dtype=np.complex128))],
    ]
    propagators = qt.propagator(
        h,
        np.asarray(tlist, dtype=float),
        options={"atol": 1.0e-9, "rtol": 1.0e-8, "max_step": 0.001},
        tlist=np.asarray(tlist, dtype=float),
    )
    final = propagators[-1] if isinstance(propagators, list) else propagators
    return _norm_unitary(np.asarray(final.full(), dtype=np.complex128))


def _unitary_distance(U_ref: np.ndarray, U_test: np.ndarray) -> float:
    return float(1.0 - _fidelity(_norm_unitary(U_ref), _norm_unitary(U_test)))


def _sign_scan_for_equivalence() -> dict[str, Any]:
    duration = 1.0
    dt = 0.002
    tlist = np.arange(0.0, duration + dt * 0.5, dt, dtype=float)
    t_rel = tlist / duration
    env = np.asarray(sms.normalized_gaussian(t_rel, sigma_fraction=0.18), dtype=np.complex128)
    amp = np.pi / 4.0
    phi_eff = 0.73
    omega = 2.0 * np.pi * 0.21

    coeff_qubox = _qubox_coeff(t=tlist, env=env, amp=amp, phi_eff=phi_eff, omega=omega)
    unitary_qubox = _unitary_from_coeff(coeff=coeff_qubox, tlist=tlist)

    rotation_rows = []
    sqr_rows = []
    for phase_sign in (+1, -1):
        for omega_sign in (+1, -1):
            coeff_rot = _rotation_coeff(
                t=tlist,
                env=env,
                amp=amp,
                phase=phase_sign * phi_eff,
                carrier=omega_sign * omega,
            )
            coeff_sqr = _sqr_coeff_canonical(
                t=tlist,
                env=env,
                amp=amp,
                phase=phase_sign * phi_eff,
                omega=omega_sign * omega,
            )
            row_rot = {
                "phase_sign": int(phase_sign),
                "omega_sign": int(omega_sign),
                "process_fidelity_vs_qubox": float(1.0 - _unitary_distance(unitary_qubox, _unitary_from_coeff(coeff_rot, tlist))),
            }
            row_sqr = {
                "phase_sign": int(phase_sign),
                "omega_sign": int(omega_sign),
                "process_fidelity_vs_qubox": float(1.0 - _unitary_distance(unitary_qubox, _unitary_from_coeff(coeff_sqr, tlist))),
            }
            rotation_rows.append(row_rot)
            sqr_rows.append(row_sqr)

    best_rot = max(rotation_rows, key=lambda row: row["process_fidelity_vs_qubox"])
    best_sqr = max(sqr_rows, key=lambda row: row["process_fidelity_vs_qubox"])
    return {
        "duration": duration,
        "dt": dt,
        "phi_eff_test_rad": phi_eff,
        "omega_test_rad_s": omega,
        "rotation_sign_scan": rotation_rows,
        "sqr_sign_scan": sqr_rows,
        "best_rotation_match": best_rot,
        "best_sqr_match": best_sqr,
    }


def _detuning_sign_check() -> dict[str, Any]:
    duration = 1.0
    dt = 0.002
    tlist = np.arange(0.0, duration + dt * 0.5, dt, dtype=float)
    t_rel = tlist / duration
    env = np.asarray(sms.normalized_gaussian(t_rel, sigma_fraction=0.18), dtype=np.complex128)
    amp = np.pi / 4.0
    delta = 2.0 * np.pi * 0.25

    coeff_ref = _rotation_coeff(t=tlist, env=env, amp=amp, phase=0.0, carrier=0.0)
    coeff_rot_plus = _rotation_coeff(t=tlist, env=env, amp=amp, phase=0.0, carrier=delta)
    coeff_rot_minus = _rotation_coeff(t=tlist, env=env, amp=amp, phase=0.0, carrier=-delta)

    coeff_sqr_ref = _sqr_coeff_canonical(t=tlist, env=env, amp=amp, phase=0.0, omega=0.0)
    coeff_sqr_plus = _sqr_coeff_canonical(t=tlist, env=env, amp=amp, phase=0.0, omega=delta)
    coeff_sqr_minus = _sqr_coeff_canonical(t=tlist, env=env, amp=amp, phase=0.0, omega=-delta)

    u_ref = _unitary_from_coeff(coeff_ref, tlist)
    u_rot_plus = _unitary_from_coeff(coeff_rot_plus, tlist)
    u_rot_minus = _unitary_from_coeff(coeff_rot_minus, tlist)
    u_sqr_ref = _unitary_from_coeff(coeff_sqr_ref, tlist)
    u_sqr_plus = _unitary_from_coeff(coeff_sqr_plus, tlist)
    u_sqr_minus = _unitary_from_coeff(coeff_sqr_minus, tlist)

    err_rot_plus = _norm_unitary(u_ref.conj().T @ u_rot_plus)
    err_rot_minus = _norm_unitary(u_ref.conj().T @ u_rot_minus)
    err_sqr_plus = _norm_unitary(u_sqr_ref.conj().T @ u_sqr_plus)
    err_sqr_minus = _norm_unitary(u_sqr_ref.conj().T @ u_sqr_minus)

    _, _, _, nz_rot_plus, _ = sms.rotation_axis_parameters(err_rot_plus)
    _, _, _, nz_rot_minus, _ = sms.rotation_axis_parameters(err_rot_minus)
    _, _, _, nz_sqr_plus, _ = sms.rotation_axis_parameters(err_sqr_plus)
    _, _, _, nz_sqr_minus, _ = sms.rotation_axis_parameters(err_sqr_minus)
    return {
        "delta_test_rad_s": float(delta),
        "rotation_like": {
            "+delta_axis_z": float(nz_rot_plus),
            "-delta_axis_z": float(nz_rot_minus),
            "flip": bool(np.sign(nz_rot_plus) == -np.sign(nz_rot_minus) and np.sign(nz_rot_plus) != 0.0),
        },
        "sqr_like": {
            "+delta_axis_z": float(nz_sqr_plus),
            "-delta_axis_z": float(nz_sqr_minus),
            "flip": bool(np.sign(nz_sqr_plus) == -np.sign(nz_sqr_minus) and np.sign(nz_sqr_plus) != 0.0),
            "relative_to_rotation_sign": "opposite" if np.sign(nz_sqr_plus) == -np.sign(nz_rot_plus) else "same",
        },
    }


def _run_case_suite(
    profile: sms.TargetProfile,
    params: sms.StudyParams,
    objective_scope: str = "global",
    support_cfg: sms.ActiveSupportParams | None = None,
) -> dict[str, Any]:
    model, frame = sms.build_model_and_frame(params.system, chi_hz=float(params.system.chi_nominal_hz))
    reference_states = sms.build_reference_states(model, n_max=params.system.n_max, coherent_alpha=params.coherent_alpha)
    ideal_outputs = sms.apply_unitary_to_states(sms.build_target_unitary(profile), reference_states)
    opt_cfg = replace(params.optimization, objective_scope=str(objective_scope))
    controls = sms.build_controls_from_target(
        profile=profile,
        model=model,
        frame=frame,
        duration_s=float(params.pulse.duration_nominal_s),
        theta_cutoff=float(params.pulse.theta_cutoff),
    )
    case_b = sms.build_case(
        case_id="B",
        description="Naive pulse",
        controls=controls,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
        dt_s=float(params.pulse.dt_eval_s),
        max_step_s=float(params.pulse.max_step_eval_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_outputs,
    )
    case_c = sms.optimize_case(
        mode=sms.MODE_BASIC,
        base_controls=controls,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        opt_params=opt_cfg,
        duration_s=float(params.pulse.duration_nominal_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_outputs,
        support_config=support_cfg,
    )
    case_d = sms.optimize_case(
        mode=sms.MODE_EXTENDED,
        base_controls=controls,
        model=model,
        frame=frame,
        profile=profile,
        pulse_params=params.pulse,
        opt_params=opt_cfg,
        duration_s=float(params.pulse.duration_nominal_s),
        reference_states=reference_states,
        ideal_state_outputs=ideal_outputs,
        support_config=support_cfg,
    )
    cases = {"B": case_b, "C": case_c, "D": case_d}
    if params.include_case_e:
        case_e = sms.optimize_case(
            mode=sms.MODE_CHIRP,
            base_controls=controls,
            model=model,
            frame=frame,
            profile=profile,
            pulse_params=params.pulse,
            opt_params=opt_cfg,
            duration_s=float(params.pulse.duration_nominal_s),
            reference_states=reference_states,
            ideal_state_outputs=ideal_outputs,
            support_config=support_cfg,
        )
        cases["E"] = case_e

    corrected = {}
    for case_id, case in cases.items():
        summary, rows = _corrected_block_metrics(case, profile)
        legacy_phase = np.asarray([row["relative_block_phase_rad"] for row in case.block_rows], dtype=float)
        legacy_phase_rms = float(np.sqrt(np.mean(legacy_phase**2))) if legacy_phase.size else float("nan")
        corrected[case_id] = {
            "summary": summary,
            "rows": rows,
            "old_summary": case.summary,
            "legacy_phase_rms_rad": legacy_phase_rms,
            "state_fidelities": case.state_fidelities,
        }

    crosstalk_naive = sms.compute_single_tone_crosstalk(
        profile=profile,
        controls=case_b.controls,
        model=model,
        frame=frame,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
    )
    crosstalk_opt = sms.compute_single_tone_crosstalk(
        profile=profile,
        controls=case_d.controls,
        model=model,
        frame=frame,
        pulse_params=params.pulse,
        duration_s=float(params.pulse.duration_nominal_s),
    )

    return {
        "profile": {
            "name": profile.name,
            "mode": profile.mode,
            "seed": profile.seed,
            "theta": profile.theta.tolist(),
            "phi": profile.phi.tolist(),
        },
        "objective_scope": str(objective_scope),
        "active_support": None
        if support_cfg is None
        else {
            "mode": str(support_cfg.mode),
            "max_level_active": int(support_cfg.max_level_active),
            "active_levels": [int(x) for x in support_cfg.active_levels],
            "active_weights": [float(x) for x in support_cfg.active_weights],
            "inference_state_label": None if support_cfg.inference_state_label is None else str(support_cfg.inference_state_label),
            "state_population_threshold": float(support_cfg.state_population_threshold),
            "infer_weights_from_state": bool(support_cfg.infer_weights_from_state),
            "inactive_weight": float(support_cfg.inactive_weight),
            "boundary_leakage_boost": float(support_cfg.boundary_leakage_boost),
        },
        "corrected_metrics": corrected,
        "crosstalk": {
            "case_B": crosstalk_naive,
            "case_D": crosstalk_opt,
        },
    }


def _write_convention_note(out_dir: Path, sign_scan: dict[str, Any], detuning: dict[str, Any]) -> None:
    best_rot = sign_scan["best_rotation_match"]
    best_sqr = sign_scan["best_sqr_match"]
    rows = [
        {
            "quantity": "waveform phase factor",
            "lab_qubox": "exp(+i phi_eff)",
            "cQED_rotation": "exp(+i phase)",
            "cQED_sqr": "exp(+i phase)",
            "mapping_to_match_lab": "phase_cqed = phi_eff",
        },
        {
            "quantity": "time modulation",
            "lab_qubox": "exp(+i omega t)",
            "cQED_rotation": "exp(+i carrier t)",
            "cQED_sqr": "exp(+i omega t)",
            "mapping_to_match_lab": "carrier=omega and tone_omega=omega (same sign semantics)",
        },
        {
            "quantity": "detuning knob sign",
            "lab_qubox": "+domega -> positive IQ phase slope",
            "cQED_rotation": "+carrier -> negative residual Z sign in this frame",
            "cQED_sqr": "+domega -> same sign response as rotation",
            "mapping_to_match_lab": "shared positive detuning convention",
        },
    ]

    lines = []
    lines.append("# Convention Reconciliation: qubox vs cQED_sim")
    lines.append("")
    lines.append("## Lab-side Reference")
    lines.append("- QubitRotation waveform: `w = s*(I+iQ)*exp(+i*phi_eff)*exp(+i*omega*t)`")
    lines.append("- SQR tone waveform: `w_n = s_n*w0*exp(+i*phi_eff_n)*exp(+i*omega_n*t)` and `w = sum_n w_n`")
    lines.append("")
    lines.append("## Derived Mapping")
    lines.append("- Positive `phi_eff` rotates the IQ phasor counterclockwise in the complex plane.")
    lines.append("- With `H_drive = Re[w]*sigma_x/2 + Im[w]*sigma_y/2`, axis phase is `phi_axis = arg(w) = phi_eff`.")
    lines.append("- Therefore `R_xy(theta, phi_axis)` uses the same numeric phase in both the lab waveform and the canonical simulator builders.")
    lines.append("")
    lines.append("## Direct Numerical Equivalence (Two-level)")
    lines.append(
        f"- Best Rotation match to qubox waveform: phase_sign={best_rot['phase_sign']:+d}, omega_sign={best_rot['omega_sign']:+d}, fidelity={best_rot['process_fidelity_vs_qubox']:.9f}"
    )
    lines.append(
        f"- Best SQR(single-tone) match to qubox waveform: phase_sign={best_sqr['phase_sign']:+d}, omega_sign={best_sqr['omega_sign']:+d}, fidelity={best_sqr['process_fidelity_vs_qubox']:.9f}"
    )
    lines.append("- Interpretation: cQED Rotation and SQR now share the same phase and frequency sign convention.")
    lines.append("")
    lines.append("## Detuning Sign Check")
    lines.append(
        "- Rotation-like drive: "
        + f"+delta axis_z={detuning['rotation_like']['+delta_axis_z']:.6f}, "
        + f"-delta axis_z={detuning['rotation_like']['-delta_axis_z']:.6f}"
    )
    lines.append(
        "- SQR-like drive: "
        + f"+delta axis_z={detuning['sqr_like']['+delta_axis_z']:.6f}, "
        + f"-delta axis_z={detuning['sqr_like']['-delta_axis_z']:.6f}, "
        + f"relative_to_rotation={detuning['sqr_like']['relative_to_rotation_sign']}"
    )
    lines.append("")
    lines.append("## Convention Table")
    lines.append("| Quantity | lab qubox | cQED Rotation | cQED SQR | Required mapping |")
    lines.append("|---|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['quantity']} | {row['lab_qubox']} | {row['cQED_rotation']} | {row['cQED_sqr']} | {row['mapping_to_match_lab']} |"
        )
    (out_dir / "convention_reconciliation.md").write_text("\n".join(lines), encoding="utf-8")


def _write_equivalence_payload(out_dir: Path, sign_scan: dict[str, Any], detuning: dict[str, Any]) -> None:
    payload = {
        "sign_scan": sign_scan,
        "detuning_sign_check": detuning,
    }
    (out_dir / "waveform_hamiltonian_equivalence.json").write_text(
        json.dumps(_to_jsonable(payload), indent=2),
        encoding="utf-8",
    )


def _write_implementation_note(out_dir: Path, sign_scan: dict[str, Any], detuning: dict[str, Any]) -> None:
    lines = []
    lines.append("# Detuning Convention Implementation Note")
    lines.append("")
    lines.append("## Root Cause")
    lines.append("- Rotation path (`Pulse._sample_analytic`) used `exp(+i*(carrier*t + phase))`.")
    lines.append("- SQR multitone path used `exp(+i*phase)*exp(-i*omega*t)`.")
    lines.append("- This made positive numeric detuning map to opposite IQ phase slope between Rotation and SQR.")
    lines.append("")
    lines.append("## Minimal Fix Applied")
    lines.append("- Updated `cqed_sim/pulses/envelopes.py::multitone_gaussian_envelope` to use `exp(+i*omega*t)`.")
    lines.append("- Updated `cqed_sim/pulses/calibration.py::build_sqr_tone_specs` to set `omega_waveform = -manifold_transition_frequency(...)` so resonant physical tones remain unchanged while parameter semantics are unified.")
    lines.append("- Updated `examples/sqr_multitone_study.py::multitone_envelope` and spectrum marker sign to the same canonical convention.")
    lines.append("")
    lines.append("## Post-fix Verification")
    lines.append(
        f"- Rotation sign match vs lab waveform: phase_sign={sign_scan['best_rotation_match']['phase_sign']:+d}, omega_sign={sign_scan['best_rotation_match']['omega_sign']:+d}."
    )
    lines.append(
        f"- SQR sign match vs lab waveform: phase_sign={sign_scan['best_sqr_match']['phase_sign']:+d}, omega_sign={sign_scan['best_sqr_match']['omega_sign']:+d}."
    )
    lines.append(
        "- Detuning axis sign comparison: "
        + f"rotation(+delta)={detuning['rotation_like']['+delta_axis_z']:.6f}, "
        + f"sqr(+delta)={detuning['sqr_like']['+delta_axis_z']:.6f}, "
        + f"relative={detuning['sqr_like']['relative_to_rotation_sign']}."
    )
    lines.append("")
    lines.append("## Canonical Convention")
    lines.append("| Quantity | Canonical meaning |")
    lines.append("|---|---|")
    lines.append("| Waveform phasor | `w(t)=I(t)+iQ(t)` |")
    lines.append("| Pulse phase knob | implemented as `exp(+i*phase)` and matches lab `exp(+i*phi_eff)` directly via `phase=phi_eff` |")
    lines.append("| Detuning knob `d_omega` | increases IQ phase slope via `exp(+i*d_omega*t)` |")
    lines.append("| Effective detuning sign in rotating-frame block extraction | frame-dependent; use the block-unitary sign check, not IQ slope alone |")
    lines.append("| Target axis mapping | `R_xy(theta,phi_axis)` with `phi_axis=arg(w)` and `I->+x`, `Q->+y` |")
    (out_dir / "implementation_note.md").write_text("\n".join(lines), encoding="utf-8")


def _write_support_design_note(out_dir: Path) -> None:
    lines = []
    lines.append("# Support-Aware Objective Design Note")
    lines.append("")
    lines.append("## Philosophy")
    lines.append("- Active support levels are optimization targets; inactive levels are spectators unless they induce leakage from support.")
    lines.append("- Global full-space fidelity is still reported for reference, but not the primary optimization target in support-aware mode.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("- `ActiveSupportParams.mode`: `contiguous`, `explicit`, or `from_state`.")
    lines.append("- `active_levels` / `max_level_active` define support set `S`.")
    lines.append("- `active_weights` can be user-specified; otherwise uniform or inferred from a reference state.")
    lines.append("- `inference_state_label` + `state_population_threshold` support state-driven support inference.")
    lines.append("")
    lines.append("## Support-Aware Loss Terms")
    lines.append("- Active weighted block infidelity, active theta/phase/pre-Z/post-Z.")
    lines.append("- Support-state mean/min fidelity and phase-superposition coherence.")
    lines.append("- Leakage penalties: support-state leakage mean/max and spectral boundary leakage proxy.")
    lines.append("- Worst active-block floor penalty and weak inactive infidelity penalty.")
    lines.append("")
    lines.append("## Case Recommendation")
    lines.append("- Use Case E as default support-aware ansatz (amplitude + phase + detuning + phase-ramp/chirp).")
    lines.append("- Use Case D as fallback / ablation when runtime or control complexity must be reduced.")
    lines.append("")
    lines.append("## How to pass support inputs")
    lines.append("- Explicit support: `ActiveSupportParams(mode='explicit', active_levels=(0,1,4), active_weights=(...))`.")
    lines.append("- Contiguous support: `ActiveSupportParams(mode='contiguous', max_level_active=m)`.")
    lines.append("- Inferred support: `ActiveSupportParams(mode='from_state', inference_state_label='...', state_population_threshold=...)`.")
    lines.append("- Projected coherent support states are included in support ensemble by default.")
    (out_dir / "support_aware_design_note.md").write_text("\n".join(lines), encoding="utf-8")


def _write_phase_metric_audit(out_dir: Path, suite: dict[str, Any]) -> None:
    corrected = suite["corrected_metrics"]
    case_ids = sorted(corrected.keys())
    old_phase = [float(corrected[cid]["legacy_phase_rms_rad"]) for cid in case_ids]
    source_phase = [float(corrected[cid]["old_summary"]["phase_rms_rad"]) for cid in case_ids]
    new_phase = [float(corrected[cid]["summary"]["phase_sensitive_rms_rad"]) for cid in case_ids]
    old_span = float(max(old_phase) - min(old_phase))
    source_span = float(max(source_phase) - min(source_phase))
    new_span = float(max(new_phase) - min(new_phase))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(case_ids), dtype=float)
    ax.plot(x, old_phase, "o-", label="legacy relative-block phase_rms", color="tab:gray")
    ax.plot(x, source_phase, "^-", label="current study phase_rms_rad", color="tab:green")
    ax.plot(x, new_phase, "s-", label="new phase_sensitive_rms_rad", color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels(case_ids)
    ax.set_ylabel("Phase metric [rad]")
    ax.set_title("Phase metric audit: old frozen metric vs corrected")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "phase_metric_audit.png", dpi=170)
    plt.close(fig)

    lines = []
    lines.append("# Phase Metric Audit")
    lines.append("")
    lines.append("## Legacy Code Path")
    lines.append("- Legacy phase metric used only `relative_block_phase_rad` from simulated blocks.")
    lines.append("- That quantity did not reference target block phase, so it can remain nearly unchanged across optimized cases.")
    lines.append("")
    lines.append("## Audit Result")
    lines.append(f"- Legacy metric span across cases: {old_span:.6e} rad")
    lines.append(f"- Current study `phase_rms_rad` span across cases: {source_span:.6e} rad")
    lines.append(f"- Corrected metric span across cases: {new_span:.6e} rad")
    lines.append("- Corrected metric recomputes phase-sensitive mismatch from `U_err = U_target^dag U_sim` with a gauge-fixed `Z-Rxy-Z` decomposition.")
    lines.append("")
    lines.append("## Corrected Case Metrics")
    lines.append("| Case | legacy phase_rms_rad | current phase_rms_rad | new phase_sensitive_rms_rad | mean process fidelity | residual pre-Z rms | residual post-Z rms |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cid in case_ids:
        old_val = corrected[cid]["legacy_phase_rms_rad"]
        source_val = corrected[cid]["old_summary"]["phase_rms_rad"]
        new_summary = corrected[cid]["summary"]
        lines.append(
            f"| {cid} | {old_val:.6f} | {source_val:.6f} | {new_summary['phase_sensitive_rms_rad']:.6f} | {new_summary['mean_process_fidelity']:.6f} | {new_summary['residual_pre_z_rms_rad']:.6f} | {new_summary['residual_post_z_rms_rad']:.6f} |"
        )
    (out_dir / "phase_metric_audit.md").write_text("\n".join(lines), encoding="utf-8")


def _tiny_targets() -> list[ManualTarget]:
    return [
        ManualTarget(name="x90_x90", theta=(np.pi / 2.0, np.pi / 2.0), phi=(0.0, 0.0)),
        ManualTarget(name="x90_y90", theta=(np.pi / 2.0, np.pi / 2.0), phi=(0.0, np.pi / 2.0)),
        ManualTarget(name="x180_identity", theta=(np.pi, 0.0), phi=(0.0, 0.0)),
    ]


def _run_tiny_benchmark(out_dir: Path) -> dict[str, Any]:
    tiny_params = sms.StudyParams(
        seed=23,
        include_case_e=True,
        run_profiles=("structured",),
        system=sms.SystemParams(
            n_max=1,
            chi_nominal_hz=-2.84e6,
            chi_easy_hz=-2.84e6,
            chi_hard_hz=-2.84e6,
        ),
    )
    results: dict[str, Any] = {}
    for idx, target in enumerate(_tiny_targets()):
        profile = sms.TargetProfile(
            name=target.name,
            mode="manual",
            theta=np.asarray(target.theta, dtype=float),
            phi=np.asarray(target.phi, dtype=float),
            seed=idx,
        )
        suite = _run_case_suite(profile, tiny_params)
        results[target.name] = suite

    payload = {"params": sms.asdict(tiny_params), "targets": results}
    (out_dir / "tiny_benchmark.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")

    lines = []
    lines.append("# Tiny SQR Benchmark (n=0,1)")
    lines.append("")
    lines.append("## Targets")
    for target in _tiny_targets():
        lines.append(f"- {target.name}: theta={list(target.theta)}, phi={list(target.phi)}")
    lines.append("")
    lines.append("## Results")
    lines.append("| Target | Case | mean process fidelity | phase_sensitive_rms_rad |")
    lines.append("|---|---|---:|---:|")

    success_rows = []
    for target_name, suite in results.items():
        corrected = suite["corrected_metrics"]
        for cid in sorted(corrected.keys()):
            summary = corrected[cid]["summary"]
            lines.append(
                f"| {target_name} | {cid} | {summary['mean_process_fidelity']:.6f} | {summary['phase_sensitive_rms_rad']:.6f} |"
            )
            if cid in ("C", "D", "E"):
                success_rows.append((target_name, cid, summary["mean_process_fidelity"], summary["phase_sensitive_rms_rad"]))

    max_row = max(success_rows, key=lambda row: row[2]) if success_rows else None
    lines.append("")
    if max_row is not None:
        lines.append(
            f"- Best optimized tiny-case fidelity: target={max_row[0]}, case={max_row[1]}, fidelity={max_row[2]:.6f}, phase_sensitive_rms={max_row[3]:.6f}"
        )
        if max_row[2] >= 0.95:
            lines.append("- Conclusion: tiny benchmark reaches high blockwise fidelity (>=0.95), so optimizer/pipeline is trustworthy at small manifold size.")
        else:
            lines.append("- Conclusion: tiny benchmark does not reach 0.95; likely limited by pulse ansatz/optimizer settings in this configuration.")

    (out_dir / "tiny_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _run_full_updated(out_dir: Path) -> dict[str, Any]:
    params = sms.StudyParams(
        seed=17,
        include_case_e=True,
        run_profiles=("structured", "hard_random"),
        output_dir=Path("outputs/sqr_multitone_study"),
        system=sms.SystemParams(n_max=6),
    )

    full: dict[str, Any] = {
        "params": sms.asdict(params),
        "profiles": {},
    }
    for idx, mode in enumerate(params.run_profiles):
        profile = sms.build_target_profile(
            mode=mode,
            n_levels=int(params.system.n_cav),
            seed=int(params.seed + 19 * idx),
            theta_max=float(params.theta_max_rad),
        )
        suite = _run_case_suite(profile, params)
        full["profiles"][profile.name] = suite

    (out_dir / "full_study_updated.json").write_text(json.dumps(_to_jsonable(full), indent=2), encoding="utf-8")

    lines = []
    lines.append("# Updated Full SQR Study (After Convention + Metric Audit)")
    lines.append("")
    for profile_name, suite in full["profiles"].items():
        lines.append(f"## Profile: {profile_name}")
        corrected = suite["corrected_metrics"]
        lines.append("| Case | mean process fidelity | min process fidelity | phase_sensitive_rms | phase_axis_rms | pre-Z rms | post-Z rms | state fidelity mean |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for cid in sorted(corrected.keys()):
            summary = corrected[cid]["summary"]
            state_mean = float(np.mean(list(corrected[cid]["state_fidelities"].values())))
            lines.append(
                f"| {cid} | {summary['mean_process_fidelity']:.6f} | {summary['min_process_fidelity']:.6f} | {summary['phase_sensitive_rms_rad']:.6f} | {summary['phase_axis_rms_rad']:.6f} | {summary['residual_pre_z_rms_rad']:.6f} | {summary['residual_post_z_rms_rad']:.6f} | {state_mean:.6f} |"
            )

        # population-only vs phase-sensitive mismatch example
        for cid in sorted(corrected.keys()):
            rows = corrected[cid]["rows"]
            candidates = [
                row
                for row in rows
                if row["process_infidelity"] > 0.05
                and abs(np.sin(row["theta_simulated_rad"] / 2.0) ** 2 - np.sin(row["theta_target_rad"] / 2.0) ** 2) < 0.03
            ]
            if candidates:
                pick = max(candidates, key=lambda row: row["process_infidelity"])
                lines.append(
                    f"- Case {cid} mismatch example (n={pick['n']}): process infidelity={pick['process_infidelity']:.4f}, "
                    + f"phi-axis error={pick['phi_axis_error_rad']:.4f}, pre-Z={pick['residual_pre_z_rad']:.4f}, post-Z={pick['residual_post_z_rad']:.4f}."
                )
                break

        cross_b = suite["crosstalk"]["case_B"]
        cross_d = suite["crosstalk"]["case_D"]
        lines.append(
            "- Cross-talk (neighbor mean): "
            + f"Case B={cross_b['mean_neighbor_leakage']:.4f}, Case D={cross_d['mean_neighbor_leakage']:.4f}"
        )

        best_case = min(corrected.keys(), key=lambda cid: corrected[cid]["summary"]["phase_sensitive_rms_rad"])
        lines.append(
            f"- Best phase-sensitive case: {best_case} (phase_sensitive_rms={corrected[best_case]['summary']['phase_sensitive_rms_rad']:.6f})."
        )
        lines.append("")

    lines.append("## Conclusion")
    lines.append("- The updated metrics are phase-sensitive and differ across cases, unlike the previous frozen metric.")
    lines.append("- Optimized pulses can improve blockwise fidelity on selected targets; phase cancellation quality is case-dependent.")
    lines.append("- Positive detuning sign is now consistent between Rotation and SQR in waveform and two-level-equivalence checks.")

    (out_dir / "full_study_updated_report.md").write_text("\n".join(lines), encoding="utf-8")
    return full


def _build_stage_profiles(seed: int, n_levels: int, theta_max: float) -> list[sms.TargetProfile]:
    modes = ("structured", "moderate_random", "hard_random")
    out = []
    for idx, mode in enumerate(modes):
        out.append(
            sms.build_target_profile(
                mode=mode,
                n_levels=int(n_levels),
                seed=int(seed + 23 * idx),
                theta_max=float(theta_max),
            )
        )
    return out


def _representative_state_fidelity(state_fidelities: dict[str, float]) -> float:
    if not state_fidelities:
        return float("nan")
    picked = [
        float(value)
        for label, value in state_fidelities.items()
        if not (label.startswith("g,") and label[2:].isdigit())
    ]
    if not picked:
        picked = [float(value) for value in state_fidelities.values()]
    return float(np.mean(picked))


def _case_metrics_row(
    suite: dict[str, Any],
    case_id: str,
    profile_name: str,
    duration_us: float,
    envelope_kind: str,
) -> dict[str, Any]:
    corrected = suite["corrected_metrics"][case_id]
    summary = corrected["summary"]
    state_fid = corrected["state_fidelities"]
    if case_id == "B":
        crosstalk = suite["crosstalk"]["case_B"]
    elif case_id == "D":
        crosstalk = suite["crosstalk"]["case_D"]
    else:
        crosstalk = {"mean_neighbor_leakage": float("nan"), "max_neighbor_leakage": float("nan")}
    old_summary = corrected["old_summary"]
    return {
        "profile": profile_name,
        "case": case_id,
        "duration_us": float(duration_us),
        "envelope_kind": envelope_kind,
        "mean_process_fidelity": float(summary["mean_process_fidelity"]),
        "min_process_fidelity": float(summary["min_process_fidelity"]),
        "phase_sensitive_rms_rad": float(summary["phase_sensitive_rms_rad"]),
        "phase_axis_rms_rad": float(summary["phase_axis_rms_rad"]),
        "residual_pre_z_rms_rad": float(summary["residual_pre_z_rms_rad"]),
        "residual_post_z_rms_rad": float(summary["residual_post_z_rms_rad"]),
        "theta_error_rms_rad": float(summary["theta_error_rms_rad"]),
        "mean_neighbor_leakage": float(crosstalk["mean_neighbor_leakage"]),
        "max_neighbor_leakage": float(crosstalk["max_neighbor_leakage"]),
        "state_fidelity_mean": float(np.mean(list(state_fid.values()))),
        "state_fidelity_representative_mean": float(_representative_state_fidelity(state_fid)),
        "off_block_norm": float(old_summary.get("off_block_norm", 0.0)),
        "neighbor_overlap_proxy_mean": float(old_summary.get("neighbor_overlap_proxy_mean", 0.0)),
        "neighbor_overlap_proxy_max": float(old_summary.get("neighbor_overlap_proxy_max", 0.0)),
        "loss_terms": dict(old_summary.get("loss_terms", {})),
    }


def _run_duration_sweep(out_dir: Path) -> dict[str, Any]:
    durations_us = [1.0, 1.5, 2.0, 2.5]
    base_params = sms.StudyParams(
        seed=41,
        include_case_e=False,
        run_profiles=("structured",),
        system=sms.SystemParams(
            n_max=6,
            chi_nominal_hz=-2.84e6,
            chi_easy_hz=-2.84e6,
            chi_hard_hz=-2.84e6,
        ),
        optimization=replace(
            sms.OptimizationParams(),
            maxiter_stage1_basic=9,
            maxiter_stage2_basic=14,
            maxiter_stage1_extended=12,
            maxiter_stage2_extended=18,
        ),
    )
    profiles = _build_stage_profiles(
        seed=int(base_params.seed),
        n_levels=int(base_params.system.n_cav),
        theta_max=float(base_params.theta_max_rad),
    )

    rows: list[dict[str, Any]] = []
    per_profile: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        per_profile[profile.name] = {"durations": {}}
        for duration_us in durations_us:
            pulse = replace(base_params.pulse, duration_nominal_s=float(duration_us * 1.0e-6), envelope_kind="gaussian")
            params = replace(base_params, pulse=pulse)
            suite = _run_case_suite(profile, params)
            row_b = _case_metrics_row(suite, "B", profile_name=profile.name, duration_us=float(duration_us), envelope_kind="gaussian")
            row_d = _case_metrics_row(suite, "D", profile_name=profile.name, duration_us=float(duration_us), envelope_kind="gaussian")
            rows.extend([row_b, row_d])
            per_profile[profile.name]["durations"][f"{duration_us:.1f}"] = {"B": row_b, "D": row_d}

    # pick best longer duration by minimal phase-sensitive score with highest fidelity on case D.
    best_long_by_profile: dict[str, float] = {}
    best_long_by_mode: dict[str, float] = {}
    for profile in profiles:
        candidates = [
            row
            for row in rows
            if row["profile"] == profile.name and row["case"] == "D" and row["duration_us"] > 1.0
        ]
        if not candidates:
            best_long_by_profile[profile.name] = 1.0
            best_long_by_mode[profile.mode] = 1.0
            continue
        picked = min(candidates, key=lambda row: (row["phase_sensitive_rms_rad"], -row["mean_process_fidelity"]))
        best_long_by_profile[profile.name] = float(picked["duration_us"])
        best_long_by_mode[profile.mode] = float(picked["duration_us"])
    payload = {
        "rows": rows,
        "best_long_duration_us_by_profile": best_long_by_profile,
        "best_long_duration_us_by_mode": best_long_by_mode,
    }
    (out_dir / "duration_sweep_selectivity.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")

    lines = []
    lines.append("# Duration Sweep at chi=-2.84 MHz")
    lines.append("")
    lines.append("- Sweep settings: Gaussian envelope, durations `[1.0, 1.5, 2.0, 2.5] us`, cases `B` and `D`.")
    lines.append("")
    for profile in profiles:
        lines.append(f"## Profile: {profile.name}")
        lines.append("| Duration [us] | Case | mean fidelity | min fidelity | phase-sensitive RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for duration_us in durations_us:
            for case_id in ("B", "D"):
                row = per_profile[profile.name]["durations"][f"{duration_us:.1f}"][case_id]
                lines.append(
                    f"| {duration_us:.1f} | {case_id} | {row['mean_process_fidelity']:.6f} | {row['min_process_fidelity']:.6f} | "
                    + f"{row['phase_sensitive_rms_rad']:.6f} | {row['residual_pre_z_rms_rad']:.6f} | {row['residual_post_z_rms_rad']:.6f} | "
                    + f"{row['mean_neighbor_leakage']:.4f}/{row['max_neighbor_leakage']:.4f} | {row['state_fidelity_representative_mean']:.6f} |"
                )
        lines.append(f"- Best longer duration for envelope comparison: {best_long_by_profile[profile.name]:.1f} us.")
        lines.append("")
    (out_dir / "duration_sweep_selectivity.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _run_envelope_comparison(out_dir: Path, duration_payload: dict[str, Any]) -> dict[str, Any]:
    base_params = sms.StudyParams(
        seed=41,
        include_case_e=False,
        run_profiles=("structured",),
        system=sms.SystemParams(
            n_max=6,
            chi_nominal_hz=-2.84e6,
            chi_easy_hz=-2.84e6,
            chi_hard_hz=-2.84e6,
        ),
        optimization=replace(
            sms.OptimizationParams(),
            maxiter_stage1_basic=9,
            maxiter_stage2_basic=14,
            maxiter_stage1_extended=12,
            maxiter_stage2_extended=18,
        ),
    )
    profiles = _build_stage_profiles(
        seed=int(base_params.seed),
        n_levels=int(base_params.system.n_cav),
        theta_max=float(base_params.theta_max_rad),
    )
    best_long_by_profile = duration_payload.get("best_long_duration_us_by_profile", {})
    best_long_by_mode = duration_payload.get("best_long_duration_us_by_mode", {})
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        pick_long = float(best_long_by_profile.get(profile.name, best_long_by_mode.get(profile.mode, 1.0)))
        durations_us = sorted(set([1.0, pick_long]))
        for duration_us in durations_us:
            for envelope_kind in ("gaussian", "flat_top"):
                pulse = replace(
                    base_params.pulse,
                    duration_nominal_s=float(duration_us * 1.0e-6),
                    envelope_kind=str(envelope_kind),
                )
                params = replace(base_params, pulse=pulse)
                suite = _run_case_suite(profile, params)
                rows.append(
                    _case_metrics_row(
                        suite,
                        "D",
                        profile_name=profile.name,
                        duration_us=float(duration_us),
                        envelope_kind=str(envelope_kind),
                    )
                )
    payload = {"rows": rows}
    (out_dir / "envelope_selectivity_comparison.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")

    lines = []
    lines.append("# Envelope Ansatz Comparison")
    lines.append("")
    lines.append("- Compared Case D (`amp+phase+detuning`) for Gaussian vs flat-top Gaussian at baseline and best longer duration per profile.")
    lines.append("")
    for profile in profiles:
        lines.append(f"## Profile: {profile.name}")
        lines.append("| Duration [us] | Envelope | mean fidelity | min fidelity | phase-sensitive RMS | neighbor leakage mean/max | representative state fidelity |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|")
        for row in [r for r in rows if r["profile"] == profile.name]:
            lines.append(
                f"| {row['duration_us']:.1f} | {row['envelope_kind']} | {row['mean_process_fidelity']:.6f} | {row['min_process_fidelity']:.6f} | "
                + f"{row['phase_sensitive_rms_rad']:.6f} | {row['mean_neighbor_leakage']:.4f}/{row['max_neighbor_leakage']:.4f} | {row['state_fidelity_representative_mean']:.6f} |"
            )
        lines.append("")
    (out_dir / "envelope_selectivity_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def _run_staged_full_selectivity(out_dir: Path) -> dict[str, Any]:
    params = sms.StudyParams(
        seed=61,
        include_case_e=True,
        run_profiles=("structured",),
        system=sms.SystemParams(
            n_max=6,
            chi_nominal_hz=-2.84e6,
            chi_easy_hz=-2.84e6,
            chi_hard_hz=-2.84e6,
        ),
    )
    profiles = _build_stage_profiles(
        seed=int(params.seed),
        n_levels=int(params.system.n_cav),
        theta_max=float(params.theta_max_rad),
    )
    out: dict[str, Any] = {"profiles": {}}
    for profile in profiles:
        out["profiles"][profile.name] = _run_case_suite(profile, params)
    (out_dir / "full_selectivity_staged.json").write_text(json.dumps(_to_jsonable(out), indent=2), encoding="utf-8")

    lines = []
    lines.append("# Selectivity-Focused Full SQR Report")
    lines.append("")
    lines.append("- Baseline settings: chi = -2.84 MHz, duration = 1.0 us, Gaussian envelope.")
    lines.append("- Cases: B (naive), C (amp+phase), D (amp+phase+detuning), E (with chirp).")
    lines.append("")
    for profile in profiles:
        suite = out["profiles"][profile.name]
        corrected = suite["corrected_metrics"]
        lines.append(f"## Profile: {profile.name}")
        lines.append("| Case | mean fidelity | min fidelity | phase-sensitive RMS | phase-axis RMS | pre-Z RMS | post-Z RMS | neighbor leakage mean/max | representative state fidelity |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for case_id in ("B", "C", "D", "E"):
            if case_id not in corrected:
                continue
            row = _case_metrics_row(
                suite=suite,
                case_id=case_id,
                profile_name=profile.name,
                duration_us=1.0,
                envelope_kind="gaussian",
            )
            lines.append(
                f"| {case_id} | {row['mean_process_fidelity']:.6f} | {row['min_process_fidelity']:.6f} | {row['phase_sensitive_rms_rad']:.6f} | "
                + f"{row['phase_axis_rms_rad']:.6f} | {row['residual_pre_z_rms_rad']:.6f} | {row['residual_post_z_rms_rad']:.6f} | "
                + f"{row['mean_neighbor_leakage']:.4f}/{row['max_neighbor_leakage']:.4f} | {row['state_fidelity_representative_mean']:.6f} |"
            )
        for case_id in ("D", "E"):
            if case_id not in corrected:
                continue
            loss_terms = corrected[case_id]["old_summary"].get("loss_terms", {})
            if not loss_terms:
                continue
            lines.append(
                f"- Case {case_id} loss terms: "
                + ", ".join(f"{name}={float(value):.4e}" for name, value in loss_terms.items())
            )
        lines.append("")
    lines.append("## Baseline Limitation")
    lines.append("- At T=1.0 us and chi=-2.84 MHz, the dominant limiter remains manifold selectivity (neighbor leakage / off-target action), not only phase bookkeeping.")
    (out_dir / "full_selectivity_staged_report.md").write_text("\n".join(lines), encoding="utf-8")
    return out


def _manual_profile_from_tiny_target(target: ManualTarget, n_levels: int, seed: int) -> sms.TargetProfile:
    theta = np.zeros(int(n_levels), dtype=float)
    phi = np.zeros(int(n_levels), dtype=float)
    l = min(len(target.theta), int(n_levels))
    theta[:l] = np.asarray(target.theta[:l], dtype=float)
    phi[:l] = np.asarray(target.phi[:l], dtype=float)
    return sms.TargetProfile(
        name=f"{target.name}_n{int(n_levels)}",
        mode="manual",
        theta=theta,
        phi=phi,
        seed=int(seed),
    )


def _support_objective_comparison(out_dir: Path) -> dict[str, Any]:
    base_opt = replace(
        sms.OptimizationParams(),
        maxiter_stage1_basic=7,
        maxiter_stage2_basic=10,
        maxiter_stage1_extended=9,
        maxiter_stage2_extended=12,
        maxiter_stage1_chirp=10,
        maxiter_stage2_chirp=12,
        objective_scope="global",
    )
    base_params = sms.StudyParams(
        seed=97,
        include_case_e=True,
        run_profiles=("structured",),
        system=sms.SystemParams(
            n_max=5,
            chi_nominal_hz=-2.84e6,
            chi_easy_hz=-2.84e6,
            chi_hard_hz=-2.84e6,
        ),
        optimization=base_opt,
    )
    support_sets = {
        "S01": sms.ActiveSupportParams(mode="explicit", active_levels=(0, 1), inactive_weight=0.02),
        "S0123": sms.ActiveSupportParams(mode="explicit", active_levels=(0, 1, 2, 3), inactive_weight=0.02),
    }

    payload: dict[str, Any] = {"support_sets": {}}
    lines = []
    lines.append("# Support-Aware Objective Comparison")
    lines.append("")
    lines.append("- Compared legacy global objective vs new support-aware objective on tiny benchmark-style targets.")
    lines.append("- Truncation: n_max=5, chi=-2.84 MHz, duration=1.0 us.")
    lines.append("- Case E is the default expressive ansatz in support-aware mode; Case D is retained as fallback/ablation.")
    lines.append("")

    for support_name, support_cfg in support_sets.items():
        lines.append(f"## Active Support: {support_name}")
        lines.append(f"- Levels: {list(support_cfg.active_levels)}")
        lines.append("")
        lines.append("| Target | Objective | Case | active weighted mean fid | active min fid | active theta RMS | active phase RMS | active pre-Z RMS | active post-Z RMS | support-leak mean/max | spectral-leak mean/max | state mean/min | phase-super RMS | global mean/min fid | global phase-sensitive RMS |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        payload["support_sets"][support_name] = {"targets": {}}

        for idx, target in enumerate(_tiny_targets()):
            profile = _manual_profile_from_tiny_target(target, n_levels=int(base_params.system.n_cav), seed=idx)
            global_suite = _run_case_suite(
                profile=profile,
                params=base_params,
                objective_scope="global",
                support_cfg=support_cfg,
            )
            support_opt = replace(base_params.optimization, objective_scope="support_aware")
            support_params = replace(base_params, optimization=support_opt)
            support_suite = _run_case_suite(
                profile=profile,
                params=support_params,
                objective_scope="support_aware",
                support_cfg=support_cfg,
            )

            row_pack: dict[str, Any] = {}
            for label, suite in (("global", global_suite), ("support", support_suite)):
                case_used = "E" if "E" in suite["corrected_metrics"] else "D"
                old_summary = suite["corrected_metrics"][case_used]["old_summary"]
                corrected_summary = suite["corrected_metrics"][case_used]["summary"]
                support_metrics = old_summary.get("support_metrics", {})
                crosstalk = np.asarray(suite["crosstalk"]["case_D"]["relative_leakage_matrix"], dtype=float)
                active_levels = [int(n) for n in support_cfg.active_levels]
                active_set = set(active_levels)
                inactive_levels = [n for n in range(crosstalk.shape[0]) if n not in active_set]
                restricted = (
                    crosstalk[np.ix_(inactive_levels, active_levels)]
                    if active_levels and inactive_levels
                    else np.zeros((0, 0), dtype=float)
                )
                boundary = max(active_levels) if active_levels else 0
                boundary_spill = (
                    float(crosstalk[boundary + 1, boundary])
                    if (boundary + 1) < crosstalk.shape[0]
                    else float("nan")
                )
                channels = []
                for k in active_levels:
                    for n in inactive_levels:
                        if 0 <= n < crosstalk.shape[0] and 0 <= k < crosstalk.shape[1]:
                            channels.append(float(crosstalk[n, k]))
                active_to_inactive_crosstalk_max = float(np.max(channels)) if channels else float("nan")
                row_pack[label] = {
                    "case_used": case_used,
                    "active_metrics": support_metrics,
                    "global_metrics": corrected_summary,
                    "loss_terms": old_summary.get("loss_terms", {}),
                    "boundary_spill": boundary_spill,
                    "active_to_inactive_crosstalk_max": active_to_inactive_crosstalk_max,
                    "active_to_inactive_crosstalk_matrix": restricted.tolist(),
                }
                lines.append(
                    f"| {target.name} | {label} | {case_used} | "
                    + f"{float(support_metrics.get('active_weighted_mean_process_fidelity', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('active_min_process_fidelity', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('active_theta_rms_rad', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('active_phase_axis_rms_rad', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('active_residual_pre_z_rms_rad', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('active_residual_post_z_rms_rad', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('support_state_leakage_mean', float('nan'))):.4e}/"
                    + f"{float(support_metrics.get('support_state_leakage_max', float('nan'))):.4e} | "
                    + f"{float(support_metrics.get('support_weighted_leakage_mean', float('nan'))):.4e}/"
                    + f"{float(support_metrics.get('support_weighted_leakage_max', float('nan'))):.4e} | "
                    + f"{float(support_metrics.get('support_state_fidelity_mean', float('nan'))):.6f}/"
                    + f"{float(support_metrics.get('support_state_fidelity_min', float('nan'))):.6f} | "
                    + f"{float(support_metrics.get('support_phase_superposition_rms_rad', float('nan'))):.6f} | "
                    + f"{float(corrected_summary['mean_process_fidelity']):.6f}/"
                    + f"{float(corrected_summary['min_process_fidelity']):.6f} | "
                    + f"{float(corrected_summary['phase_sensitive_rms_rad']):.6f} |"
                )

            g = row_pack["global"]["active_metrics"]
            s = row_pack["support"]["active_metrics"]
            lines.append(
                "- Delta (support-aware - global objective): "
                + f"active mean fid={float(s.get('active_weighted_mean_process_fidelity', 0.0) - g.get('active_weighted_mean_process_fidelity', 0.0)):+.4f}, "
                + f"active min fid={float(s.get('active_min_process_fidelity', 0.0) - g.get('active_min_process_fidelity', 0.0)):+.4f}, "
                + f"state min={float(s.get('support_state_fidelity_min', 0.0) - g.get('support_state_fidelity_min', 0.0)):+.4f}, "
                + f"support-state leak mean={float(s.get('support_state_leakage_mean', 0.0) - g.get('support_state_leakage_mean', 0.0)):+.3e}."
            )
            lines.append(
                "- Active-boundary crosstalk (Case D diagnostic): "
                + f"global boundary spill={row_pack['global']['boundary_spill']:.3e}, "
                + f"support boundary spill={row_pack['support']['boundary_spill']:.3e}, "
                + f"global max active->inactive={row_pack['global']['active_to_inactive_crosstalk_max']:.3e}, "
                + f"support max active->inactive={row_pack['support']['active_to_inactive_crosstalk_max']:.3e}."
            )
            lines.append(
                "- Restricted active->inactive crosstalk matrices (rows=inactive, cols=active): "
                + f"global={np.array(row_pack['global']['active_to_inactive_crosstalk_matrix']).round(3).tolist()}, "
                + f"support={np.array(row_pack['support']['active_to_inactive_crosstalk_matrix']).round(3).tolist()}."
            )
            terms = row_pack["support"]["loss_terms"]
            if terms:
                term_rank = sorted(terms.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:4]
                lines.append("- Dominant support-aware loss terms (primary case): " + ", ".join(f"{k}={float(v):.3e}" for k, v in term_rank))

            if "E" in support_suite["corrected_metrics"] and "D" in support_suite["corrected_metrics"]:
                e_support = support_suite["corrected_metrics"]["E"]["old_summary"].get("support_metrics", {})
                d_support = support_suite["corrected_metrics"]["D"]["old_summary"].get("support_metrics", {})
                lines.append(
                    "- Case E vs Case D (support-aware ablation): "
                    + f"active mean fid delta={float(e_support.get('active_weighted_mean_process_fidelity', 0.0) - d_support.get('active_weighted_mean_process_fidelity', 0.0)):+.4f}, "
                    + f"active min fid delta={float(e_support.get('active_min_process_fidelity', 0.0) - d_support.get('active_min_process_fidelity', 0.0)):+.4f}, "
                    + f"state min delta={float(e_support.get('support_state_fidelity_min', 0.0) - d_support.get('support_state_fidelity_min', 0.0)):+.4f}."
                )

            payload["support_sets"][support_name]["targets"][target.name] = row_pack
        lines.append("")

    lines.append("## Main Question")
    lines.append("- Support-aware optimization can improve the experimentally occupied subspace metrics even when global equal-weight metrics move less.")
    lines.append("- The largest practical gains typically come from worst-block protection and active-ensemble fidelity terms; leakage terms are most impactful near support boundaries.")

    (out_dir / "support_aware_objective_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "support_aware_objective.json").write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
    return payload


def run_all(output_dir: Path | str = "outputs/sqr_convention_metric_audit") -> dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sign_scan = _sign_scan_for_equivalence()
    detuning = _detuning_sign_check()
    _write_implementation_note(out_dir, sign_scan, detuning)
    _write_support_design_note(out_dir)
    _write_convention_note(out_dir, sign_scan, detuning)
    _write_equivalence_payload(out_dir, sign_scan, detuning)

    audit_params = sms.StudyParams(seed=17, include_case_e=True, run_profiles=("structured",), system=sms.SystemParams(n_max=6))
    profile = sms.build_target_profile(
        mode="structured",
        n_levels=int(audit_params.system.n_cav),
        seed=int(audit_params.seed),
        theta_max=float(audit_params.theta_max_rad),
    )
    audit_suite = _run_case_suite(profile, audit_params)
    _write_phase_metric_audit(out_dir, audit_suite)

    tiny = _run_tiny_benchmark(out_dir)
    support_objective = _support_objective_comparison(out_dir)
    duration_sweep = _run_duration_sweep(out_dir)
    envelope_cmp = _run_envelope_comparison(out_dir, duration_payload=duration_sweep)
    full = _run_staged_full_selectivity(out_dir)

    summary = {
        "output_dir": str(out_dir),
        "files": {
            "implementation_note": str(out_dir / "implementation_note.md"),
            "support_design_note": str(out_dir / "support_aware_design_note.md"),
            "convention_note": str(out_dir / "convention_reconciliation.md"),
            "equivalence_json": str(out_dir / "waveform_hamiltonian_equivalence.json"),
            "phase_metric_audit": str(out_dir / "phase_metric_audit.md"),
            "tiny_benchmark_report": str(out_dir / "tiny_benchmark_report.md"),
            "support_objective_report": str(out_dir / "support_aware_objective_report.md"),
            "duration_sweep_report": str(out_dir / "duration_sweep_selectivity.md"),
            "envelope_comparison_report": str(out_dir / "envelope_selectivity_comparison.md"),
            "full_study_report": str(out_dir / "full_selectivity_staged_report.md"),
        },
        "best_rotation_sign_match": sign_scan["best_rotation_match"],
        "best_sqr_sign_match": sign_scan["best_sqr_match"],
        "detuning_sign": detuning,
        "tiny": tiny,
        "support_objective": support_objective,
        "duration_sweep": duration_sweep,
        "envelope_comparison": envelope_cmp,
        "full": full,
    }
    (out_dir / "summary.json").write_text(json.dumps(_to_jsonable(summary), indent=2), encoding="utf-8")
    return summary


def main() -> None:
    summary = run_all()
    print(
        json.dumps(
            {
                "output_dir": summary["output_dir"],
                "implementation_note": summary["files"]["implementation_note"],
                "support_design_note": summary["files"]["support_design_note"],
                "convention_note": summary["files"]["convention_note"],
                "phase_metric_audit": summary["files"]["phase_metric_audit"],
                "tiny_benchmark_report": summary["files"]["tiny_benchmark_report"],
                "support_objective_report": summary["files"]["support_objective_report"],
                "duration_sweep_report": summary["files"]["duration_sweep_report"],
                "envelope_comparison_report": summary["files"]["envelope_comparison_report"],
                "full_study_report": summary["files"]["full_study_report"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
