from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import (
    drive_frequency_for_transition_frequency,
    internal_carrier_from_drive_frequency,
    manifold_transition_frequency,
)
from cqed_sim.core.ideal_gates import embed_qubit_op, qubit_rotation_axis
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.extractors import bloch_xyz_from_qubit_state, conditioned_bloch_xyz, qubit_density_from_bloch_xyz, reduced_qubit_state
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _gaussian_norm(t_rel: np.ndarray, sigma_rel: float = 0.17) -> np.ndarray:
    x = np.exp(-0.5 * ((t_rel - 0.5) / sigma_rel) ** 2)
    return x / np.mean(x)


@dataclass
class QubitPulseCal:
    amp90: float
    y_phase: float = np.pi / 2
    drag: float = 0.0
    detuning: float = 0.0
    duration_ns: float = 16.0

    @staticmethod
    def nominal() -> "QubitPulseCal":
        # For our convention, theta ~ 2 * amp * duration, so amp90 = pi/(4*duration)
        return QubitPulseCal(amp90=np.pi / (4 * 16.0), y_phase=np.pi / 2, drag=0.0, detuning=0.0)

    def amp(self, label: str) -> float:
        if label in ("x90", "y90"):
            return self.amp90
        if label in ("x180", "y180"):
            return 2 * self.amp90
        if label == "i":
            return 0.0
        raise ValueError(f"Unknown pulse label '{label}'.")

    def phase(self, label: str) -> float:
        if label.startswith("x"):
            return 0.0
        if label.startswith("y"):
            return self.y_phase
        return 0.0


ALL_XY_21: list[tuple[str, str]] = [
    ("i", "i"),
    ("x180", "x180"),
    ("y180", "y180"),
    ("x180", "y180"),
    ("y180", "x180"),
    ("x90", "i"),
    ("y90", "i"),
    ("x90", "y90"),
    ("y90", "x90"),
    ("x90", "y180"),
    ("y90", "x180"),
    ("x180", "y90"),
    ("y180", "x90"),
    ("x90", "x180"),
    ("x180", "x90"),
    ("y90", "y180"),
    ("y180", "y90"),
    ("x180", "i"),
    ("y180", "i"),
    ("x90", "x90"),
    ("y90", "y90"),
]


def _apply_ideal_label(state: qt.Qobj, label: str, n_cav: int) -> qt.Qobj:
    if label == "i":
        return state
    if label == "x90":
        return embed_qubit_op(qubit_rotation_axis(np.pi / 2, "x"), n_cav) * state
    if label == "y90":
        return embed_qubit_op(qubit_rotation_axis(np.pi / 2, "y"), n_cav) * state
    if label == "x180":
        return embed_qubit_op(qubit_rotation_axis(np.pi, "x"), n_cav) * state
    if label == "y180":
        return embed_qubit_op(qubit_rotation_axis(np.pi, "y"), n_cav) * state
    raise ValueError(label)


def _pulse_for_label(label: str, t0: float, cal: QubitPulseCal, carrier: float = 0.0) -> Pulse:
    return Pulse(
        channel="q",
        t0=t0,
        duration=cal.duration_ns,
        envelope=_gaussian_norm,
        carrier=carrier + cal.detuning,
        phase=cal.phase(label),
        amp=cal.amp(label),
        drag=cal.drag,
    )


def run_all_xy(
    model: DispersiveTransmonCavityModel,
    cal: QubitPulseCal,
    dt_ns: float = 0.2,
    frame: FrameSpec | None = None,
    noise: NoiseSpec | None = None,
    simulation_config: SimulationConfig | None = None,
) -> dict[str, np.ndarray]:
    frame = frame or FrameSpec(omega_q_frame=model.omega_q)
    compiler = SequenceCompiler(dt=dt_ns)
    config = SimulationConfig(frame=frame) if simulation_config is None else replace(simulation_config, frame=frame)
    init = model.basis_state(0, 0)
    measured = []
    expected = []
    for a, b in ALL_XY_21:
        pulses = []
        t = 0.0
        for lbl in (a, b):
            pulses.append(_pulse_for_label(lbl, t, cal))
            t += cal.duration_ns
        compiled = compiler.compile(pulses, t_end=t + dt_ns)
        res = simulate_sequence(model, compiled, init, {"q": "qubit"}, config, noise=noise, e_ops={})
        rho_q = reduced_qubit_state(res.final_state)
        measured.append(float(np.real((rho_q * qt.sigmaz()).tr())))

        psi_ideal = init
        psi_ideal = _apply_ideal_label(psi_ideal, a, model.n_cav)
        psi_ideal = _apply_ideal_label(psi_ideal, b, model.n_cav)
        rho_q_ideal = reduced_qubit_state(psi_ideal)
        expected.append(float(np.real((rho_q_ideal * qt.sigmaz()).tr())))
    measured_arr = np.asarray(measured, dtype=float)
    expected_arr = np.asarray(expected, dtype=float)
    rms = float(np.sqrt(np.mean((measured_arr - expected_arr) ** 2)))
    return {"measured_z": measured_arr, "expected_z": expected_arr, "rms_error": rms}


def autocalibrate_all_xy(
    model: DispersiveTransmonCavityModel,
    initial_cal: QubitPulseCal,
    dt_ns: float = 0.2,
    max_iter: int = 12,
    target_rms: float = 0.08,
) -> tuple[QubitPulseCal, dict]:
    best = initial_cal
    best_res = run_all_xy(model, best, dt_ns=dt_ns)
    step_amp = 0.2 * initial_cal.amp90
    step_phi = 0.25
    for _ in range(max_iter):
        candidates = [
            QubitPulseCal(best.amp90 + da, best.y_phase + dp, best.drag, best.detuning, best.duration_ns)
            for da in (-step_amp, 0.0, step_amp)
            for dp in (-step_phi, 0.0, step_phi)
            if best.amp90 + da > 0
        ]
        for cand in candidates:
            res = run_all_xy(model, cand, dt_ns=dt_ns)
            if res["rms_error"] < best_res["rms_error"]:
                best, best_res = cand, res
        step_amp *= 0.65
        step_phi *= 0.65
        if best_res["rms_error"] <= target_rms:
            break
    return best, best_res


def selective_qubit_drive_frequency(model: DispersiveTransmonCavityModel, n: int) -> float:
    frame = FrameSpec(omega_q_frame=model.omega_q)
    transition = manifold_transition_frequency(model, n=n, frame=frame)
    return drive_frequency_for_transition_frequency(transition, frame.omega_q_frame)


def selective_qubit_freq(model: DispersiveTransmonCavityModel, n: int) -> float:
    return internal_carrier_from_drive_frequency(selective_qubit_drive_frequency(model, n), model.omega_q)


def selective_pi_pulse(n: int, t0_ns: float, duration_ns: float, amp: float, model: DispersiveTransmonCavityModel, drag: float = 0.0) -> Pulse:
    drive_frequency = selective_qubit_drive_frequency(model, n)
    return Pulse(
        channel="q",
        t0=t0_ns,
        duration=duration_ns,
        envelope=_gaussian_norm,
        carrier=internal_carrier_from_drive_frequency(drive_frequency, model.omega_q),
        phase=0.0,
        amp=amp,
        drag=drag,
    )


def _pre_rotation_pulses(axis: str, t0_ns: float, cal: QubitPulseCal) -> list[Pulse]:
    # Maps sigma_a measurement onto sigma_z readout.
    if axis == "z":
        return []
    if axis == "x":
        # Ry(-pi/2)
        return [Pulse("q", t0_ns, cal.duration_ns, _gaussian_norm, phase=cal.y_phase + np.pi, amp=cal.amp90, drag=cal.drag)]
    if axis == "y":
        # Rx(+pi/2) convention.
        return [Pulse("q", t0_ns, cal.duration_ns, _gaussian_norm, phase=0.0, amp=cal.amp90, drag=cal.drag)]
    raise ValueError(axis)


@dataclass
class FockTomographyResult:
    v_hat: dict[str, np.ndarray]
    p_n: np.ndarray
    conditioned_bloch: dict[int, np.ndarray]
    v_rec: dict[str, np.ndarray] | None = None


def run_fock_resolved_tomo(
    model: DispersiveTransmonCavityModel,
    state_prep: Callable[[], qt.Qobj],
    n_max: int,
    cal: QubitPulseCal,
    tag_duration_ns: float = 1000.0,
    tag_amp: float = 0.0015,
    dt_ns: float = 1.0,
    noise: NoiseSpec | None = None,
    ideal_tag: bool = False,
    pre_rotation_mode: str = "pulse",
    leakage_cal: tuple[np.ndarray, dict[str, np.ndarray]] | None = None,
    unmix_lambda: float = 1e-2,
    simulation_config: SimulationConfig | None = None,
) -> FockTomographyResult:
    axes = ("x", "y", "z")
    v_hat = {a: np.zeros(n_max + 1, dtype=float) for a in axes}
    p_n = np.zeros(n_max + 1, dtype=float)
    frame = FrameSpec(omega_q_frame=model.omega_q)
    compiler = SequenceCompiler(dt=dt_ns)
    config = SimulationConfig(frame=frame) if simulation_config is None else replace(simulation_config, frame=frame)
    for n in range(n_max + 1):
        rho0 = state_prep()
        for a in axes:
            if pre_rotation_mode == "ideal":
                if a == "z":
                    off_state = rho0
                elif a == "x":
                    u_pre = embed_qubit_op(qubit_rotation_axis(-np.pi / 2, "y"), model.n_cav)
                    off_state = u_pre * rho0 * u_pre.dag() if rho0.isoper else u_pre * rho0
                elif a == "y":
                    u_pre = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "x"), model.n_cav)
                    off_state = u_pre * rho0 * u_pre.dag() if rho0.isoper else u_pre * rho0
                else:
                    raise ValueError(a)
                # matched idle
                comp_off = compiler.compile([], t_end=tag_duration_ns + dt_ns)
                off = simulate_sequence(model, comp_off, off_state, {}, config, noise=noise, e_ops={})
                s_off = float(np.real((reduced_qubit_state(off.final_state) * qt.sigmaz()).tr()))
                t_pre = 0.0
                pre = []
            else:
                pre = _pre_rotation_pulses(a, 0.0, cal)
                t_pre = sum(p.duration for p in pre)
                # OFF branch: matched idle equal to tag duration.
                comp_off = compiler.compile(pre, t_end=t_pre + tag_duration_ns + dt_ns)
                off = simulate_sequence(model, comp_off, rho0, {"q": "qubit"} if pre else {}, config, noise=noise, e_ops={})
                s_off = float(np.real((reduced_qubit_state(off.final_state) * qt.sigmaz()).tr()))

            if ideal_tag:
                pn = qt.basis(model.n_cav, n) * qt.basis(model.n_cav, n).dag()
                tag_u = qt.tensor(qt.qeye(model.n_tr), qt.qeye(model.n_cav) - pn) + qt.tensor(
                    qubit_rotation_axis(np.pi, "x"), pn
                )
                on_state = tag_u * off.final_state * tag_u.dag() if off.final_state.isoper else tag_u * off.final_state
                rho_q_on = reduced_qubit_state(on_state)
                s_on = float(np.real((rho_q_on * qt.sigmaz()).tr()))
            else:
                tag = selective_pi_pulse(n=n, t0_ns=t_pre, duration_ns=tag_duration_ns, amp=tag_amp, model=model, drag=cal.drag)
                comp_on = compiler.compile(pre + [tag], t_end=t_pre + tag_duration_ns + dt_ns)
                on = simulate_sequence(model, comp_on, rho0, {"q": "qubit"}, config, noise=noise, e_ops={})
                s_on = float(np.real((reduced_qubit_state(on.final_state) * qt.sigmaz()).tr()))

            v_hat[a][n] = 0.5 * (s_off - s_on)

        # P(n) via conditioned projector from original prepared state.
        rho_prep = state_prep()
        if not rho_prep.isoper:
            rho_prep = rho_prep.proj()
        proj_n = qt.tensor(qt.qeye(model.n_tr), qt.basis(model.n_cav, n) * qt.basis(model.n_cav, n).dag())
        p_n[n] = float(np.real((rho_prep * proj_n).tr()))

    v_rec = None
    if leakage_cal is not None:
        w, b = leakage_cal
        wt = w.T
        reg = wt @ w + unmix_lambda * np.eye(w.shape[1])
        v_rec = {a: np.linalg.solve(reg, wt @ (v_hat[a] - b[a])) for a in axes}

    cond = {}
    src = v_rec if v_rec is not None else v_hat
    for n in range(n_max + 1):
        if p_n[n] > 1e-12:
            cond[n] = np.array([src["x"][n], src["y"][n], src["z"][n]], dtype=float) / p_n[n]
        else:
            cond[n] = np.array([np.nan, np.nan, np.nan], dtype=float)
    return FockTomographyResult(v_hat=v_hat, p_n=p_n, conditioned_bloch=cond, v_rec=v_rec)


def true_fock_resolved_vectors(state: qt.Qobj, n_max: int) -> dict[str, np.ndarray]:
    rho = state if state.isoper else state.proj()
    n_cav = rho.dims[0][1]
    out = {a: np.zeros(n_max + 1, dtype=float) for a in ("x", "y", "z")}
    for n in range(n_max + 1):
        if n >= n_cav:
            continue
        pn = qt.basis(n_cav, n) * qt.basis(n_cav, n).dag()
        block = qt.tensor(qt.qeye(2), pn) * rho * qt.tensor(qt.qeye(2), pn)
        rho_q_tilde = qt.ptrace(block, 0)
        x, y, z = bloch_xyz_from_qubit_state(rho_q_tilde)
        out["x"][n] = x
        out["y"][n] = y
        out["z"][n] = z
    return out


def calibrate_leakage_matrix(
    model: DispersiveTransmonCavityModel,
    n_max: int,
    alphas: list[complex],
    bloch_states: list[np.ndarray],
    cal: QubitPulseCal,
    tag_duration_ns: float = 1000.0,
    tag_amp: float = 0.0015,
    dt_ns: float = 1.0,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    # Build linear system per axis: v_hat_a = W v_true_a + b_a.
    rows = {a: [] for a in ("x", "y", "z")}
    y = {a: [] for a in ("x", "y", "z")}
    for alpha in alphas:
        coh = qt.coherent(model.n_cav, alpha)
        for r in bloch_states:
            rq = qubit_density_from_bloch_xyz(r[0], r[1], r[2])
            rho = qt.tensor(rq, coh.proj())
            v_true = true_fock_resolved_vectors(rho, n_max=n_max)
            tomo = run_fock_resolved_tomo(
                model=model,
                state_prep=lambda rho=rho: rho,
                n_max=n_max,
                cal=cal,
                tag_duration_ns=tag_duration_ns,
                tag_amp=tag_amp,
                dt_ns=dt_ns,
                ideal_tag=False,
            )
            for a in ("x", "y", "z"):
                rows[a].append(np.concatenate([v_true[a], [1.0]]))
                y[a].append(tomo.v_hat[a])
    # Solve each output n independently for W row and bias.
    w_axes = {a: np.zeros((n_max + 1, n_max + 1), dtype=float) for a in ("x", "y", "z")}
    b = {a: np.zeros(n_max + 1, dtype=float) for a in ("x", "y", "z")}
    conds = []
    for axis in ("x", "y", "z"):
        a_mat = np.stack(rows[axis], axis=0)  # samples x (n_max+2)
        conds.append(np.linalg.cond(a_mat[:, :-1]))
        y_axis = np.stack(y[axis], axis=0)  # samples x (n_max+1)
        for n in range(n_max + 1):
            coef, *_ = np.linalg.lstsq(a_mat, y_axis[:, n], rcond=None)
            w_axes[axis][n, :] = coef[:-1]
            b[axis][n] = coef[-1]
    w = (w_axes["x"] + w_axes["y"] + w_axes["z"]) / 3.0
    cond = float(max(conds))
    return w, b, cond
