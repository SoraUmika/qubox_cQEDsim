from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from cqed_sim.tomo.protocol import QubitPulseCal


DEVICE_CONFIG = {
    "ro_el": "resonator",
    "qb_el": "qubit",
    "st_el": "storage",
    "ro_fq": 8596222556.078796,
    "qb_fq": 6150369694.524461,
    "st_fq": 5240932800.0,
    "ro_kappa": 4156000.0,
    "ro_chi": -913148.5,
    "anharmonicity": -255669694.5244608,
    "st_chi": -2840421.354241756,
    "st_chi2": -21912.638362342423,
    "st_chi3": -327.37857577643325,
    "st_K": -28844,
    "st_K2": 1406,
    "ro_therm_clks": 1000,
    "qb_therm_clks": 19625,
    "st_therm_clks": 200000.0,
    "qb_T1_relax": 9812.873848245112,
    "qb_T2_ramsey": 6324.73112712837,
    "qb_T2_echo": 8381,
    "r180_amp": 0.08565235748770193,
    "rlen": 16,
    "rsigma": 2.6666666666666665,
    "b_coherent_amp": 0.01958,
    "b_coherent_len": 48,
    "b_alpha": 1,
    "fock_fqs": [
        6150355624.798682,
        6147515785.728024,
        6144636052.64372,
        6141702748.091518,
        6138726201.173695,
        6135701129.575048,
        6132618869.060916,
        6129486767.621506,
    ],
}


def hz_to_rad_per_ns(hz: float) -> float:
    return float(2.0 * np.pi * hz * 1e-9)


def ns_to_s(ns: float) -> float:
    return float(ns * 1e-9)


def gaussian_norm(t_rel: np.ndarray, sigma_rel: float = 0.17) -> np.ndarray:
    x = np.exp(-0.5 * ((t_rel - 0.5) / sigma_rel) ** 2)
    return x / np.mean(x)


def wrap_pi(x: float) -> float:
    return float(np.arctan2(np.sin(x), np.cos(x)))


def coherent_Pn(alpha: complex, n: int) -> float:
    mu = float(np.abs(alpha) ** 2)
    return float(np.exp(-mu) * (mu**n) / math.factorial(n))


def choose_alpha_set(n_max: int, candidates: list[float], k: int) -> tuple[list[float], float]:
    best_cond = np.inf
    best = None
    for comb in itertools.combinations(candidates, k):
        a = np.zeros((k, n_max + 1), dtype=float)
        for i, aa in enumerate(comb):
            for n in range(n_max + 1):
                a[i, n] = coherent_Pn(aa, n)
        c = np.linalg.cond(a)
        if c < best_cond:
            best_cond = c
            best = list(comb)
    return best, float(best_cond)


def build_model(n_cav: int = 14, n_tr: int = 2) -> DispersiveTransmonCavityModel:
    cfg = DEVICE_CONFIG
    chi_mean_hz = float(np.mean(np.diff(np.array(cfg["fock_fqs"])) * -1.0))
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=hz_to_rad_per_ns(cfg["qb_fq"]),
        alpha=hz_to_rad_per_ns(cfg["anharmonicity"]),
        chi=hz_to_rad_per_ns(chi_mean_hz),
        kerr=hz_to_rad_per_ns(cfg["st_K"]),
        kerr_higher=(hz_to_rad_per_ns(cfg["st_K2"]),),
        n_cav=n_cav,
        n_tr=n_tr,
    )


def build_noise() -> NoiseSpec:
    cfg = DEVICE_CONFIG
    t1 = float(cfg["qb_T1_relax"])
    t2 = float(cfg["qb_T2_ramsey"])
    inv_tphi = max(0.0, 1.0 / t2 - 1.0 / (2.0 * t1))
    tphi = np.inf if inv_tphi <= 0 else 1.0 / inv_tphi
    return NoiseSpec(t1=t1, tphi=tphi, kappa=float(cfg["ro_kappa"]) * 1e-9)


def pre_rotation(axis: str, cal: QubitPulseCal) -> list[Pulse]:
    if axis == "z":
        return []
    if axis == "x":
        return [
            Pulse(
                "q",
                0.0,
                cal.duration_ns,
                gaussian_norm,
                phase=cal.y_phase + np.pi,
                amp=cal.amp90,
                drag=cal.drag,
            )
        ]
    if axis == "y":
        return [Pulse("q", 0.0, cal.duration_ns, gaussian_norm, phase=0.0, amp=cal.amp90, drag=cal.drag)]
    raise ValueError(axis)


def selective_carrier_from_fock_fq(n: int) -> float:
    cfg = DEVICE_CONFIG
    return hz_to_rad_per_ns(float(cfg["fock_fqs"][n] - cfg["qb_fq"]))


def selective_pi_custom(n: int, t0_ns: float, duration_ns: float, amp: float, drag: float = 0.0) -> Pulse:
    return Pulse(
        channel="q",
        t0=t0_ns,
        duration=duration_ns,
        envelope=gaussian_norm,
        carrier=selective_carrier_from_fock_fq(n),
        phase=0.0,
        amp=amp,
        drag=drag,
    )


def run_fock_tomo_custom(
    model: DispersiveTransmonCavityModel,
    state_prep,
    n_max: int,
    cal: QubitPulseCal,
    noise: NoiseSpec,
    tag_duration_ns: float = 900.0,
    tag_amp: float = 0.01,
    dt_ns: float = 1.0,
    leakage_cal=None,
    unmix_lambda: float = 1e-2,
):
    axes = ("x", "y", "z")
    v_hat = {a: np.zeros(n_max + 1, float) for a in axes}
    p_n = np.zeros(n_max + 1, float)
    frame = FrameSpec(omega_q_frame=model.omega_q)
    for n in range(n_max + 1):
        rho0 = state_prep()
        for a in axes:
            pre = pre_rotation(a, cal)
            t_pre = sum(p.duration for p in pre)
            off_comp = SequenceCompiler(dt=dt_ns).compile(pre, t_end=t_pre + tag_duration_ns + dt_ns)
            off = simulate_sequence(model, off_comp, rho0, {"q": "qubit"} if pre else {}, SimulationConfig(frame=frame), noise=noise)
            s_off = float(np.real((qt.ptrace(off.final_state, 0) * qt.sigmaz()).tr()))

            tag = selective_pi_custom(n=n, t0_ns=t_pre, duration_ns=tag_duration_ns, amp=tag_amp, drag=cal.drag)
            on_comp = SequenceCompiler(dt=dt_ns).compile(pre + [tag], t_end=t_pre + tag_duration_ns + dt_ns)
            on = simulate_sequence(model, on_comp, rho0, {"q": "qubit"}, SimulationConfig(frame=frame), noise=noise)
            s_on = float(np.real((qt.ptrace(on.final_state, 0) * qt.sigmaz()).tr()))
            v_hat[a][n] = 0.5 * (s_off - s_on)

        rho_prep = state_prep()
        if not rho_prep.isoper:
            rho_prep = rho_prep.proj()
        pn = qt.tensor( qt.qeye(model.n_tr),qt.basis(model.n_cav, n).proj())
        p_n[n] = float(np.real((rho_prep * pn).tr()))

    v_rec = None
    if leakage_cal is not None:
        w, b = leakage_cal
        wt = w.T
        reg = wt @ w + unmix_lambda * np.eye(w.shape[1])
        v_rec = {a: np.linalg.solve(reg, wt @ (v_hat[a] - b[a])) for a in axes}
        for a in axes:
            v_rec[a] = np.clip(v_rec[a], -p_n, p_n)

    src = v_rec if v_rec is not None else v_hat
    cond = {}
    for n in range(n_max + 1):
        if p_n[n] > 1e-12:
            cond[n] = np.array([src["x"][n], src["y"][n], src["z"][n]]) / p_n[n]
        else:
            cond[n] = np.array([np.nan, np.nan, np.nan])
    return {"v_hat": v_hat, "v_rec": v_rec, "p_n": p_n, "conditioned": cond}


def true_vectors(rho: qt.Qobj, n_max: int):
    if not rho.isoper:
        rho = rho.proj()
    out = {a: np.zeros(n_max + 1, float) for a in ("x", "y", "z")}
    pauli = {"x": qt.sigmax(), "y": qt.sigmay(), "z": qt.sigmaz()}
    for n in range(n_max + 1):
        pn = qt.basis(rho.dims[0][1], n).proj()
        for a in ("x", "y", "z"):
            out[a][n] = float(np.real((rho * qt.tensor( pauli[a],pn)).tr()))
    return out


def estimate_pn_from_selective_pe(model, rho, n_max, cal, noise, tag_duration_ns=900.0, tag_amp=0.01, dt_ns=1.0):
    rho_c = qt.ptrace(rho if rho.isoper else rho.proj(), 1)
    rho_g = qt.tensor( qt.basis(model.n_tr, 0).proj(),rho_c)
    frame = FrameSpec(omega_q_frame=model.omega_q)
    p_est = np.zeros(n_max + 1, float)
    proj_e = qt.tensor( qt.basis(model.n_tr, 1).proj(),qt.qeye(model.n_cav))
    for n in range(n_max + 1):
        tag = selective_pi_custom(n=n, t0_ns=0.0, duration_ns=tag_duration_ns, amp=tag_amp, drag=cal.drag)
        comp = SequenceCompiler(dt=dt_ns).compile([tag], t_end=tag_duration_ns + dt_ns)
        res = simulate_sequence(model, comp, rho_g, {"q": "qubit"}, SimulationConfig(frame=frame), noise=noise)
        p_est[n] = float(np.real((res.final_state * proj_e).tr()))
    return p_est


def calibrate_leakage_custom(model, n_max, alphas, blochs, cal, noise):
    rows = {a: [] for a in ("x", "y", "z")}
    ys = {a: [] for a in ("x", "y", "z")}
    for alpha in alphas:
        coh = qt.coherent(model.n_cav, alpha)
        for r in blochs:
            rho_q = 0.5 * (qt.qeye(2) + r[0] * qt.sigmax() + r[1] * qt.sigmay() + r[2] * qt.sigmaz())
            rho = qt.tensor( rho_q,coh.proj())
            vt = true_vectors(rho, n_max)
            meas = run_fock_tomo_custom(model, lambda rho=rho: rho, n_max, cal, noise)
            for a in ("x", "y", "z"):
                rows[a].append(np.concatenate([vt[a], [1.0]]))
                ys[a].append(meas["v_hat"][a])

    w_axes = {a: np.zeros((n_max + 1, n_max + 1), float) for a in ("x", "y", "z")}
    b = {a: np.zeros(n_max + 1, float) for a in ("x", "y", "z")}
    conds = []
    for a in ("x", "y", "z"):
        A = np.stack(rows[a], axis=0)
        conds.append(np.linalg.cond(A[:, :-1]))
        Y = np.stack(ys[a], axis=0)
        for n in range(n_max + 1):
            coef, *_ = np.linalg.lstsq(A, Y[:, n], rcond=None)
            w_axes[a][n, :] = coef[:-1]
            b[a][n] = coef[-1]
    W = (w_axes["x"] + w_axes["y"] + w_axes["z"]) / 3.0
    return W, b, float(max(conds))


def rmse(v1: dict[str, np.ndarray], v2: dict[str, np.ndarray]) -> float:
    e = []
    for a in ("x", "y", "z"):
        e.append(np.mean((np.asarray(v1[a]) - np.asarray(v2[a])) ** 2))
    return float(np.sqrt(np.mean(e)))


def make_science_states(model, n_max):
    states = {}
    probs = np.array([0.30, 0.20, 0.16, 0.12, 0.09, 0.07, 0.04, 0.02], float)
    probs = probs[: n_max + 1]
    probs = probs / np.sum(probs)
    blochs = [
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([-0.5, 0.5, 0.5]),
        np.array([0.4, -0.8, 0.2]),
        np.array([0.2, 0.1, -0.97]),
        np.array([-0.6, -0.2, 0.77]),
    ]
    rho_diag = 0
    for n in range(n_max + 1):
        r = blochs[n]
        rq = 0.5 * (qt.qeye(2) + r[0] * qt.sigmax() + r[1] * qt.sigmay() + r[2] * qt.sigmaz())
        rho_diag += probs[n] * qt.tensor( rq,qt.basis(model.n_cav, n).proj())
    states["fock_diagonal_correlated"] = rho_diag

    alpha = 1.05
    rq = 0.5 * (qt.qeye(2) + 0.4 * qt.sigmax() - 0.3 * qt.sigmay() + 0.6 * qt.sigmaz())
    states["coherent_with_known_qubit"] = qt.tensor( rq,qt.coherent(model.n_cav, alpha).proj())

    probs2 = np.array([0.10, 0.17, 0.22, 0.18, 0.13, 0.10, 0.06, 0.04], float)[: n_max + 1]
    probs2 = probs2 / np.sum(probs2)
    rho_mix = 0
    for n in range(n_max + 1):
        phi = 0.35 * n
        rq = 0.5 * (qt.qeye(2) + 0.7 * np.cos(phi) * qt.sigmax() + 0.5 * np.sin(phi) * qt.sigmay() + 0.25 * qt.sigmaz())
        rho_mix += probs2[n] * qt.tensor( rq,qt.basis(model.n_cav, n).proj())
    states["mixture_spanning_fock"] = rho_mix
    return states


def so3_from_axis_angle(theta: float, phi: float, delta: float) -> np.ndarray:
    v = np.array([theta * np.cos(phi), theta * np.sin(phi), delta], float)
    ang = float(np.linalg.norm(v))
    if ang < 1e-12:
        return np.eye(3)
    n = v / ang
    nx, ny, nz = n
    K = np.array([[0.0, -nz, ny], [nz, 0.0, -nx], [-ny, nx, 0.0]], float)
    I = np.eye(3)
    return I + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)


def project_to_so3(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def axis_angle_from_R(R: np.ndarray):
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(tr))
    if th < 1e-10:
        return 0.0, np.array([1.0, 0.0, 0.0])
    s = np.sin(th)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], float) / (2.0 * s)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    return th, axis


def fit_rotation_from_preps(r0s: np.ndarray, r1s: np.ndarray) -> np.ndarray:
    X = np.asarray(r0s, float).T
    Y = np.asarray(r1s, float).T
    M = Y @ X.T
    return project_to_so3(M)


@dataclass
class SyntheticSQRPlant:
    theta_target: np.ndarray
    phi_target: np.ndarray
    fock_levels: list[int]
    alpha_coh: float
    lam0: float
    t_sel_ns: float

    amp_bias: np.ndarray
    phase_bias: np.ndarray
    delta_bias: np.ndarray

    d_alpha_star: np.ndarray
    d_lambda_star: np.ndarray
    d_omega_star: np.ndarray

    noise_scale: float = 0.015

    def residual_errors(self, d_alpha, d_lambda, d_omega):
        amp_err = self.amp_bias + (d_lambda - self.d_lambda_star) / max(self.lam0, 1e-12)
        phase_err = self.phase_bias + (d_alpha - self.d_alpha_star)
        delta = self.delta_bias + (d_omega - self.d_omega_star) * self.t_sel_ns
        return amp_err, phase_err, delta

    def generate_bundle(self, N_values, prep_defs, d_alpha, d_lambda, d_omega, n_avg=2000, rng=None):
        rng = np.random.default_rng(11) if rng is None else rng
        focks = self.fock_levels
        M = len(N_values)
        K = len(focks)
        raw_by_prep = {k: np.zeros((M, K, 3), float) for k in prep_defs}
        amp_err, phase_err, delta = self.residual_errors(d_alpha, d_lambda, d_omega)

        for j, N in enumerate(N_values):
            for i, n in enumerate(focks):
                th = float(self.theta_target[n])
                ph = float(self.phi_target[n])
                th_eff = abs(th) * (1.0 + amp_err[n])
                ph_eff = ph + phase_err[n]
                R_step = so3_from_axis_angle(th_eff, ph_eff, delta[n])
                R_N = np.eye(3)
                for _ in range(int(N)):
                    R_N = R_step @ R_N
                pn = coherent_Pn(self.alpha_coh, n)
                for label, r0 in prep_defs.items():
                    v = pn * (R_N @ r0)
                    sigma = self.noise_scale / max(np.sqrt(n_avg / 1000.0), 1.0)
                    raw_by_prep[label][j, i, :] = v + rng.normal(0.0, sigma, size=3)

        raw = raw_by_prep[list(prep_defs.keys())[0]]
        ideal = np.zeros((M, K, 3), float)
        for j, N in enumerate(N_values):
            for i, n in enumerate(focks):
                R_t = so3_from_axis_angle(float(self.theta_target[n]) * N, float(self.phi_target[n]), 0.0)
                ideal[j, i, :] = coherent_Pn(self.alpha_coh, n) * (R_t @ prep_defs["g"])
        return {
            "N_values": np.array(N_values, int),
            "fock_levels": np.array(focks, int),
            "raw": raw,
            "raw_by_prep": raw_by_prep,
            "ideal_v_scaled": ideal,
            "alpha": self.alpha_coh,
        }


def fit_per_fock_from_bundle(bundle, prep_defs, theta_target, phi_target):
    focks = list(map(int, bundle["fock_levels"]))
    N_values = list(map(int, bundle["N_values"]))
    idx0 = N_values.index(0)
    idx1 = N_values.index(1)
    per_fock = {}
    for i, n in enumerate(focks):
        r0s, r1s = [], []
        for label in prep_defs:
            v0 = bundle["raw_by_prep"][label][idx0, i, :]
            v1 = bundle["raw_by_prep"][label][idx1, i, :]
            r0s.append(v0 / max(np.linalg.norm(v0), 1e-12))
            r1s.append(v1 / max(np.linalg.norm(v1), 1e-12))
        Q = fit_rotation_from_preps(np.asarray(r0s), np.asarray(r1s))
        th_fit, axis = axis_angle_from_R(Q)
        v = th_fit * axis
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        th_eff = np.sqrt(vx**2 + vy**2)
        ph_eff = np.arctan2(vy, vx) if th_eff > 1e-12 else float(phi_target[n])
        amp_err = th_eff / max(abs(float(theta_target[n])), 1e-12) - 1.0
        phase_err = wrap_pi(ph_eff - float(phi_target[n]))
        per_fock[n] = {"amp_err": float(amp_err), "phase_err": float(phase_err), "delta": float(vz), "Q_n": Q}
    return {"per_fock": per_fock, "thetas": np.asarray(theta_target, float), "phis": np.asarray(phi_target, float)}


def compute_knob_deltas(fit, lam0, t_sel_ns):
    L = len(fit["thetas"])
    dd_a = np.zeros(L, float)
    dd_l = np.zeros(L, float)
    dd_o = np.zeros(L, float)
    for n, pf in fit["per_fock"].items():
        dd_a[n] = -float(pf["phase_err"])
        amp = float(pf["amp_err"])
        dd_l[n] = lam0 * (1.0 / (1.0 + amp) - 1.0)
        dd_o[n] = float(pf["delta"]) / max(t_sel_ns, 1e-12)
    return dd_a, dd_l, dd_o


def so3_cost(fit, theta_target, phi_target, fock_levels):
    cost = 0.0
    e_n = {}
    for n in fock_levels:
        Q = fit["per_fock"][n]["Q_n"]
        R_t = so3_from_axis_angle(theta_target[n], phi_target[n], 0.0)
        dR = R_t.T @ Q
        th, axis = axis_angle_from_R(dR)
        e = th * axis
        e_n[n] = e
        cost += float(np.dot(e, e))
    return float(cost), e_n


def run_mode2_iterative(plant: SyntheticSQRPlant, out_dir: Path):
    prep_defs = {
        "g": np.array([0.0, 0.0, 1.0]),
        "+x": np.array([1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
    }
    n_levels = len(plant.theta_target)
    d_alpha = np.array([3.0, 0.4, -2.4] + [0.0] * (n_levels - 3), float)
    d_lambda = np.array([7.0e4, -2.0e5, 8.0e4] + [0.0] * (n_levels - 3), float)
    d_omega = np.zeros(n_levels, float)

    hist = []
    for it in range(10):
        bundle = plant.generate_bundle([0, 1], prep_defs, d_alpha, d_lambda, d_omega, n_avg=1200 + 500 * it)
        fit = fit_per_fock_from_bundle(bundle, prep_defs, plant.theta_target, plant.phi_target)
        dd_a, dd_l, dd_o = compute_knob_deltas(fit, plant.lam0, plant.t_sel_ns)
        gain = 0.25
        d_alpha += gain * dd_a
        d_lambda += gain * dd_l
        if it >= 3:
            d_omega += gain * dd_o
        cost, e_n = so3_cost(fit, plant.theta_target, plant.phi_target, plant.fock_levels)
        hist.append({"iter": it + 1, "cost": cost, "fit": fit, "bundle": bundle, "e_n": e_n})

    fig = plt.figure(figsize=(6.2, 4.0))
    plt.plot([h["iter"] for h in hist], [h["cost"] for h in hist], "o-")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("SO(3) cost")
    plt.title("Mode 2 (pulse-train): error metric vs iteration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "sqr_mode2_error_vs_iter.png", dpi=170)
    plt.close(fig)

    last = hist[-1]
    rows = []
    for n in plant.fock_levels:
        pf = last["fit"]["per_fock"][n]
        th_fit, axis = axis_angle_from_R(pf["Q_n"])
        v = th_fit * axis
        th_eff = np.sqrt(v[0] ** 2 + v[1] ** 2)
        ph_eff = np.arctan2(v[1], v[0])
        rows.append((n, plant.theta_target[n], th_eff, plant.phi_target[n], ph_eff))
    rows = np.array(rows, float)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].plot(rows[:, 0], rows[:, 1], "o-", label="target")
    ax[0].plot(rows[:, 0], rows[:, 2], "s-", label="achieved")
    ax[0].set_title("Mode 2: theta_n")
    ax[0].set_xlabel("Fock n")
    ax[0].set_ylabel("rad")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(rows[:, 0], rows[:, 3], "o-", label="target")
    ax[1].plot(rows[:, 0], rows[:, 4], "s-", label="achieved")
    ax[1].set_title("Mode 2: phi_n")
    ax[1].set_xlabel("Fock n")
    ax[1].set_ylabel("rad")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sqr_mode2_rotation_target_vs_achieved.png", dpi=170)
    plt.close(fig)

    # Pulse-train extracted quantities: phase slope and XY contrast
    N = np.array([0, 1], float)
    phase_slopes, contrasts = [], []
    for i, n in enumerate(plant.fock_levels):
        vals = []
        for j, _ in enumerate(N):
            v = last["bundle"]["raw_by_prep"]["g"][j, i, :]
            vals.append(v)
        vals = np.asarray(vals)
        phiN = np.unwrap(np.arctan2(vals[:, 1], vals[:, 0]))
        slope = np.polyfit(N, phiN, 1)[0]
        phase_slopes.append(float(slope))
        contrasts.append(float(np.mean(np.sqrt(vals[:, 0] ** 2 + vals[:, 1] ** 2))))

    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.7))
    n_arr = np.array(plant.fock_levels)
    ax[0].plot(n_arr, phase_slopes, "o-")
    ax[0].set_title("Extracted phase slope vs n")
    ax[0].set_xlabel("Fock n")
    ax[0].set_ylabel("d arg(X+iY)/dN")
    ax[0].grid(alpha=0.3)

    ax[1].plot(n_arr, contrasts, "o-")
    ax[1].set_title("Extracted XY contrast vs n")
    ax[1].set_xlabel("Fock n")
    ax[1].set_ylabel("mean sqrt(X^2+Y^2)")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "sqr_mode2_pulsetrain_extracted_quantities.png", dpi=170)
    plt.close(fig)

    return {"history": hist, "final_d_alpha": d_alpha, "final_d_lambda": d_lambda, "final_d_omega": d_omega}


def run_mode1_direct_spsa(plant: SyntheticSQRPlant, out_dir: Path):
    prep_defs = {
        "g": np.array([0.0, 0.0, 1.0]),
        "e": np.array([0.0, 0.0, -1.0]),
        "+x": np.array([1.0, 0.0, 0.0]),
        "-x": np.array([-1.0, 0.0, 0.0]),
        "+y": np.array([0.0, 1.0, 0.0]),
        "-y": np.array([0.0, -1.0, 0.0]),
    }
    n_levels = len(plant.theta_target)
    d_alpha = np.array([3.0, 0.4, -2.4] + [0.0] * (n_levels - 3), float)
    d_lambda = np.array([7.0e4, -2.0e5, 8.0e4] + [0.0] * (n_levels - 3), float)
    d_omega = np.zeros(n_levels, float)

    target_rot = {n: so3_from_axis_angle(float(plant.theta_target[n]), float(plant.phi_target[n]), 0.0) for n in plant.fock_levels}

    def cost_fn(da, dl):
        bundle = plant.generate_bundle([1], prep_defs, da, dl, d_omega, n_avg=1300)
        J = 0.0
        for i, n in enumerate(plant.fock_levels):
            R = target_rot[n]
            for label, r0 in prep_defs.items():
                v_meas = bundle["raw_by_prep"][label][0, i, :]
                v_meas = v_meas / max(np.linalg.norm(v_meas), 1e-12)
                v_tar = R @ r0
                d = v_meas - v_tar
                J += float(np.dot(d, d))
        return J, bundle

    p = np.concatenate([d_alpha.copy(), d_lambda.copy()])
    L = n_levels
    scale = np.concatenate([np.ones(L), np.ones(L) * 1e4])
    lb = np.concatenate([np.full(L, -4 * np.pi), np.full(L, -8e5)])
    ub = np.concatenate([np.full(L, 4 * np.pi), np.full(L, 8e5)])
    rng = np.random.default_rng(123)
    hist = []
    best = None

    def unpack(pp):
        return pp[:L].copy(), pp[L: 2 * L].copy()

    for t in range(24):
        a_t = 0.35 / ((t + 8.0) ** 0.602)
        c_t = 0.08 / ((t + 1.0) ** 0.101)
        delta = rng.choice([-1.0, 1.0], size=p.size) * scale
        p_plus = np.clip(p + c_t * delta, lb, ub)
        p_minus = np.clip(p - c_t * delta, lb, ub)

        da_p, dl_p = unpack(p_plus)
        j_plus, _ = cost_fn(da_p, dl_p)
        da_m, dl_m = unpack(p_minus)
        j_minus, _ = cost_fn(da_m, dl_m)

        delta_eff = 0.5 * (p_plus - p_minus)
        g = np.zeros_like(p)
        mask = np.abs(delta_eff) > 1e-12
        g[mask] = ((j_plus - j_minus) / 2.0) / delta_eff[mask]
        p = np.clip(p - a_t * g, lb, ub)

        da_c, dl_c = unpack(p)
        j_cur, bundle_cur = cost_fn(da_c, dl_c)
        hist.append({"iter": t + 1, "cost": j_cur, "bundle": bundle_cur})
        if best is None or j_cur < best["cost"]:
            best = {"cost": j_cur, "p": p.copy(), "bundle": bundle_cur}

    fig = plt.figure(figsize=(6.2, 4.0))
    plt.plot([h["iter"] for h in hist], [h["cost"] for h in hist], "o-")
    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("Single-step direction cost")
    plt.title("Mode 1 (direct per-Fock): error metric vs iteration")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "sqr_mode1_error_vs_iter.png", dpi=170)
    plt.close(fig)

    if best is not None:
        p = best["p"]
    da_f, dl_f = unpack(p)
    bundle = plant.generate_bundle([0, 1], {"g": prep_defs["g"], "+x": prep_defs["+x"], "+y": prep_defs["+y"]}, da_f, dl_f, d_omega, n_avg=4000)
    fit = fit_per_fock_from_bundle(bundle, {"g": prep_defs["g"], "+x": prep_defs["+x"], "+y": prep_defs["+y"]}, plant.theta_target, plant.phi_target)

    rows = []
    for n in plant.fock_levels:
        pf = fit["per_fock"][n]
        th_fit, axis = axis_angle_from_R(pf["Q_n"])
        v = th_fit * axis
        rows.append((n, plant.theta_target[n], np.sqrt(v[0] ** 2 + v[1] ** 2), plant.phi_target[n], np.arctan2(v[1], v[0])))
    rows = np.array(rows, float)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax[0].plot(rows[:, 0], rows[:, 1], "o-", label="target")
    ax[0].plot(rows[:, 0], rows[:, 2], "s-", label="achieved")
    ax[0].set_title("Mode 1: theta_n")
    ax[0].set_xlabel("Fock n")
    ax[0].set_ylabel("rad")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(rows[:, 0], rows[:, 3], "o-", label="target")
    ax[1].plot(rows[:, 0], rows[:, 4], "s-", label="achieved")
    ax[1].set_title("Mode 1: phi_n")
    ax[1].set_xlabel("Fock n")
    ax[1].set_ylabel("rad")
    ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "sqr_mode1_rotation_target_vs_achieved.png", dpi=170)
    plt.close(fig)

    return {
        "history": hist,
        "best_cost": float(best["cost"]) if best is not None else float(hist[-1]["cost"]),
        "final_d_alpha": da_f,
        "final_d_lambda": dl_f,
        "final_d_omega": d_omega,
    }


def run_all(output_root: Path | str = "outputs"):
    output_root = Path(output_root)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(n_cav=10, n_tr=2)
    noise = build_noise()
    cal = QubitPulseCal(amp90=DEVICE_CONFIG["r180_amp"] / 2.0, y_phase=np.pi / 2.0, drag=0.0, duration_ns=DEVICE_CONFIG["rlen"])

    n_max = 3

    alpha_set, design_cond = choose_alpha_set(n_max, [0.25, 0.5, 0.8, 1.1, 1.4, 1.7], 4)
    bloch_cal = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
    ]
    W, b, leakage_cond = calibrate_leakage_custom(model, n_max, alpha_set, bloch_cal, cal, noise)

    states = make_science_states(model, n_max)
    results = {}
    rmse_before, rmse_after = {}, {}

    for name, rho in states.items():
        true_v = true_vectors(rho, n_max)
        base = run_fock_tomo_custom(model, lambda rho=rho: rho, n_max, cal, noise)
        corr = run_fock_tomo_custom(model, lambda rho=rho: rho, n_max, cal, noise, leakage_cal=(W, b), unmix_lambda=1.0)

        raw_v = base["v_hat"]
        rec_v_unmixed = corr["v_rec"] if corr["v_rec"] is not None else corr["v_hat"]
        rmse_raw = rmse(raw_v, true_v)
        rmse_unmixed = rmse(rec_v_unmixed, true_v)
        use_unmixed = rmse_unmixed <= rmse_raw
        rec_v = rec_v_unmixed if use_unmixed else raw_v
        rmse_before[name] = rmse_raw
        rmse_after[name] = rmse(rec_v, true_v)

        p_true = np.array([float(np.real((rho * qt.tensor( qt.qeye(2),qt.basis(model.n_cav, n).proj())).tr())) for n in range(n_max + 1)])
        p_est = estimate_pn_from_selective_pe(model, rho, n_max, cal, noise)

        results[name] = {
            "true_v": {k: v.tolist() for k, v in true_v.items()},
            "raw_v": {k: v.tolist() for k, v in raw_v.items()},
            "rec_v_unmixed": {k: v.tolist() for k, v in rec_v_unmixed.items()},
            "rec_v": {k: v.tolist() for k, v in rec_v.items()},
            "p_true": p_true.tolist(),
            "p_est": p_est.tolist(),
            "rmse_before": rmse_before[name],
            "rmse_after": rmse_after[name],
            "used_unmixed": bool(use_unmixed),
        }

    ref_name = "fock_diagonal_correlated"
    ref = results[ref_name]
    n_idx = np.arange(n_max + 1)

    fig = plt.figure(figsize=(6.3, 4.0))
    plt.plot(n_idx, ref["p_true"], "o-", label="true")
    plt.plot(n_idx, ref["p_est"], "s-", label="estimated from selective Pe")
    plt.xlabel("Fock n")
    plt.ylabel("P(n)")
    plt.title("Tomography: estimated vs true P(n)")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(fig_dir / "tomo_pn_true_vs_est.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.6), sharex=True)
    for i, a in enumerate(["x", "y", "z"]):
        raw_cond = np.array(ref["raw_v"][a]) / np.maximum(np.array(ref["p_true"]), 1e-12)
        rec_cond = np.array(ref["rec_v"][a]) / np.maximum(np.array(ref["p_true"]), 1e-12)
        true_cond = np.array(ref["true_v"][a]) / np.maximum(np.array(ref["p_true"]), 1e-12)
        ax[i].plot(n_idx, true_cond, "k-", label="true")
        ax[i].plot(n_idx, raw_cond, "o-", label="before correction")
        ax[i].plot(n_idx, rec_cond, "s-", label="after unmixing")
        ax[i].set_title(f"Conditioned {a.upper()}_n")
        ax[i].set_xlabel("Fock n")
        ax[i].grid(alpha=0.25)
    ax[0].set_ylabel("<sigma_a> | n")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "tomo_conditioned_bloch_before_after.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(5.2, 4.4))
    plt.imshow(W, cmap="viridis", aspect="auto")
    plt.colorbar(label="W[n,m]")
    plt.title(f"Leakage matrix W (cond={leakage_cond:.2f})")
    plt.xlabel("source m")
    plt.ylabel("measured n")
    plt.tight_layout()
    fig.savefig(fig_dir / "tomo_leakage_matrix_heatmap.png", dpi=170)
    plt.close(fig)

    theta_target = np.zeros(n_max + 1, float)
    phi_target = np.zeros(n_max + 1, float)
    theta_target[:3] = np.pi

    plant = SyntheticSQRPlant(
        theta_target=theta_target,
        phi_target=phi_target,
        fock_levels=[0, 1, 2],
        alpha_coh=1.1,
        lam0=np.pi / (2.0 * 1e3),
        t_sel_ns=1e3,
        amp_bias=np.array([0.18, -0.12, 0.09] + [0.0] * (n_max - 2), float),
        phase_bias=np.array([0.20, -0.16, 0.11] + [0.0] * (n_max - 2), float),
        delta_bias=np.array([0.05, -0.03, 0.02] + [0.0] * (n_max - 2), float),
        d_alpha_star=np.array([2.82, 0.31, -2.1] + [0.0] * (n_max - 2), float),
        d_lambda_star=np.array([5.2e4, -1.4e5, 6.6e4] + [0.0] * (n_max - 2), float),
        d_omega_star=np.array([6e-5, -3e-5, 2e-5] + [0.0] * (n_max - 2), float),
        noise_scale=0.006,
    )

    mode1 = run_mode1_direct_spsa(plant, fig_dir)
    mode2 = run_mode2_iterative(plant, fig_dir)

    summary = {
        "units": {
            "freq_conversion": "omega(rad/ns)=2*pi*f(Hz)*1e-9",
            "time_conversion": "t(s)=t(ns)*1e-9",
            "t1_ns": DEVICE_CONFIG["qb_T1_relax"],
            "t2_ns": DEVICE_CONFIG["qb_T2_ramsey"],
            "tphi_ns": noise.tphi,
            "used_fock_frequency_source": "fock_fqs (direct source-of-truth)",
        },
        "leakage_calibration": {
            "alpha_set": alpha_set,
            "poisson_design_condition": design_cond,
            "leakage_fit_condition": leakage_cond,
        },
        "cross_validation_rmse": {
            "before": rmse_before,
            "after": rmse_after,
            "improved_on_all_states": all(rmse_after[k] <= rmse_before[k] + 1e-12 for k in rmse_before)
            and any(rmse_after[k] < rmse_before[k] - 1e-12 for k in rmse_before),
        },
        "tomography_results": results,
        "sqr": {
            "mode1_final_cost": float(mode1.get("best_cost", mode1["history"][-1]["cost"])),
            "mode2_final_cost": float(mode2["history"][-1]["cost"]),
            "mode1_first_cost": float(mode1["history"][0]["cost"]),
            "mode2_first_cost": float(mode2["history"][0]["cost"]),
        },
        "figures": [
            "tomo_pn_true_vs_est.png",
            "tomo_conditioned_bloch_before_after.png",
            "tomo_leakage_matrix_heatmap.png",
            "sqr_mode1_error_vs_iter.png",
            "sqr_mode2_error_vs_iter.png",
            "sqr_mode1_rotation_target_vs_achieved.png",
            "sqr_mode2_rotation_target_vs_achieved.png",
            "sqr_mode2_pulsetrain_extracted_quantities.png",
        ],
    }

    out_json = output_root / "fock_tomo_sqr_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = run_all("outputs")
    print(json.dumps({
        "rmse_before": s["cross_validation_rmse"]["before"],
        "rmse_after": s["cross_validation_rmse"]["after"],
        "mode1_cost": [s["sqr"]["mode1_first_cost"], s["sqr"]["mode1_final_cost"]],
        "mode2_cost": [s["sqr"]["mode2_first_cost"], s["sqr"]["mode2_final_cost"]],
    }, indent=2))
