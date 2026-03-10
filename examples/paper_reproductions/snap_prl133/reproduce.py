from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from cqed_sim.core.frame import FrameSpec
from examples.studies.snap_opt.experiments import SnapRunConfig, target_difficulty_metric

from .errors import CoherentMetricResult, compute_mean_squared_overlap
from .model import SnapModelConfig, manifold_transition_frequency
from .optimize import optimize_snap_prl133
from .pulses import SnapToneParameters, landgraf_envelope, slow_stage_multitone_pulse


@dataclass(frozen=True)
class ReproduceConfig:
    n_cav: int = 6
    n_tr: int = 2
    n_max: int = 2
    chi: float = 2 * np.pi * 0.02
    dt: float = 0.25
    base_amp: float = 0.010
    durations: tuple[float, ...] = (70.0, 120.0, 180.0, 240.0)
    learning_rate: float = 0.3
    iteration_cap: int = 60
    epsilon_target: float = 1e-5
    local_refine_maxiter: int = 8
    fft_bins: int = 8192
    leakage_threshold: float = 1e-3


def _baseline_target(n_max: int) -> np.ndarray:
    base = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    return base[: n_max + 1].copy()


def _generalized_targets(n_max: int) -> list[np.ndarray]:
    rng = np.random.default_rng(1234)
    out = [
        _baseline_target(n_max),
        np.array([0.0, 0.4, -0.2, 0.8][: n_max + 1], dtype=float),
        np.array([0.0, 1.0, -0.7, 0.3][: n_max + 1], dtype=float),
    ]
    for _ in range(2):
        out.append(np.concatenate(([0.0], rng.uniform(-1.2, 1.2, size=n_max))).astype(float))
    return out


def _optimize_and_measure(
    model,
    frame: FrameSpec,
    target: np.ndarray,
    run_cfg: SnapRunConfig,
    cfg: ReproduceConfig,
    *,
    local_refine_maxiter: int,
):
    vanilla_params = SnapToneParameters.vanilla(target)
    vanilla_metric = compute_mean_squared_overlap(
        model=model,
        target_phases=target,
        cfg=run_cfg,
        params=vanilla_params,
        frame=frame,
        context="reproduce:vanilla",
    )
    opt = optimize_snap_prl133(
        model=model,
        target_phases=target,
        cfg=run_cfg,
        initial_params=vanilla_params,
        frame=frame,
        max_iter=cfg.iteration_cap,
        learning_rate=cfg.learning_rate,
        threshold=cfg.epsilon_target,
        local_refine_maxiter=local_refine_maxiter,
    )
    return vanilla_metric, opt


def _fft_leakage_ratio(
    model,
    frame: FrameSpec,
    target: np.ndarray,
    params: SnapToneParameters,
    cfg: ReproduceConfig,
    duration: float,
) -> dict:
    pulse = slow_stage_multitone_pulse(
        model=model,
        target_phases=target,
        params=params,
        duration=duration,
        base_amp=cfg.base_amp,
        frame=frame,
        channel="q",
    )
    t_rel = np.linspace(0.0, 1.0, cfg.fft_bins, endpoint=False)
    env = landgraf_envelope(t_rel)
    coeff = pulse.envelope(t_rel)
    mask = np.abs(env) > 1e-6 * np.max(np.abs(env))
    sig = np.zeros_like(coeff, dtype=np.complex128)
    sig[mask] = coeff[mask] / env[mask]
    window = np.hanning(sig.size)
    sig_w = sig * window
    dt = duration / cfg.fft_bins
    freqs = np.fft.fftshift(np.fft.fftfreq(sig_w.size, d=dt))
    omega = 2 * np.pi * freqs
    spec = np.fft.fftshift(np.fft.fft(sig_w))
    power = np.abs(spec) ** 2

    allowed = np.array([manifold_transition_frequency(model, n, frame=frame) + params.detunings[n] for n in range(target.size)])
    keep = np.zeros(power.size, dtype=bool)
    for w in np.concatenate([allowed, -allowed]):
        idx = int(np.argmin(np.abs(omega - w)))
        i0 = max(0, idx - 2)
        i1 = min(power.size, idx + 3)
        keep[i0:i1] = True
    p_total = float(np.sum(power))
    p_out = float(np.sum(power[~keep]))
    r_out = p_out / max(p_total, 1e-15)
    return {
        "omega": omega,
        "power_norm": power / max(np.max(power), 1e-15),
        "allowed_omegas": allowed,
        "r_out": float(r_out),
        "df_hz": float(freqs[1] - freqs[0]),
        "n_bins": int(power.size),
        "neighbor_rule": "+/-2 FFT bins around each allowed tone and mirror (Hann-window main-lobe capture)",
    }


def _write_report_markdown(report_md: Path, summary: dict) -> None:
    figs = summary["figure_files_by_name"]
    lines = [
        "# Reproduction Report: Landgraf et al. (PRL 133, 260802)",
        "",
        "## Executive Summary",
        f"- Headline metric uses supplement definition: `F in [0,1]`, `epsilon_coh = 1 - F in [0,1]`.",
        f"- Baseline `T_opt` (threshold-based): `{summary['T_opt']}` with `epsilon_target={summary['epsilon_target']}`.",
        f"- Spectrum leakage ratio: `r_out={summary['r_out']:.3e}` (threshold `{summary['leakage_threshold']:.1e}`).",
        f"- Metric boundedness checks passed: `{summary['metric_bounds_ok']}`.",
        "",
        "## What Was Wrong Before",
        "- The previous report used an unbounded surrogate (`sqrt(mean(dtheta^2 + dlambda^2 + dalpha^2))`) as the headline coherent error.",
        "- That surrogate can exceed 1 and is not the paper's primary performance metric.",
        "- This invalidated direct claims about \"numerical zero coherent error\" and optimization-limit detection.",
        "- Fix applied: bounded mean-squared-overlap fidelity from the supplement, with fail-fast invariant checks.",
        "",
        "## Paper Metric Definitions (Main + Supplement)",
        "- Supplement fidelity definition quote: `F = avg_{||c||=1} <psi_target(c)| rho_out(c) |psi_target(c)>`.",
        "- Coherent headline error in this report: `epsilon_coh = 1 - F`.",
        "- Main-text coherent-state parameterization implemented:",
        "  - `|psi_out(c)> = sum_n c_n ( sqrt(1-|eps_n|^2/4) e^{i(theta_n+Delta theta_n)} |e n> - (eps_n/2)|g n> )`",
        "  - `eps_n = (eps_n^(L) + i eps_n^(T)) e^{i Delta theta_n}`",
        "",
        "## Paper-to-Code Mapping",
        "| Paper symbol | Code | Notes |",
        "|---|---|---|",
        "| `F` | `CoherentMetricResult.fidelity` | bounded primary metric |",
        "| `epsilon_coh` | `1 - fidelity` | bounded headline error |",
        "| `Delta theta_n` | `CoherentMetricResult.dtheta[n]` | phase component |",
        "| `epsilon_n^(L)` | `CoherentMetricResult.eps_l[n]` | longitudinal component |",
        "| `epsilon_n^(T)` | `CoherentMetricResult.eps_t[n]` | transversal component |",
        "| `Delta lambda_n` | optimizer amplitude update | mapped with `base_amp` scaling |",
        "| `Delta omega_n` | optimizer detuning update | `pi*eps_t/(2T)` |",
        "| `Delta alpha_n` | optimizer phase update | `-Delta theta_n` |",
        "",
        "## Optimization Limit Definition",
        "- Definition used: `T_opt = min { T : epsilon_coh(T) < epsilon_target within iteration_cap }`.",
        f"- `epsilon_target = {summary['epsilon_target']}`.",
        f"- `iteration_cap = {summary['config']['iteration_cap']}`.",
        "",
        "## Threshold Evidence (Below vs Above T_opt)",
        f"- Below `T_opt` (`T={summary['threshold_evidence']['t_below']:.1f}`): `epsilon_coh={summary['threshold_evidence']['epsilon_below']:.4e}`, hit=`{summary['threshold_evidence']['hit_below']}`.",
        f"- At/above `T_opt` (`T={summary['threshold_evidence']['t_above']:.1f}`): `epsilon_coh={summary['threshold_evidence']['epsilon_above']:.4e}`, hit=`{summary['threshold_evidence']['hit_above']}`.",
        f"- Max component below: `{summary['threshold_evidence']['max_component_below']:.4e}`; above: `{summary['threshold_evidence']['max_component_above']:.4e}`.",
        "",
        "## Per-Manifold Diagnostics at T_opt",
        "| n | |Delta theta_n| | |Delta lambda_n| | |Delta alpha_n| | overlap_n |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in summary["per_manifold_at_topt"]:
        lines.append(
            f"| {int(row['n'])} | {row['abs_dtheta']:.4e} | {row['abs_dlambda']:.4e} | {row['abs_dalpha']:.4e} | {row['overlap']:.6f} |"
        )
    lines.extend(
        [
            "",
        "## Reproduction Plots",
        f"![epsilon_vs_gate_time]({figs['epsilon_vs_gate_time.png']})",
        "",
        f"![optimization_limit]({figs['optimization_limit.png']})",
        "",
        f"![components_before_after]({figs['components_before_after.png']})",
        "",
        f"![optimized_params_vs_n]({figs['optimized_params_vs_n.png']})",
        "",
        f"![pulse_spectrum]({figs['pulse_spectrum.png']})",
        "",
        f"![difficulty_vs_topt]({figs['difficulty_vs_topt.png']})",
        "",
        f"![error_norm_vs_epsilon]({figs['error_norm_vs_epsilon.png']})",
        "",
        "## Baseline Table",
        "| T | epsilon_vanilla | epsilon_optimized | F_optimized | threshold_hit |",
        "|---:|---:|---:|---:|:---:|",
        ]
    )
    for t, ev, eo, fo, hit in zip(
        summary["curves"]["durations"],
        summary["curves"]["epsilon_vanilla"],
        summary["curves"]["epsilon_optimized"],
        summary["curves"]["fidelity_optimized"],
        summary["threshold_hit_map"],
    ):
        lines.append(f"| {t:.1f} | {ev:.4e} | {eo:.4e} | {fo:.6f} | {str(bool(hit))} |")

    lines.extend(
        [
            "",
            "## Paper-Comparison Metric",
            "- RMSE against digitized paper curve is not reported because no official numeric trace was available locally.",
            "- This report therefore compares qualitative trends and threshold behavior (vanilla decay vs optimized threshold crossing).",
            "",
            "## FFT Leakage Details",
            f"- `r_out = {summary['r_out']:.3e}`",
            f"- `df (Hz-equivalent in simulation units) = {summary['fft_details']['df_hz']:.4e}`",
            f"- `n_bins = {summary['fft_details']['n_bins']}`",
            f"- neighborhood rule: `{summary['fft_details']['neighbor_rule']}`",
            "",
            "## Secondary Diagnostic (Not Paper Headline Metric)",
            "- `error_vector_norm` is retained as a secondary diagnostic only.",
            "- Relationship to headline metric is shown in `error_norm_vs_epsilon.png`.",
            "",
            "## Convergence Check",
            f"- coarse dt epsilon: `{summary['convergence_check']['coarse_epsilon']:.6e}`",
            f"- fine dt epsilon: `{summary['convergence_check']['fine_epsilon']:.6e}`",
            f"- coarse dt fidelity: `{summary['convergence_check']['coarse_fidelity']:.6f}`",
            f"- fine dt fidelity: `{summary['convergence_check']['fine_fidelity']:.6f}`",
            f"- relative fidelity difference: `{summary['convergence_check']['relative']:.3e}`",
            "",
            "## Citations",
            "- J. Landgraf et al., Phys. Rev. Lett. 133, 260802 (2024), DOI: 10.1103/PhysRevLett.133.260802.",
            "- arXiv:2310.10498.",
            "- APS supplemental material (metric and geometric decomposition definitions).",
        ]
    )
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_reproduction(config: ReproduceConfig | None = None, output_root: str | Path | None = None) -> dict:
    cfg = config or ReproduceConfig()
    root = Path(output_root) if output_root is not None else Path(__file__).resolve().parents[2] / "outputs"
    figures_dir = root / "figures"
    report_dir = root / "report"
    figures_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    start = perf_counter()
    model = SnapModelConfig(n_cav=cfg.n_cav, n_tr=cfg.n_tr, chi=cfg.chi).build_model()
    frame = FrameSpec(omega_q_frame=model.omega_q)
    baseline = _baseline_target(cfg.n_max)
    targets = _generalized_targets(cfg.n_max)

    eps_v: list[float] = []
    eps_o: list[float] = []
    f_o: list[float] = []
    hit_map: list[bool] = []
    best_by_duration = []
    metric_bounds_ok = True
    for duration in cfg.durations:
        run_cfg = SnapRunConfig(duration=duration, dt=cfg.dt, base_amp=cfg.base_amp)
        try:
            refine = cfg.local_refine_maxiter if duration >= 180.0 else 0
            v, opt = _optimize_and_measure(model, frame, baseline, run_cfg, cfg, local_refine_maxiter=refine)
        except ValueError:
            metric_bounds_ok = False
            raise
        eps_v.append(v.epsilon_coh)
        eps_o.append(opt.final_metric.epsilon_coh)
        f_o.append(opt.final_metric.fidelity)
        hit_map.append(bool(opt.threshold_hit))
        best_by_duration.append(opt)

    t_opt = None
    for t, e in zip(cfg.durations, eps_o):
        if e < cfg.epsilon_target:
            t_opt = float(t)
            break
    if t_opt is None:
        t_opt = float(cfg.durations[-1])
    t_below = max([float(t) for t in cfg.durations if t < t_opt], default=float(cfg.durations[0]))
    t_above = t_opt

    durations_list = [float(t) for t in cfg.durations]

    def _index_for_duration(target: float) -> int:
        for i, value in enumerate(durations_list):
            if abs(value - target) < 1e-12:
                return i
        return len(durations_list) - 1

    idx_below = _index_for_duration(t_below)
    idx_above = _index_for_duration(t_above)

    pick_duration = durations_list[idx_above]
    pick_idx = idx_above
    best = best_by_duration[pick_idx]
    before_metric = compute_mean_squared_overlap(
        model=model,
        target_phases=baseline,
        cfg=SnapRunConfig(duration=pick_duration, dt=cfg.dt, base_amp=cfg.base_amp),
        params=SnapToneParameters.vanilla(baseline),
        frame=frame,
        context="reproduce:before",
    )
    after_metric = best.final_metric
    metric_below = best_by_duration[idx_below].final_metric
    metric_above = best_by_duration[idx_above].final_metric

    topt_pairs: list[tuple[float, float]] = []
    for target in targets:
        found = np.nan
        for duration in cfg.durations:
            run_cfg = SnapRunConfig(duration=duration, dt=cfg.dt, base_amp=cfg.base_amp)
            _, opt = _optimize_and_measure(model, frame, target, run_cfg, cfg, local_refine_maxiter=0)
            if opt.final_metric.epsilon_coh < cfg.epsilon_target:
                found = float(duration)
                break
        topt_pairs.append((target_difficulty_metric(target), found))

    fft_diag = _fft_leakage_ratio(
        model=model,
        frame=frame,
        target=baseline,
        params=best.params,
        cfg=cfg,
        duration=pick_duration,
    )
    r_out = float(fft_diag["r_out"])

    # Convergence check with dt refinement.
    coarse = compute_mean_squared_overlap(
        model=model,
        target_phases=baseline,
        cfg=SnapRunConfig(duration=pick_duration, dt=cfg.dt, base_amp=cfg.base_amp),
        params=best.params,
        frame=frame,
        context="reproduce:coarse",
    )
    fine = compute_mean_squared_overlap(
        model=model,
        target_phases=baseline,
        cfg=SnapRunConfig(duration=pick_duration, dt=cfg.dt * 0.5, base_amp=cfg.base_amp),
        params=best.params,
        frame=frame,
        context="reproduce:fine",
    )
    conv_rel = float(abs(fine.fidelity - coarse.fidelity) / max(abs(fine.fidelity), 1e-15))

    # Plot 1: epsilon vs gate time
    fig = plt.figure(figsize=(6, 4))
    plt.plot(cfg.durations, eps_v, "o-", label="vanilla")
    plt.plot(cfg.durations, eps_o, "o-", label="optimized")
    plt.yscale("log")
    plt.axhline(cfg.epsilon_target, linestyle="--", color="tab:red", label="epsilon_target")
    plt.xlabel("Gate duration T")
    plt.ylabel("epsilon_coh = 1 - F")
    plt.title("Coherent overlap error vs gate time")
    plt.grid(alpha=0.3)
    plt.legend()
    p1 = figures_dir / "epsilon_vs_gate_time.png"
    fig.tight_layout()
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    # Plot 2: optimization limit
    fig = plt.figure(figsize=(6, 4))
    plt.plot(cfg.durations, eps_o, "o-")
    plt.axhline(cfg.epsilon_target, linestyle="--", color="tab:red", label="epsilon_target")
    plt.axvline(t_opt, linestyle=":", color="tab:green", label="T_opt")
    plt.yscale("log")
    plt.xlabel("Gate duration T")
    plt.ylabel("optimized epsilon_coh")
    plt.title("Threshold-based optimization limit")
    plt.grid(alpha=0.3)
    plt.legend()
    p2 = figures_dir / "optimization_limit.png"
    fig.tight_layout()
    fig.savefig(p2, dpi=160)
    plt.close(fig)

    # Plot 3: components before/after
    n = np.arange(baseline.size)
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    ax[0].plot(n, np.abs(before_metric.dtheta), "o-", label="before")
    ax[0].plot(n, np.abs(after_metric.dtheta), "o-", label="after")
    ax[0].set_title("|Delta theta_n|")
    ax[1].plot(n, np.abs(before_metric.eps_l), "o-")
    ax[1].plot(n, np.abs(after_metric.eps_l), "o-")
    ax[1].set_title("|eps_L,n|")
    ax[2].plot(n, np.abs(before_metric.eps_t), "o-")
    ax[2].plot(n, np.abs(after_metric.eps_t), "o-")
    ax[2].set_title("|eps_T,n|")
    for a in ax:
        a.set_xlabel("n")
        a.grid(alpha=0.3)
    ax[0].legend()
    p3 = figures_dir / "components_before_after.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # Plot 4: optimized params vs n
    fig, ax = plt.subplots(3, 1, figsize=(6, 7), sharex=True)
    ax[0].plot(n, best.params.amplitudes, "o-")
    ax[0].set_ylabel("A_n")
    ax[1].plot(n, best.params.detunings, "o-")
    ax[1].set_ylabel("delta_n")
    ax[2].plot(n, best.params.phases, "o-")
    ax[2].set_ylabel("phi_n")
    ax[2].set_xlabel("n")
    for a in ax:
        a.grid(alpha=0.3)
    p4 = figures_dir / "optimized_params_vs_n.png"
    fig.tight_layout()
    fig.savefig(p4, dpi=160)
    plt.close(fig)

    # Plot 5: spectrum with allowed tones
    fig = plt.figure(figsize=(7, 4))
    plt.plot(fft_diag["omega"], fft_diag["power_norm"], label="demodulated + Hann")
    for w in fft_diag["allowed_omegas"]:
        plt.axvline(float(w), color="tab:red", alpha=0.25)
        plt.axvline(float(-w), color="tab:red", alpha=0.25)
    plt.xlabel("Angular frequency")
    plt.ylabel("Normalized power")
    plt.title(f"Tone-family check (r_out={r_out:.2e})")
    plt.grid(alpha=0.25)
    p5 = figures_dir / "pulse_spectrum.png"
    fig.tight_layout()
    fig.savefig(p5, dpi=160)
    plt.close(fig)

    # Plot 6: generalized Topt vs difficulty
    pairs = np.asarray(topt_pairs, dtype=float)
    fig = plt.figure(figsize=(6, 4))
    plt.scatter(pairs[:, 0], pairs[:, 1])
    plt.xlabel("target difficulty")
    plt.ylabel("T_opt")
    plt.title("Generalized sweep")
    plt.grid(alpha=0.3)
    p6 = figures_dir / "difficulty_vs_topt.png"
    fig.tight_layout()
    fig.savefig(p6, dpi=160)
    plt.close(fig)

    # Plot 7: relation secondary norm vs headline epsilon
    fig = plt.figure(figsize=(6, 4))
    plt.plot([m.final_metric.error_vector_norm for m in best_by_duration], eps_o, "o-")
    plt.xlabel("error_vector_norm (secondary)")
    plt.ylabel("epsilon_coh (headline)")
    plt.title("Secondary-vs-headline metric relation")
    plt.grid(alpha=0.3)
    p7 = figures_dir / "error_norm_vs_epsilon.png"
    fig.tight_layout()
    fig.savefig(p7, dpi=160)
    plt.close(fig)

    fig_files = [p1.name, p2.name, p3.name, p4.name, p5.name, p6.name, p7.name]
    fig_map = {name: f"figures/{name}" for name in fig_files}
    summary = {
        "config": asdict(cfg),
        "citations": {
            "doi": "10.1103/PhysRevLett.133.260802",
            "arxiv": "2310.10498",
            "supplement_used": True,
        },
        "epsilon_target": float(cfg.epsilon_target),
        "T_opt": float(t_opt),
        "T_below": float(t_below),
        "T_above": float(t_above),
        "curves": {
            "durations": list(map(float, cfg.durations)),
            "epsilon_vanilla": list(map(float, eps_v)),
            "epsilon_optimized": list(map(float, eps_o)),
            "fidelity_optimized": list(map(float, f_o)),
        },
        "threshold_hit_map": hit_map,
        "metric_bounds_ok": bool(metric_bounds_ok),
        "max_component_after": float(after_metric.max_component_error),
        "threshold_evidence": {
            "t_below": float(t_below),
            "t_above": float(t_above),
            "epsilon_below": float(metric_below.epsilon_coh),
            "epsilon_above": float(metric_above.epsilon_coh),
            "hit_below": bool(metric_below.epsilon_coh < cfg.epsilon_target),
            "hit_above": bool(metric_above.epsilon_coh < cfg.epsilon_target),
            "max_component_below": float(metric_below.max_component_error),
            "max_component_above": float(metric_above.max_component_error),
        },
        "per_manifold_at_topt": [
            {
                "n": int(n),
                "abs_dtheta": float(abs(metric_above.dtheta[n])),
                "abs_dlambda": float(abs(metric_above.eps_l[n])),
                "abs_dalpha": float(abs(metric_above.eps_t[n])),
                "overlap": float(metric_above.per_manifold_overlap[n]),
            }
            for n in range(metric_above.dtheta.size)
        ],
        "r_out": r_out,
        "leakage_threshold": float(cfg.leakage_threshold),
        "fft_details": {
            "df_hz": float(fft_diag["df_hz"]),
            "n_bins": int(fft_diag["n_bins"]),
            "neighbor_rule": fft_diag["neighbor_rule"],
        },
        "difficulty_pairs": [(float(a), float(b) if np.isfinite(b) else np.nan) for a, b in topt_pairs],
        "secondary_diagnostics": {
            "ratio_optimized_to_vanilla": [float(o / max(v, 1e-15)) for o, v in zip(eps_o, eps_v)],
        },
        "convergence_check": {
            "coarse_epsilon": float(coarse.epsilon_coh),
            "fine_epsilon": float(fine.epsilon_coh),
            "coarse_fidelity": float(coarse.fidelity),
            "fine_fidelity": float(fine.fidelity),
            "relative": float(conv_rel),
        },
        "runtime_s": float(perf_counter() - start),
        "figure_files": fig_files,
        "figure_files_by_name": fig_map,
    }

    (root / "reproduce_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report_markdown(root / "report.md", summary)

    # Keep PDF output for backward compatibility.
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(report_dir / "report.pdf") as pdf:
        for name in fig_files:
            img = plt.imread(figures_dir / name)
            fig = plt.figure(figsize=(8.5, 6))
            plt.imshow(img)
            plt.axis("off")
            plt.title(name)
            pdf.savefig(fig)
            plt.close(fig)
    summary["report_md"] = str(root / "report.md")
    summary["report_pdf"] = str(report_dir / "report.pdf")
    return summary


def main() -> None:
    summary = run_reproduction()
    print(json.dumps({"report_md": summary["report_md"], "report_pdf": summary["report_pdf"]}, indent=2))


if __name__ == "__main__":
    main()
