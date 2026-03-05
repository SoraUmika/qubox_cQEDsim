# Figure Caption Style Guide

## General Rules

1. **First sentence:** Describe what is plotted and the key takeaway.
2. **Axes:** Name all axes with units in parentheses.
3. **Parameters:** State key simulation parameters: χ/2π, T_SQR, N_levels, max_n.
4. **Method:** Mention the optimizer (Powell + L-BFGS-B) and metric (process fidelity).
5. **Statistics:** If showing distributions, state: N targets, percentile bands, seed.
6. **Length:** 3–5 sentences. Technical but readable.

## Templates by Figure Type

### Duration sweep (sqr_benchmark_duration_summary.png)
"Logical fidelity $F_{\mathrm{logical}}$ (top), guard leakage
$\epsilon_{\mathrm{guard}}$ (middle), and joint success probability (bottom)
versus SQR pulse duration $T$ for [N] seeded random targets. Shaded bands show
25th–75th percentile range. Success is defined as
$F_{\mathrm{logical}} \geq [threshold]$ and
$\epsilon_{\mathrm{guard}} \leq [threshold]$. [N_logical] logical levels and
[N_guard] guard levels calibrated with dispersive shift
$\chi/2\pi = [value]$\,MHz."

### Per-manifold angles (sqr_benchmark_representative_angles.png)
"Achieved conditional rotation angle $\theta_n$ (top) and axis phase $\phi_n$
(bottom) for a representative SQR target across [N_durations] pulse durations.
Dashed line: target values. Markers: optimized values per Fock level $n$."

### Infidelity heatmap (sqr_benchmark_infidelity_heatmap.png)
"Per-manifold process infidelity $1 - F_{\mathrm{proc}}^{(n)}$ as a function of
Fock level $n$ (horizontal) and SQR pulse duration (vertical) for a
representative target. Colorscale: magma. Lower values indicate better agreement
between simulated and target conditional qubit unitaries."

### Convergence trace (sqr_benchmark_convergence.png)
"Optimizer convergence traces for a representative SQR target at [N_durations]
pulse durations. Top: logical fidelity; middle: guard leakage; bottom: total
objective $\mathcal{L} = (1 - F_{\mathrm{logical}}) +
\lambda_{\mathrm{guard}} \epsilon_{\mathrm{guard}}$. Horizontal axis: objective
evaluation count."

### Calibration loss improvement (Section 8 plot)
"SQR calibration process infidelity before (dashed) and after (solid)
per-manifold optimization for gate \texttt{[name]}. Vertical axis: log-scale
infidelity. [N] levels calibrated; skipped levels (identity target) omitted."
