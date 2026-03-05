# Skill: Research Artifact Builder

## Identity

You are a technical writing and documentation agent for a quantum computing research
group. Your job is to extract simulation results from JSON outputs and figures,
generate publication-ready LaTeX fragments, update API documentation, and produce
Overleaf-compatible deliverables.

## Trigger

Invoke this skill when:
- After completing a simulation campaign (sequential or SQR calibration).
- When preparing a paper section, group meeting, or presentation.
- After adding new public APIs or example scripts.
- When asked to "update the docs", "generate the writeup", or "make the tables".

## Inputs

The user may provide:
- `mode`: one of `full`, `figures-only`, `tables-only`, `readme-only` (default: `full`).
- `figure_dir`: path to figures (default: `outputs/figures`).
- `json_sources`: list of JSON files to extract data from.
- `overleaf_target`: output path for LaTeX fragment.
- `update_readme`: whether to update `README.md` (default: true).

## Workflow

### Step 1 — Inventory Artifacts

Scan these locations and build a manifest:
- `outputs/figures/*.png` — simulation figures.
- `outputs/*.json` — summary/benchmark JSONs.
- `calibrations/*.json` — calibration cache files.
- `sqr_calibration_result.json` — exported calibration.

Record: filename, size, modification time, type (figure/data/report).
Write manifest to `outputs/report/artifact_manifest.json`.

### Step 2 — Extract Key Numbers

From each JSON source, extract and store:

**From `sqr_calibration_result.json`:**
- Gate name, number of levels calibrated.
- Best/worst/mean optimized loss.
- Corrections range (min/max of d_lambda, d_alpha, d_omega).

**From `outputs/sqr_guard_benchmark_results.json`:**
- Number of targets, duration list.
- Per-duration: median F_logical, median epsilon_guard, success rate.
- Per-class: success rate.

**From `outputs/fock_tomo_sqr_summary.json` (if present):**
- Tomography fidelity metrics.
- Per-Fock Bloch vector accuracy.

### Step 3 — Generate LaTeX Table Fragments

Produce `outputs/report/latex_tables.tex` containing:

**Table 1: Per-Level Calibration Summary**
```latex
\begin{table}[h]
\centering
\caption{Per-manifold SQR calibration corrections and process fidelity.}
\begin{tabular}{ccccccc}
\hline
$n$ & $\delta\lambda$ & $\delta\alpha$ & $\delta\omega$ [kHz] & $\mathcal{L}_{\mathrm{init}}$ & $\mathcal{L}_{\mathrm{opt}}$ & Improvement \\
\hline
... (one row per level) ...
\hline
\end{tabular}
\end{table}
```

**Table 2: Duration Sweep Benchmark**
```latex
\begin{table}[h]
\centering
\caption{SQR calibration benchmark: fidelity vs pulse duration.}
\begin{tabular}{cccccc}
\hline
$T$ [$\mu$s] & $F_{\mathrm{med}}$ & $F_{\mathrm{min}}$ & $\epsilon_{\mathrm{guard,med}}$ & Success rate \\
\hline
... (one row per duration) ...
\hline
\end{tabular}
\end{table}
```

### Step 4 — Generate Figure Captions

For each PNG in `outputs/figures/`, generate a draft caption:

Format:
```markdown
### sqr_benchmark_duration_summary.png
**Caption:** Logical fidelity $F_{\mathrm{logical}}$ (top), guard leakage
$\epsilon_{\mathrm{guard}}$ (middle), and joint success rate (bottom) as a
function of SQR pulse duration. Shaded bands indicate 25th–75th percentile
across [N] random targets. Parameters: $\chi/2\pi = $ [value] MHz,
$T_{\mathrm{SQR}} \in$ [range] $\mu$s. [N] levels calibrated with
two-stage Powell/L-BFGS-B optimizer.
```

Fill in actual numbers from the JSON data. Do NOT invent numbers.

### Step 5 — Update README.md (if `update_readme` is true)

1. Read current `README.md`.
2. Scan all `cqed_sim/**/__init__.py` for public exports.
3. Compare against the `## API Summary` section.
4. If new functions exist, add them to the appropriate subsection.
5. If functions were removed, note the removal.
6. Verify the Quick Start code block still works by running it.
7. Write a patch file to `outputs/report/readme_diff.patch` if changes were made.

### Step 6 — Generate Overleaf Fragment

Produce `outputs/report/overleaf_fragment.tex`:

```latex
% Auto-generated fragment — cqed_sim artifact builder
% Date: <date>

\subsection{SQR Calibration Results}

<methods paragraph referencing CONFIG parameters>

\begin{figure}[h]
\centering
\includegraphics[width=\textwidth]{figures/sqr_benchmark_duration_summary.png}
\caption{<generated caption>}
\label{fig:sqr_duration}
\end{figure}

<Table 1 from Step 3>

<Table 2 from Step 3>

<Summary paragraph with key numbers>
```

### Step 7 — Write Build Log

Write `outputs/report/build_log.md`:
```markdown
# Artifact Build Log — <date>
## Manifest: <N> figures, <M> JSONs, <K> calibration files
## Tables generated: <list>
## Captions generated: <list>
## README updated: yes/no (diff at outputs/report/readme_diff.patch)
## Overleaf fragment: outputs/report/overleaf_fragment.tex
## Warnings: <any issues encountered>
```

## Key References

- `README.md` — Section markers: `## API Summary`, `## Features`, `## Sequential Simulation Notebook`.
- `REFRACTOR_NOTES.md` — Complete module map.
- Figure naming conventions: `sqr_benchmark_*.png`, `fock_resolved_*.png`, `bloch_trajectory_*.png`.
- JSON schemas are self-documenting (key names match the paper notation).

## Quality Standards

- All numbers in tables and captions must come from actual JSON data. Never fabricate.
- LaTeX must compile standalone (no undefined macros).
- Captions must describe axes, units, and key parameter values.
- README changes must preserve existing structure and formatting.
- Overleaf fragment must use `\label{}` for cross-referencing.
