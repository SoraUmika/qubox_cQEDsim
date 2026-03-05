# Skills + MCP Deployment Blueprint for `cqed_sim`

> Generated: 2026-03-04 — Tailored to `qubox_cQEDsim` (cqed-sim 0.1.0)

---

## Table of Contents

1. [Skill 1 — Codebase Refactor Reviewer](#skill-1--codebase-refactor-reviewer)
2. [Skill 2 — Calibration / Experiment Audit](#skill-2--calibration--experiment-audit)
3. [Skill 3 — Research Artifact Builder](#skill-3--research-artifact-builder)
4. [MCP Integration 1 — Filesystem Access](#mcp-1--filesystem-access)
5. [MCP Integration 2 — Git / Diff Access](#mcp-2--git--diff-access)
6. [MCP Integration 3 — Shell / Command Runner](#mcp-3--shell--command-runner)
7. [MCP Integration 4 — Documentation / Knowledge Access](#mcp-4--documentation--knowledge-access)
8. [Rollout Order](#rollout-order)
9. [Daily Workflow — Before vs After](#daily-workflow--before-vs-after)
10. [Starter Template and Setup](#starter-template-and-setup)

---

## Skill 1 — Codebase Refactor Reviewer

### Purpose

Provide a repeatable, checklist-driven review of any structural change (module move, API rename, dependency rewire) inside `cqed_sim/`. The skill replaces ad hoc prompts like "check if the refactor broke anything" with a deterministic workflow that audits imports, tests, public API surface, and notebook compatibility before the change is committed.

### When to Invoke

- Before merging any branch that touches `cqed_sim/` module structure.
- After running `outputs/generate_sequential_simulation_notebook.py` or `outputs/generate_sqr_calibration_notebook.py`.
- When renaming or relocating a public function listed in `REFRACTOR_NOTES.md`.
- When adding a new sub-package under `cqed_sim/`.

### Workflow (step-by-step)

| Step | Action | Tools / Context |
|------|--------|-----------------|
| 1 | **Snapshot public API surface.** Read every `__init__.py` re-export and the README API Summary. Build `api_before.json`. | MCP-Filesystem: read `cqed_sim/**/__init__.py`, `README.md` |
| 2 | **Run the full test suite.** Execute `pytest tests/ -q --tb=short --junitxml=outputs/test_results.xml`. Capture baseline pass/fail counts. | MCP-Shell |
| 3 | **Diff analysis.** Collect `git diff main --stat` and `git diff main --name-only`. Classify changed files as *API*, *Internal*, *Test*, *Notebook*, *Config*. | MCP-Git |
| 4 | **Import audit.** For every changed `.py` file, grep all `from cqed_sim...` and `import cqed_sim...` lines across the entire repo. Flag any import that references a moved/deleted symbol. | MCP-Filesystem + grep |
| 5 | **Notebook compatibility check.** Parse `sequential_simulation.ipynb` and `SQR_calibration.ipynb` cell sources. Verify all `from cqed_sim.*` imports resolve. | MCP-Filesystem |
| 6 | **Re-run tests.** Run `pytest tests/ -q --tb=short`. Compare pass/fail counts against baseline. | MCP-Shell |
| 7 | **Generate review report.** Produce a structured Markdown report with sections: *Scope*, *API changes*, *Broken imports*, *Test delta*, *Notebook status*, *Recommendations*. | Agent output |
| 8 | **Propose patches.** For each broken import, emit a concrete diff suggestion. | Agent output |

### Input Format

```yaml
skill: refactor-reviewer
branch: feature/move-calibration-cache    # or "working-tree"
scope:                                     # optional, narrows analysis
  - cqed_sim/calibration/
  - cqed_sim/io/
baseline_ref: main                         # git ref for diff base
run_tests: true
check_notebooks: true
```

### Output Format

```
outputs/report/refactor_review_<branch>_<timestamp>.md
```

Contents:

```markdown
# Refactor Review — <branch>
## Scope: <files touched>
## API Surface Delta
| Symbol | Status | Old location | New location |
## Broken Import Sites
| File | Line | Import | Resolution |
## Test Delta
| Suite | Before | After | Regression? |
## Notebook Compatibility
| Notebook | Cell | Import | Status |
## Recommendations
- ...
```

### Included Resources / Checklists

- `REFRACTOR_NOTES.md` (source-of-truth for module map).
- `README.md` API Summary section.
- `pyproject.toml` declared entry points.
- Checklist template: `.copilot/skills/refactor-reviewer/checklist.md`.

---

## Skill 2 — Calibration / Experiment Audit

### Purpose

Audit any SQR calibration run or Fock-tomography experiment for physics consistency, numerical convergence, and parameter sanity. This skill replaces manual inspection of calibration JSONs and notebook outputs with an automated evidence-collection-and-verdict pipeline.

### When to Invoke

- After running cells in `SQR_calibration.ipynb` (Sections 5–9).
- After producing a new `calibrations/sqr_*.json` cache file.
- After running `examples/simulate_fock_tomo_and_sqr_calibration.py`.
- Before citing calibration results in a paper or Overleaf doc.

### Workflow (step-by-step)

| Step | Action | Tools / Context |
|------|--------|-----------------|
| 1 | **Load calibration artifact.** Read the target JSON (e.g., `sqr_calibration_result.json` or a cache under `calibrations/`). Parse all per-level correction vectors `(d_lambda, d_alpha, d_omega)`. | MCP-Filesystem |
| 2 | **Parameter bounds check.** For each Fock level, verify corrections are within the CONFIG bounds: `d_lambda ∈ (-0.5, 0.5)`, `d_alpha ∈ (-π, π)`, `d_omega_hz ∈ (-2 MHz, 2 MHz)`. Flag any saturation. | Agent logic |
| 3 | **Convergence check.** Compare `initial_loss` vs `optimized_loss` per level. Flag levels where improvement < 10× or where `optimized_loss > 1e-3`. | Agent logic |
| 4 | **Physics consistency.** Verify `chi`, `chi2`, `chi3` values used match `experiment_mapping.md`. Compare detuning convention `Δ(n) = 2π(χn + χ₂n² + χ₃n³)` against the solver. | MCP-Filesystem: read `experiment_mapping.md`, source code |
| 5 | **Guard-band audit.** If benchmark results exist (`outputs/sqr_guard_benchmark_results.json`), load and verify: logical fidelity ≥ threshold, guard leakage ≤ threshold, success rate per class. | MCP-Filesystem |
| 6 | **Cross-reference experiment mapping.** Check that the pulse model assumptions (Gaussian envelope, sigma fraction, duration) match the physical device parameters listed in `experiment_mapping.md`. | MCP-Filesystem |
| 7 | **Generate audit report.** Emit a structured verdict with per-level tables, pass/fail flags, and warnings. | Agent output |
| 8 | **Suggest next steps.** If any level fails, recommend: increase `maxiter`, widen bounds, check chi convention, or re-run with `force_recompute=True`. | Agent output |

### Input Format

```yaml
skill: calibration-audit
calibration_file: sqr_calibration_result.json
benchmark_file: outputs/sqr_guard_benchmark_results.json   # optional
config_source: SQR_calibration.ipynb::Section 2             # for extracting CONFIG
experiment_ref: experiment_mapping.md
```

### Output Format

```
outputs/report/calibration_audit_<gate_name>_<timestamp>.md
```

Contents:

```markdown
# Calibration Audit — <gate_name>
## Parameter Bounds
| Level | d_lambda | d_alpha | d_omega_hz | Saturated? |
## Convergence
| Level | Initial loss | Final loss | Improvement | Pass? |
## Physics Consistency
| Parameter | Expected | Used | Match? |
## Guard-Band Benchmark
| Metric | Value | Threshold | Pass? |
## Verdict: PASS / WARN / FAIL
## Recommendations
- ...
```

### Included Resources

- `experiment_mapping.md` — canonical device ↔ sim parameter mapping.
- Expected bounds from CONFIG dict in `SQR_calibration.ipynb` Section 2.
- Convergence thresholds: `improvement_factor >= 10`, `final_loss <= 1e-3` (configurable).
- Template: `.copilot/skills/calibration-audit/checklist.md`.

---

## Skill 3 — Research Artifact Builder

### Purpose

Automate the generation of publication-ready deliverables: LaTeX-ready figure exports, summary tables, README/API doc updates, and Overleaf-compatible writeup fragments. This skill replaces manually copying numbers from notebook outputs into documents.

### When to Invoke

- After completing a simulation campaign (sequential or SQR calibration).
- When preparing a paper section or group meeting presentation.
- After adding a new public API or example script.
- When asked to "update the docs" or "generate the writeup".

### Workflow (step-by-step)

| Step | Action | Tools / Context |
|------|--------|-----------------|
| 1 | **Inventory artifacts.** Scan `outputs/figures/`, `outputs/*.json`, `calibrations/`, `sqr_calibration_result.json`. Build a manifest of available data. | MCP-Filesystem |
| 2 | **Extract key numbers.** From JSON summaries, extract: best fidelity, worst fidelity, median guard leakage, number of levels calibrated, benchmark success rates. | MCP-Filesystem + Agent logic |
| 3 | **Generate LaTeX table fragments.** Emit `\begin{tabular}...` blocks for: per-level calibration summary, benchmark duration sweep, success-by-class. | Agent output |
| 4 | **Generate figure captions.** For each PNG in `outputs/figures/`, produce a draft caption with axis descriptions, parameter annotations, and methodology reference. | Agent logic |
| 5 | **Update README.md.** Refresh the API Summary section if any new public functions were added. Verify Quick Start still runs. | MCP-Filesystem + MCP-Shell |
| 6 | **Generate Overleaf fragment.** Produce a self-contained `.tex` snippet that includes figures (via `\includegraphics`), tables, and a methods paragraph referencing the simulation parameters from CONFIG. | Agent output |
| 7 | **Export all artifacts.** Write outputs to `outputs/report/`. | MCP-Filesystem |

### Input Format

```yaml
skill: artifact-builder
mode: full                  # or "figures-only", "tables-only", "readme-only"
figure_dir: outputs/figures
json_sources:
  - sqr_calibration_result.json
  - outputs/sqr_guard_benchmark_results.json
  - outputs/fock_tomo_sqr_summary.json
overleaf_target: outputs/report/overleaf_fragment.tex
update_readme: true
```

### Output Format

```
outputs/report/
├── artifact_manifest.json
├── latex_tables.tex
├── figure_captions.md
├── overleaf_fragment.tex
├── readme_diff.patch         (if README was updated)
└── build_log.md
```

### Included Resources

- LaTeX table templates: `.copilot/skills/artifact-builder/templates/table_template.tex`.
- Caption style guide: `.copilot/skills/artifact-builder/templates/caption_style.md`.
- README section markers: look for `## API Summary`, `## Features`, `## Sequential Simulation Notebook`.
- Figure naming conventions: `sqr_benchmark_*.png`, `fock_resolved_*.png`, `bloch_trajectory_*.png`.

---

## MCP 1 — Filesystem Access

### What It Exposes

Read/write/list operations on the workspace directory tree. Specifically:

- `read_file(path)` — read any source, JSON, markdown, config file.
- `write_file(path, content)` — create or overwrite files (reports, patches, configs).
- `list_directory(path)` — enumerate contents of any subdir.
- `glob(pattern)` — e.g., `calibrations/sqr_*.json`, `cqed_sim/**/__init__.py`.
- `search_text(pattern, paths)` — regex/literal search across files.

### Tasks It Enables

- Skill 1: scanning all imports, reading `__init__.py` re-exports.
- Skill 2: loading calibration JSONs, reading `experiment_mapping.md`.
- Skill 3: inventorying figures, writing report files.
- Ad hoc: "find all uses of `load_or_calibrate_sqr_gate`", "read CONFIG from the notebook".

### Risks / Permissions

| Risk | Mitigation |
|------|-----------|
| Accidental overwrite of calibration data | Restrict writes to `outputs/` and `.copilot/` only; calibration cache is read-only to the agent. |
| Reading credentials or sensitive paths | Project has no secrets; Box sync path is the only concern — agent should never write outside the workspace root. |
| Large file reads | Limit read size to 500 KB per call; JSON calibration files are ~10 KB. |

### Minimum Setup

**Already available** in VS Code Copilot via the built-in workspace tools (`read_file`, `grep_search`, `list_dir`, `create_file`, `replace_string_in_file`). No additional MCP server is needed.

For standalone / CLI agent use, install the reference Filesystem MCP server:

```bash
npx -y @modelcontextprotocol/server-filesystem <workspace_root>
```

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "filesystem": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
    }
  }
}
```

### How It Improves This Workflow

- Eliminates manual copy-paste of file contents into prompts.
- Enables multi-file cross-referencing (imports, configs, tests) in a single agent turn.
- Lets Skills write their reports directly to `outputs/report/`.

---

## MCP 2 — Git / Diff Access

### What It Exposes

- `git_status()` — working tree status.
- `git_diff(ref, paths)` — unified diff against a ref (e.g., `main`, `HEAD~3`).
- `git_log(n, paths)` — recent commit messages and SHAs.
- `git_diff_stat(ref)` — summary of changed files with line counts.
- `git_show(sha, path)` — file content at a specific commit.
- `git_blame(path, lines)` — authorship per line.

### Tasks It Enables

- Skill 1 step 3: automated diff classification.
- Branch review: "show me what changed in the calibration module since last week".
- Regression triage: "which commit introduced the chi convention change?"
- Pre-commit gating: "validate that all changed files pass import audit before I push".

### Risks / Permissions

| Risk | Mitigation |
|------|-----------|
| Exposing commit history with sensitive messages | This repo is research-internal; no secrets in commit messages. |
| Agent executing `git push` or `git reset` | MCP server should expose **read-only** git operations only. No write commands (`push`, `reset`, `checkout`, `commit`). |
| Performance on large diffs | Limit diff output to 200 KB; use `--stat` first to preview scope. |

### Minimum Setup

Install the Git MCP server:

```bash
npx -y @modelcontextprotocol/server-git
```

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "git": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-git", "--repository", "${workspaceFolder}"]
    }
  }
}
```

Alternatively, use the GitHub Copilot built-in `get_changed_files` tool plus terminal `git diff` commands — these already cover 80% of use cases with zero additional setup.

### How It Improves This Workflow

- Skill 1 gets structured diff data without you pasting `git diff` output.
- "Review my branch" becomes a one-shot Skill invocation instead of 5+ manual prompts.
- Regression tracking connects test failures to specific commits automatically.

---

## MCP 3 — Shell / Command Runner

### What It Exposes

- `run_command(cmd, cwd, timeout)` — execute a shell command and return stdout/stderr.
- `run_background(cmd, cwd)` — launch a long-running process; return a handle.
- `get_output(handle)` — poll output from a background process.

### Tasks It Enables

- Running `pytest tests/ -q --tb=short --junitxml=outputs/test_results.xml`.
- Running `python outputs/generate_sequential_simulation_notebook.py` to regenerate notebooks.
- Running `python -c "from cqed_sim.calibration.sqr import calibrate_sqr_gate; print('OK')"` for smoke tests.
- Running `pip install -e .[dev]` for environment setup.
- Running `python examples/sanity_run.py` for quick validation.

### Risks / Permissions

| Risk | Mitigation |
|------|-----------|
| Arbitrary code execution | Restrict to an allowlist of commands: `pytest`, `python`, `pip`, `git`. Block `rm`, `del`, `format`, network-facing commands. |
| Long-running simulations blocking the agent | Set default timeout = 300 s. Use background mode for anything expected to take > 60 s. |
| Accidental `pip install` of untrusted packages | Only allow `pip install -e .` and `pip install -e .[dev]` within the project. |
| Environment pollution | Run in the same venv/conda env as the notebook kernel. |

### Minimum Setup

**Already available** in VS Code Copilot via the built-in `run_in_terminal` tool. No additional MCP server needed.

For external agents, use a sandboxed shell MCP server. Reference implementation:

```bash
npx -y @anthropic/mcp-shell --allowed-commands "pytest,python,pip,git"
```

### How It Improves This Workflow

- Skill 1 runs the full test suite and captures results without you switching to terminal.
- Skill 3 can verify `python -c "from cqed_sim.io.gates import load_gate_sequence"` still works after README update.
- "Run tests on just the calibration module" becomes `pytest tests/ -k calibration -q`.

---

## MCP 4 — Documentation / Knowledge Access

### What It Exposes

A semantic search over a curated knowledge base assembled from your repo:

- `search_docs(query)` — returns relevant snippets from indexed documentation.
- `get_api_reference(symbol)` — returns docstring + signature for a public function.
- `get_experiment_mapping(topic)` — returns relevant section from `experiment_mapping.md`.

Indexed sources:

| Source | Content |
|--------|---------|
| `README.md` | API summary, features, quick start |
| `REFRACTOR_NOTES.md` | Module map, refactor history |
| `experiment_mapping.md` | Device-to-sim parameter mapping |
| `cqed_sim/**/*.py` docstrings | Function signatures and docstrings |
| `outputs/report/*.md` | Previous audit/review reports |

### Tasks It Enables

- Skill 2: cross-referencing chi convention against `experiment_mapping.md`.
- Skill 3: pulling function signatures for API docs.
- Ad hoc: "what is the convention for sideband swap timing?" → returns relevant README section.
- Onboarding: "explain the calibration pipeline" → returns structured answer from indexed docs.

### Risks / Permissions

| Risk | Mitigation |
|------|-----------|
| Stale index | Re-index on every git pull or when Skills are invoked. |
| Indexing large files | Exclude `__pycache__/`, `.egg-info/`, `outputs/figures/` from indexing. |
| Hallucinated API references | Always verify with a live `grep_search` if the doc index returns uncertain results. |

### Minimum Setup

Two practical options:

**Option A — Built-in VS Code semantic search (zero setup):**
The `semantic_search` tool already indexes the workspace. This is sufficient for most queries.

**Option B — Dedicated docs MCP server (for richer retrieval):**

```bash
pip install mcp-server-docs
mcp-server-docs index --root . --exclude "__pycache__,*.egg-info,outputs/figures"
```

Add to `.vscode/mcp.json`:

```json
{
  "servers": {
    "docs": {
      "type": "stdio",
      "command": "mcp-server-docs",
      "args": ["serve", "--index-dir", ".copilot/docs_index"]
    }
  }
}
```

**Recommended: start with Option A** (already works), upgrade to Option B only if you find retrieval quality insufficient.

### How It Improves This Workflow

- Skills can look up conventions ("what units does `chi` use?") without you providing context.
- Reduces hallucination risk by grounding answers in actual repo documentation.
- Enables the Artifact Builder to auto-generate accurate method descriptions.

---

## Rollout Order

Prioritized by: fastest payoff × lowest friction × maximum immediate usefulness.

| Phase | Item | Effort | Payoff | Notes |
|-------|------|--------|--------|-------|
| **Phase 0** (Day 1) | MCP-Filesystem | Already done | High | Built-in VS Code tools cover this. |
| **Phase 0** (Day 1) | MCP-Shell | Already done | High | Built-in `run_in_terminal` covers this. |
| **Phase 0** (Day 1) | MCP-Docs (Option A) | Already done | Medium | Built-in `semantic_search` covers this. |
| **Phase 1** (Day 1–2) | **Skill 1: Refactor Reviewer** | ~2 hours | **Highest** | Write the prompt/instructions file + checklist. Immediately usable for your ongoing refactoring work. |
| **Phase 2** (Day 2–3) | MCP-Git | ~30 min | High | `npx` one-liner install + 5 lines in `mcp.json`. |
| **Phase 2** (Day 2–3) | **Skill 2: Calibration Audit** | ~2 hours | High | Directly validates your SQR calibration pipeline. |
| **Phase 3** (Day 4–5) | **Skill 3: Artifact Builder** | ~3 hours | Medium-High | Most valuable near paper deadlines, but templates can be built incrementally. |
| **Phase 4** (Week 2+) | MCP-Docs (Option B) | ~1 hour | Medium | Only if built-in search proves insufficient. |

**Key insight:** Three of the four MCP integrations are already functional via VS Code's built-in tools. Your primary effort should go into writing the Skill instruction files.

---

## Daily Workflow — Before vs After

### Scenario 1: Refactor Review

**Before (manual prompting):**
```
You: "I moved calibration cache logic from sqr.py to a new cache.py. Can you check if anything broke?"
Agent: "Can you paste the diff?"
You: <paste 200 lines of git diff>
Agent: "Can you show me the imports in sequential_simulation.ipynb?"
You: <paste notebook cell>
Agent: "Looks like line 47 imports from the old path."
You: "Can you check test_17 too?"
... (5-8 round trips)
```

**After (Skill 1):**
```
You: "@workspace /refactor-reviewer branch=feature/move-cache"
Agent: [runs Skill 1 autonomously]
  → reads all diffs via MCP-Git
  → scans all imports via MCP-Filesystem
  → runs pytest via MCP-Shell
  → generates outputs/report/refactor_review_move-cache_20260304.md
You: [reads structured report, applies suggested patches]
```

### Scenario 2: SQR Calibration Validation

**Before:**
```
You: "I just ran the SQR calibration notebook. The results are in sqr_calibration_result.json. Can you check if the calibration looks good?"
Agent: "Can you paste the JSON?"
You: <paste 300 lines>
Agent: "Level 5 has high loss. What were the bounds?"
You: <scrolls through notebook, pastes CONFIG>
... (4-6 round trips)
```

**After (Skill 2):**
```
You: "@workspace /calibration-audit calibration_file=sqr_calibration_result.json"
Agent: [runs Skill 2 autonomously]
  → loads JSON, extracts per-level data
  → cross-references CONFIG bounds
  → checks experiment_mapping.md consistency
  → generates outputs/report/calibration_audit_SQR_20260304.md with PASS/WARN/FAIL verdict
You: [reads structured audit, addresses warnings]
```

### Scenario 3: Generate Paper Tables

**Before:**
```
You: "I need a LaTeX table of the benchmark results for the paper."
Agent: "Can you paste the benchmark JSON?"
You: <paste JSON>
Agent: <generates table>
You: "Now do figure captions for all PNGs in outputs/figures/."
You: <lists 8 files manually>
... (3-5 round trips)
```

**After (Skill 3):**
```
You: "@workspace /artifact-builder mode=full"
Agent: [runs Skill 3 autonomously]
  → inventories all outputs
  → generates latex_tables.tex, figure_captions.md, overleaf_fragment.tex
  → updates README if new APIs detected
You: [copies overleaf_fragment.tex into Overleaf project]
```

### Scenario 4: Running Tests After a Change

**Before:**
```
You: <switches to terminal, types pytest, scrolls output, copies failures back into chat>
```

**After:**
```
You: "Run the test suite and summarize failures."
Agent: [runs pytest via MCP-Shell, parses JUnit XML, returns structured summary]
```

### Scenario 5: Branch Diff Inspection

**Before:**
```
You: <runs git diff, copies 500 lines into chat>
```

**After:**
```
You: "What changed on my branch vs main?"
Agent: [calls MCP-Git diff_stat, classifies files, summarizes by module]
```

---

## Starter Template and Setup

### Folder Structure

```
.copilot/
├── skills/
│   ├── refactor-reviewer/
│   │   ├── skill.md              ← main Skill instructions (prompt)
│   │   └── checklist.md          ← review checklist template
│   ├── calibration-audit/
│   │   ├── skill.md
│   │   └── checklist.md
│   └── artifact-builder/
│       ├── skill.md
│       └── templates/
│           ├── table_template.tex
│           └── caption_style.md
├── docs_index/                    ← (Phase 4) semantic index cache
└── README.md                      ← overview of available Skills
.vscode/
├── settings.json                  ← existing
└── mcp.json                       ← MCP server configuration
outputs/
└── report/                        ← Skill output directory (already exists)
```

### Windows Setup Plan

```
Step 1: Verify Node.js is available (needed for MCP servers)
  > node --version
  If missing: winget install OpenJS.NodeJS.LTS

Step 2: Create .copilot/ directory structure
  > mkdir .copilot\skills\refactor-reviewer
  > mkdir .copilot\skills\calibration-audit
  > mkdir .copilot\skills\artifact-builder\templates

Step 3: Install Git MCP server (Phase 2)
  > npx -y @modelcontextprotocol/server-git --help

Step 4: Create .vscode/mcp.json (Phase 2)
  (see file created below)

Step 5: Verify existing tools work
  > python -c "from cqed_sim.calibration.sqr import calibrate_sqr_gate; print('OK')"
  > pytest tests/ -q --tb=line -x
```

### Invocation Examples (Once System Is Ready)

**Refactor Reviewer:**
```
@workspace /refactor-reviewer
```
or with explicit parameters:
```
Invoke the refactor-reviewer skill on the current working tree against main.
Check all cqed_sim/ imports, run tests, and audit both notebooks.
```

**Calibration Audit:**
```
@workspace /calibration-audit
Audit sqr_calibration_result.json against the CONFIG in SQR_calibration.ipynb Section 2.
Cross-reference experiment_mapping.md for physics consistency.
```

**Artifact Builder:**
```
@workspace /artifact-builder mode=full
Generate LaTeX tables from outputs/sqr_guard_benchmark_results.json,
figure captions for all PNGs in outputs/figures/,
and an Overleaf-ready fragment.
```

---

## Security Summary

| Concern | Recommendation |
|---------|---------------|
| File writes | Restrict agent writes to `outputs/`, `.copilot/`, and temp directories. Never write to `calibrations/` (cache integrity). |
| Shell access | Allowlist: `pytest`, `python`, `pip install -e .`, `git` (read-only). Block destructive commands. |
| Git operations | Read-only MCP server. No push/reset/force operations exposed. |
| Secrets | This repo contains no secrets. Box sync path in CONFIG is a local path, not sensitive. |
| Network | No MCP server should make outbound network requests. All operations are local. |

---

## Extension Points (Future)

| Extension | Description | When |
|-----------|-------------|------|
| Skill 4: Test Failure Debugger | Auto-triages pytest failures with stack traces, suggests fixes | When test suite grows past 30 files |
| Skill 5: Notebook Recovery | Repairs broken/stale notebooks by regenerating from generators | After generator script changes |
| MCP 5: Overleaf API | Push LaTeX fragments directly to Overleaf project | When paper writing is active |
| MCP 6: Experiment Database | Query device calibration history from lab database | When connecting sim to real device data |
