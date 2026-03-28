---
name: literature-review
description: Search for and record the canonical reference papers for a physics feature, algorithm, or result being added to cqed_sim. Use this before implementing any new feature or when a tutorial/example needs an external citation.
---

You are performing a literature review for a cQED simulation feature, tutorial, or result. Follow this workflow exactly:

## Step 1 — Identify the subject

From `$ARGUMENTS` (or from context if no arguments are given), extract:
- The physics concept, gate, protocol, or algorithm to be referenced (e.g., "SNAP gates", "GRAPE optimal control", "dispersive readout", "self-Kerr collapse and revival").
- Whether this is for: (a) a new implementation, (b) a tutorial demonstrating an effect, (c) a test reproducing a figure or result, or (d) a convention check.

## Step 2 — Search for the canonical reference

Use WebSearch to find:
1. The **original paper** introducing the concept (search for the concept name + "cQED" or "superconducting qubit" or "cavity QED" or "quantum optics" as appropriate).
2. A **review article or textbook chapter** that presents the standard derivation or convention.
3. Any **paper providing the specific numerical result or figure** being reproduced (if applicable).

Run at least two searches with different query terms to ensure coverage. Prefer:
- Physical Review Letters, Physical Review A, Physical Review X
- Nature, Science, npj Quantum Information
- Applied Physics Letters, PRX Quantum
- arXiv (quant-ph section)

## Step 3 — Extract the key information

For each relevant paper found, record:
- Full citation: Author(s), "Title," Journal, Volume, Pages/Article number, Year. DOI.
- The specific equation, figure, or section relevant to this task.
- The conventions used (sign conventions, units, Hamiltonian form, parameter definitions).
- The key numerical result or qualitative behavior being reproduced (if applicable).

## Step 4 — Check against project paper_summary/

Read the `paper_summary/` directory listing to see if a summary for any of these papers already exists. If yes, read it and note whether the existing summary is consistent with what you found. If a summary is missing for an important paper, note that one should be added.

## Step 5 — Check against physics_and_conventions/

Check whether the conventions in the found paper match those documented in `physics_and_conventions/physics_conventions_report.tex`. If there is a discrepancy, flag it explicitly.

## Step 6 — Output a structured reference block

Produce a reference block in this format, suitable for pasting directly into a tutorial `.md` file, test header, or code comment:

```
## References

[1] Author(s), "Title," Journal/arXiv, Year. DOI: <doi>
    Relevant for: <what this paper establishes for this task>
    Key equation: <equation number and brief description>
    Convention note: <any sign/unit/frame convention to be aware of>

[2] ...
```

Also produce the short inline code-comment form:
```python
# Implements <description>, following Eq. (<N>) in [Author et al., Year].
# DOI: <doi>
```

## Step 7 — State any gaps

If no authoritative external reference can be found after searching, state that explicitly so the user can decide how to proceed. Do not silently omit the citation step.
