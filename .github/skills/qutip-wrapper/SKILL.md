---
name: qutip-wrapper
description: "Add or refactor QuTiP-backed functionality using the repo's QuTiP-native-first policy. Use when evaluating whether QuTiP already supports a solver, operator, transformation, visualization, or analysis and when exposing that capability through a project-level wrapper with explicit dimensions, basis ordering, frames, units, defaults, and documentation."
argument-hint: "Describe the needed QuTiP capability, the target cqed_sim module, and any dimension or convention constraints."
---

# QuTiP Wrapper

Use this skill when the task touches QuTiP integration or risks re-implementing something QuTiP already provides.

## When to Use

- Adding a solver, operator, transformation, visualization helper, or analysis flow.
- Replacing ad hoc QuTiP calls with a structured wrapper.
- Auditing dimension handling, subsystem ordering, or project-specific conventions around QuTiP usage.

## Core Policy

- Check QuTiP first.
- Reuse the native QuTiP capability if it satisfies the task.
- Wrap it in the appropriate `cqed_sim` layer so project conventions stay explicit and centralized.
- Implement custom logic only for the project-specific behavior QuTiP does not cover.

## Common Failure Modes to Prevent

- Re-implementing a QuTiP feature that already exists.
- Scattering raw QuTiP calls across unrelated modules instead of creating a stable wrapper.
- Flattening composite `dims` and breaking downstream solver behavior.
- Exposing raw QuTiP naming that hides this repo's frame, truncation, or ordering conventions.

## Procedure

1. Identify the native QuTiP capability.
   - Verify whether QuTiP already provides the solver, operator, transform, or utility.
   - Document any gap that still requires custom project logic.
2. Pick the right wrapper boundary.
   - Place the wrapper in the relevant `cqed_sim` module instead of calling QuTiP ad hoc from high-level code.
   - Keep the public API aligned with existing project naming and return types.
3. Make repo conventions explicit.
   - Specify basis ordering, truncation, rotating-frame assumptions, parameter naming, and units.
   - Preserve composite subsystem `dims` instead of flattening them unless the downstream contract truly requires it.
4. Add validation.
   - Test behavior at the wrapper boundary, not just the raw QuTiP call.
   - Include regression coverage for dimension structure, frame assumptions, and convention-sensitive outputs.
5. Synchronize docs.
   - Update `API_REFERENCE.md` if the wrapper is part of the public or developer-facing API.
   - Update `documentations/` if user workflows or examples change.
   - Update `physics_and_conventions/physics_conventions_report.tex` if the wrapper carries physics assumptions or changes convention-sensitive meaning.

## Completion Checklist

- QuTiP-native functionality was evaluated before custom implementation.
- QuTiP usage is wrapped in a project-level interface.
- Basis ordering, dimensions, units, and frame assumptions are explicit.
- Tests cover wrapper behavior and dimension handling.
- Relevant API and physics docs are synchronized.