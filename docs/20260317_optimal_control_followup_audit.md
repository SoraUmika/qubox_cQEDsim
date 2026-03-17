# 2026-03-17 Optimal-Control Follow-up Audit

## Current Support

The current `cqed_sim.optimal_control` package now supports:

- solver-agnostic direct-control problem definitions through `ControlProblem`
- dense piecewise-constant schedules and pulse export back into the standard runtime path
- state-transfer and unitary objectives, including retained-subspace targets
- amplitude, slew-rate, and subspace leakage penalties
- exact closed-system GRAPE gradients via Fr\'echet derivatives of the slice propagators
- nominal and simple robust ensemble optimization through `mean` and `worst` aggregation
- top-level API exports, one script example, and one tutorial notebook path in the docs

## What Was Already Validated

The initial implementation already validated:

- model-backed unitary optimization
- pulse export plus runtime replay
- state-preparation optimization
- worst-case ensemble improvement
- top-level export completeness

The focused validation that had already been run before this follow-up was limited to the new API-export test and the dedicated GRAPE regression file.

## Lightly Tested Areas

The parts that still appear lightly tested are:

- interaction between optimized schedules and open-system simulation through `NoiseSpec`
- post-optimization evaluation workflows that compare closed-system objective values against noisy runtime replay
- larger multi-slice optimization behavior outside the one-slice regression cases
- benchmark-style usage and result serialization for future performance comparisons
- the tutorial notebook artifact itself, which currently exists on disk but is empty

## Main Integration Risks

The most likely current integration risks are:

- closed-system optimization results being interpreted too strongly without a built-in noisy replay path
- drift between optimal-control docs and the live notebook/example assets
- hidden convention mismatches at the pulse-export/runtime boundary if broader simulator tests reveal them
- adjacent breakage in unitary-synthesis helpers because the optimal-control objectives reuse that target/subspace machinery

## Adjacent Areas Most Likely To Break

The highest-risk neighboring areas for this follow-up validation are:

- `cqed_sim.sim` runtime and `NoiseSpec` handling
- `cqed_sim.pulses` and `SequenceCompiler` pulse semantics
- `cqed_sim.core` / universal-model operator conventions
- `cqed_sim.unitary_synthesis` subspace, metric, and target helpers
- tutorial/example-facing surfaces that depend on notebook content or documented public API names

## Follow-up Direction

The cleanest next extension is not a second optimizer backend. The more useful next step is a simulator-backed open-system evaluation layer built on top of the existing `ControlProblem` and pulse export path. That would let the repository distinguish clearly between:

- what is optimized under the current closed-system GRAPE assumptions, and
- what is evaluated under realistic Lindblad noise using the existing simulator.