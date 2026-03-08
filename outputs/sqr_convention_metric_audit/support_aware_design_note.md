# Support-Aware Objective Design Note

## Philosophy
- Active support levels are optimization targets; inactive levels are spectators unless they induce leakage from support.
- Global full-space fidelity is still reported for reference, but not the primary optimization target in support-aware mode.

## Configuration
- `ActiveSupportParams.mode`: `contiguous`, `explicit`, or `from_state`.
- `active_levels` / `max_level_active` define support set `S`.
- `active_weights` can be user-specified; otherwise uniform or inferred from a reference state.
- `inference_state_label` + `state_population_threshold` support state-driven support inference.

## Support-Aware Loss Terms
- Active weighted block infidelity, active theta/phase/pre-Z/post-Z.
- Support-state mean/min fidelity and phase-superposition coherence.
- Leakage penalties: support-state leakage mean/max and spectral boundary leakage proxy.
- Worst active-block floor penalty and weak inactive infidelity penalty.

## Case Recommendation
- Use Case E as default support-aware ansatz (amplitude + phase + detuning + phase-ramp/chirp).
- Use Case D as fallback / ablation when runtime or control complexity must be reduced.

## How to pass support inputs
- Explicit support: `ActiveSupportParams(mode='explicit', active_levels=(0,1,4), active_weights=(...))`.
- Contiguous support: `ActiveSupportParams(mode='contiguous', max_level_active=m)`.
- Inferred support: `ActiveSupportParams(mode='from_state', inference_state_label='...', state_population_threshold=...)`.
- Projected coherent support states are included in support ensemble by default.