# RL Hybrid Control Tutorial

The RL-ready hybrid control walkthrough lives in:

- `tutorials/30_advanced_protocols/05_rl_hybrid_control_environment.ipynb`

It demonstrates:

- constructing a `HybridCQEDEnv`
- configuring the reduced dispersive regime
- choosing a benchmark task from the benchmark suite and configuring an action space
- using measurement-like observations, history stacking, and measurement-proxy rewards
- running scripted and random rollouts
- plotting simulator-side diagnostics
- evaluating the same controller under domain randomization

The notebook is interactive rather than artifact-only: it builds the environment directly inside the notebook, runs the rollout path in-place, and then visualizes the returned diagnostics.

This tutorial is intended as the starting template for future simulator-trained bosonic-ancilla control studies inside `cqed_sim`.