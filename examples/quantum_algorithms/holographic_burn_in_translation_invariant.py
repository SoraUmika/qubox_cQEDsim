from __future__ import annotations

import numpy as np

from cqed_sim.quantum_algorithms.holographic_sim import (
    BurnInConfig,
    HolographicSampler,
    ObservableSchedule,
    partial_swap_channel,
    pauli_z,
)


def main() -> None:
    channel = partial_swap_channel(theta=0.35)
    excited_left_boundary = np.array([0.0, 1.0], dtype=np.complex128)
    sampler = HolographicSampler(
        channel,
        left_state=excited_left_boundary,
        burn_in=BurnInConfig(steps=12, label="bulk_preparation"),
    )
    summary = sampler.summarize_burn_in()
    observable = ObservableSchedule([{"step": 1, "operator": pauli_z()}], total_steps=1, label="bulk_Z")
    estimate = sampler.sample_correlator(observable, shots=8_000, seed=11)

    print("Translation-invariant burn-in example")
    print(f"Burn-in steps: {summary.steps}")
    print(f"Final bond population |0><0|: {np.real(summary.final_state[0, 0]):.6f}")
    print(f"Max residual: {summary.max_residual:.6e}")
    print(f"Bulk <Z>: {estimate.mean.real:+.6f} +/- {estimate.stderr:.6f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
