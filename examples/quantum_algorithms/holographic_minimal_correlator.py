from __future__ import annotations

import numpy as np

from cqed_sim.quantum_algorithms.holographic_sim import (
    HolographicSampler,
    ObservableSchedule,
    hadamard_reference_channel,
    pauli_z,
)


def main() -> None:
    channel = hadamard_reference_channel()
    schedule = ObservableSchedule(
        [
            {"step": 1, "operator": pauli_z()},
            {"step": 2, "operator": pauli_z()},
        ]
    )
    sampler = HolographicSampler(channel)
    exact = sampler.enumerate_correlator(schedule)
    sampled = sampler.sample_correlator(schedule, shots=10_000, seed=7)

    print("Minimal ideal correlator example")
    print(f"Exact mean:   {exact.mean.real:+.6f}")
    print(f"Sampled mean: {sampled.mean.real:+.6f} +/- {sampled.stderr:.6f}")
    print(f"Normalization error: {exact.normalization_error:.3e}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
