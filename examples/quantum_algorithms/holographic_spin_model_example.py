from __future__ import annotations

from cqed_sim.quantum_algorithms.holographic_sim import (
    BurnInConfig,
    HolographicSampler,
    ObservableSchedule,
    ising_transfer_channel,
    pauli_x,
    pauli_z,
)


def main() -> None:
    channel = ising_transfer_channel(zz_coupling=0.8, hx_physical=0.25, hx_bond=0.15, time=0.7)
    schedule = ObservableSchedule(
        [
            {"step": 2, "operator": pauli_z(), "label": "Z_bulk"},
            {"step": 5, "operator": pauli_x(), "label": "X_probe"},
        ],
        total_steps=6,
        label="spin_like_correlator",
    )
    sampler = HolographicSampler(channel, burn_in=BurnInConfig(steps=10))
    exact = sampler.enumerate_correlator(schedule)
    sampled = sampler.sample_correlator(schedule, shots=12_000, seed=123)

    print("Spin-inspired holographic example")
    print(f"Exact correlator:   {exact.mean.real:+.6f}")
    print(f"Sampled correlator: {sampled.mean.real:+.6f} +/- {sampled.stderr:.6f}")
    print(f"Accepted probability: {exact.accepted_probability_sum:.6f}")


if __name__ == "__main__":
    main()
