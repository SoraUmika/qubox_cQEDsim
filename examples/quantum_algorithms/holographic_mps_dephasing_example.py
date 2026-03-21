from __future__ import annotations

import numpy as np

from cqed_sim.quantum_algorithms.holographic_sim import (
    BondNoiseChannel,
    HolographicSampler,
    ObservableSchedule,
    pauli_z,
)


def _ghz_state(num_sites: int = 4) -> np.ndarray:
    state = np.zeros(2 ** int(num_sites), dtype=np.complex128)
    state[0] = 1.0
    state[-1] = 1.0
    state /= np.linalg.norm(state)
    return state.reshape((2,) * int(num_sites))


def main() -> None:
    state = _ghz_state(4)
    ideal_sampler = HolographicSampler.from_mps_state(state, site=1)
    dephasing = BondNoiseChannel.dephasing(bond_dim=ideal_sampler.channel.bond_dim, probability=0.15)
    noisy_sampler = HolographicSampler.from_mps_state(state, site=1, bond_noise=dephasing)
    schedule = ObservableSchedule(
        [
            {"step": 1, "operator": pauli_z()},
            {"step": 2, "operator": pauli_z()},
        ],
        total_steps=2,
    )

    ideal = ideal_sampler.enumerate_correlator(schedule)
    noisy = noisy_sampler.enumerate_correlator(schedule)

    print("Transfer-channel workflow from a selected GHZ MPS site")
    print(f"Bond dimension: {ideal_sampler.channel.bond_dim}")
    print(f"Ideal exact mean:     {ideal.mean.real:+.6f}")
    print(f"Dephased exact mean:  {noisy.mean.real:+.6f}")
    print(f"Per-step dephasing p: {dephasing.metadata['probability']:.3f}")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()