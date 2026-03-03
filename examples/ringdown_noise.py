from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def main():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=16, n_tr=2
    )
    psi0 = qt.tensor(qt.coherent(model.n_cav, 1.2), qt.basis(model.n_tr, 0))
    compiled = SequenceCompiler(dt=0.05).compile([], t_end=8.0)
    result = simulate_sequence(
        model,
        compiled,
        psi0,
        {},
        config=SimulationConfig(),
        noise=NoiseSpec(kappa=0.25, nth=0.0),
    )
    print("Initial <n>:", float(abs(1.2) ** 2))
    print("Final <n>:", float(result.expectations["n_c"][-1]))


if __name__ == "__main__":
    main()

