from __future__ import annotations

import numpy as np

from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def square(x):
    return np.ones_like(x, dtype=np.complex128)


def main():
    g = 0.35
    t_pi = np.pi / (2 * g)
    model = DispersiveTransmonCavityModel(
        omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2
    )
    pulse = Pulse("sb", 0.0, t_pi, square, amp=g)
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=t_pi)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state( 1,0),
        {"sb": "sideband"},
        config=SimulationConfig(),
    )
    p_g1 = abs(model.basis_state( 0,1).overlap(result.final_state)) ** 2
    p_e0 = abs(model.basis_state( 1,0).overlap(result.final_state)) ** 2
    print("P(g,1):", float(p_g1))
    print("P(e,0):", float(p_e0))


if __name__ == "__main__":
    main()

