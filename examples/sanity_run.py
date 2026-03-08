from __future__ import annotations

import numpy as np

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.diagnostics import channel_norms
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def gauss(t):
    return gaussian_envelope(t, sigma=0.2)


def main():
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=-2 * np.pi * 0.25,
        chi=2 * np.pi * 0.02,
        kerr=-2 * np.pi * 0.005,
        n_cav=12,
        n_tr=3,
    )
    pulses = [
        Pulse("c", 0.0, 2.4, gauss, amp=0.18),
        Pulse("q", 1.0, 1.2, gauss, amp=0.9, phase=np.pi / 2),
    ]
    compiled = SequenceCompiler(dt=0.02).compile(pulses, t_end=3.0)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state( 0,0),
        {"c": "cavity", "q": "qubit"},
        config=SimulationConfig(frame=FrameSpec()),
    )
    print("Final P_e:", float(result.expectations["P_e"][-1]))
    print("Final <n_c>:", float(result.expectations["n_c"][-1]))
    print("Channel norms:", channel_norms(compiled))


if __name__ == "__main__":
    main()

