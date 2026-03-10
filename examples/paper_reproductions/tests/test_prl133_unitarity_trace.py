from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from examples.paper_reproductions.snap_prl133.model import SnapModelConfig
from examples.paper_reproductions.snap_prl133.pulses import SnapToneParameters, slow_stage_multitone_pulse
from cqed_sim.sequence.scheduler import SequenceCompiler


def test_prl133_closed_system_unitarity_and_trace():
    model = SnapModelConfig(n_cav=6, n_tr=2, chi=2 * np.pi * 0.02).build_model()
    target = np.array([0.0, -np.pi / 4.0, np.pi / 2.0], dtype=float)
    cfg = SimulationConfig(frame=FrameSpec(omega_q_frame=model.omega_q), store_states=True, atol=1e-9, rtol=1e-8)
    pulse = slow_stage_multitone_pulse(
        model=model,
        target_phases=target,
        params=SnapToneParameters.vanilla(target),
        duration=120.0,
        base_amp=0.010,
        frame=cfg.frame,
        channel="q",
    )
    compiled = SequenceCompiler(dt=0.25).compile([pulse], t_end=120.25)
    psi0 = model.basis_state( 0,0)
    res_ket = simulate_sequence(model, compiled, psi0, {"q": "qubit"}, config=cfg)
    assert res_ket.states is not None
    norms = [float(s.norm()) for s in res_ket.states]
    assert max(abs(n - 1.0) for n in norms) < 1e-7

    rho0 = qt.ket2dm(psi0)
    res_rho = simulate_sequence(model, compiled, rho0, {"q": "qubit"}, config=cfg)
    assert res_rho.states is not None
    traces = [float(np.real(s.tr())) for s in res_rho.states]
    assert max(abs(t - 1.0) for t in traces) < 1e-7

