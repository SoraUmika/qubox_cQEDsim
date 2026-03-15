from __future__ import annotations

import numpy as np

from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.unitary_synthesis import CQEDSystemAdapter, GateSequence, QubitRotation, SQR, Displacement, Subspace
from cqed_sim.unitary_synthesis.backends import simulate_sequence
from cqed_sim.unitary_synthesis.waveform_bridge import waveform_sequence_from_gates


def _test_model() -> tuple[DispersiveTransmonCavityModel, FrameSpec]:
    model = DispersiveTransmonCavityModel(
        omega_c=2.0 * np.pi * 5.0e9,
        omega_q=2.0 * np.pi * 6.0e9,
        alpha=2.0 * np.pi * (-180.0e6),
        chi=2.0 * np.pi * (-2.4e6),
        kerr=2.0 * np.pi * (-28.0e3),
        n_cav=3,
        n_tr=2,
    )
    frame = FrameSpec(omega_c_frame=2.0 * np.pi * 5.0e9, omega_q_frame=2.0 * np.pi * 6.0e9)
    return model, frame


def test_waveform_bridge_converts_supported_gate_sequence() -> None:
    seq = GateSequence(
        gates=[
            QubitRotation(name="rot", theta=np.pi / 6.0, phi=0.1, duration=16.0e-9),
            SQR(name="sqr", theta_n=[0.08, -0.04, 0.0], phi_n=[0.0, 0.2, 0.0], duration=1.0e-6),
            Displacement(name="disp", alpha=0.03 + 0.02j, duration=48.0e-9),
        ],
        n_cav=3,
    )
    converted = waveform_sequence_from_gates(seq)
    assert len(converted.gates) == 3
    assert all(gate.type == "PrimitiveGate" for gate in converted.gates)
    assert converted.n_cav == seq.n_cav


def test_waveform_bridge_supports_open_system_state_propagation() -> None:
    model, frame = _test_model()
    seq = GateSequence(
        gates=[
            QubitRotation(name="rot", theta=np.pi / 8.0, phi=0.0, duration=16.0e-9),
            SQR(name="sqr", theta_n=[0.05, -0.03, 0.0], phi_n=[0.0, 0.15, 0.0], duration=1.0e-6),
            Displacement(name="disp", alpha=0.01 + 0.0j, duration=48.0e-9),
        ],
        n_cav=3,
    )
    converted = waveform_sequence_from_gates(seq, frame=frame)
    subspace = Subspace.qubit_cavity_block(n_match=1, n_cav=3)
    psi0 = model.basis_state(0, 0)
    result = simulate_sequence(
        converted,
        subspace,
        backend="pulse",
        system=CQEDSystemAdapter(model=model),
        state_inputs=[psi0],
        need_operator=False,
        dt=4.0e-9,
        frame=frame,
        noise=NoiseSpec(t1=60.0e-6, tphi=80.0e-6),
    )
    assert result.state_outputs is not None
    assert len(result.state_outputs) == 1
    assert result.full_operator is None
    rho = result.state_outputs[0]
    assert rho.shape == (6, 6) or rho.shape == (6, 1)