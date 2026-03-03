from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.ideal_gates import displacement_op, embed_qubit_op, qubit_rotation_axis
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.extractors import cavity_moments
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(x):
    return np.ones_like(x, dtype=np.complex128)


def test_pulse_x90_matches_ideal_rotation_in_linear_regime():
    m = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=2, n_tr=2)
    p = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    c = SequenceCompiler(dt=0.01).compile([p], t_end=1.1)
    sim = simulate_sequence(m, c, m.basis_state(0, 0), {"q": "qubit"}, SimulationConfig(frame=FrameSpec()))
    u = embed_qubit_op(qubit_rotation_axis(np.pi / 2, "x"), 2)
    ideal = u * m.basis_state(0, 0)
    f = abs(ideal.overlap(sim.final_state)) ** 2
    assert f > 0.995


@pytest.mark.slow
def test_pulse_displacement_matches_ideal_displacement_k0():
    m = DispersiveTransmonCavityModel(
        omega_c=2 * np.pi * 0.9, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=20, n_tr=2
    )
    amp = 0.10
    dur = 2.4
    p = Pulse("c", 0.0, dur, _square, amp=amp)
    c = SequenceCompiler(dt=0.02).compile([p], t_end=dur + 0.1)
    sim = simulate_sequence(
        m,
        c,
        m.basis_state(0, 0),
        {"c": "cavity"},
        SimulationConfig(frame=FrameSpec(omega_c_frame=m.omega_c, omega_q_frame=0.0)),
    )
    alpha = -1j * amp * dur
    ideal_c = displacement_op(m.n_cav, alpha) * qt.basis(m.n_cav, 0)
    joint_ideal = qt.tensor(ideal_c, qt.basis(2, 0))
    m_sim = cavity_moments(sim.final_state)
    m_id = cavity_moments(joint_ideal)
    assert np.isclose(m_sim["n"], m_id["n"], rtol=0.12, atol=0.02)
    assert np.isclose(m_sim["a"], m_id["a"], rtol=0.12, atol=0.03)

