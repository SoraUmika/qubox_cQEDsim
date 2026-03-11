from __future__ import annotations

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.ideal_gates import embed_cavity_op, embed_qubit_op
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.extractors import conditioned_bloch_xyz, reduced_cavity_state, reduced_qubit_state
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def test_basis_state_flat_indices_follow_qubit_then_cavity():
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=4, n_tr=2)

    expected = {
        (0, 0): 0,
        (1, 0): 4,
        (0, 1): 1,
    }
    for (q, n), flat_index in expected.items():
        psi = model.basis_state(q, n)
        manual = qt.tensor(qt.basis(2, q), qt.basis(4, n))
        assert (psi - manual).norm() < 1e-12
        assert int(np.argmax(np.abs(np.asarray(psi.full()).ravel()))) == flat_index


def test_operator_embeddings_act_on_qubit_and_cavity_slots():
    n_cav = 5
    state = qt.tensor(qt.basis(2, 1), qt.basis(n_cav, 2))

    z_q = embed_qubit_op(qt.sigmaz(), n_cav)
    n_c = embed_cavity_op(qt.num(n_cav), 2)

    assert np.isclose(qt.expect(z_q, state), -1.0, atol=1e-12)
    assert np.isclose(qt.expect(n_c, state), 2.0, atol=1e-12)


def test_static_hamiltonian_diagonal_uses_qubit_cavity_order():
    omega_c = 1.7
    omega_q = 5.2
    chi = 0.4
    kerr = -0.15
    n_cav = 4

    model = DispersiveTransmonCavityModel(
        omega_c=omega_c,
        omega_q=omega_q,
        alpha=0.0,
        chi=chi,
        kerr=kerr,
        n_cav=n_cav,
        n_tr=2,
    )
    diag = np.real(np.diag(np.asarray(model.static_hamiltonian().full(), dtype=np.complex128)))

    assert np.isclose(diag[0], 0.0, atol=1e-12)  # |g,0>
    assert np.isclose(diag[1], omega_c, atol=1e-12)  # |g,1>
    assert np.isclose(diag[2], 2.0 * omega_c - kerr, atol=1e-12)  # |g,2>
    assert np.isclose(diag[n_cav], omega_q, atol=1e-12)  # |e,0>
    assert np.isclose(diag[n_cav + 1], omega_q + omega_c - chi, atol=1e-12)  # |e,1>


def test_conditioned_bloch_uses_fock_projector_on_second_subsystem():
    n_cav = 4
    state = qt.tensor((qt.basis(2, 0) + qt.basis(2, 1)).unit(), qt.basis(n_cav, 2))

    x2, y2, z2, p2, valid2 = conditioned_bloch_xyz(state, n=2)
    x0, y0, z0, p0, valid0 = conditioned_bloch_xyz(state, n=0, fallback="nan")

    assert valid2
    assert np.isclose(p2, 1.0, atol=1e-12)
    assert np.allclose([x2, y2, z2], [1.0, 0.0, 0.0], atol=1e-12)
    assert not valid0
    assert np.isclose(p0, 0.0, atol=1e-12)
    assert np.isnan(x0) and np.isnan(y0) and np.isnan(z0)


def test_state_prep_evolution_and_measurement_keep_qubit_cavity_consistent():
    model = DispersiveTransmonCavityModel(omega_c=0.0, omega_q=0.0, alpha=0.0, chi=0.0, kerr=0.0, n_cav=3, n_tr=2)
    pulse = Pulse("q", 0.0, 1.0, _square, amp=np.pi / 4.0)
    compiled = SequenceCompiler(dt=0.01).compile([pulse], t_end=1.1)

    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 1),
        {"q": "qubit"},
        SimulationConfig(frame=FrameSpec()),
    )

    rho_q = reduced_qubit_state(result.final_state)
    rho_c = reduced_cavity_state(result.final_state)

    assert np.isclose(float(np.real((rho_c * qt.basis(3, 1).proj()).tr())), 1.0, atol=1e-6)
    assert np.isclose(float(np.real((rho_c * qt.num(3)).tr())), 1.0, atol=1e-6)
    assert np.allclose(
        [
            float(np.real((rho_q * qt.sigmax()).tr())),
            float(np.real((rho_q * qt.sigmay()).tr())),
            float(np.real((rho_q * qt.sigmaz()).tr())),
        ],
        [0.0, -1.0, 0.0],
        atol=5e-2,
    )
