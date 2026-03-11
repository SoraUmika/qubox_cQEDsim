from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.backends import JaxBackend, NumPyBackend
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import NoiseSpec, SimulationConfig, simulate_sequence


def _free_model() -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=2.0,
        omega_q=3.0,
        alpha=0.0,
        chi=0.0,
        kerr=0.0,
        n_cav=2,
        n_tr=2,
    )


def test_numpy_backend_matches_qutip_for_closed_system_free_evolution():
    model = _free_model()
    compiled = SequenceCompiler(dt=0.01).compile([], t_end=0.5)
    initial = (model.basis_state(0, 0) + model.basis_state(1, 0)).unit()

    reference = simulate_sequence(model, compiled, initial, {}, SimulationConfig(frame=FrameSpec()))
    backend_result = simulate_sequence(
        model,
        compiled,
        initial,
        {},
        SimulationConfig(frame=FrameSpec(), backend=NumPyBackend()),
    )

    overlap = abs(reference.final_state.overlap(backend_result.final_state))
    assert np.isclose(overlap, 1.0, atol=1.0e-6)


@pytest.mark.skipif(JaxBackend is None, reason="JAX is not installed.")
def test_jax_backend_matches_numpy_backend_for_open_system_evolution():
    model = _free_model()
    compiled = SequenceCompiler(dt=0.02).compile([], t_end=0.4)
    initial = model.basis_state(1, 0)
    noise = NoiseSpec(t1=1.5, tphi=3.0)

    numpy_result = simulate_sequence(
        model,
        compiled,
        initial,
        {},
        SimulationConfig(frame=FrameSpec(), backend=NumPyBackend()),
        noise=noise,
    )
    jax_result = simulate_sequence(
        model,
        compiled,
        initial,
        {},
        SimulationConfig(frame=FrameSpec(), backend=JaxBackend()),
        noise=noise,
    )

    assert (numpy_result.final_state - jax_result.final_state).norm() < 1.0e-6
