from __future__ import annotations

import numpy as np

from cqed_sim.gates import blue_sideband, conditional_displacement, jaynes_cummings
from cqed_sim.unitary_synthesis import (
    BlueSidebandExchange,
    ConditionalDisplacement,
    JaynesCummingsExchange,
)


def _dense(op) -> np.ndarray:
    return np.asarray(op.full(), dtype=np.complex128)


def test_conditional_displacement_matches_ideal_gate() -> None:
    gate = ConditionalDisplacement(name="CD", alpha=0.21 + 0.13j, duration=160.0e-9, optimize_time=False)
    actual = _dense(gate.ideal_unitary(6))
    target = _dense(conditional_displacement(alpha=0.21 + 0.13j, cavity_dim=6))
    assert np.allclose(actual, target)


def test_jaynes_cummings_exchange_matches_gate_library_at_zero_phase() -> None:
    gate = JaynesCummingsExchange(
        name="JC",
        coupling=2.0 * np.pi * 3.5e6,
        duration=55.0e-9,
        phase=0.0,
        optimize_time=False,
    )
    actual = _dense(gate.ideal_unitary(5))
    target = _dense(jaynes_cummings(2.0 * np.pi * 3.5e6, 55.0e-9, cavity_dim=5))
    assert np.allclose(actual, target)


def test_blue_sideband_exchange_matches_gate_library_at_zero_phase() -> None:
    gate = BlueSidebandExchange(
        name="BSB",
        coupling=2.0 * np.pi * 2.2e6,
        duration=48.0e-9,
        phase=0.0,
        optimize_time=False,
    )
    actual = _dense(gate.ideal_unitary(5))
    target = _dense(blue_sideband(2.0 * np.pi * 2.2e6, 48.0e-9, cavity_dim=5))
    assert np.allclose(actual, target)
