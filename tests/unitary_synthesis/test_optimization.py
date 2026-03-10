from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis import Subspace
from cqed_sim.unitary_synthesis.backends import simulate_sequence
from cqed_sim.unitary_synthesis.optim import TimeMapper, UnitarySynthesizer
from cqed_sim.unitary_synthesis.sequence import GateSequence, QubitRotation


def test_e1_duration_mapping_stability() -> None:
    mapper = TimeMapper(20e-9, 2000e-9)
    x = np.asarray([-1e6, -10.0, 0.0, 10.0, 1e6], dtype=float)
    t = mapper.map(x)
    g = mapper.grad(x)
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(g))
    assert np.all(t >= 20e-9)
    assert np.all(t <= 2000e-9)


def test_e2_time_changes_dynamics_smoothly() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    gate = QubitRotation(name="r", theta=np.pi / 2, phi=0.0, duration=100e-9)
    seq = GateSequence(gates=[gate], n_cav=3)
    vals = []
    for t in [60e-9, 80e-9, 100e-9, 120e-9, 140e-9]:
        gate.duration = t
        vals.append(simulate_sequence(seq, sub, backend="pulse").subspace_operator.copy())
    diffs = [np.linalg.norm(vals[k + 1] - vals[k]) for k in range(len(vals) - 1)]
    assert all(d > 1e-5 for d in diffs)


def test_e3_optimizer_uses_time_to_improve_fidelity() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target_gate = QubitRotation(name="target", theta=np.pi / 2, phi=0.0, duration=100e-9)
    target = GateSequence(gates=[target_gate], n_cav=3).unitary(backend="pulse")

    synth = UnitarySynthesizer(
        subspace=sub,
        backend="pulse",
        gateset=["QubitRotation"],
        optimize_times=True,
        time_bounds={"default": (20e-9, 200e-9)},
        leakage_weight=0.0,
        time_reg_weight=0.0,
        seed=1,
    )
    synth.sequence.gates[0].theta = np.pi / 2
    synth.sequence.gates[0].duration = 30e-9
    result = synth.fit(target=target, multistart=2, maxiter=200)
    assert result.report["metrics"]["fidelity"] > 0.999


def test_e4_time_regularization_keeps_durations_near_initial() -> None:
    sub = Subspace.qubit_cavity_block(n_match=2)
    target_gate = QubitRotation(name="target", theta=np.pi / 2, phi=0.0, duration=100e-9)
    target = GateSequence(gates=[target_gate], n_cav=3).unitary(backend="pulse")

    synth = UnitarySynthesizer(
        subspace=sub,
        backend="pulse",
        gateset=["QubitRotation"],
        optimize_times=True,
        time_bounds={"default": (20e-9, 200e-9)},
        leakage_weight=0.0,
        time_reg_weight=1e12,
        seed=2,
    )
    t_init = synth.sequence.gates[0].duration
    res = synth.fit(target=target, multistart=1, maxiter=120)
    t_final = res.sequence.gates[0].duration
    assert abs(t_final - t_init) < 5e-9
