from __future__ import annotations

import numpy as np

from cqed_sim.unitary_synthesis.sequence import Displacement, GateSequence, SQR


def _make_multi_gate_sequence(n_cav: int = 4) -> GateSequence:
    gates = [
        SQR(name="sqr_a", theta_n=[0.1] * n_cav, phi_n=[0.0] * n_cav, duration=100e-9),
        Displacement(name="disp_a", alpha=0.02 + 0.01j, duration=90e-9),
        SQR(name="sqr_b", theta_n=[0.2] * n_cav, phi_n=[0.1] * n_cav, duration=120e-9),
        Displacement(name="disp_b", alpha=-0.01 + 0.03j, duration=80e-9),
    ]
    return GateSequence(gates=gates, n_cav=n_cav)


def test_time_policy_enforcement_modes() -> None:
    policy = {
        "default": {"optimize": True, "bounds": (20e-9, 2e-6), "init": 200e-9},
        "SQR": {"optimize": True, "bounds": (50e-9, 1.2e-6), "init": 400e-9},
        "Displacement": {"optimize": True, "bounds": (30e-9, 0.9e-6), "init": 300e-9},
    }

    seq_type = _make_multi_gate_sequence()
    seq_type.configure_time_parameters(policy, mode="per-type")
    sqr_groups = [gate.time_group for gate in seq_type.gates if gate.type == "SQR"]
    assert len(set(sqr_groups)) == 1

    sqr_param_idx = next(i for i, param in enumerate(seq_type.time_params) if param.group == sqr_groups[0])
    vals = seq_type.get_time_vector(active_only=False)
    vals[sqr_param_idx] = 900e-9
    seq_type.set_time_vector(vals, active_only=False)
    sqr_durations = [gate.duration for gate in seq_type.gates if gate.type == "SQR"]
    assert np.allclose(sqr_durations, [900e-9, 900e-9])

    seq_instance = _make_multi_gate_sequence()
    seq_instance.configure_time_parameters(policy, mode="per-instance")
    sqr_groups_inst = [gate.time_group for gate in seq_instance.gates if gate.type == "SQR"]
    assert len(set(sqr_groups_inst)) == 2

    idx0 = next(i for i, param in enumerate(seq_instance.time_params) if param.group == sqr_groups_inst[0])
    idx1 = next(i for i, param in enumerate(seq_instance.time_params) if param.group == sqr_groups_inst[1])
    vals = seq_instance.get_time_vector(active_only=False)
    old_1 = vals[idx1]
    vals[idx0] = 850e-9
    seq_instance.set_time_vector(vals, active_only=False)
    assert np.isclose(seq_instance.time_params[idx0].value, 850e-9)
    assert np.isclose(seq_instance.time_params[idx1].value, old_1)

    freeze_policy = {
        "default": {"optimize": True, "bounds": (20e-9, 2e-6), "init": 200e-9},
        "SQR": {"optimize": False, "bounds": (50e-9, 1.2e-6), "init": 400e-9},
        "Displacement": {"optimize": True, "bounds": (30e-9, 0.9e-6), "init": 300e-9},
    }
    seq_freeze = _make_multi_gate_sequence()
    seq_freeze.configure_time_parameters(freeze_policy, mode="per-type")
    active_types = {param.gate_type for param in seq_freeze.active_time_params()}
    assert "SQR" not in active_types
    assert "Displacement" in active_types

    seq_hybrid = _make_multi_gate_sequence()
    seq_hybrid.configure_time_parameters(
        policy,
        mode="hybrid",
        shared_groups={"sqr_a": "shared_sqr", "sqr_b": "shared_sqr"},
    )
    sqr_groups_hybrid = [gate.time_group for gate in seq_hybrid.gates if gate.type == "SQR"]
    disp_groups_hybrid = [gate.time_group for gate in seq_hybrid.gates if gate.type == "Displacement"]
    assert len(set(sqr_groups_hybrid)) == 1
    assert len(set(disp_groups_hybrid)) == 2