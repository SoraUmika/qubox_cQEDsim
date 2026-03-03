from __future__ import annotations

from typing import Any, Mapping

import qutip as qt

from cqed_sim.core.ideal_gates import displacement_op, qubit_rotation_xy, sqr_op
from cqed_sim.io.gates import DisplacementGate, Gate, RotationGate, SQRGate
from cqed_sim.operators.basic import embed_cavity_op, embed_qubit_op
from cqed_sim.pulses.calibration import pad_sqr_angles
from cqed_sim.simulators.common import build_initial_state, finalize_track, snapshot_from_state


def ideal_gate_unitary(gate: Gate, n_cav_dim: int) -> qt.Qobj:
    if isinstance(gate, DisplacementGate):
        return embed_cavity_op(displacement_op(n_cav_dim, gate.alpha), n_tr=2)
    if isinstance(gate, RotationGate):
        return embed_qubit_op(qubit_rotation_xy(gate.theta, gate.phi), n_cav_dim)
    if isinstance(gate, SQRGate):
        theta, phi = pad_sqr_angles(gate.theta, gate.phi, n_cav_dim)
        return sqr_op(theta, phi)
    raise TypeError(f"Unsupported gate type '{type(gate).__name__}'.")


def run_case_a(gates: list[Gate], config: Mapping[str, Any], case_label: str = "Case A") -> dict[str, Any]:
    state = build_initial_state(config, n_cav_dim=int(config["n_cav_dim"]))
    snapshots = [snapshot_from_state(state, 0, None, config, case_label=case_label)]
    for step_index, gate in enumerate(gates, start=1):
        unitary = ideal_gate_unitary(gate, int(config["n_cav_dim"]))
        state = unitary * state if not state.isoper else unitary * state * unitary.dag()
        snapshots.append(snapshot_from_state(state, step_index, gate, config, case_label=case_label))
    return finalize_track(case_label, snapshots, metadata={"solver": "instantaneous_unitary"})
