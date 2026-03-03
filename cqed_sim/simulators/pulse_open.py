from __future__ import annotations

from typing import Any, Mapping

from cqed_sim.io.gates import Gate
from cqed_sim.simulators.pulse_unitary import run_pulse_case


def run_case_c(gates: list[Gate], config: Mapping[str, Any], case_label: str = "Case C") -> dict[str, Any]:
    return run_pulse_case(gates, config, include_dissipation=True, case_label=case_label)
