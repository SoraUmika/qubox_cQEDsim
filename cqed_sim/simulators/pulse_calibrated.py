from __future__ import annotations

from typing import Any, Mapping

from cqed_sim.calibration.sqr import calibrate_all_sqr_gates
from cqed_sim.io.gates import Gate
from cqed_sim.simulators.pulse_unitary import run_pulse_case


def run_case_d(
    gates: list[Gate],
    config: Mapping[str, Any],
    case_label: str = "Case D",
) -> dict[str, Any]:
    include_dissipation = bool(config.get("case_d_include_dissipation", True))
    calibration_map = calibrate_all_sqr_gates(gates, config, cache_dir=config.get("calibration_cache_dir", "calibrations"))
    track = run_pulse_case(
        gates,
        config,
        include_dissipation=include_dissipation,
        case_label=case_label,
        sqr_calibration_map=calibration_map,
    )
    track["metadata"]["calibration_results"] = calibration_map
    track["metadata"]["calibration_summaries"] = {
        name: result.improvement_summary() for name, result in calibration_map.items()
    }
    return track
