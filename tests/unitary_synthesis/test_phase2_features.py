from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cqed_sim.core import (
    BosonicModeSpec,
    DispersiveCouplingSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
)
from cqed_sim.pulses import Pulse, square_envelope
from cqed_sim.sim import NoiseSpec
from cqed_sim.unitary_synthesis import (
    LeakagePenalty,
    MultiObjective,
    ParameterDistribution,
    ParetoFrontResult,
    PrimitiveGate,
    Subspace,
    SynthesisConstraints,
    TargetStateMapping,
    TargetUnitary,
    UnitarySynthesizer,
)
from cqed_sim.unitary_synthesis.metrics import subspace_unitary_fidelity


def _rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _two_level_model() -> UniversalCQEDModel:
    return UniversalCQEDModel(
        transmon=TransmonModeSpec(
            omega=2.0 * np.pi * 6.0e9,
            dim=2,
            alpha=0.0,
            label="qubit",
            aliases=("qubit", "transmon"),
            frame_channel="q",
        ),
        bosonic_modes=(
            BosonicModeSpec(
                label="storage",
                omega=2.0 * np.pi * 5.0e9,
                dim=2,
                kerr=2.0 * np.pi * (-2.0e3),
                aliases=("storage", "cavity"),
                frame_channel="c",
            ),
        ),
        dispersive_couplings=(DispersiveCouplingSpec(mode="storage", chi=2.0 * np.pi * (-2.4e6)),),
    )


@dataclass
class _RobustAngleModel:
    scale: float = 1.0
    subsystem_dims: tuple[int, ...] = (2,)


class _FixedSamples:
    def __init__(self, values: list[float]) -> None:
        self.values = list(values)
        self.index = 0

    def sample(self, rng) -> float:  # pragma: no cover - exercised via ParameterDistribution.
        value = self.values[self.index % len(self.values)]
        self.index += 1
        return float(value)

    def nominal(self) -> float:
        return float(self.values[0])


def test_constraint_limited_optimization_and_warm_start_export(tmp_path) -> None:
    primitive = PrimitiveGate(
        name="idle",
        duration=80.0e-9,
        matrix=np.eye(2, dtype=np.complex128),
        parameters={"duration": 80.0e-9},
        parameter_bounds={"duration": (20.0e-9, 100.0e-9)},
        hilbert_dim=2,
    )
    synth = UnitarySynthesizer(
        primitives=[primitive],
        target=TargetUnitary(np.eye(2, dtype=np.complex128)),
        objectives=MultiObjective(fidelity_weight=1.0, duration_weight=1.0),
        synthesis_constraints=SynthesisConstraints(max_duration=40.0e-9, duration_mode="hard"),
        seed=3,
    )
    result = synth.fit(maxiter=6)
    assert result.sequence.total_duration() <= 40.0e-9 + 1.0e-12
    assert result.report["constraint_violations"]["duration_violation"] <= 1.0e-12

    save_path = tmp_path / "phase2_solution.json"
    result.save(save_path)

    warm_start = UnitarySynthesizer(
        primitives=[
            PrimitiveGate(
                name="idle",
                duration=80.0e-9,
                matrix=np.eye(2, dtype=np.complex128),
                parameters={"duration": 80.0e-9},
                parameter_bounds={"duration": (20.0e-9, 100.0e-9)},
                hilbert_dim=2,
            )
        ],
        target=TargetUnitary(np.eye(2, dtype=np.complex128)),
        warm_start=save_path,
        seed=3,
    )
    warm_result = warm_start.fit(maxiter=1)
    assert abs(warm_result.sequence.total_duration() - result.sequence.total_duration()) <= 1.0e-12


def test_open_system_unitary_targets_use_probe_state_fidelity() -> None:
    model = _two_level_model()

    def waveform(params, model):
        pulse = Pulse(
            "qubit",
            0.0,
            float(params["duration"]),
            square_envelope,
            amp=float(params["amp"]),
            phase=float(params.get("phase", 0.0)),
        )
        return [pulse], {"qubit": "qubit"}

    primitive = PrimitiveGate(
        name="drive",
        duration=20.0e-9,
        waveform=waveform,
        parameters={"amp": 0.01, "phase": 0.0, "duration": 20.0e-9},
        parameter_bounds={"amp": (-0.1, 0.1), "phase": (-np.pi, np.pi), "duration": (10.0e-9, 40.0e-9)},
        hilbert_dim=4,
    )
    synth = UnitarySynthesizer(
        model=model,
        backend="pulse",
        primitives=[primitive],
        target=TargetUnitary(np.eye(4, dtype=np.complex128), ignore_global_phase=True),
        simulation_options={"noise": NoiseSpec(t1=40.0e-6, tphi=30.0e-6), "dt": 2.0e-9},
        seed=5,
    )
    result = synth.fit(maxiter=1)
    assert result.report["target"]["type"] == "unitary"
    assert np.isfinite(result.report["metrics"]["fidelity"])
    assert result.report["target"]["open_system_probe_strategy"] == "basis_plus_uniform"


def test_robust_optimization_improves_worst_case_fidelity() -> None:
    scales = [0.7, 0.8, 0.9, 1.0]
    target = TargetUnitary(_rotation_y(np.pi / 2.0), ignore_global_phase=True)

    def make_primitive() -> PrimitiveGate:
        return PrimitiveGate(
            name="robust_ry",
            duration=20.0e-9,
            matrix=lambda params, model: _rotation_y(float(params["theta"]) * float(getattr(model, "scale", 1.0))),
            parameters={"theta": 0.2, "duration": 20.0e-9},
            parameter_bounds={"theta": (-2.0 * np.pi, 2.0 * np.pi), "duration": (10.0e-9, 30.0e-9)},
            hilbert_dim=2,
        )

    nominal = UnitarySynthesizer(
        model=_RobustAngleModel(),
        primitives=[make_primitive()],
        target=target,
        optimizer="powell",
        optimize_times=False,
        seed=1,
    ).fit(maxiter=40)

    robust = UnitarySynthesizer(
        model=_RobustAngleModel(),
        primitives=[make_primitive()],
        target=target,
        optimizer="powell",
        optimize_times=False,
        objectives=MultiObjective(fidelity_weight=1.0, robustness_weight=4.0),
        parameter_distribution=ParameterDistribution(
            sample_count=len(scales) - 1,
            include_nominal=False,
            aggregate="worst",
            scale=_FixedSamples(scales),
        ),
        seed=1,
    ).fit(maxiter=40)

    def worst_case_fidelity(theta: float) -> float:
        return min(subspace_unitary_fidelity(_rotation_y(theta * scale), target.matrix, gauge="global") for scale in scales)

    theta_nominal = float(nominal.sequence.gates[0].parameters["theta"])
    theta_robust = float(robust.sequence.gates[0].parameters["theta"])
    assert worst_case_fidelity(theta_robust) > worst_case_fidelity(theta_nominal)


def test_pareto_front_exposes_multiobjective_tradeoffs() -> None:
    primitive = PrimitiveGate(
        name="idle",
        duration=80.0e-9,
        matrix=np.eye(2, dtype=np.complex128),
        parameters={"duration": 80.0e-9},
        parameter_bounds={"duration": (20.0e-9, 100.0e-9)},
        hilbert_dim=2,
    )
    synth = UnitarySynthesizer(
        primitives=[primitive],
        target=TargetUnitary(np.eye(2, dtype=np.complex128)),
        seed=9,
    )
    front = synth.explore_pareto(
        [
            MultiObjective(fidelity_weight=1.0, duration_weight=0.0),
            MultiObjective(fidelity_weight=1.0, duration_weight=1.0),
        ],
        maxiter=6,
    )
    assert isinstance(front, ParetoFrontResult)
    assert len(front.results) == 2
    assert front.results[1].report["metrics"]["duration_metric"] <= front.results[0].report["metrics"]["duration_metric"]
    assert front.nondominated_indices


def test_leakage_penalty_reports_out_of_subspace_cost() -> None:
    def leakage_gate(params, model):
        phi = float(params["phi"])
        c = np.cos(phi)
        s = np.sin(phi)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, c, -s],
                [0.0, s, c],
            ],
            dtype=np.complex128,
        )

    primitive = PrimitiveGate(
        name="leak",
        duration=20.0e-9,
        matrix=leakage_gate({"phi": 0.5}, None),
        parameters={"duration": 20.0e-9},
        parameter_bounds={"duration": (19.0e-9, 21.0e-9)},
        hilbert_dim=3,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(3, [0, 1]),
        primitives=[primitive],
        target=TargetStateMapping(
            initial_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
            target_state=np.array([0.0, 1.0, 0.0], dtype=np.complex128),
        ),
        leakage_penalty=LeakagePenalty(weight=1.0),
        objectives=MultiObjective(fidelity_weight=1.0, leakage_weight=1.0),
        optimize_times=False,
        seed=2,
    )
    result = synth.fit(maxiter=1)
    assert result.report["metrics"]["leakage_worst"] > 0.0
    assert result.report["objective"]["leakage_term"] > 0.0
