from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.experiment.measurement import QubitMeasurementResult, QubitMeasurementSpec, measure_qubit
from cqed_sim.experiment.state_prep import StatePreparationSpec, prepare_state
from cqed_sim.pulses.hardware import HardwareConfig
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import CompiledSequence, SequenceCompiler
from cqed_sim.sim.noise import NoiseSpec
from cqed_sim.sim.runner import SimulationConfig, SimulationResult, simulate_sequence


@dataclass
class ExperimentMetadata:
    label: str = ""
    target_state: qt.Qobj | None = None
    target_unitary: qt.Qobj | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    initial_state: qt.Qobj
    compiled: CompiledSequence
    simulation: SimulationResult
    measurement: QubitMeasurementResult | None
    metadata: ExperimentMetadata


@dataclass
class SimulationExperiment:
    model: Any
    pulses: list[Pulse]
    drive_ops: dict[str, str]
    dt: float
    t_end: float | None = None
    frame: FrameSpec = field(default_factory=FrameSpec)
    initial_state: qt.Qobj | None = None
    state_prep: StatePreparationSpec | None = None
    noise: NoiseSpec | None = None
    e_ops: dict[str, qt.Qobj] | None = None
    measurement: QubitMeasurementSpec | None = None
    hardware: dict[str, HardwareConfig] | None = None
    crosstalk_matrix: dict[str, dict[str, float]] | None = None
    metadata: ExperimentMetadata = field(default_factory=ExperimentMetadata)

    def resolve_initial_state(self) -> qt.Qobj:
        if self.initial_state is not None:
            return self.initial_state
        return prepare_state(self.model, self.state_prep)

    def compile(self) -> CompiledSequence:
        compiler = SequenceCompiler(
            dt=self.dt,
            hardware=self.hardware,
            crosstalk_matrix=self.crosstalk_matrix,
        )
        return compiler.compile(self.pulses, t_end=self.t_end)

    def run(self) -> ExperimentResult:
        initial = self.resolve_initial_state()
        compiled = self.compile()
        simulation = simulate_sequence(
            self.model,
            compiled,
            initial,
            self.drive_ops,
            config=SimulationConfig(frame=self.frame),
            noise=self.noise,
            e_ops=self.e_ops,
        )
        measurement = None if self.measurement is None else measure_qubit(simulation.final_state, self.measurement)
        return ExperimentResult(
            initial_state=initial,
            compiled=compiled,
            simulation=simulation,
            measurement=measurement,
            metadata=self.metadata,
        )
