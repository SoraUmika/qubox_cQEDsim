"""Hardware-faithful time-domain cQED pulse simulator."""

from .core.ideal_gates import displacement_op, embed_cavity_op, embed_qubit_op, qubit_rotation_xy, sqr_op
from .core.frame import FrameSpec
from .core.model import DispersiveTransmonCavityModel
from .io.gates import load_gate_sequence, render_gate_table
from .pulses.pulse import Pulse
from .sequence.scheduler import SequenceCompiler
from .simulators.common import build_frame, build_initial_state, build_model, build_noise_spec
from .simulators.ideal import ideal_gate_unitary, run_case_a
from .simulators.pulse_open import run_case_c
from .simulators.pulse_unitary import run_case_b
from .sim.noise import NoiseSpec
from .sim.runner import SimulationConfig, simulate_sequence
from .tomo.device import DeviceParameters
from .tomo.protocol import QubitPulseCal, autocalibrate_all_xy, run_all_xy, run_fock_resolved_tomo
from .snap_opt.optimizer import optimize_snap_parameters
from .snap_opt.pulses import SnapToneParameters

__all__ = [
    "DispersiveTransmonCavityModel",
    "FrameSpec",
    "Pulse",
    "SequenceCompiler",
    "SimulationConfig",
    "NoiseSpec",
    "simulate_sequence",
    "load_gate_sequence",
    "render_gate_table",
    "build_initial_state",
    "build_model",
    "build_frame",
    "build_noise_spec",
    "ideal_gate_unitary",
    "run_case_a",
    "run_case_b",
    "run_case_c",
    "qubit_rotation_xy",
    "displacement_op",
    "sqr_op",
    "embed_qubit_op",
    "embed_cavity_op",
    "DeviceParameters",
    "QubitPulseCal",
    "run_all_xy",
    "autocalibrate_all_xy",
    "run_fock_resolved_tomo",
    "SnapToneParameters",
    "optimize_snap_parameters",
]
