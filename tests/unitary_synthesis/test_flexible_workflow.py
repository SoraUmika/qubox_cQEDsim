from __future__ import annotations

import numpy as np

from cqed_sim.core import (
    BosonicModeSpec,
    DispersiveCouplingSpec,
    FrameSpec,
    TransmonModeSpec,
    UniversalCQEDModel,
)
from cqed_sim.pulses import Pulse, square_envelope
from cqed_sim.unitary_synthesis import (
    CQEDSystemAdapter,
    GateSequence,
    PrimitiveGate,
    Subspace,
    TargetStateMapping,
    TargetUnitary,
    UnitarySynthesizer,
)
from cqed_sim.unitary_synthesis.backends import simulate_sequence


def _rotation_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -np.sin(theta / 2.0)],
            [np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _two_level_model() -> tuple[UniversalCQEDModel, FrameSpec]:
    model = UniversalCQEDModel(
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
    frame = FrameSpec(omega_c_frame=2.0 * np.pi * 5.0e9, omega_q_frame=2.0 * np.pi * 6.0e9)
    return model, frame


def test_matrix_defined_primitive_matches_identity_target() -> None:
    sub = Subspace.custom(2, range(2))
    primitive = PrimitiveGate(
        name="identity",
        duration=40.0e-9,
        matrix=np.eye(2, dtype=np.complex128),
        hilbert_dim=2,
    )
    synth = UnitarySynthesizer(
        subspace=sub,
        primitives=[primitive],
        target=TargetUnitary(np.eye(2, dtype=np.complex128)),
        optimize_times=False,
        seed=7,
    )
    result = synth.fit(maxiter=5)
    assert result.report["metrics"]["fidelity"] > 0.999999


def test_waveform_defined_primitive_returns_unitary_operator() -> None:
    model, frame = _two_level_model()

    def waveform(params, model):
        duration = float(params["duration"])
        pulse = Pulse("qubit", 0.0, duration, square_envelope, amp=float(params["amp"]), phase=float(params["phase"]))
        return [pulse], {"qubit": "qubit"}

    primitive = PrimitiveGate(
        name="wf_identity",
        duration=20.0e-9,
        waveform=waveform,
        parameters={"amp": 0.0, "phase": 0.0, "duration": 20.0e-9},
        parameter_bounds={"amp": (-1.0, 1.0), "phase": (-np.pi, np.pi), "duration": (10.0e-9, 50.0e-9)},
        hilbert_dim=4,
    )
    sequence = GateSequence(gates=[primitive], full_dim=4)
    result = simulate_sequence(
        sequence,
        Subspace.custom(4, range(4)),
        backend="pulse",
        model=model,
        frame=frame,
        dt=5.0e-9,
    )
    ident = np.eye(4, dtype=np.complex128)
    assert np.linalg.norm(result.full_operator.conj().T @ result.full_operator - ident) < 1.0e-8


def test_waveform_defined_primitive_supports_cqed_system_adapter() -> None:
    model, frame = _two_level_model()

    def waveform(params, model):
        duration = float(params["duration"])
        pulse = Pulse("qubit", 0.0, duration, square_envelope, amp=float(params["amp"]), phase=float(params["phase"]))
        return [pulse], {"qubit": "qubit"}

    primitive = PrimitiveGate(
        name="wf_identity",
        duration=20.0e-9,
        waveform=waveform,
        parameters={"amp": 0.0, "phase": 0.0, "duration": 20.0e-9},
        parameter_bounds={"amp": (-1.0, 1.0), "phase": (-np.pi, np.pi), "duration": (10.0e-9, 50.0e-9)},
        hilbert_dim=4,
    )
    sequence = GateSequence(gates=[primitive], full_dim=4)
    result = simulate_sequence(
        sequence,
        Subspace.custom(4, range(4)),
        backend="pulse",
        system=CQEDSystemAdapter(model=model),
        frame=frame,
        dt=5.0e-9,
    )
    ident = np.eye(4, dtype=np.complex128)
    assert np.linalg.norm(result.full_operator.conj().T @ result.full_operator - ident) < 1.0e-8


def test_unitary_target_optimization_converges_with_matrix_primitive() -> None:
    sub = Subspace.custom(2, range(2))
    primitive = PrimitiveGate(
        name="ry",
        duration=30.0e-9,
        matrix=lambda params, model: _rotation_y(float(params["theta"])),
        parameters={"theta": 0.1, "duration": 30.0e-9},
        parameter_bounds={"theta": (-np.pi, np.pi), "duration": (10.0e-9, 50.0e-9)},
        hilbert_dim=2,
    )
    target = TargetUnitary(_rotation_y(np.pi))
    synth = UnitarySynthesizer(
        subspace=sub,
        primitives=[primitive],
        target=target,
        optimizer="nelder_mead",
        optimize_times=False,
        seed=11,
    )
    result = synth.fit(maxiter=80)
    assert result.report["metrics"]["fidelity"] > 0.99


def test_state_mapping_target_optimization_converges() -> None:
    sub = Subspace.custom(2, range(2))
    primitive = PrimitiveGate(
        name="ry",
        duration=30.0e-9,
        matrix=lambda params, model: _rotation_y(float(params["theta"])),
        parameters={"theta": 0.1, "duration": 30.0e-9},
        parameter_bounds={"theta": (-np.pi, np.pi), "duration": (10.0e-9, 50.0e-9)},
        hilbert_dim=2,
    )
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)
    phi0 = np.array([0.0, 1.0], dtype=np.complex128)
    synth = UnitarySynthesizer(
        subspace=sub,
        primitives=[primitive],
        target=TargetStateMapping(initial_state=psi0, target_state=phi0),
        optimizer="differential_evolution",
        optimize_times=False,
        seed=19,
    )
    result = synth.fit(maxiter=6)
    assert result.report["metrics"]["state_fidelity_mean"] > 0.99


def test_model_backed_waveform_state_mapping_accepts_arbitrary_cqed_model() -> None:
    model, frame = _two_level_model()

    def waveform(params, model):
        duration = float(params["duration"])
        pulse = Pulse("qubit", 0.0, duration, square_envelope, amp=float(params["amp"]), phase=float(params["phase"]))
        return [pulse], {"qubit": "qubit"}

    primitive = PrimitiveGate(
        name="wf_ground",
        duration=20.0e-9,
        waveform=waveform,
        parameters={"amp": 0.0, "phase": 0.0, "duration": 20.0e-9},
        parameter_bounds={"amp": (-1.0, 1.0), "phase": (-np.pi, np.pi), "duration": (10.0e-9, 50.0e-9)},
        hilbert_dim=4,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(4, range(4)),
        model=model,
        primitives=[primitive],
        target=TargetStateMapping(
            initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            target_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        ),
        optimize_times=False,
        simulation_options={"frame": frame, "dt": 5.0e-9},
        seed=23,
    )
    result = synth.fit(maxiter=4)
    assert result.report["metrics"]["state_fidelity_mean"] > 0.999
    assert result.report["system"]["kind"] == "CQEDSystemAdapter"


def test_system_backed_waveform_state_mapping_accepts_cqed_adapter() -> None:
    model, frame = _two_level_model()

    def waveform(params, model):
        duration = float(params["duration"])
        pulse = Pulse("qubit", 0.0, duration, square_envelope, amp=float(params["amp"]), phase=float(params["phase"]))
        return [pulse], {"qubit": "qubit"}

    primitive = PrimitiveGate(
        name="wf_ground",
        duration=20.0e-9,
        waveform=waveform,
        parameters={"amp": 0.0, "phase": 0.0, "duration": 20.0e-9},
        parameter_bounds={"amp": (-1.0, 1.0), "phase": (-np.pi, np.pi), "duration": (10.0e-9, 50.0e-9)},
        hilbert_dim=4,
    )
    synth = UnitarySynthesizer(
        subspace=Subspace.custom(4, range(4)),
        system=CQEDSystemAdapter(model=model),
        primitives=[primitive],
        target=TargetStateMapping(
            initial_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            target_state=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
        ),
        optimize_times=False,
        simulation_options={"frame": frame, "dt": 5.0e-9},
        seed=29,
    )
    result = synth.fit(maxiter=4)
    assert result.report["metrics"]["state_fidelity_mean"] > 0.999
    assert result.report["system"]["kind"] == "CQEDSystemAdapter"
