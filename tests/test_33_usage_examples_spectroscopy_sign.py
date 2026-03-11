from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    Pulse,
    SequenceCompiler,
    SimulationConfig,
    carrier_for_transition_frequency,
    simulate_sequence,
)


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def _peak_transition_detuning(
    model: DispersiveTransmonCavityModel,
    cavity_level: int,
    transition_scan: np.ndarray,
) -> float:
    response = []
    for transition_detuning in transition_scan:
        pulse = Pulse(
            "q",
            t0=0.0,
            duration=14.0,
            envelope=_square,
            amp=0.06,
            carrier=carrier_for_transition_frequency(float(transition_detuning)),
        )
        compiled = SequenceCompiler(dt=0.1).compile([pulse], t_end=14.2)
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, cavity_level),
            {"q": "qubit"},
            SimulationConfig(frame=FrameSpec(omega_q_frame=model.omega_q)),
        )
        response.append(float(result.expectations["P_e"][-1]))
    response = np.asarray(response, dtype=float)
    return float(transition_scan[int(np.argmax(response))])


def test_negative_chi_transition_detuning_axis_moves_left_with_photon_number():
    chi = -2.0 * np.pi * 0.03
    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=chi,
        kerr=0.0,
        n_cav=8,
        n_tr=2,
    )
    frame = FrameSpec(omega_q_frame=model.omega_q)
    transition_scan = np.linspace(-2.8 * abs(chi), 0.8 * abs(chi), 33)

    expected = np.array(
        [model.manifold_transition_frequency(n, frame=frame) for n in range(3)],
        dtype=float,
    )
    observed = np.array(
        [_peak_transition_detuning(model, cavity_level=n, transition_scan=transition_scan) for n in range(3)],
        dtype=float,
    )

    assert np.allclose(expected, np.array([0.0, chi, 2.0 * chi]), atol=1.0e-12)
    assert np.isclose(observed[0], expected[0], atol=0.02)
    assert np.isclose(observed[1], expected[1], rtol=0.15, atol=0.02)
    assert np.isclose(observed[2], expected[2], rtol=0.15, atol=0.03)
    assert observed[0] > observed[1] > observed[2]


def test_usage_examples_notebook_uses_transition_detuning_not_raw_carrier():
    notebook = Path("usage_examples.ipynb")
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    content = "\n".join("".join(cell.get("source", [])) for cell in nb["cells"])

    assert "carrier_for_transition_frequency(MHz(detuning_mhz))" in content
    assert content.count("carrier_for_transition_frequency(MHz(detuning_mhz))") >= 3
    assert "predicted_lines_mhz = np.arange(6) * chi_mhz" not in content
    assert "dispersive_model.manifold_transition_frequency(n, frame=dispersive_frame)" in content
    assert "negative `chi` moves the `n`-resolved qubit lines to lower transition detuning" in content
