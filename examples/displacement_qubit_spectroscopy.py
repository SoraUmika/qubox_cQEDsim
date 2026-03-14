from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import carrier_for_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import gaussian_envelope, square_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence


ns = 1.0e-9


def MHz(value: float) -> float:
    return float(2.0 * np.pi * value * 1.0e6)


def angular_to_mhz(omega: float) -> float:
    return float(omega / (2.0 * np.pi * 1.0e6))


def gauss_sigma_018(t_rel: np.ndarray) -> np.ndarray:
    return gaussian_envelope(t_rel, sigma=0.18)


def build_model(chi_mhz: float = -2.84, n_cav: int = 24) -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=MHz(chi_mhz),
        kerr=0.0,
        n_cav=n_cav,
        n_tr=2,
    )


def simulate_post_displacement_state(
    model: DispersiveTransmonCavityModel,
    disp_amp: float,
    disp_duration_s: float,
    dt_s: float,
) -> qt.Qobj:
    disp = Pulse(
        channel="c",
        t0=0.0,
        duration=disp_duration_s,
        envelope=gauss_sigma_018,
        carrier=0.0,
        phase=0.0,
        amp=disp_amp,
    )
    compiled = SequenceCompiler(dt=dt_s).compile([disp], t_end=disp_duration_s + dt_s)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0),
        drive_ops={"c": "cavity"},
        config=SimulationConfig(frame=FrameSpec()),
    )
    return result.final_state


def run_displacement_then_qubit_spectroscopy() -> dict[str, object]:
    chi_mhz = -2.84
    model = build_model(chi_mhz=chi_mhz, n_cav=24)
    q_frame = FrameSpec(omega_q_frame=model.omega_q)

    dt_s = 0.5 * ns
    disp_duration_s = 80.0 * ns
    spec_duration_s = 1200.0 * ns
    gap_s = 20.0 * ns
    disp_amp = 0.035 / ns
    spec_amp = 0.004 / ns
    transition_detunings_mhz = np.linspace(-12.0, 2.0, 71)

    rho_after_disp = simulate_post_displacement_state(
        model=model,
        disp_amp=disp_amp,
        disp_duration_s=disp_duration_s,
        dt_s=dt_s,
    )

    nbar = float(np.real(qt.expect(model.operators()["n_c"], rho_after_disp)))
    rho_cav = qt.ptrace(rho_after_disp, 1)
    if not rho_cav.isoper:
        rho_cav = rho_cav.proj()
    p_n = np.clip(np.real(np.diag(rho_cav.full())), 0.0, 1.0)
    p_n = p_n / max(np.sum(p_n), 1e-12)

    pe = np.zeros_like(transition_detunings_mhz)
    for i, det_mhz in enumerate(transition_detunings_mhz):
        spec = Pulse(
            channel="q",
            t0=disp_duration_s + gap_s,
            duration=spec_duration_s,
            envelope=square_envelope,
            carrier=carrier_for_transition_frequency(MHz(float(det_mhz))),
            phase=0.0,
            amp=spec_amp,
        )
        t_end = disp_duration_s + gap_s + spec_duration_s + dt_s
        compiled = SequenceCompiler(dt=dt_s).compile(
            [
                Pulse(
                    channel="c",
                    t0=0.0,
                    duration=disp_duration_s,
                    envelope=gauss_sigma_018,
                    carrier=0.0,
                    phase=0.0,
                    amp=disp_amp,
                ),
                spec,
            ],
            t_end=t_end,
        )
        # Re-run the full displacement + probe schedule so the idle gap and carrier
        # phase reference match the experiment at each spectroscopy point.
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 0),
            drive_ops={"c": "cavity", "q": "qubit"},
            config=SimulationConfig(frame=q_frame),
        )
        pe[i] = float(np.real(result.expectations["P_e"][-1]))

    peak_idx = int(np.argmax(pe))
    peak_detuning_mhz = float(transition_detunings_mhz[peak_idx])

    top_n = int(min(8, model.n_cav))
    predicted_lines_mhz = [
        angular_to_mhz(model.manifold_transition_frequency(n, frame=q_frame))
        for n in range(top_n)
    ]
    predicted_weights = [float(p_n[n]) for n in range(top_n)]

    output = {
        "chi_mhz": float(chi_mhz),
        "displacement": {
            "amp": float(disp_amp),
            "duration_ns": float(disp_duration_s / ns),
            "nbar_after_displacement": float(nbar),
            "p_n_first_8": [float(x) for x in p_n[:8]],
        },
        "spectroscopy": {
            "amp": float(spec_amp),
            "duration_ns": float(spec_duration_s / ns),
            "gap_ns": float(gap_s / ns),
            "transition_detunings_mhz": [float(x) for x in transition_detunings_mhz],
            "pe_final": [float(x) for x in pe],
            "peak_detuning_mhz": peak_detuning_mhz,
        },
        "predicted_transition_lines_mhz": predicted_lines_mhz,
        "predicted_line_weights": predicted_weights,
        "units": {
            "transition_detuning": "MHz relative to the qubit rotating frame",
            "chi": "MHz",
            "carrier": "internal waveform carrier in rad/s; carrier = -transition_detuning",
            "pulse_amplitude": "rad/s",
        },
    }
    return output


def save_artifacts(result: dict[str, object]) -> tuple[Path, Path]:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "displacement_qubit_spectroscopy_chi_minus_2p84MHz.json"
    fig_path = out_dir / "displacement_qubit_spectroscopy_chi_minus_2p84MHz.png"

    det = np.asarray(result["spectroscopy"]["transition_detunings_mhz"], dtype=float)
    pe = np.asarray(result["spectroscopy"]["pe_final"], dtype=float)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(det, pe, "o-", ms=3.0, lw=1.2, label="simulated $P_e$")

    lines = np.asarray(result["predicted_transition_lines_mhz"], dtype=float)
    weights = np.asarray(result["predicted_line_weights"], dtype=float)
    for ln, wt in zip(lines, weights):
        if wt < 1e-4:
            continue
        ax.axvline(ln, color="tab:red", alpha=min(0.9, 0.2 + 1.5 * wt), lw=1.0)

    ax.set_xlabel("Qubit transition detuning relative to frame (MHz)")
    ax.set_ylabel("Final excited-state probability $P_e$")
    ax.set_title(r"Displacement $\rightarrow$ qubit spectroscopy ($\chi=-2.84$ MHz)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=170)
    plt.close(fig)

    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return json_path, fig_path


def main() -> None:
    result = run_displacement_then_qubit_spectroscopy()
    json_path, fig_path = save_artifacts(result)
    peak_mhz = float(result["spectroscopy"]["peak_detuning_mhz"])
    nbar = float(result["displacement"]["nbar_after_displacement"])
    print(f"chi = {result['chi_mhz']:+.3f} MHz")
    print(f"post-displacement <n> = {nbar:.3f}")
    print(f"spectroscopy peak detuning = {peak_mhz:+.3f} MHz")
    print(f"saved: {json_path}")
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
