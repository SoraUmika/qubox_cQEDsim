from __future__ import annotations

from functools import partial
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import DisplacementGate, build_displacement_pulse
from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.frequencies import carrier_for_transition_frequency
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.pulses.envelopes import gaussian_envelope
from cqed_sim.pulses.pulse import Pulse
from cqed_sim.sequence.scheduler import SequenceCompiler
from cqed_sim.sim.extractors import reduced_cavity_state
from cqed_sim.sim.runner import SimulationConfig, simulate_sequence
from tutorials.tutorial_support import gaussian_selective_spectrum_response


ns = 1.0e-9


def _to_jsonable(value: object) -> object:
    if isinstance(value, complex):
        return {"real": float(np.real(value)), "imag": float(np.imag(value))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def MHz(value: float) -> float:
    return float(2.0 * np.pi * value * 1.0e6)


def angular_to_mhz(omega: float) -> float:
    return float(omega / (2.0 * np.pi * 1.0e6))


def selective_gaussian_envelope(t_rel: np.ndarray, sigma_fraction: float = 0.18) -> np.ndarray:
    return gaussian_envelope(t_rel, sigma=sigma_fraction)


def build_model(chi_mhz: float = -2.84, n_cav: int = 24) -> DispersiveTransmonCavityModel:
    return DispersiveTransmonCavityModel(
        omega_c=MHz(5.0e3),
        omega_q=MHz(6.0e3),
        alpha=0.0,
        chi=MHz(chi_mhz),
        kerr=0.0,
        n_cav=n_cav,
        n_tr=2,
    )


def simulate_post_displacement_state(
    model: DispersiveTransmonCavityModel,
    target_alpha: complex,
    disp_duration_s: float,
    dt_s: float,
    *,
    frame: FrameSpec,
) -> qt.Qobj:
    pulses, drive_ops, _ = build_displacement_pulse(
        DisplacementGate(index=0, name="displace", re=float(np.real(target_alpha)), im=float(np.imag(target_alpha))),
        {"duration_displacement_s": disp_duration_s},
    )
    compiled = SequenceCompiler(dt=dt_s).compile(pulses, t_end=disp_duration_s + dt_s)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0),
        drive_ops=drive_ops,
        config=SimulationConfig(frame=frame, max_step=dt_s),
    )
    return result.final_state


def _extract_resolved_peak_heights(
    scan_mhz: np.ndarray,
    response: np.ndarray,
    line_positions_mhz: np.ndarray,
    *,
    half_window_points: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    peak_detunings = []
    peak_heights = []
    for line_position in np.asarray(line_positions_mhz, dtype=float):
        center_index = int(np.argmin(np.abs(scan_mhz - line_position)))
        lo = max(0, center_index - int(half_window_points))
        hi = min(scan_mhz.size, center_index + int(half_window_points) + 1)
        local_index = lo + int(np.argmax(response[lo:hi]))
        peak_detunings.append(float(scan_mhz[local_index]))
        peak_heights.append(float(response[local_index]))
    return np.asarray(peak_detunings, dtype=float), np.asarray(peak_heights, dtype=float)


def run_displacement_then_qubit_spectroscopy() -> dict[str, object]:
    chi_mhz = -2.84
    model = build_model(chi_mhz=chi_mhz, n_cav=24)
    frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

    dt_s = 2.0 * ns
    target_alpha = 2.0 + 0.0j
    displacement_duration_s = 120.0 * ns
    probe_duration_s = 2.5 * ns * 1.0e3
    probe_gap_s = 40.0 * ns
    probe_amp_rad_s = MHz(0.04)
    probe_sigma_fraction = 0.18
    transition_detunings_mhz = np.linspace(-16.0, 2.0, 181)

    rho_after_disp = simulate_post_displacement_state(
        model=model,
        target_alpha=target_alpha,
        disp_duration_s=displacement_duration_s,
        dt_s=dt_s,
        frame=frame,
    )

    rho_cav = reduced_cavity_state(rho_after_disp)
    p_n = np.array(np.clip(np.real(np.diag(rho_cav.full())), 0.0, 1.0), dtype=float)
    p_n = p_n / max(np.sum(p_n), 1e-12)
    nbar = float(np.real((rho_cav * qt.num(model.n_cav)).tr()))
    cavity_mean = complex((rho_cav * qt.destroy(model.n_cav)).tr())
    displacement_pulses, displacement_drive_ops, displacement_meta = build_displacement_pulse(
        DisplacementGate(index=0, name="displace", re=float(np.real(target_alpha)), im=float(np.imag(target_alpha))),
        {"duration_displacement_s": displacement_duration_s},
    )
    probe_envelope = partial(selective_gaussian_envelope, sigma_fraction=probe_sigma_fraction)

    pe = np.zeros_like(transition_detunings_mhz)
    for i, det_mhz in enumerate(transition_detunings_mhz):
        spec = Pulse(
            channel="qubit",
            t0=displacement_duration_s + probe_gap_s,
            duration=probe_duration_s,
            envelope=probe_envelope,
            carrier=carrier_for_transition_frequency(MHz(float(det_mhz))),
            phase=0.0,
            amp=probe_amp_rad_s,
        )
        t_end = displacement_duration_s + probe_gap_s + probe_duration_s + dt_s
        compiled = SequenceCompiler(dt=dt_s).compile(
            displacement_pulses + [spec],
            t_end=t_end,
        )
        result = simulate_sequence(
            model,
            compiled,
            model.basis_state(0, 0),
            drive_ops={**displacement_drive_ops, "qubit": "qubit"},
            config=SimulationConfig(frame=frame, max_step=dt_s),
        )
        pe[i] = float(np.real(result.expectations["P_e"][-1]))

    top_n = int(min(8, model.n_cav))
    predicted_lines_mhz = np.asarray([
        angular_to_mhz(model.manifold_transition_frequency(n, frame=frame))
        for n in range(top_n)
    ], dtype=float)
    predicted_weights = np.asarray([float(p_n[n]) for n in range(top_n)], dtype=float)
    peak_detunings_mhz, peak_heights = _extract_resolved_peak_heights(
        transition_detunings_mhz,
        pe,
        predicted_lines_mhz,
    )
    selected_peak_weights = peak_heights / max(float(np.sum(peak_heights)), 1.0e-15)
    selected_photon_weights = predicted_weights / max(float(np.sum(predicted_weights)), 1.0e-15)

    detunings_rad_s = 2.0 * np.pi * 1.0e6 * transition_detunings_mhz
    line_centers_rad_s = 2.0 * np.pi * 1.0e6 * predicted_lines_mhz
    sigma_time_s = probe_duration_s * probe_sigma_fraction
    theory_unscaled = gaussian_selective_spectrum_response(
        detunings_rad_s,
        line_centers_rad_s,
        predicted_weights,
        sigma_time_s,
    )
    theory_scale = float(np.dot(theory_unscaled, pe) / max(np.dot(theory_unscaled, theory_unscaled), 1.0e-18))
    theory_response = gaussian_selective_spectrum_response(
        detunings_rad_s,
        line_centers_rad_s,
        predicted_weights,
        sigma_time_s,
        scale=theory_scale,
    )
    theory_rms = float(np.sqrt(np.mean((pe - theory_response) ** 2)))
    peak_weight_correlation = float(np.corrcoef(selected_photon_weights[:6], selected_peak_weights[:6])[0, 1])

    output = {
        "chi_mhz": float(chi_mhz),
        "displacement": {
            "target_alpha_real": float(np.real(target_alpha)),
            "target_alpha_imag": float(np.imag(target_alpha)),
            "duration_ns": float(displacement_duration_s / ns),
            "final_alpha_real": float(np.real(cavity_mean)),
            "final_alpha_imag": float(np.imag(cavity_mean)),
            "nbar_after_displacement": float(nbar),
            "p_n_first_8": [float(x) for x in p_n[:8]],
            "builder_meta": _to_jsonable(displacement_meta),
        },
        "spectroscopy": {
            "amp_rad_s": float(probe_amp_rad_s),
            "duration_ns": float(probe_duration_s / ns),
            "gap_ns": float(probe_gap_s / ns),
            "sigma_fraction": float(probe_sigma_fraction),
            "transition_detunings_mhz": [float(x) for x in transition_detunings_mhz],
            "pe_final": [float(x) for x in pe],
            "theory_response": [float(x) for x in theory_response],
            "theory_rms": theory_rms,
        },
        "predicted_transition_lines_mhz": [float(x) for x in predicted_lines_mhz],
        "predicted_line_weights": [float(x) for x in predicted_weights],
        "extracted_peak_detunings_mhz": [float(x) for x in peak_detunings_mhz],
        "extracted_peak_heights": [float(x) for x in peak_heights],
        "normalized_peak_weights": [float(x) for x in selected_peak_weights],
        "peak_weight_correlation": peak_weight_correlation,
        "units": {
            "transition_detuning": "MHz relative to the qubit rotating frame",
            "chi": "MHz",
            "carrier": "internal waveform carrier in rad/s; carrier = -transition_detuning",
            "pulse_amplitude": "rad/s",
            "selective_theory": "weak-drive Gaussian spectrum sum scale * sum_n p_n exp(-(sigma_t * (Delta - Delta_n))^2)",
        },
    }
    return output


def save_artifacts(result: dict[str, object]) -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = repo_root / "documentations" / "assets" / "images" / "tutorials"
    docs_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "displacement_qubit_spectroscopy_chi_minus_2p84MHz.json"
    fig_path = out_dir / "displacement_qubit_spectroscopy_chi_minus_2p84MHz.png"
    docs_fig_path = docs_dir / "displacement_spectroscopy.png"

    det = np.asarray(result["spectroscopy"]["transition_detunings_mhz"], dtype=float)
    pe = np.asarray(result["spectroscopy"]["pe_final"], dtype=float)
    theory = np.asarray(result["spectroscopy"]["theory_response"], dtype=float)
    lines = np.asarray(result["predicted_transition_lines_mhz"], dtype=float)
    weights = np.asarray(result["predicted_line_weights"], dtype=float)
    extracted = np.asarray(result["normalized_peak_weights"], dtype=float)
    alpha_real = float(result["displacement"]["target_alpha_real"])
    alpha_imag = float(result["displacement"]["target_alpha_imag"])
    n_levels = np.arange(weights.size, dtype=int)

    fig, (ax_spectrum, ax_weights) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    ax_spectrum.plot(det, pe, color="#1f77b4", lw=1.6, label="pulse-level simulation")
    ax_spectrum.plot(det, theory, "--", color="black", lw=1.1, alpha=0.9, label="weak-drive Gaussian theory")
    for line, weight in zip(lines, weights, strict=True):
        if weight < 1.0e-4:
            continue
        ax_spectrum.axvline(line, color="#d62728", alpha=min(0.95, 0.2 + 2.0 * float(weight)), lw=0.9)
    ax_spectrum.set_xlabel("Qubit drive detuning relative to frame (MHz)")
    ax_spectrum.set_ylabel("Final excited-state probability $P_e$")
    ax_spectrum.set_title(rf"Selective Gaussian number splitting after $D(\alpha={alpha_real:+.1f}{alpha_imag:+.1f}i)$")
    ax_spectrum.grid(alpha=0.25)
    ax_spectrum.legend(loc="upper left")

    width = 0.38
    ax_weights.bar(n_levels - width / 2.0, weights, width=width, color="#72B7B2", label="displaced-state Fock weight")
    ax_weights.bar(n_levels + width / 2.0, extracted, width=width, color="#F58518", label="normalized peak height")
    ax_weights.set_xlabel("Photon number manifold $n$")
    ax_weights.set_ylabel("Normalized weight")
    ax_weights.set_title("Resolved peak amplitudes recover cavity populations")
    ax_weights.set_xticks(n_levels)
    ax_weights.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(fig_path, dpi=170)
    fig.savefig(docs_fig_path, dpi=170)
    plt.close(fig)

    json_path.write_text(json.dumps(_to_jsonable(result), indent=2), encoding="utf-8")
    return json_path, fig_path


def main() -> None:
    result = run_displacement_then_qubit_spectroscopy()
    json_path, fig_path = save_artifacts(result)
    nbar = float(result["displacement"]["nbar_after_displacement"])
    peak_correlation = float(result["peak_weight_correlation"])
    print(f"chi = {result['chi_mhz']:+.3f} MHz")
    print(f"post-displacement <n> = {nbar:.3f}")
    print(f"final cavity mean field = {result['displacement']['final_alpha_real']:+.3f}{result['displacement']['final_alpha_imag']:+.3f}i")
    print(f"peak-weight correlation = {peak_correlation:.6f}")
    print(f"theory RMS = {result['spectroscopy']['theory_rms']:.3e}")
    print(f"saved: {json_path}")
    print(f"saved: {fig_path}")


if __name__ == "__main__":
    main()
