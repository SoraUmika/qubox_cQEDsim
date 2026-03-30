from __future__ import annotations

import json
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "tutorials"

BOOTSTRAP = """
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = next(
    (
        candidate
        for candidate in (Path.cwd(), *Path.cwd().parents)
        if (candidate / "pyproject.toml").exists() and (candidate / "cqed_sim").is_dir()
    ),
    None,
)
if REPO_ROOT is None:
    raise RuntimeError("Could not resolve the repository root from the notebook working directory.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from tutorials.workflow_tutorial_support import configure_notebook_style

configure_notebook_style()
"""


def normalize(text: str) -> list[str]:
    text = textwrap.dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": normalize(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": normalize(text),
    }


def write_notebook(path: Path, cells: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")


def workflow_notebook(
    *,
    title: str,
    intro: str,
    imports_code: str,
    model_code: str,
    pulse_code: str,
    simulation_code: str,
    analysis_code: str,
    interpretation: str,
    next_steps: str,
) -> list[dict]:
    return [
        md(f"# {title}\n\n{intro}"),
        md("## Imports"),
        code(BOOTSTRAP + "\n" + textwrap.dedent(imports_code).strip("\n")),
        md("## Physics / model definition"),
        code(model_code),
        md("## Pulse / sequence construction"),
        code(pulse_code),
        md("## Simulation"),
        code(simulation_code),
        md("## Analysis / visualization"),
        code(analysis_code),
        md("## Interpretation"),
        md(interpretation),
        md("## Variations / exercises"),
        md(next_steps),
    ]


def build_protocol_style_simulation() -> None:
    path = TUTORIALS / "00_getting_started" / "01_protocol_style_simulation.ipynb"
    cells = workflow_notebook(
        title="Protocol-Style End-to-End Simulation",
        intro="""
This notebook rewrites `examples/protocol_style_simulation.py` as a first practical `cqed_sim` workflow. We model a qubit-storage device prepared in `|g,0>`, apply a calibrated `x90` pulse, compile the schedule, simulate the dynamics, and then sample a qubit measurement.

The physical goal is simple but realistic: show how the public model, pulse-builder, compiler, solver, and measurement APIs fit together in one top-to-bottom workflow. Because the frame is matched to the bare mode frequencies, the pulse area controls the qubit rotation and the expected outcome is a final excited-state probability near `0.5`.
""",
        imports_code="""
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    QubitMeasurementSpec,
    RotationGate,
    SequenceCompiler,
    SimulationConfig,
    StatePreparationSpec,
    build_rotation_pulse,
    fock_state,
    measure_qubit,
    prepare_state,
    qubit_state,
    simulate_sequence,
    storage_photon_number,
    transmon_level_populations,
)
from tutorials.tutorial_support import GHz, MHz, ns
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-2.5),
    kerr=MHz(-0.002),
    n_cav=4,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

initial_state = prepare_state(
    model,
    StatePreparationSpec(
        qubit=qubit_state("g"),
        storage=fock_state(0),
    ),
)

gate = RotationGate(index=0, name="x90", theta=np.pi / 2.0, phi=0.0)
""",
        pulse_code="""
pulses, drive_ops, pulse_meta = build_rotation_pulse(
    gate,
    {
        "duration_rotation_s": 64.0 * ns,
        "rotation_sigma_fraction": 0.18,
    },
)

compiled = SequenceCompiler(dt=1.0 * ns).compile(pulses, t_end=70.0 * ns)
print("Pulse metadata:", pulse_meta)
print("Compiled samples:", compiled.tlist.size)
""",
        simulation_code="""
result = simulate_sequence(
    model,
    compiled,
    initial_state,
    drive_ops,
    config=SimulationConfig(frame=frame, store_states=True),
)
measurement = measure_qubit(result.final_state, QubitMeasurementSpec(shots=2048, seed=7))

exact_probabilities = measurement.probabilities
sampled_probabilities = {
    key: value / 2048.0
    for key, value in measurement.counts.items()
}
transmon_populations = transmon_level_populations(result.final_state)
storage_n = storage_photon_number(result.final_state)

print("Exact qubit probabilities:", exact_probabilities)
print("Sampled qubit counts:", measurement.counts)
print("Final transmon populations:", transmon_populations)
print("Final storage <n>:", storage_n)
""",
        analysis_code="""
labels = ["g", "e"]
exact = [exact_probabilities[label] for label in labels]
sampled = [sampled_probabilities[label] for label in labels]

fig, ax = plt.subplots(figsize=(6.2, 3.8))
x = np.arange(len(labels), dtype=float)
width = 0.35
ax.bar(x - width / 2.0, exact, width=width, label="exact probability", color="#4C78A8")
ax.bar(x + width / 2.0, sampled, width=width, label="sampled frequency", color="#F58518")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("Probability")
ax.set_title("End-to-end x90 workflow")
ax.legend(loc="upper right")
plt.show()
""",
        interpretation="""
The qubit ends near an equal superposition because the Gaussian pulse builder calibrates the pulse area for a `pi/2` rotation. The cavity remains in vacuum because we never drive the storage channel.

This is the smallest realistic `cqed_sim` workflow to keep in mind when you adapt the library for your own experiments:

`model -> frame -> initial state -> pulse builder -> compiler -> simulate_sequence -> measure_qubit`
""",
        next_steps="""
- Change `gate.theta` to `np.pi` and confirm that the final qubit population moves close to `P_e = 1`.
- Replace the builder with a manually defined `Pulse` if you want explicit control over the envelope.
- Add cavity displacement before the qubit pulse to turn this into a cavity-conditioned protocol.
""",
    )
    write_notebook(path, cells)


def build_displacement_then_spectroscopy() -> None:
    path = TUTORIALS / "10_core_workflows" / "01_displacement_then_qubit_spectroscopy.ipynb"
    cells = workflow_notebook(
        title="Displacement Then Qubit Spectroscopy",
        intro="""
This notebook rewrites `examples/displacement_qubit_spectroscopy.py` as a guided number-splitting tutorial. We first prepare a calibrated cavity displacement with target $\\alpha = 2$, then apply a long weak Gaussian qubit probe and read out the final excited-state population.

The physical signature is photon-number-dependent qubit spectroscopy: with dispersive `chi < 0`, each extra cavity photon shifts the qubit transition to lower transition detuning in the matched rotating frame. When the probe is selective enough, the resolved peak heights directly track the displaced cavity's Fock-state weights.
""",
        imports_code="""
from functools import partial

from cqed_sim import (
    DisplacementGate,
    DispersiveTransmonCavityModel,
    drive_frequency_for_transition_frequency,
    FrameSpec,
    internal_carrier_from_drive_frequency,
    Pulse,
    SequenceCompiler,
    SimulationConfig,
    build_displacement_pulse,
    reduced_cavity_state,
    simulate_sequence,
)
from cqed_sim.pulses.envelopes import gaussian_envelope
from tutorials.tutorial_support import (
    GHz,
    MHz,
    angular_to_mhz,
    gaussian_selective_spectrum_response,
    ns,
    us,
)
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-2.84),
    kerr=MHz(-0.002),
    n_cav=18,
    n_tr=2,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)

target_alpha = 2.0 + 0.0j
probe_duration_s = 2.5 * us
probe_gap_s = 40.0 * ns
probe_amp_rad_s = MHz(0.04)
probe_sigma_fraction = 0.18
transition_scan_mhz = np.linspace(-16.0, 2.0, 181)
fock_levels = np.arange(8, dtype=int)
predicted_lines_mhz = np.asarray(
    [angular_to_mhz(model.manifold_transition_frequency(int(n), frame=frame)) for n in fock_levels],
    dtype=float,
)
""",
        pulse_code="""
displacement_pulses, displacement_drive_ops, displacement_meta = build_displacement_pulse(
    DisplacementGate(index=0, name="displace_alpha_2", re=float(np.real(target_alpha)), im=float(np.imag(target_alpha))),
    {"duration_displacement_s": 120.0 * ns},
)
displacement_duration_s = float(displacement_meta["duration_s"])
dt_s = 2.0 * ns
probe_envelope = partial(gaussian_envelope, sigma=probe_sigma_fraction)

def final_excited_population_for_detuning(detuning_mhz: float) -> float:
    transition_frequency = MHz(detuning_mhz)
    drive_frequency = drive_frequency_for_transition_frequency(transition_frequency, frame.omega_q_frame)
    probe = Pulse(
        channel="qubit",
        t0=displacement_duration_s + probe_gap_s,
        duration=probe_duration_s,
        envelope=probe_envelope,
        amp=probe_amp_rad_s,
        carrier=internal_carrier_from_drive_frequency(drive_frequency, frame.omega_q_frame),
        label=f"probe_{detuning_mhz:+.2f}_MHz",
    )
    compiled = SequenceCompiler(dt=dt_s).compile(
        displacement_pulses + [probe],
        t_end=displacement_duration_s + probe_gap_s + probe_duration_s + dt_s,
    )
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(0, 0),
        {**displacement_drive_ops, "qubit": "qubit"},
        config=SimulationConfig(frame=frame, max_step=dt_s),
    )
    return float(np.real(result.expectations["P_e"][-1]))

print("Displacement metadata:", displacement_meta)
""",
        simulation_code="""
displacement_compiled = SequenceCompiler(dt=dt_s).compile(
    displacement_pulses,
    t_end=displacement_duration_s + dt_s,
)

displacement_only = simulate_sequence(
    model,
    displacement_compiled,
    model.basis_state(0, 0),
    displacement_drive_ops,
    config=SimulationConfig(frame=frame, max_step=dt_s),
)

rho_storage = reduced_cavity_state(displacement_only.final_state)

photon_weights = np.clip(np.real(np.diag(rho_storage.full())), 0.0, 1.0)
photon_weights = photon_weights / max(np.sum(photon_weights), 1.0e-12)
nbar = float(np.real((rho_storage * qt.num(model.n_cav)).tr()))
cavity_mean = complex((rho_storage * qt.destroy(model.n_cav)).tr())

spectroscopy_response = np.asarray(
    [final_excited_population_for_detuning(point_mhz) for point_mhz in transition_scan_mhz],
    dtype=float,
)

def peak_height_near(line_mhz: float, half_window_points: int = 4) -> float:
    center_index = int(np.argmin(np.abs(transition_scan_mhz - line_mhz)))
    lo = max(0, center_index - int(half_window_points))
    hi = min(transition_scan_mhz.size, center_index + int(half_window_points) + 1)
    return float(np.max(spectroscopy_response[lo:hi]))

selected_weights = photon_weights[: fock_levels.size]
peak_heights = np.asarray([peak_height_near(line_mhz) for line_mhz in predicted_lines_mhz], dtype=float)
normalized_peak_weights = peak_heights / max(float(np.sum(peak_heights)), 1.0e-15)
normalized_photon_weights = selected_weights / max(float(np.sum(selected_weights)), 1.0e-15)

sigma_time_s = probe_duration_s * probe_sigma_fraction
theory_unscaled = gaussian_selective_spectrum_response(
    2.0 * np.pi * 1.0e6 * transition_scan_mhz,
    2.0 * np.pi * 1.0e6 * predicted_lines_mhz,
    selected_weights,
    sigma_time_s,
)
theory_scale = float(
    np.dot(theory_unscaled, spectroscopy_response)
    / max(np.dot(theory_unscaled, theory_unscaled), 1.0e-18)
)
theory_response = gaussian_selective_spectrum_response(
    2.0 * np.pi * 1.0e6 * transition_scan_mhz,
    2.0 * np.pi * 1.0e6 * predicted_lines_mhz,
    selected_weights,
    sigma_time_s,
    scale=theory_scale,
)

print("Post-displacement <n>:", nbar)
print("Post-displacement <a>:", cavity_mean)
print("Predicted n-resolved qubit lines [MHz]:", predicted_lines_mhz)
""",
        analysis_code="""
fig, (ax_spec, ax_weights) = plt.subplots(1, 2, figsize=(11.0, 4.0))

ax_spec.plot(transition_scan_mhz, spectroscopy_response, color="#4C78A8", lw=1.5, label="pulse-level simulation")
ax_spec.plot(transition_scan_mhz, theory_response, "--", color="black", lw=1.1, label="weak-drive Gaussian theory")
for line_mhz, weight in zip(predicted_lines_mhz, selected_weights, strict=True):
    ax_spec.axvline(
        line_mhz,
        color="#E45756",
        alpha=min(0.95, 0.25 + 2.0 * float(weight)),
        lw=1.1,
    )
ax_spec.set_xlabel("Qubit transition detuning relative to frame (MHz)")
ax_spec.set_ylabel("Final excited-state probability $P_e$")
ax_spec.set_title("Selective Gaussian number splitting after calibrated displacement")
ax_spec.legend(loc="upper left")

width = 0.38
ax_weights.bar(fock_levels - width / 2.0, normalized_photon_weights, width=width, color="#72B7B2", label="displaced-state Fock weight")
ax_weights.bar(fock_levels + width / 2.0, normalized_peak_weights, width=width, color="#F58518", label="normalized peak height")
ax_weights.set_xlabel("Storage Fock level $n$")
ax_weights.set_ylabel("Normalized weight")
ax_weights.set_title(rf"Resolved peaks recover cavity populations, $\\langle n \\rangle = {nbar:.2f}$")
ax_weights.legend(loc="upper right")

plt.tight_layout()
plt.show()
""",
        interpretation="""
The cavity displacement prepares a coherent-state-like Fock distribution, so the spectroscopy curve is not a single broad feature. Each photon number manifold contributes its own line, separated by `chi`, and the long weak Gaussian probe narrows those lines enough to resolve them cleanly.

Because this repository defines `chi` as the per-photon qubit-transition pull, negative `chi` moves the `n`-resolved qubit lines to lower transition detuning. In the weak-drive limit the line envelope is accurately modeled by a weighted Gaussian sum, and the resolved peak heights recover the displaced-state photon weights.
""",
        next_steps="""
- Replace the Gaussian envelope with a shorter square probe and compare how the peaks merge together.
- Increase the displacement amplitude to make higher-`n` manifolds visible.
- Compare the extracted peak heights with the ideal coherent-state Poisson weights for the same target `alpha`.
""",
    )
    write_notebook(path, cells)


def build_kerr_free_evolution() -> None:
    path = TUTORIALS / "10_core_workflows" / "02_kerr_free_evolution.ipynb"
    cells = workflow_notebook(
        title="Kerr Free Evolution of a Storage Coherent State",
        intro="""
This notebook rewrites `examples/kerr_free_evolution.py` as a workflow tutorial for storage self-Kerr. We prepare a coherent cavity state with target $\\alpha = 2$ and let it evolve under the static Hamiltonian with no applied drive.

The physical result is nonlinear phase winding: the cavity mean field rotates, but more importantly the Wigner function bends and eventually departs from a Gaussian coherent-state shape. The notebook stores Wigner snapshots directly in coherent-state $\\alpha$ coordinates so the $t = 0$ peak stays centered at the prepared displacement.
""",
        imports_code="""
from cqed_sim import coherent_state
from examples.workflows.kerr_free_evolution import (
    plot_kerr_wigner_snapshots,
    run_kerr_free_evolution,
    times_us_to_seconds,
)
""",
        model_code="""
    times_us = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0], dtype=float)
    wigner_times_us = np.array([0.0, 125.0, 250.0, 375.0], dtype=float)
    initial_cavity_state = coherent_state(2.0)
""",
        pulse_code="""
times_s = times_us_to_seconds(times_us)
wigner_times_s = times_us_to_seconds(wigner_times_us)
print("This protocol uses no external drive pulses: the cavity simply evolves under the static self-Kerr Hamiltonian.")
""",
        simulation_code="""
result = run_kerr_free_evolution(
    times_s,
    cavity_state=initial_cavity_state,
    parameter_set="phase_evolution",
    n_cav=30,
    wigner_times_s=wigner_times_s,
    wigner_n_points=91,
    wigner_extent=4.6,
    wigner_coordinate="alpha",
)

mean_field = np.asarray([snapshot.cavity_mean for snapshot in result.snapshots], dtype=np.complex128)
phase_rad = np.unwrap(np.angle(mean_field))
photon_number = np.asarray([snapshot.cavity_photon_number for snapshot in result.snapshots], dtype=float)

t0_snapshot = next(snapshot for snapshot in result.snapshots if snapshot.wigner is not None and np.isclose(snapshot.time_us, 0.0))
t0_wigner = t0_snapshot.wigner["w"]
t0_index = np.unravel_index(np.argmax(t0_wigner), t0_wigner.shape)
t0_peak_alpha = (
    float(t0_snapshot.wigner["xvec"][t0_index[1]]),
    float(t0_snapshot.wigner["yvec"][t0_index[0]]),
)

print("Parameter set:", result.parameter_set.name)
print("Kerr coefficient [Hz]:", result.parameter_set.kerr_hz)
print("t = 0 Wigner peak in alpha coordinates:", t0_peak_alpha)
""",
        analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
axes[0].plot(times_us, phase_rad, "o-", color="#4C78A8")
axes[0].set_xlabel("Free-evolution time (us)")
axes[0].set_ylabel("Unwrapped phase of <a> (rad)")
axes[0].set_title("Mean-field phase evolution")

axes[1].plot(times_us, photon_number, "o-", color="#54A24B")
axes[1].set_xlabel("Free-evolution time (us)")
axes[1].set_ylabel("Storage photon number")
axes[1].set_title("Photon number is approximately conserved")
plt.tight_layout()
plt.show()

fig = plot_kerr_wigner_snapshots(result, max_cols=4, show_colorbar=True, coordinate="alpha")
for axis, label in zip(
    fig.axes[:4],
    [r"$t = 0$", r"$t = T_K/4$", r"$t = T_K/2$", r"$t = 3T_K/4$"],
    strict=True,
):
    axis.set_title(label)
plt.show()
""",
        interpretation="""
The photon number stays nearly constant because this is unitary free evolution, not ringdown. What changes is the phase picked up by each Fock component, so the coherent state shears in phase space.

That shearing direction is controlled by the sign of the Kerr term documented in `physics_and_conventions/physics_conventions_report.tex`. The notebook also makes the phase-space convention explicit: plotting in coherent-state `alpha` coordinates keeps the initial Wigner peak centered near `(2, 0)`, whereas quadrature coordinates would place the same state near $(\\sqrt{2}\\,\\alpha, 0)$ instead.
""",
        next_steps="""
- Switch `coordinate="quadrature"` in the plotting helper and verify the expected $\\sqrt{2}$ rescaling.
- Change the initial coherent-state amplitude to see how larger `n` content accelerates non-Gaussian distortion.
- Add cavity loss with `NoiseSpec(kappa=...)` if you want to compare Kerr distortion and ringdown.
""",
    )
    write_notebook(path, cells)


def build_phase_space_coordinates_and_wigner_conventions() -> None:
    path = TUTORIALS / "10_core_workflows" / "03_phase_space_coordinates_and_wigner_conventions.ipynb"
    cells = workflow_notebook(
    title="Phase-Space Coordinates and Wigner Conventions",
    intro="""
This companion notebook explains a common source of confusion in bosonic tutorials: the same cavity state can be plotted either in quadrature coordinates `(x, p)` or in coherent-state coordinates `alpha`.

For a coherent state centered at $\\alpha = 2$, the Wigner peak sits near $x = \\sqrt{2} \\alpha \\approx 2.83$ in quadrature coordinates, but near $\\alpha = 2$ when the plotting routine is asked to use coherent-state coordinates directly. The underlying state is the same; only the axis convention changes.
""",
    imports_code="""
from cqed_sim import coherent_state, cavity_wigner
""",
    model_code="""
alpha = 2.0 + 0.0j
rho_cavity = coherent_state(alpha).proj()
expected_quadrature_center = np.sqrt(2.0) * np.real(alpha)
expected_alpha_center = np.real(alpha)
""",
    pulse_code="""
print("No control pulse is needed here: we are only changing the plotting convention for the same prepared coherent state.")
""",
    simulation_code="""
x_quad, y_quad, w_quad = cavity_wigner(rho_cavity, n_points=121, extent=4.6, coordinate="quadrature")
x_alpha, y_alpha, w_alpha = cavity_wigner(rho_cavity, n_points=121, extent=4.6, coordinate="alpha")

quad_index = np.unravel_index(np.argmax(w_quad), w_quad.shape)
alpha_index = np.unravel_index(np.argmax(w_alpha), w_alpha.shape)
quadrature_peak = (float(x_quad[quad_index[1]]), float(y_quad[quad_index[0]]))
alpha_peak = (float(x_alpha[alpha_index[1]]), float(y_alpha[alpha_index[0]]))

print("Expected quadrature-space center:", expected_quadrature_center)
print("Numerical quadrature-space peak:", quadrature_peak)
print("Expected alpha-space center:", expected_alpha_center)
print("Numerical alpha-space peak:", alpha_peak)
""",
    analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
images = []
for axis, xvec, yvec, values, title, xlabel, ylabel, xmark in [
    (axes[0], x_quad, y_quad, w_quad, "Quadrature coordinates", "x", "p", expected_quadrature_center),
    (axes[1], x_alpha, y_alpha, w_alpha, "Coherent-state coordinates", r"Re($\\alpha$)", r"Im($\\alpha$)", expected_alpha_center),
]:
    image = axis.imshow(
    values,
    origin="lower",
    extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
    cmap="RdBu_r",
    aspect="equal",
    )
    images.append(image)
    axis.axvline(xmark, color="#E45756", ls="--", lw=1.0)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)

fig.colorbar(images[-1], ax=axes.tolist(), shrink=0.86, label="Wigner value")
plt.tight_layout()
plt.show()
""",
    interpretation="""
This notebook is a sanity check on plotting conventions, not a different physical preparation. The two panels differ only by the phase-space coordinate system used by the plotting routine.

Use `coordinate="alpha"` when you want the axes to match coherent-state amplitudes directly. Use `coordinate="quadrature"` when you want canonical oscillator axes. Both are valid, but the interpretation of the peak position changes by a factor of $\\sqrt{2}$.
""",
    next_steps="""
- Repeat the comparison for a complex displacement such as `alpha = 1.5 + 0.8j`.
- Overlay the expected centers by hand and confirm the numerical peak follows the same rule.
- Use this notebook as a quick sanity check whenever a bosonic tutorial plot looks shifted.
""",
    )
    write_notebook(path, cells)


def build_selective_gaussian_number_splitting() -> None:
    path = TUTORIALS / "10_core_workflows" / "04_selective_gaussian_number_splitting.ipynb"
    cells = workflow_notebook(
    title="Selective Gaussian Number Splitting",
    intro="""
This companion notebook focuses on interpretation rather than pulse construction. It reuses the calibrated displacement-plus-spectroscopy workflow and asks a narrower question: when do the resolved peak heights become faithful proxies for cavity photon-number weights?

The answer in this regime is: when the qubit probe is weak and spectrally selective enough that each dispersive manifold produces its own approximately Gaussian line. Then the measured peak amplitudes track the displaced-state Fock distribution.
""",
    imports_code="""
from examples.displacement_qubit_spectroscopy import run_displacement_then_qubit_spectroscopy
from tutorials.tutorial_support import coherent_state_fock_probabilities
""",
    model_code="""
result = run_displacement_then_qubit_spectroscopy()
alpha = complex(
    result["displacement"]["target_alpha_real"],
    result["displacement"]["target_alpha_imag"],
)
predicted_lines_mhz = np.asarray(result["predicted_transition_lines_mhz"], dtype=float)
predicted_weights = np.asarray(result["predicted_line_weights"], dtype=float)
normalized_peak_weights = np.asarray(result["normalized_peak_weights"], dtype=float)
ideal_poisson = coherent_state_fock_probabilities(alpha, predicted_weights.size)
ideal_poisson = ideal_poisson / max(float(np.sum(ideal_poisson)), 1.0e-15)
""",
    pulse_code="""
print("This notebook reuses the calibrated displacement and selective Gaussian probe from the workflow example so that the interpretation stays aligned with the published docs image.")
print("Target alpha:", alpha)
print("Peak-weight correlation:", result["peak_weight_correlation"])
""",
    simulation_code="""
transition_scan_mhz = np.asarray(result["spectroscopy"]["transition_detunings_mhz"], dtype=float)
spectroscopy_response = np.asarray(result["spectroscopy"]["pe_final"], dtype=float)
theory_response = np.asarray(result["spectroscopy"]["theory_response"], dtype=float)
nbar = float(result["displacement"]["nbar_after_displacement"])
""",
    analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

axes[0].plot(transition_scan_mhz, spectroscopy_response, color="#4C78A8", lw=1.5, label="pulse-level simulation")
axes[0].plot(transition_scan_mhz, theory_response, "--", color="black", lw=1.1, label="weak-drive Gaussian theory")
for line_mhz in predicted_lines_mhz:
    axes[0].axvline(line_mhz, color="#E45756", alpha=0.35, lw=0.9)
axes[0].set_xlabel("Qubit transition detuning relative to frame (MHz)")
axes[0].set_ylabel("Final excited-state probability $P_e$")
axes[0].set_title("Resolved manifolds under a selective Gaussian probe")
axes[0].legend(loc="upper left")

levels = np.arange(predicted_weights.size, dtype=int)
width = 0.25
axes[1].bar(levels - width, predicted_weights, width=width, color="#72B7B2", label="simulated displaced-state weight")
axes[1].bar(levels, normalized_peak_weights, width=width, color="#F58518", label="normalized peak height")
axes[1].bar(levels + width, ideal_poisson, width=width, color="#54A24B", label="ideal coherent-state Poisson weight")
axes[1].set_xlabel("Photon number manifold $n$")
axes[1].set_ylabel("Normalized weight")
axes[1].set_title(rf"Weight recovery for $D(\\alpha={np.real(alpha):+.1f}{np.imag(alpha):+.1f}i)$ with $\\langle n \\rangle={nbar:.2f}$")
axes[1].set_xticks(levels)
axes[1].legend(loc="upper right")

plt.tight_layout()
plt.show()
""",
    interpretation="""
The simulated displaced-state weights and the resolved peak heights agree closely because the probe is long, weak, and Gaussian. In that limit the qubit is effectively sampling one dispersive manifold at a time.

The ideal coherent-state Poisson bars are a useful reference, but the physically relevant comparison is the simulation's actual post-displacement cavity state. Small differences between the two come from finite truncation and pulse-level preparation details.
""",
    next_steps="""
- Shorten the probe duration and watch the peaks broaden into one another.
- Increase the probe amplitude and check when the weak-drive Gaussian theory starts to fail.
- Change the target displacement amplitude and confirm that the recovered weights move with the expected Poisson mean $|\\alpha|^2$.
""",
    )
    write_notebook(path, cells)


def build_sideband_swap() -> None:
    path = TUTORIALS / "20_bosonic_and_sideband" / "01_sideband_swap.ipynb"
    cells = workflow_notebook(
        title="Red Sideband Swap Between `|f,0>` and `|g,1>`",
        intro="""
This notebook rewrites `examples/sideband_swap_demo.py` as a tutorial for the effective sideband interface. We use a three-level transmon, a storage mode, and a red sideband drive that swaps population between `|f,0>` and `|g,1>`.

The physical objective is the canonical bosonic-transfer primitive used in many cQED protocols: move one excitation from the transmon ladder into the cavity while staying within an effective rotating-wave sideband model.
""",
        imports_code="""
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    build_sideband_pulse,
    carrier_for_transition_frequency,
    simulate_sequence,
)
from tutorials.tutorial_support import GHz, MHz, ns
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-0.8),
    kerr=0.0,
    n_cav=4,
    n_tr=3,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
g_sb = MHz(8.0)
t_swap = np.pi / (2.0 * g_sb)

initial_state = model.basis_state(2, 0)
target_state = model.basis_state(0, 1)
""",
        pulse_code="""
omega_sb = model.sideband_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=2,
    sideband="red",
    frame=frame,
)
pulses, drive_ops, pulse_meta = build_sideband_pulse(
    SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2, sideband="red"),
    duration_s=t_swap,
    amplitude_rad_s=g_sb,
    channel="sb",
    carrier=carrier_for_transition_frequency(omega_sb),
    label="gf_red_sideband",
)
compiled = SequenceCompiler(dt=t_swap / 300.0).compile(pulses, t_end=t_swap)
print("Pulse metadata:", pulse_meta)
""",
        simulation_code="""
result = simulate_sequence(
    model,
    compiled,
    initial_state,
    drive_ops,
    config=SimulationConfig(frame=frame, store_states=True),
)

times_ns = np.asarray(compiled.tlist, dtype=float) / ns
p_g1 = np.asarray([abs(target_state.overlap(state)) ** 2 for state in result.states], dtype=float)
p_f0 = np.asarray([abs(initial_state.overlap(state)) ** 2 for state in result.states], dtype=float)

print("Final |g,1> population:", float(p_g1[-1]))
print("Final |f,0> population:", float(p_f0[-1]))
""",
        analysis_code="""
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(times_ns, p_g1, label=r"$P_{|g,1\\rangle}$", lw=2.0, color="#4C78A8")
ax.plot(times_ns, p_f0, label=r"$P_{|f,0\\rangle}$", lw=1.6, color="#E45756")
ax.axvline(t_swap / ns, color="black", ls="--", lw=1.0, label=r"$t_{\\mathrm{swap}}$")
ax.set_xlabel("Time (ns)")
ax.set_ylabel("Population")
ax.set_title("Red sideband population transfer")
ax.legend(loc="best")
plt.show()
""",
        interpretation="""
At the swap time, almost all amplitude has moved from `|f,0>` into `|g,1>`. This is the effective `gf` red-sideband primitive that many storage-control protocols build on.

The notebook explicitly computes the effective rotating-frame sideband transition frequency and then feeds that reduced quantity through `carrier_for_transition_frequency(...)`, which is appropriate here because the workflow is already formulated directly in the sideband rotating-frame language rather than in a single-mode positive physical drive frequency.
""",
        next_steps="""
- Shorten the pulse to observe under-rotation and Rabi-like oscillations.
- Add `NoiseSpec(...)` to see how transmon relaxation or cavity decay spoils the swap.
- Repeat the calculation with a different initial storage Fock level and compare the matrix-element scaling.
""",
    )
    write_notebook(path, cells)


def build_detuned_sideband_sync() -> None:
    path = TUTORIALS / "20_bosonic_and_sideband" / "02_detuned_sideband_synchronization.ipynb"
    cells = workflow_notebook(
        title="Detuned Sideband Synchronization",
        intro="""
This notebook rewrites `examples/detuned_sideband_sync_demo.py` as a tutorial on branch synchronization. We intentionally drive a `gf` sideband in a dispersive model where the `n=0` and `n=1` branches do not stay perfectly synchronized under a naive resonant pulse.

The physical question is practical: if two branches see slightly different sideband frequencies, can a small drive detuning and a longer pulse improve the transfer fidelity of a superposition? The answer is yes, and the notebook shows that tradeoff explicitly.
""",
        imports_code="""
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    Pulse,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    carrier_for_transition_frequency,
    simulate_sequence,
)
from cqed_sim.pulses import square_envelope
from tutorials.tutorial_support import MHz, ns
""",
        model_code="""
scale = MHz(10.0)
g_sb = 0.35 * scale
chi = -0.2 * scale

model = DispersiveTransmonCavityModel(
    omega_c=0.0,
    omega_q=0.0,
    alpha=0.0,
    chi=chi,
    kerr=0.0,
    n_cav=5,
    n_tr=3,
)
frame = FrameSpec()
target_spec = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2)

initial_state = (model.basis_state(2, 0) + model.basis_state(2, 1)).unit()
target_state = (model.basis_state(0, 1) + model.basis_state(2, 1)).unit()

base_frequency = model.sideband_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=2,
    sideband="red",
    frame=frame,
)
naive_duration_s = np.pi / (2.0 * g_sb)
optimized_detuning_rad_s = 0.16 * scale
optimized_duration_s = 5.11 / scale
dt_s = 0.02 / scale
""",
        pulse_code="""
def run_sideband_gate(*, detuning_rad_s: float, duration_s: float) -> tuple[qt.Qobj, float]:
    pulse = Pulse(
        "sb",
        0.0,
        duration_s,
        square_envelope,
        amp=g_sb,
        carrier=carrier_for_transition_frequency(base_frequency + detuning_rad_s),
    )
    compiled = SequenceCompiler(dt=dt_s).compile([pulse], t_end=duration_s)
    result = simulate_sequence(
        model,
        compiled,
        initial_state,
        {"sb": target_spec},
        config=SimulationConfig(frame=frame),
    )
    fidelity = abs(target_state.overlap(result.final_state)) ** 2
    return result.final_state, float(fidelity)
""",
        simulation_code="""
naive_state, naive_fidelity = run_sideband_gate(detuning_rad_s=0.0, duration_s=naive_duration_s)
optimized_state, optimized_fidelity = run_sideband_gate(
    detuning_rad_s=optimized_detuning_rad_s,
    duration_s=optimized_duration_s,
)

detuning_scan_mhz = np.linspace(-4.0, 4.0, 31)
detuning_scan_fidelity = np.asarray(
    [
        run_sideband_gate(detuning_rad_s=MHz(detuning_mhz), duration_s=optimized_duration_s)[1]
        for detuning_mhz in detuning_scan_mhz
    ],
    dtype=float,
)

print("Naive fidelity:", naive_fidelity)
print("Optimized fidelity:", optimized_fidelity)
""",
        analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))

axes[0].bar(
    ["naive resonant", "detuned sync"],
    [naive_fidelity, optimized_fidelity],
    color=["#E45756", "#4C78A8"],
)
axes[0].set_ylim(0.0, 1.0)
axes[0].set_ylabel("Target-state fidelity")
axes[0].set_title("One-shot comparison")

axes[1].plot(detuning_scan_mhz, detuning_scan_fidelity, "o-", color="#4C78A8")
axes[1].axvline(optimized_detuning_rad_s / MHz(1.0), color="black", ls="--", lw=1.0)
axes[1].set_xlabel("Applied detuning relative to branch-0 sideband (MHz)")
axes[1].set_ylabel("Target-state fidelity")
axes[1].set_title("Detuning scan at the synchronized duration")

plt.tight_layout()
plt.show()
""",
        interpretation="""
The dispersive shift makes different Fock branches accumulate phase at slightly different rates, so a simple resonant pulse is not always the best compromise for a superposition. A modest drive detuning plus a longer pulse can improve the overlap with the desired synchronized target.

This example stays within the current effective sideband abstraction. It is not a microscopic coupler model, and the notebook is honest about that abstraction level.
""",
        next_steps="""
- Sweep the pulse duration together with the detuning to reproduce the optimization surface more fully.
- Start from a different superposition and see whether the same detuning remains optimal.
- Compare this branch-synchronization idea with the simpler single-branch sideband swap in the previous notebook.
""",
    )
    write_notebook(path, cells)


def build_sequential_sideband_reset() -> None:
    path = TUTORIALS / "20_bosonic_and_sideband" / "03_sequential_sideband_reset.ipynb"
    cells = workflow_notebook(
        title="Sequential Sideband Reset Workflow",
        intro="""
This notebook rewrites `examples/sequential_sideband_reset.py` as a guided workflow for a three-mode storage-transmon-readout model. The protocol uses repeated storage and readout sideband steps plus readout ringdown to remove storage excitations one level at a time.

The physical scenario is a reset recipe: map storage excitations through the transmon into the readout mode, let the readout dissipate, and repeat. This notebook is built on the reusable helper module in `examples/workflows/sequential_sideband_reset.py`, which is the canonical repo-side implementation of this workflow.
""",
        imports_code="""
from cqed_sim import readout_photon_number, storage_photon_number, transmon_level_populations
from examples.workflows.sequential_sideband_reset import (
    SequentialSidebandResetCalibration,
    SequentialSidebandResetDevice,
    build_sideband_reset_frame,
    build_sideband_reset_model,
    build_sideband_reset_noise,
    run_sequential_sideband_reset,
)
from tutorials.tutorial_support import ns
""",
        model_code="""
device = SequentialSidebandResetDevice(
    readout_frequency_hz=8_596_222_556.078796,
    qubit_frequency_hz=6_150_358_764.4830475,
    storage_frequency_hz=5_240_932_800.0,
    readout_kappa_hz=4.156e6,
    qubit_anharmonicity_hz=-255_669_694.5244608,
    chi_storage_hz=-2_840_421.0,
    chi_readout_hz=-3.0e6,
    storage_gf_sideband_frequency_hz=6_803_533_628.0,
    storage_t1_s=250.0e-6,
    storage_t2_ramsey_s=150.0e-6,
)
calibration = SequentialSidebandResetCalibration(
    storage_sideband_rate_hz=8.0e6,
    readout_sideband_rate_hz=10.0e6,
    ef_rate_hz=12.0e6,
    ringdown_multiple=3.0,
)

model = build_sideband_reset_model(device, n_storage=5, n_readout=3)
frame = build_sideband_reset_frame(model)
noise = build_sideband_reset_noise(device)
initial_state = model.basis_state(0, 2, 0)
""",
        pulse_code="""
pulse_dt_s = 0.25e-9
ringdown_dt_s = 4.0e-9
print("Reset protocol stages: storage sideband -> readout sideband -> ringdown, repeated for each storage excitation.")
""",
        simulation_code="""
result = run_sequential_sideband_reset(
    model,
    initial_state,
    calibration=calibration,
    initial_storage_level=2,
    frame=frame,
    noise=noise,
    pulse_dt_s=pulse_dt_s,
    ringdown_dt_s=ringdown_dt_s,
)

stage_labels = [f"C{record.cycle_index}:{record.stage}" for record in result.stage_records]
stage_storage = np.asarray(
    [storage_photon_number(record.simulation.final_state) for record in result.stage_records],
    dtype=float,
)
stage_readout = np.asarray(
    [readout_photon_number(record.simulation.final_state) for record in result.stage_records],
    dtype=float,
)
stage_duration_ns = np.asarray([record.duration_s / ns for record in result.stage_records], dtype=float)
cycle_index = np.arange(1, result.cycle_final_storage_photon_number.size + 1, dtype=int)

print("Final transmon populations:", transmon_level_populations(result.final_state))
print("Final storage <n>:", storage_photon_number(result.final_state))
print("Final readout <n>:", readout_photon_number(result.final_state))
""",
        analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

axes[0].plot(stage_storage, "o-", label="storage <n>", color="#4C78A8")
axes[0].plot(stage_readout, "o-", label="readout <n>", color="#E45756")
axes[0].set_xticks(np.arange(len(stage_labels)))
axes[0].set_xticklabels(stage_labels, rotation=45, ha="right")
axes[0].set_ylabel("Photon number")
axes[0].set_title("Stage-by-stage reset trajectory")
axes[0].legend(loc="best")

axes[1].plot(cycle_index, result.cycle_final_storage_photon_number, "o-", label="storage", color="#4C78A8")
axes[1].plot(cycle_index, result.cycle_final_readout_photon_number, "o-", label="readout", color="#E45756")
axes[1].set_xlabel("Reset cycle")
axes[1].set_ylabel("Photon number after cycle")
axes[1].set_title("Cycle summaries")
axes[1].legend(loc="best")

plt.tight_layout()
plt.show()

print("Stage durations [ns]:", stage_duration_ns)
""",
        interpretation="""
The storage photon number drops after each cycle because the protocol repeatedly maps storage excitations into a dissipative readout channel. The readout population spikes during the dump step and then decays during ringdown.

This notebook is intentionally workflow-oriented rather than minimal. It shows how to use the repo-side helper module while keeping the physics assumptions explicit: matched rotating frame, effective sideband transitions, and Lindblad readout decay.
""",
        next_steps="""
- Change `initial_storage_level` to see how the number of cycles scales with the initial excitation.
- Increase `ringdown_multiple` to study the tradeoff between reset time and residual readout population.
- Run the same workflow with `noise=None` to separate coherent protocol errors from open-system degradation.
""",
    )
    write_notebook(path, cells)


def build_shelving_isolation() -> None:
    path = TUTORIALS / "20_bosonic_and_sideband" / "04_shelving_isolation.ipynb"
    cells = workflow_notebook(
        title="Shelving Isolation With a Multilevel Sideband",
        intro="""
This notebook rewrites `examples/shelving_isolation_demo.py` as a short study of shelving. We prepare a superposition with population in `|e,0>` and `|f,0>`, then use a `gf` red sideband to move only the `|f,0>` component into the cavity while leaving the shelved `|e,0>` population isolated.

The physical point is selective access: multilevel structure lets us move one branch of the wavefunction while protecting another branch from the same control tone.
""",
        imports_code="""
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    build_sideband_pulse,
    carrier_for_transition_frequency,
    compute_shelving_leakage,
    simulate_sequence,
    subsystem_level_population,
)
from tutorials.tutorial_support import GHz, MHz, ns
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-0.8),
    kerr=0.0,
    n_cav=4,
    n_tr=3,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
duration_s = 80.0 * ns
amplitude_rad_s = np.pi / (2.0 * duration_s)

initial_state = (
    np.sqrt(0.4) * model.basis_state(1, 0)
    + np.sqrt(0.6) * model.basis_state(2, 0)
).unit()
""",
        pulse_code="""
omega_sb = model.sideband_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=2,
    sideband="red",
    frame=frame,
)
pulses, drive_ops, pulse_meta = build_sideband_pulse(
    SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2, sideband="red"),
    duration_s=duration_s,
    amplitude_rad_s=amplitude_rad_s,
    sigma_fraction=0.18,
    channel="sb",
    carrier=carrier_for_transition_frequency(omega_sb),
    label="gaussian_gf_sideband",
)
compiled = SequenceCompiler(dt=1.0 * ns).compile(pulses, t_end=duration_s)
print("Pulse metadata:", pulse_meta)
""",
        simulation_code="""
result = simulate_sequence(
    model,
    compiled,
    initial_state,
    drive_ops,
    config=SimulationConfig(frame=frame),
)

initial_plot = np.asarray([0.4, 0.6, 0.0], dtype=float)
final_plot = np.asarray(
    [
        subsystem_level_population(result.final_state, "transmon", 1),
        abs(model.basis_state(2, 0).overlap(result.final_state)) ** 2,
        abs(model.basis_state(0, 1).overlap(result.final_state)) ** 2,
    ],
    dtype=float,
)

shelving_leakage = compute_shelving_leakage(initial_state, result.final_state, shelved_level=1)
print("Shelving leakage:", shelving_leakage)
""",
        analysis_code="""
labels = [r"$|e,0\\rangle$", r"$|f,0\\rangle$", r"$|g,1\\rangle$"]
x = np.arange(len(labels), dtype=float)
width = 0.35

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar(x - width / 2.0, initial_plot, width=width, label="initial", color="#B279A2")
ax.bar(x + width / 2.0, final_plot, width=width, label="final", color="#4C78A8")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0.0, 1.0)
ax.set_ylabel("Population")
ax.set_title("Shelving isolates the |e,0> branch")
ax.legend(loc="best")
plt.show()
""",
        interpretation="""
The `|e,0>` population stays nearly unchanged because the chosen sideband couples `|f,0>` to `|g,1>` and does not directly address the shelved `|e,0>` branch. The leakage metric quantifies how well that isolation worked.

This is a small but practical example of why multilevel transmon structure matters: selective control often depends on choosing which ladder transition participates in the bosonic transfer.
""",
        next_steps="""
- Increase the pulse amplitude to see when the shelved branch starts to leak appreciably.
- Compare Gaussian and square sideband envelopes.
- Add weak dissipation to see whether shelving remains useful once the protocol is no longer unitary.
""",
    )
    write_notebook(path, cells)


def build_multimode_crosskerr() -> None:
    path = TUTORIALS / "30_advanced_protocols" / "01_multimode_crosskerr.ipynb"
    cells = workflow_notebook(
        title="Multimode Cross-Kerr Phase Accumulation",
        intro="""
This notebook rewrites `examples/multimode_crosskerr_demo.py` as an advanced three-mode tutorial. We prepare a superposition of storage Fock states while the readout mode contains one photon, then let the system evolve freely under a storage-readout cross-Kerr interaction.

The physical observable is a conditional phase. Cross-Kerr does not transfer population here; it changes the relative phase between the storage branches at a rate set by `chi_sr`.
""",
        imports_code="""
from cqed_sim import (
    DispersiveReadoutTransmonStorageModel,
    FrameSpec,
    SequenceCompiler,
    SimulationConfig,
    simulate_sequence,
)
from tutorials.tutorial_support import GHz, MHz, ns
""",
        model_code="""
chi_sr = MHz(1.5)
final_phase_rad = 1.12
evolution_time_s = final_phase_rad / chi_sr

model = DispersiveReadoutTransmonStorageModel(
    omega_s=GHz(5.0),
    omega_r=GHz(7.5),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi_s=0.0,
    chi_r=0.0,
    chi_sr=chi_sr,
    kerr_s=0.0,
    kerr_r=0.0,
    n_storage=3,
    n_readout=3,
    n_tr=2,
)
frame = FrameSpec(
    omega_c_frame=model.omega_s,
    omega_q_frame=model.omega_q,
    omega_r_frame=model.omega_r,
)
initial_state = (model.basis_state(0, 0, 1) + model.basis_state(0, 1, 1)).unit()
times_ns = np.linspace(0.0, evolution_time_s / ns, 31)
""",
        pulse_code="""
def relative_phase_after(time_s: float) -> float:
    if time_s <= 0.0:
        state = initial_state
    else:
        compiled = SequenceCompiler(dt=max(0.5 * ns, time_s / 150.0)).compile([], t_end=time_s)
        result = simulate_sequence(
            model,
            compiled,
            initial_state,
            {},
            config=SimulationConfig(frame=frame),
        )
        state = result.final_state
    amp_ref = model.basis_state(0, 0, 1).overlap(state)
    amp_shifted = model.basis_state(0, 1, 1).overlap(state)
    return float(np.angle(amp_shifted / amp_ref))
""",
        simulation_code="""
relative_phase_rad = np.asarray([relative_phase_after(time_ns * ns) for time_ns in times_ns], dtype=float)
expected_phase_rad = np.asarray(
    [float(np.angle(np.exp(-1j * chi_sr * time_ns * ns))) for time_ns in times_ns],
    dtype=float,
)

print("Final simulated phase [rad]:", float(relative_phase_rad[-1]))
print("Final expected phase [rad]:", float(expected_phase_rad[-1]))
""",
        analysis_code="""
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(times_ns, relative_phase_rad, "o-", label="simulation", color="#4C78A8")
ax.plot(times_ns, expected_phase_rad, "--", label=r"$\\angle e^{-i \\chi_{sr} t}$", color="#E45756")
ax.set_xlabel("Free-evolution time (ns)")
ax.set_ylabel("Relative phase (rad)")
ax.set_title("Cross-Kerr conditional phase accumulation")
ax.legend(loc="best")
plt.show()
""",
        interpretation="""
The storage and readout populations stay fixed, but the branch with one storage photon accumulates an extra phase because the readout is occupied. That is exactly what a cross-Kerr term is supposed to do in this effective model.

This notebook is a useful sanity check whenever you need to interpret three-mode conditional phases or when you want to calibrate a wait-based controlled-phase primitive.
""",
        next_steps="""
- Repeat the calculation with the readout in vacuum to confirm that the conditional phase disappears.
- Add storage self-Kerr or readout self-Kerr to separate self- and cross-nonlinearity effects.
- Extend the initial state to include two storage photons and compare the phase scaling.
""",
    )
    write_notebook(path, cells)


def build_open_system_sideband_degradation() -> None:
    path = TUTORIALS / "30_advanced_protocols" / "02_open_system_sideband_degradation.ipynb"
    cells = workflow_notebook(
        title="Open-System Degradation of a Sideband Swap",
        intro="""
This notebook rewrites `examples/open_system_sideband_degradation.py` as a tutorial on realism. We compare an ideal `gf` sideband swap against the same pulse in the presence of transmon relaxation, dephasing, and cavity loss.

The physical message is simple but important: a pulse that looks nearly perfect in closed-system simulation can lose fidelity once realistic dissipation is added, even when the control waveform itself is unchanged.
""",
        imports_code="""
from cqed_sim import (
    DispersiveTransmonCavityModel,
    FrameSpec,
    NoiseSpec,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    build_sideband_pulse,
    carrier_for_transition_frequency,
    simulate_sequence,
)
from tutorials.tutorial_support import GHz, MHz, ns
""",
        model_code="""
g_sb = MHz(8.0)
scale = g_sb / 0.3
duration_s = np.pi / (2.0 * g_sb)

model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-0.8),
    kerr=0.0,
    n_cav=4,
    n_tr=3,
)
frame = FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
initial_state = model.basis_state(2, 0)
target_state = model.basis_state(0, 1)

noise = NoiseSpec(
    transmon_t1=(120.0 / scale, 35.0 / scale),
    tphi=90.0 / scale,
    kappa=0.02 * scale,
)
""",
        pulse_code="""
omega_sb = model.sideband_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=2,
    sideband="red",
    frame=frame,
)
pulses, drive_ops, pulse_meta = build_sideband_pulse(
    SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2, sideband="red"),
    duration_s=duration_s,
    amplitude_rad_s=g_sb,
    channel="sb",
    carrier=carrier_for_transition_frequency(omega_sb),
    label="gf_swap",
)
compiled = SequenceCompiler(dt=duration_s / 250.0).compile(pulses, t_end=duration_s)
print("Noise parameters:", noise)
""",
        simulation_code="""
ideal = simulate_sequence(
    model,
    compiled,
    initial_state,
    drive_ops,
    config=SimulationConfig(frame=frame, store_states=True),
)
open_system = simulate_sequence(
    model,
    compiled,
    initial_state,
    drive_ops,
    config=SimulationConfig(frame=frame, store_states=True),
    noise=noise,
)

target_projector = target_state.proj()

def projector_population(state: qt.Qobj) -> float:
    if state.isoper:
        return float(np.real((target_projector * state).tr()))
    return float(abs(target_state.overlap(state)) ** 2)

ideal_target_population = np.asarray([projector_population(state) for state in ideal.states], dtype=float)
open_target_population = np.asarray([projector_population(state) for state in open_system.states], dtype=float)

print("Ideal final target population:", float(ideal_target_population[-1]))
print("Open-system final target population:", float(open_target_population[-1]))
""",
        analysis_code="""
times_ns = np.asarray(compiled.tlist, dtype=float) / ns

fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(times_ns, ideal_target_population, label="closed system", lw=2.0, color="#4C78A8")
ax.plot(times_ns, open_target_population, label="with dissipation", lw=2.0, color="#E45756")
ax.set_xlabel("Time (ns)")
ax.set_ylabel(r"Population in $|g,1\\rangle$")
ax.set_title("Noise degrades the same sideband pulse")
ax.legend(loc="best")
plt.show()
""",
        interpretation="""
Both simulations use the same control pulse. The only difference is the Lindblad noise model, and that difference is enough to reduce the final transfer probability noticeably.

For realistic protocol design, it is therefore not enough to report the coherent unitary result alone. You need to know how far the hardware coherence times sit above the control time scale.
""",
        next_steps="""
- Increase or decrease the sideband rate to see how faster control trades against rotating-wave accuracy.
- Sweep `transmon_t1` or `kappa` to identify the dominant error channel.
- Compare the open-system result with the sequential-reset notebook, where dissipation is part of the protocol rather than an unwanted error.
""",
    )
    write_notebook(path, cells)


def build_unitary_synthesis_workflow() -> None:
    path = TUTORIALS / "30_advanced_protocols" / "03_unitary_synthesis_workflow.ipynb"
    cells = workflow_notebook(
        title="Unitary Synthesis Workflow",
        intro="""
This notebook rewrites `examples/unitary_synthesis_demo.py` as a guided tour of `cqed_sim.unitary_synthesis`. Instead of simulating a fixed pulse sequence, we ask the optimizer to assemble a gate sequence that approximates a target unitary inside a qubit-cavity subspace.

This is an advanced control-design workflow rather than a hardware experiment notebook. The main learning goal is how to define the target, choose a gate set, run the synthesizer, and interpret the resulting fidelity, leakage, and timing report.
""",
        imports_code="""
from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer, make_target
from cqed_sim.unitary_synthesis.reporting import make_run_report
from tutorials.tutorial_support import ns
""",
        model_code="""
subspace = Subspace.qubit_cavity_block(n_match=3)
target_name = "cluster"
u_target = make_target(target_name, n_match=3, variant="mps")
""",
        pulse_code="""
synthesizer = UnitarySynthesizer(
    subspace=subspace,
    backend="pulse",
    gateset=["QubitRotation", "SQR", "SNAP", "Displacement"],
    optimize_times=True,
    time_bounds={"default": (20e-9, 2000e-9)},
    leakage_weight=10.0,
    time_reg_weight=1e-2,
    seed=1234,
)
""",
        simulation_code="""
result = synthesizer.fit(target=u_target, init_guess="heuristic", multistart=4, maxiter=120)
report = make_run_report(result.report, result.simulation.subspace_operator)

history_objective = np.asarray([entry["objective_total"] for entry in result.history], dtype=float)
history_fidelity = np.asarray([entry["metrics"]["fidelity_subspace"] for entry in result.history], dtype=float)
gate_labels = [f"{idx + 1}. {gate.type}" for idx, gate in enumerate(result.sequence.gates)]
gate_durations_ns = np.asarray([gate.duration / ns for gate in result.sequence.gates], dtype=float)
total_duration_ns = float(np.sum(gate_durations_ns))

print("Final fidelity:", report["metrics"]["fidelity"])
print("Worst leakage:", report["metrics"]["leakage_worst"])
print("Total duration [ns]:", total_duration_ns)
""",
        analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

axes[0].plot(history_objective, "o-", label="objective", color="#E45756")
axes[0].plot(1.0 - history_fidelity, "o--", label="1 - fidelity", color="#4C78A8")
axes[0].set_xlabel("Recorded optimizer step")
axes[0].set_ylabel("Loss-like metric")
axes[0].set_title("Synthesis progress")
axes[0].legend(loc="best")

axes[1].barh(gate_labels, gate_durations_ns, color="#72B7B2")
axes[1].set_xlabel("Duration (ns)")
axes[1].set_title("Optimized gate durations")

plt.tight_layout()
plt.show()
""",
        interpretation="""
The synthesized sequence is a compromise between subspace fidelity, leakage suppression, and time regularization. A higher-fidelity answer is often possible, but only if you spend more optimization budget or enlarge the gate set.

The important workflow habit is to read the report, not just the final fidelity. Leakage, duration, and the per-gate composition all matter when you decide whether a synthesized sequence is usable.
""",
        next_steps="""
- Swap `target_name` to `"easy"` or `"ghz"` and compare the optimization difficulty.
- Change the gate set to see which primitives matter most for your chosen target.
- Use the resulting sequence as a starting point for a more hardware-constrained optimization study.
""",
    )
    write_notebook(path, cells)


def build_snap_optimization_workflow() -> None:
    path = TUTORIALS / "30_advanced_protocols" / "04_snap_optimization_workflow.ipynb"
    cells = workflow_notebook(
        title="SNAP Optimization Workflow",
        intro="""
This notebook rewrites `examples/run_snap_optimization_demo.py` as a study-style tutorial. It uses the repo-side helper module under `examples/studies/snap_opt` to optimize a selective multitone pulse for a target set of per-manifold phases.

The physical abstraction is narrower than the full runtime simulator: the study helper works in a matched rotating frame with a simplified model of manifold-selective control. That makes it ideal for teaching how the optimization loop behaves, while staying honest that this is a repo-side study workflow rather than a canonical package entry point.
""",
        imports_code="""
from examples.studies.snap_opt import (
    SnapModelConfig,
    SnapRunConfig,
    SnapToneParameters,
    optimize_snap_parameters,
    target_difficulty_metric,
)
""",
        model_code="""
model = SnapModelConfig(n_cav=7, n_tr=2, chi=2.0 * np.pi * 0.02).build_model()
target_phases = np.array([0.0, 1.1, -0.7, 0.4], dtype=float)
run_config = SnapRunConfig(duration=170.0, dt=0.2, base_amp=0.010)
initial_params = SnapToneParameters.vanilla(target_phases)
difficulty = target_difficulty_metric(target_phases)

print("Target difficulty metric:", difficulty)
""",
        pulse_code="""
print("Initial amplitudes:", initial_params.amplitudes)
print("Initial detunings:", initial_params.detunings)
print("Initial phases:", initial_params.phases)
""",
        simulation_code="""
result = optimize_snap_parameters(
    model,
    target_phases,
    run_config,
    initial_params=initial_params,
    max_iter=40,
    learning_rate=0.3,
    threshold=6e-3,
)

iterations = np.arange(len(result.history_error), dtype=int)
print("Converged:", result.converged)
print("Initial mean overlap error:", float(result.history_error[0]))
print("Final mean overlap error:", float(result.history_error[-1]))
""",
        analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

axes[0].plot(iterations, result.history_error, "o-", color="#4C78A8")
axes[0].set_xlabel("Optimization iteration")
axes[0].set_ylabel("Mean overlap error")
axes[0].set_title("SNAP optimization progress")

tone_index = np.arange(result.params.amplitudes.size, dtype=int)
axes[1].bar(tone_index - 0.25, result.params.amplitudes, width=0.25, label="amplitude")
axes[1].bar(tone_index, result.params.detunings, width=0.25, label="detuning")
axes[1].bar(tone_index + 0.25, result.params.phases, width=0.25, label="phase")
axes[1].set_xlabel("Tone / manifold index")
axes[1].set_title("Optimized tone parameters")
axes[1].legend(loc="best")

plt.tight_layout()
plt.show()
""",
        interpretation="""
The optimization history shows how quickly the selective slow-stage model can reduce coherent manifold errors for this target. The final parameter bars are not generic hardware calibrations; they are the result of this specific study model and target.

This notebook is best thought of as a bridge between the reusable package and a more specialized control study. It teaches the workflow honestly without pretending that every helper in `examples/studies` is part of the stable public API.
""",
        next_steps="""
- Change `target_phases` to see how target difficulty affects convergence.
- Increase `max_iter` and compare the cost of extra optimization budget.
- Follow this notebook with the paper-reproduction notebooks under `test_against_papers/` if you want literature-oriented validation rather than a workflow demo.
""",
    )
    write_notebook(path, cells)


def build_calibration_targets_and_fitting() -> None:
    path = TUTORIALS / "31_system_identification_and_domain_randomization" / "01_calibration_targets_and_fitting.ipynb"
    cells = workflow_notebook(
        title="Calibration Targets and Fitted Parameters",
        intro="""
This notebook introduces the calibration-target side of `cqed_sim`. We use lightweight spectroscopy, Rabi, and T1 target helpers to generate effective calibration traces, then read back the fitted device parameters.

The physical role of this notebook is not pulse-level realism. It is a workflow bridge: these targets synthesize the summary data products that a later system-identification or control stack would consume, so you can reason clearly about what is being estimated and in what units.
""",
        imports_code="""
from cqed_sim import DispersiveTransmonCavityModel, run_rabi, run_spectroscopy, run_t1
from tutorials.tutorial_support import GHz, MHz, ns, us
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-2.25),
    kerr=MHz(-0.006),
    n_cav=8,
    n_tr=2,
)

drive_frequencies_hz = np.linspace(5.9e9, 6.1e9, 121)
drive_frequencies = 2.0 * np.pi * drive_frequencies_hz
rabi_amplitudes = np.linspace(0.0, 2.0e8, 61)
t1_delays = np.linspace(0.0, 30.0, 61) * us
""",
        pulse_code="""
print("These helpers synthesize effective calibration traces and fitted summaries; they are intended for calibration / system-ID workflows rather than full pulse-level experiment replay.")
""",
        simulation_code="""
spectroscopy = run_spectroscopy(model, drive_frequencies)
rabi = run_rabi(model, rabi_amplitudes, duration=40.0 * ns)
t1 = run_t1(model, t1_delays, t1=24.0 * us)

omega_01_fit_ghz = spectroscopy.fitted_parameters["omega_01"] / (2.0 * np.pi * 1.0e9)
omega_12_fit_ghz = spectroscopy.fitted_parameters["omega_12"] / (2.0 * np.pi * 1.0e9)
omega_scale_fit = float(rabi.fitted_parameters["omega_scale"])
t1_fit_us = t1.fitted_parameters["t1"] / us

print("omega_01 fit [GHz]:", omega_01_fit_ghz)
print("omega_12 fit [GHz]:", omega_12_fit_ghz)
print("Rabi omega_scale fit:", omega_scale_fit)
print("T1 fit [us]:", t1_fit_us)
""",
        analysis_code="""
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))

axes[0].plot(drive_frequencies_hz / 1.0e9, spectroscopy.raw_data["response"], color="#4C78A8")
axes[0].axvline(omega_01_fit_ghz, color="#E45756", ls="--", lw=1.0, label=r"$\\omega_{01}$ fit")
axes[0].axvline(omega_12_fit_ghz, color="#54A24B", ls=":", lw=1.0, label=r"$\\omega_{12}$ fit")
axes[0].set_xlabel("Drive frequency (GHz)")
axes[0].set_ylabel("Effective spectroscopy response")
axes[0].set_title("Spectroscopy target")
axes[0].legend(loc="upper right")

axes[1].plot(rabi.raw_data["amplitudes"] / 1.0e8, rabi.raw_data["excited_population"], color="#F58518")
axes[1].set_xlabel(r"Drive amplitude ($10^8$ rad/s)")
axes[1].set_ylabel(r"Excited population $P_e$")
axes[1].set_title(rf"Rabi target, fitted scale = {omega_scale_fit:.3f}")

axes[2].plot(t1.raw_data["delays"] / us, t1.raw_data["excited_population"], color="#72B7B2")
axes[2].set_xlabel("Delay (us)")
axes[2].set_ylabel(r"Excited population $P_e$")
axes[2].set_title(rf"T1 target, fitted $T_1 = {t1_fit_us:.2f}$ us")

plt.tight_layout()
plt.show()
""",
        interpretation="""
Each helper returns a `CalibrationResult` with fitted parameters, uncertainties, raw synthetic data, and metadata. The point is not that these traces are the only valid calibration procedure, but that they create a consistent data structure for downstream fitting and uncertainty propagation.

For system-identification work, the fitted summaries are often more important than the raw traces. They become the posteriors or priors that later randomization and robust-control workflows will consume.
""",
        next_steps="""
- Change the spectroscopy span or linewidth and see how the fitted uncertainty changes.
- Repeat the Rabi target with a different duration to understand the fitted `omega_scale` normalization.
- Feed these fitted results into the next notebook to build calibration-informed randomization priors.
""",
    )
    write_notebook(path, cells)


def build_evidence_to_randomizer_and_env() -> None:
    path = TUTORIALS / "31_system_identification_and_domain_randomization" / "02_evidence_to_randomizer_and_env.ipynb"
    cells = workflow_notebook(
        title="Calibration Evidence, Randomization, and Environment Wiring",
        intro="""
This notebook turns fitted calibration summaries into a form that the control stack can use. We package the fitted parameters into `CalibrationEvidence`, convert that evidence into a `DomainRandomizer`, then build a `HybridCQEDEnv` whose episode reset samples from those priors.

The main workflow lesson is that system identification and robust control should not be separate stories. Calibration posteriors are what justify the train-time parameter randomization used by the RL environment.
""",
        imports_code="""
from cqed_sim import (
    CalibrationEvidence,
    DispersiveTransmonCavityModel,
    HybridCQEDEnv,
    HybridEnvConfig,
    HybridSystemConfig,
    NormalPrior,
    PrimitiveActionSpace,
    ReducedDispersiveModelConfig,
    build_observation_model,
    build_reward_model,
    fock_state_preparation_task,
    randomizer_from_calibration,
    run_rabi,
    run_spectroscopy,
    run_t1,
)
from tutorials.tutorial_support import GHz, MHz, ns, us
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-2.25),
    kerr=MHz(-0.006),
    n_cav=8,
    n_tr=2,
)

drive_frequencies_hz = np.linspace(5.9e9, 6.1e9, 51)
drive_frequencies = 2.0 * np.pi * drive_frequencies_hz
rabi_amplitudes = np.linspace(0.0, 2.0e8, 21)
t1_delays = np.linspace(0.0, 30.0, 21) * us
""",
        pulse_code="""
print("This workflow uses calibration summaries, not a new control pulse sequence. The important bridge is: fitted parameters -> CalibrationEvidence -> DomainRandomizer -> env.reset(...).")
""",
        simulation_code="""
spectroscopy = run_spectroscopy(model, drive_frequencies)
rabi = run_rabi(model, rabi_amplitudes, duration=40.0 * ns)
t1 = run_t1(model, t1_delays, t1=24.0 * us)

evidence = CalibrationEvidence(
    model_posteriors={
        "omega_q": NormalPrior(
            spectroscopy.fitted_parameters["omega_01"],
            spectroscopy.uncertainties["omega_01"] + 1.0,
        ),
        "chi": NormalPrior(MHz(-2.25), MHz(0.08)),
        "kerr": NormalPrior(MHz(-0.006), MHz(0.0015)),
    },
    noise_posteriors={
        "t1": NormalPrior(t1.fitted_parameters["t1"], t1.uncertainties["t1"] + 1.0e-12),
    },
    notes={"rabi_omega_scale": float(rabi.fitted_parameters["omega_scale"])},
)

randomizer = randomizer_from_calibration(evidence)
sample = randomizer.sample(seed=5, mode="train")

action_space = PrimitiveActionSpace(primitives=("qubit_gaussian", "cavity_displacement", "wait"))
env = HybridCQEDEnv(
    HybridEnvConfig(
        system=HybridSystemConfig(
            regime="reduced_dispersive",
            reduced_model=ReducedDispersiveModelConfig(
                omega_c=GHz(5.0),
                omega_q=GHz(6.0),
                alpha=MHz(-220.0),
                chi=MHz(-2.25),
                kerr=MHz(-0.006),
                n_cav=8,
                n_tr=2,
            ),
        ),
        task=fock_state_preparation_task(cavity_level=1),
        action_space=action_space,
        observation_model=build_observation_model("ideal_summary", action_dim=action_space.shape[0]),
        reward_model=build_reward_model("state_preparation"),
        randomizer=randomizer,
        randomization_mode="train",
        episode_horizon=2,
        seed=7,
    )
)
observation, info = env.reset(seed=11)

posterior_means = np.asarray(
    [
        evidence.model_posteriors["omega_q"].mean / (2.0 * np.pi * 1.0e9),
        evidence.model_posteriors["chi"].mean / (2.0 * np.pi * 1.0e6),
        evidence.model_posteriors["kerr"].mean / (2.0 * np.pi * 1.0e3),
        evidence.noise_posteriors["t1"].mean / us,
    ],
    dtype=float,
)
sampled_values = np.asarray(
    [
        sample.metadata["model"]["omega_q"] / (2.0 * np.pi * 1.0e9),
        sample.metadata["model"]["chi"] / (2.0 * np.pi * 1.0e6),
        sample.metadata["model"]["kerr"] / (2.0 * np.pi * 1.0e3),
        sample.metadata["noise"]["t1"] / us,
    ],
    dtype=float,
)

print("Rabi omega_scale metadata:", evidence.notes["rabi_omega_scale"])
print("Randomization metadata keys:", sorted(info["randomization"].keys()))
print("Observation shape:", observation.shape)
""",
        analysis_code="""
labels = [r"$\\omega_q$ [GHz]", r"$\\chi$ [MHz]", r"$K/2\\pi$ [kHz]", r"$T_1$ [us]"]
x = np.arange(len(labels), dtype=float)

fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

axes[0].bar(x - 0.18, posterior_means, width=0.36, color="#72B7B2", label="posterior mean")
axes[0].bar(x + 0.18, sampled_values, width=0.36, color="#F58518", label="sampled train episode")
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
axes[0].set_title("Calibration evidence becomes randomization priors")
axes[0].legend(loc="upper right")

axes[1].bar(np.arange(observation.size, dtype=int), observation, color="#4C78A8")
axes[1].set_xlabel("Observation component")
axes[1].set_ylabel("Value")
axes[1].set_title("Observation returned by env.reset(...)")

plt.tight_layout()
plt.show()
""",
        interpretation="""
The important object here is not the reset observation itself; it is the path that created it. `CalibrationEvidence` stores uncertainty in a structured way, `randomizer_from_calibration(...)` converts that uncertainty into train/eval priors, and `HybridCQEDEnv.reset(...)` samples one concrete episode model from that distribution.

That is the clean handoff between calibration and robust control. If the posterior changes after a new characterization run, the training distribution should change too.
""",
        next_steps="""
- Add additional posterior fields once you have trustworthy uncertainty estimates for decoherence, hardware, or measurement parameters.
- Compare `mode="train"` and `mode="eval"` samples if you later choose to construct separate distributions manually.
- Pair this notebook with the RL workflow notebook if you want to study how those randomization priors affect controller robustness.
""",
    )
    write_notebook(path, cells)


def build_sideband_quasienergy_scan() -> None:
    path = TUTORIALS / "50_floquet_driven_systems" / "01_sideband_quasienergy_scan.ipynb"
    cells = workflow_notebook(
        title="Driven Sideband Quasienergy Scan",
        intro="""
This notebook introduces `cqed_sim.floquet` through a transmon-storage sideband problem. We sweep a periodic drive across the red-sideband frequency, solve the Floquet problem at each point, and track the resulting quasienergy branches.

The physical question is where the drive-dressed spectrum shows the strongest avoided crossing. That minimum quasienergy gap is a compact way to locate the most resonant part of the periodic-drive sweep.
""",
        imports_code="""
from cqed_sim import FloquetConfig, FloquetProblem, SidebandDriveSpec, build_target_drive_term, run_floquet_sweep
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from tutorials.tutorial_support import GHz, MHz
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.05),
    omega_q=GHz(6.25),
    alpha=MHz(-250.0),
    chi=MHz(-15.0),
    kerr=0.0,
    n_cav=3,
    n_tr=3,
)
frame = FrameSpec()
sideband = SidebandDriveSpec(mode="storage", lower_level=0, upper_level=1, sideband="red")
center_frequency = model.sideband_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=1,
    sideband="red",
    frame=frame,
)
scan_detuning_mhz = np.linspace(-0.25, 0.25, 61)
drive_frequencies = center_frequency + 2.0 * np.pi * 1.0e6 * scan_detuning_mhz
""",
        pulse_code="""
problems = []
for frequency in drive_frequencies:
    drive = build_target_drive_term(
        model,
        sideband,
        amplitude=MHz(0.03),
        frequency=float(frequency),
        waveform="cos",
        label="effective_red_sideband_drive",
    )
    problems.append(
        FloquetProblem(
            model=model,
            frame=frame,
            periodic_terms=(drive,),
            period=2.0 * np.pi / float(frequency),
            label="transmon_storage_sideband_scan",
        )
    )

print("Reference red-sideband frequency [GHz]:", center_frequency / (2.0 * np.pi * 1.0e9))
""",
        simulation_code="""
sweep = run_floquet_sweep(
    problems,
    parameter_values=scan_detuning_mhz,
    config=FloquetConfig(n_time_samples=96),
)

tracked_quasienergies_mhz = sweep.tracked_quasienergies / (2.0 * np.pi * 1.0e6)
minimum_gap_mhz = np.min(np.abs(np.diff(tracked_quasienergies_mhz, axis=1)), axis=1)
best_index = int(np.argmin(minimum_gap_mhz))

print("Minimum tracked gap [MHz]:", float(minimum_gap_mhz[best_index]))
print("Detuning at minimum gap [MHz]:", float(scan_detuning_mhz[best_index]))
""",
        analysis_code="""
fig, axes = plt.subplots(2, 1, figsize=(9.0, 7.0), sharex=True)

for branch in range(tracked_quasienergies_mhz.shape[1]):
    axes[0].plot(scan_detuning_mhz, tracked_quasienergies_mhz[:, branch], linewidth=1.2)
axes[0].axvline(scan_detuning_mhz[best_index], color="black", linestyle="--", linewidth=1.0)
axes[0].set_ylabel("Tracked quasienergy / 2pi (MHz)")
axes[0].set_title("Sideband Floquet branch tracking")
axes[0].grid(True, alpha=0.3)

axes[1].plot(scan_detuning_mhz, minimum_gap_mhz, color="#E45756", linewidth=1.6)
axes[1].axvline(scan_detuning_mhz[best_index], color="black", linestyle="--", linewidth=1.0)
axes[1].set_xlabel("Drive detuning from red sideband (MHz)")
axes[1].set_ylabel("Minimum adjacent branch gap (MHz)")
axes[1].set_title("Avoided-crossing diagnostic")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
""",
        interpretation="""
The sweep tracks the same dressed branches continuously rather than re-sorting them independently at each parameter point. That is why the minimum-gap curve is physically interpretable: it points to the drive setting where the periodic sideband drive hybridizes the states most strongly.

This is one of the main advantages of a Floquet workflow over repeated static diagonalization. The periodic drive becomes part of the eigenproblem rather than a small perturbation added afterward.
""",
        next_steps="""
- Increase the drive amplitude to see the avoided crossing widen.
- Expand the cavity or transmon truncation to check whether the tracked branches are converged.
- Use the next Floquet notebook to interpret branch structure in terms of multiphoton resonance conditions.
""",
    )
    write_notebook(path, cells)


def build_branch_tracking_and_multiphoton_resonances() -> None:
    path = TUTORIALS / "50_floquet_driven_systems" / "02_branch_tracking_and_multiphoton_resonances.ipynb"
    cells = workflow_notebook(
        title="Branch Tracking and Multiphoton Resonances",
        intro="""
This notebook focuses on interpretation rather than sweeping. We drive the qubit at roughly half its bare transition frequency, solve a single Floquet problem, and then ask which bare energy gaps are close to integer multiples of the drive frequency.

That is the cleanest way to explain multiphoton resonances in this API: compare bare transition gaps against $n\\Omega$ and let the Floquet helper report which crossings are near resonance.
""",
        imports_code="""
from cqed_sim import FloquetConfig, FloquetProblem, build_target_drive_term, identify_multiphoton_resonances, solve_floquet
from cqed_sim.core import DispersiveTransmonCavityModel, FrameSpec
from tutorials.tutorial_support import GHz, MHz
""",
        model_code="""
model = DispersiveTransmonCavityModel(
    omega_c=GHz(5.0),
    omega_q=GHz(6.0),
    alpha=MHz(-220.0),
    chi=MHz(-2.5),
    kerr=0.0,
    n_cav=2,
    n_tr=4,
)
frame = FrameSpec()
drive_frequency = 0.5 * model.transmon_transition_frequency(
    cavity_level=0,
    lower_level=0,
    upper_level=1,
    frame=frame,
)
""",
        pulse_code="""
drive = build_target_drive_term(
    model,
    "qubit",
    amplitude=MHz(12.0),
    frequency=drive_frequency,
    waveform="cos",
    label="half_frequency_qubit_drive",
)
problem = FloquetProblem(
    model=model,
    frame=frame,
    periodic_terms=(drive,),
    period=2.0 * np.pi / drive_frequency,
    label="half_frequency_multiphoton_test",
)

print("Drive frequency / 2pi [GHz]:", drive_frequency / (2.0 * np.pi * 1.0e9))
""",
        simulation_code="""
result = solve_floquet(problem, FloquetConfig(n_time_samples=96))
resonances = identify_multiphoton_resonances(result, max_photon_order=3)

quasienergies_mhz = result.quasienergies / (2.0 * np.pi * 1.0e6)
bare_energies_ghz = result.bare_hamiltonian_eigenenergies / (2.0 * np.pi * 1.0e9)

transition_labels = []
transition_gaps_ghz = []
for upper in range(1, min(5, bare_energies_ghz.size)):
    transition_labels.append(f"0->{upper}")
    transition_gaps_ghz.append(float(bare_energies_ghz[upper] - bare_energies_ghz[0]))
transition_gaps_ghz = np.asarray(transition_gaps_ghz, dtype=float)

harmonic_orders = np.arange(1, 4, dtype=int)
harmonic_frequencies_ghz = harmonic_orders * drive_frequency / (2.0 * np.pi * 1.0e9)

print("Number of identified resonances:", len(resonances))
for item in resonances[:5]:
    print(
        f"states {item.lower_state}->{item.upper_state}, order {item.photon_order}, detuning / 2pi [MHz] = {item.detuning / (2.0 * np.pi * 1.0e6):+.3f}"
    )
""",
        analysis_code="""
fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0))

axes[0].bar(np.arange(quasienergies_mhz.size, dtype=int), quasienergies_mhz, color="#4C78A8")
axes[0].set_xlabel("Floquet mode index")
axes[0].set_ylabel("Quasienergy / 2pi (MHz)")
axes[0].set_title("Folded quasienergies for the driven problem")

axes[1].plot(np.arange(transition_gaps_ghz.size, dtype=int), transition_gaps_ghz, "o", color="#F58518", label="bare transition gap")
for order, frequency_ghz in zip(harmonic_orders, harmonic_frequencies_ghz, strict=True):
    axes[1].axhline(frequency_ghz, lw=1.0, ls="--", label=rf"${order}\\Omega / 2\\pi$")
axes[1].set_xticks(np.arange(transition_gaps_ghz.size, dtype=int))
axes[1].set_xticklabels(transition_labels)
axes[1].set_ylabel("Frequency (GHz)")
axes[1].set_title("Bare gaps compared against harmonic drive lines")
axes[1].legend(loc="best")

plt.tight_layout()
plt.show()
""",
        interpretation="""
The Floquet solver gives the dressed quasienergies, while `identify_multiphoton_resonances(...)` explains why specific branches interact strongly. In this setup the half-frequency drive makes a second-order resonance especially visible, because a bare transition gap is close to $2\\Omega$.

That split view is the right mental model for Floquet work in `cqed_sim`: quasienergies show what the periodic Hamiltonian does, and the resonance helper explains which harmonic condition is responsible.
""",
        next_steps="""
- Move the drive frequency away from the half-frequency condition and watch the resonance list disappear.
- Increase `max_photon_order` if you want to search for higher-order resonances.
- Return to the sweep notebook and compare the resonance condition with the avoided-crossing locations across detuning.
""",
    )
    write_notebook(path, cells)


def build_kerr_sign_validation() -> None:
    path = TUTORIALS / "40_validation_and_conventions" / "01_kerr_sign_and_frame_checks.ipynb"
    cells = workflow_notebook(
        title="Kerr Sign and Frame Checks",
        intro="""
This notebook rewrites `examples/kerr_sign_verification.py` as a validation notebook. Its purpose is not to introduce the API from scratch, but to confirm that the runtime Kerr sign used by `cqed_sim` matches the documented convention once the frame choice and observable phase are interpreted correctly.

This kind of notebook belongs in a validation / conventions track because its primary question is correctness of interpretation, not everyday workflow onboarding.
""",
        imports_code="""
from cqed_sim import coherent_state
from examples.workflows.kerr_free_evolution import (
    KerrParameterSet,
    resolve_kerr_parameter_set,
    run_kerr_free_evolution,
    times_us_to_seconds,
    verify_kerr_sign,
)
""",
        model_code="""
base = resolve_kerr_parameter_set("phase_evolution")
flipped = KerrParameterSet(
    name=f"{base.name}_self_kerr_flipped",
    omega_q_hz=base.omega_q_hz,
    omega_c_hz=base.omega_c_hz,
    omega_ro_hz=base.omega_ro_hz,
    alpha_q_hz=base.alpha_q_hz,
    kerr_hz=-base.kerr_hz,
    kerr2_hz=base.kerr2_hz,
    chi_hz=base.chi_hz,
    chi2_hz=base.chi2_hz,
    chi3_hz=base.chi3_hz,
)
times_us = np.array([0.0, 0.25, 0.50, 0.75, 1.00], dtype=float)
""",
        pulse_code="""
times_s = times_us_to_seconds(times_us)
initial_state = coherent_state(2.0)
print("This validation notebook uses free evolution only; the sign check comes from the static Hamiltonian, not from a driven pulse.")
""",
        simulation_code="""
documented = run_kerr_free_evolution(
    times_s,
    parameter_set=base,
    cavity_state=initial_state,
    n_cav=30,
    wigner_times_s=[],
)
flipped_result = run_kerr_free_evolution(
    times_s,
    parameter_set=flipped,
    cavity_state=initial_state,
    n_cav=30,
    wigner_times_s=[],
)
sign_check = verify_kerr_sign(comparison_time_s=1.0e-6, alpha=2.0, n_cav=30, n_tr=3)

documented_phase = np.unwrap(
    np.angle(np.asarray([snapshot.cavity_mean for snapshot in documented.snapshots], dtype=np.complex128))
)
flipped_phase = np.unwrap(
    np.angle(np.asarray([snapshot.cavity_mean for snapshot in flipped_result.snapshots], dtype=np.complex128))
)

print("Documented Kerr [Hz]:", sign_check.documented_kerr_hz)
print("Flipped Kerr [Hz]:", sign_check.flipped_kerr_hz)
print("Matches documented sign:", sign_check.matches_documented_sign)
""",
        analysis_code="""
fig, ax = plt.subplots(figsize=(6.6, 3.8))
ax.plot(times_us, documented_phase, "o-", label="documented sign", color="#4C78A8")
ax.plot(times_us, flipped_phase, "o-", label="flipped sign", color="#E45756")
ax.set_xlabel("Free-evolution time (us)")
ax.set_ylabel("Unwrapped phase of <a> (rad)")
ax.set_title("Kerr sign comparison")
ax.legend(loc="best")
plt.show()
""",
        interpretation="""
The two curves bend in opposite directions, and the documented-sign result matches the repository's runtime convention. If you ever think the sign has flipped, check the rotating frame and the plotted observable before changing the Hamiltonian.

This notebook complements the workflow tutorial on Kerr free evolution: that notebook teaches the phenomenon, while this notebook verifies that the implementation and interpretation remain aligned with the documented convention.
""",
        next_steps="""
- Repeat the check in a lab-frame calculation to separate frame offsets from Kerr-induced curvature.
- Compare the sign of the mean-field phase with the sign of the Wigner-function shear.
- Pair this notebook with `tests/test_32_kerr_sign_notebook_regression.py` when you need an automated guardrail.
""",
    )
    write_notebook(path, cells)


def main() -> None:
    build_protocol_style_simulation()
    build_displacement_then_spectroscopy()
    build_kerr_free_evolution()
    build_phase_space_coordinates_and_wigner_conventions()
    build_selective_gaussian_number_splitting()
    build_sideband_swap()
    build_detuned_sideband_sync()
    build_sequential_sideband_reset()
    build_shelving_isolation()
    build_multimode_crosskerr()
    build_open_system_sideband_degradation()
    build_unitary_synthesis_workflow()
    build_snap_optimization_workflow()
    build_calibration_targets_and_fitting()
    build_evidence_to_randomizer_and_env()
    build_kerr_sign_validation()
    build_sideband_quasienergy_scan()
    build_branch_tracking_and_multiphoton_resonances()


if __name__ == "__main__":
    main()
