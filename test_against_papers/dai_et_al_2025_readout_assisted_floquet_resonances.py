"""Literature validation for readout-assisted Floquet resonances in a driven transmon.

Paper checked:
W. Dai, S. Hazra, D. K. Weiss, P. D. Kurilovich, T. Connolly, H. K. Babla,
S. Singh, V. R. Joshi, A. Z. Ding, P. D. Parakh, J. Venkatraman, X. Xiao,
L. Frunzio, and M. H. Devoret,
"Characterization of drive-induced unwanted state transitions in superconducting circuits,"
arXiv:2506.24070, 2025. DOI: 10.48550/arXiv.2506.24070

Target results:
- Appendix E / Fig. A5: readout-mode-assisted branch analysis for the K-like
  feature at wd / (2 pi) = 10.48 GHz, where the paper associates the avoided
  crossing with the dressed branches connected to |1_t,0_r> and |7_t,1_r>.
- Appendix E / Fig. A5: readout-mode-assisted branch analysis for the L-like
  feature at wd / (2 pi) = 10.73 GHz, where the paper associates the avoided
  crossing with the dressed branches connected to |1_t,0_r> and |4_t,1_r>.

What this script validates:
- `cqed_sim.floquet` can reproduce the qualitative Appendix E readout-assisted
  avoided crossings when supplied with the paper's explicit transmon-readout
  Hamiltonian and charge-drive operator.
- The paper-fit Appendix E parameters produce well-defined undriven dressed
  states connected to |1_t,0_r>, |4_t,1_r>, and |7_t,1_r>.
- The minimum tracked branch gaps occur near the xi^2 values seen in the paper's
  fixed-frequency branch-analysis slices.

Assumptions and approximations:
- Mechanism C, readout-mode-assisted branch slices only: no TLSs, no parasitic
  package modes, and no dissipation.
- The Hamiltonian follows Eq. (E10) of the paper in a finite charge basis,
  projected into a truncated transmon eigenbasis and then coupled to a truncated
  readout resonator Fock basis.
- The xi^2 -> Ed calibration follows Eq. (C5) of the paper using the transmon-only
  omega_01, while the resonance marker here is the minimum tracked quasienergy
  gap plus dressed-state mixing rather than the paper's exact branch-number plot.
- The model uses ng = 0 and finite truncations, so this script is intended as a
  representative literature-backed validation, not a full Appendix E refit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qutip as qt

from cqed_sim.floquet import FloquetConfig, FloquetProblem, PeriodicDriveTerm, run_floquet_sweep


PAPER_EJ_GHZ = 16.40
PAPER_EC_GHZ = 0.1695
PAPER_G_GHZ = 0.153
PAPER_OMEGA_R_GHZ = 9.029

MODEL_CHARGE_CUTOFF = 35
MODEL_TRANSMON_LEVELS = 25
MODEL_RESONATOR_LEVELS = 5


def ghz_to_angular(ghz: float) -> float:
    return 2.0 * np.pi * float(ghz)


def angular_to_ghz(angular_frequency: float) -> float:
    return float(angular_frequency) / (2.0 * np.pi)


def charge_basis_operators(charge_cutoff: int) -> tuple[qt.Qobj, qt.Qobj]:
    charges = np.arange(-int(charge_cutoff), int(charge_cutoff) + 1, dtype=float)
    dimension = charges.size
    number_operator = qt.Qobj(np.diag(charges), dims=[[dimension], [dimension]])
    shift_matrix = np.zeros((dimension, dimension), dtype=np.complex128)
    for index in range(dimension - 1):
        shift_matrix[index + 1, index] = 1.0
    exp_i_phi = qt.Qobj(shift_matrix, dims=[[dimension], [dimension]])
    cosine_operator = 0.5 * (exp_i_phi + exp_i_phi.dag())
    return number_operator, cosine_operator


def build_transmon_eigenbasis_model(
    ej: float,
    ec: float,
    *,
    ng: float = 0.0,
    charge_cutoff: int = MODEL_CHARGE_CUTOFF,
    transmon_levels: int = MODEL_TRANSMON_LEVELS,
) -> tuple[np.ndarray, qt.Qobj, qt.Qobj]:
    number_operator, cosine_operator = charge_basis_operators(charge_cutoff)
    identity = qt.qeye(number_operator.shape[0])
    charge_hamiltonian = 4.0 * float(ec) * ((number_operator - float(ng) * identity) ** 2) - float(ej) * cosine_operator
    eigenvalues, eigenstates = charge_hamiltonian.eigenstates()

    level_count = int(transmon_levels)
    truncated_energies = np.asarray(eigenvalues[:level_count], dtype=float)
    truncated_energies = truncated_energies - truncated_energies[0]

    charge_matrix = np.zeros((level_count, level_count), dtype=np.complex128)
    for row in range(level_count):
        for col in range(level_count):
            charge_matrix[row, col] = complex(eigenstates[row].overlap(number_operator * eigenstates[col]))

    static_hamiltonian = qt.Qobj(np.diag(truncated_energies), dims=[[level_count], [level_count]])
    charge_drive_operator = qt.Qobj(charge_matrix, dims=[[level_count], [level_count]])
    return truncated_energies, static_hamiltonian, charge_drive_operator


def transmon_charge_zero_point_fluctuation(ej: float, ec: float) -> float:
    return float((float(ej) / (32.0 * float(ec))) ** 0.25)


def drive_amplitude_from_xi_squared(
    xi_squared: float,
    drive_angular_frequency: float,
    omega_01: float,
    ej: float,
    ec: float,
) -> float:
    xi = float(np.sqrt(max(float(xi_squared), 0.0)))
    n_zpf = transmon_charge_zero_point_fluctuation(ej, ec)
    return float(xi * (drive_angular_frequency**2 - omega_01**2) / (2.0 * n_zpf * drive_angular_frequency))


@dataclass(frozen=True)
class CompositeStateLabel:
    transmon_level: int
    resonator_level: int

    def render(self) -> str:
        return f"|{self.transmon_level}_t,{self.resonator_level}_r>"


@dataclass(frozen=True)
class DressedLabelMatch:
    label: CompositeStateLabel
    dressed_index: int
    overlap: float


@dataclass(frozen=True)
class DressedReadoutModel:
    ej: float
    ec: float
    omega_01: float
    transmon_energies: np.ndarray
    static_hamiltonian: qt.Qobj
    dressed_charge_drive_operator: qt.Qobj
    dressed_eigenstates: tuple[qt.Qobj, ...]
    transmon_levels: int
    resonator_levels: int


@dataclass(frozen=True)
class ResonanceCase:
    name: str
    paper_slice: str
    lower_label: CompositeStateLabel
    upper_label: CompositeStateLabel
    drive_frequency_ghz: float
    xi2_values: np.ndarray
    expected_xi2: float
    xi2_tolerance: float
    min_mix_score: float
    max_gap_mhz: float
    n_time_samples: int


@dataclass(frozen=True)
class ResonanceResult:
    case: ResonanceCase
    lower_match: DressedLabelMatch
    upper_match: DressedLabelMatch
    tracked_lower_branch: int
    tracked_upper_branch: int
    best_xi2: float
    min_gap_mhz: float
    mix_score: float
    lower_branch_lower_overlap: float
    lower_branch_upper_overlap: float
    upper_branch_lower_overlap: float
    upper_branch_upper_overlap: float
    xi2_values: np.ndarray
    gap_curve_mhz: np.ndarray
    mix_curve: np.ndarray


def build_dressed_readout_model() -> DressedReadoutModel:
    ej = ghz_to_angular(PAPER_EJ_GHZ)
    ec = ghz_to_angular(PAPER_EC_GHZ)
    coupling = ghz_to_angular(PAPER_G_GHZ)
    omega_r = ghz_to_angular(PAPER_OMEGA_R_GHZ)

    transmon_energies, transmon_static, transmon_charge = build_transmon_eigenbasis_model(ej, ec)
    omega_01 = float(transmon_energies[1] - transmon_energies[0])

    resonator = qt.destroy(MODEL_RESONATOR_LEVELS)
    resonator_number = resonator.dag() * resonator
    identity_t = qt.qeye(MODEL_TRANSMON_LEVELS)
    identity_r = qt.qeye(MODEL_RESONATOR_LEVELS)

    # Implements the Appendix E Hamiltonian, Eq. (E10), in a truncated dressed basis.
    coupled_hamiltonian = (
        qt.tensor(transmon_static, identity_r)
        + qt.tensor(identity_t, omega_r * resonator_number)
        - 1j * coupling * qt.tensor(transmon_charge, resonator - resonator.dag())
    )
    drive_operator = qt.tensor(transmon_charge, identity_r)

    dressed_energies, dressed_eigenstates = coupled_hamiltonian.eigenstates()
    dressed_energies = np.asarray(dressed_energies, dtype=float)
    dressed_energies = dressed_energies - dressed_energies[0]
    level_count = dressed_energies.size

    dressed_drive_matrix = np.zeros((level_count, level_count), dtype=np.complex128)
    for row in range(level_count):
        for col in range(level_count):
            dressed_drive_matrix[row, col] = complex(dressed_eigenstates[row].overlap(drive_operator * dressed_eigenstates[col]))

    return DressedReadoutModel(
        ej=ej,
        ec=ec,
        omega_01=omega_01,
        transmon_energies=np.asarray(transmon_energies, dtype=float),
        static_hamiltonian=qt.Qobj(np.diag(dressed_energies), dims=[[level_count], [level_count]]),
        dressed_charge_drive_operator=qt.Qobj(dressed_drive_matrix, dims=[[level_count], [level_count]]),
        dressed_eigenstates=tuple(dressed_eigenstates),
        transmon_levels=MODEL_TRANSMON_LEVELS,
        resonator_levels=MODEL_RESONATOR_LEVELS,
    )


def match_dressed_label(model: DressedReadoutModel, label: CompositeStateLabel) -> DressedLabelMatch:
    product_state = qt.tensor(
        qt.basis(model.transmon_levels, label.transmon_level),
        qt.basis(model.resonator_levels, label.resonator_level),
    )
    overlaps = np.asarray([abs(state.overlap(product_state)) ** 2 for state in model.dressed_eigenstates], dtype=float)
    match_index = int(np.argmax(overlaps))
    return DressedLabelMatch(label=label, dressed_index=match_index, overlap=float(overlaps[match_index]))


def build_cases() -> tuple[ResonanceCase, ...]:
    return (
        ResonanceCase(
            name="Appendix E K-like readout-assisted resonance",
            paper_slice="Fig. A5 slice at wd / (2 pi) = 10.48 GHz",
            lower_label=CompositeStateLabel(transmon_level=1, resonator_level=0),
            upper_label=CompositeStateLabel(transmon_level=7, resonator_level=1),
            drive_frequency_ghz=10.48,
            xi2_values=np.linspace(0.20, 0.40, 9),
            expected_xi2=0.325,
            xi2_tolerance=0.05,
            min_mix_score=0.08,
            max_gap_mhz=5.0,
            n_time_samples=128,
        ),
        ResonanceCase(
            name="Appendix E L-like readout-assisted resonance",
            paper_slice="Fig. A5 slice at wd / (2 pi) = 10.73 GHz",
            lower_label=CompositeStateLabel(transmon_level=1, resonator_level=0),
            upper_label=CompositeStateLabel(transmon_level=4, resonator_level=1),
            drive_frequency_ghz=10.73,
            xi2_values=np.linspace(0.10, 0.40, 13),
            expected_xi2=0.25,
            xi2_tolerance=0.06,
            min_mix_score=0.18,
            max_gap_mhz=5.0,
            n_time_samples=112,
        ),
    )


def analyze_resonance_case(case: ResonanceCase, *, model: DressedReadoutModel) -> ResonanceResult:
    lower_match = match_dressed_label(model, case.lower_label)
    upper_match = match_dressed_label(model, case.upper_label)

    drive_angular_frequency = ghz_to_angular(case.drive_frequency_ghz)
    problems = []
    for xi_squared in case.xi2_values:
        amplitude = drive_amplitude_from_xi_squared(
            float(xi_squared),
            drive_angular_frequency,
            model.omega_01,
            model.ej,
            model.ec,
        )
        problems.append(
            FloquetProblem(
                static_hamiltonian=model.static_hamiltonian,
                periodic_terms=(
                    PeriodicDriveTerm(
                        operator=model.dressed_charge_drive_operator,
                        amplitude=amplitude,
                        frequency=drive_angular_frequency,
                        waveform="cos",
                        label=f"charge_drive_{case.name}",
                    ),
                ),
                period=2.0 * np.pi / drive_angular_frequency,
                label=case.name,
            )
        )

    sweep = run_floquet_sweep(
        problems,
        parameter_values=case.xi2_values,
        config=FloquetConfig(n_time_samples=case.n_time_samples),
    )

    initial_overlaps = sweep.results[0].bare_state_overlaps
    lower_branch = int(np.argmax(initial_overlaps[:, lower_match.dressed_index]))
    upper_branch = int(np.argmax(initial_overlaps[:, upper_match.dressed_index]))

    gaps_mhz: list[float] = []
    mix_scores: list[float] = []
    for index in range(len(case.xi2_values)):
        overlaps = sweep.results[index].bare_state_overlaps[sweep.tracked_orders[index]]
        lower_row = overlaps[lower_branch]
        upper_row = overlaps[upper_branch]
        gap_ghz = abs(
            float(sweep.tracked_quasienergies[index, lower_branch] - sweep.tracked_quasienergies[index, upper_branch])
        ) / (2.0 * np.pi)
        mix_score = min(float(lower_row[lower_match.dressed_index]), float(lower_row[upper_match.dressed_index])) + min(
            float(upper_row[lower_match.dressed_index]),
            float(upper_row[upper_match.dressed_index]),
        )
        gaps_mhz.append(1.0e3 * gap_ghz)
        mix_scores.append(mix_score)

    best_index = int(np.argmin(np.asarray(gaps_mhz, dtype=float)))
    best_overlaps = sweep.results[best_index].bare_state_overlaps[sweep.tracked_orders[best_index]]
    lower_row = best_overlaps[lower_branch]
    upper_row = best_overlaps[upper_branch]
    return ResonanceResult(
        case=case,
        lower_match=lower_match,
        upper_match=upper_match,
        tracked_lower_branch=lower_branch,
        tracked_upper_branch=upper_branch,
        best_xi2=float(case.xi2_values[best_index]),
        min_gap_mhz=float(gaps_mhz[best_index]),
        mix_score=float(mix_scores[best_index]),
        lower_branch_lower_overlap=float(lower_row[lower_match.dressed_index]),
        lower_branch_upper_overlap=float(lower_row[upper_match.dressed_index]),
        upper_branch_lower_overlap=float(upper_row[lower_match.dressed_index]),
        upper_branch_upper_overlap=float(upper_row[upper_match.dressed_index]),
        xi2_values=np.asarray(case.xi2_values, dtype=float),
        gap_curve_mhz=np.asarray(gaps_mhz, dtype=float),
        mix_curve=np.asarray(mix_scores, dtype=float),
    )


def print_model_summary(model: DressedReadoutModel, *, cases: tuple[ResonanceCase, ...]) -> list[str]:
    print("Appendix E transmon-readout model in a dressed Floquet basis")
    print(f"  EJ / h: {PAPER_EJ_GHZ:.4f} GHz")
    print(f"  EC / h: {PAPER_EC_GHZ:.4f} GHz")
    print(f"  g / h: {PAPER_G_GHZ:.4f} GHz")
    print(f"  omega_r / (2 pi): {PAPER_OMEGA_R_GHZ:.4f} GHz")
    print(f"  transmon omega_01 / (2 pi): {angular_to_ghz(model.omega_01):.6f} GHz")
    print(
        "  truncation: "
        f"charge cutoff = {MODEL_CHARGE_CUTOFF}, transmon levels = {MODEL_TRANSMON_LEVELS}, resonator levels = {MODEL_RESONATOR_LEVELS}"
    )

    failures: list[str] = []
    seen_indices: dict[int, str] = {}
    all_labels = sorted(
        {case.lower_label for case in cases}.union({case.upper_label for case in cases}),
        key=lambda label: (label.transmon_level, label.resonator_level),
    )
    for label in all_labels:
        match = match_dressed_label(model, label)
        print(f"  matched {match.label.render()} -> dressed index {match.dressed_index} (overlap {match.overlap:.6f})")
        if match.overlap < 0.95:
            failures.append(f"{match.label.render()} dressed-state identification overlap dropped below 0.95.")
        if match.dressed_index in seen_indices:
            failures.append(
                f"Dressed index {match.dressed_index} was assigned to both {seen_indices[match.dressed_index]} and {match.label.render()}."
            )
        else:
            seen_indices[match.dressed_index] = match.label.render()
    return failures


def print_case_summary(result: ResonanceResult) -> list[str]:
    case = result.case
    print(f"\n{case.name}")
    print(f"  paper slice: {case.paper_slice}")
    print(f"  target avoided crossing: {case.lower_label.render()} <-> {case.upper_label.render()}")
    print(
        f"  dressed-state labels: {result.lower_match.label.render()} -> {result.lower_match.dressed_index} "
        f"(overlap {result.lower_match.overlap:.6f}), {result.upper_match.label.render()} -> {result.upper_match.dressed_index} "
        f"(overlap {result.upper_match.overlap:.6f})"
    )
    print(
        f"  tracked Floquet branches: lower = {result.tracked_lower_branch}, upper = {result.tracked_upper_branch}"
    )
    print(f"  xi^2 at minimum branch gap: {result.best_xi2:.3f} (paper slice target: {case.expected_xi2:.3f})")
    print(f"  minimum tracked gap: {result.min_gap_mhz:.3f} MHz")
    print(f"  mix score at minimum gap: {result.mix_score:.3f}")
    print(
        f"  lower tracked branch overlaps: {case.lower_label.render()}={result.lower_branch_lower_overlap:.3f}, "
        f"{case.upper_label.render()}={result.lower_branch_upper_overlap:.3f}"
    )
    print(
        f"  upper tracked branch overlaps: {case.lower_label.render()}={result.upper_branch_lower_overlap:.3f}, "
        f"{case.upper_label.render()}={result.upper_branch_upper_overlap:.3f}"
    )

    failures: list[str] = []
    if abs(result.best_xi2 - case.expected_xi2) > case.xi2_tolerance:
        failures.append(
            f"{case.name}: xi^2 minimum {result.best_xi2:.3f} fell outside the allowed window around {case.expected_xi2:.3f}."
        )
    if result.mix_score < case.min_mix_score:
        failures.append(f"{case.name}: avoided crossing was too weak (mix score {result.mix_score:.3f}).")
    if result.min_gap_mhz > case.max_gap_mhz:
        failures.append(f"{case.name}: minimum tracked gap {result.min_gap_mhz:.3f} MHz was larger than expected.")
    return failures


def save_summary_plot(results: tuple[ResonanceResult, ...], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(len(results), 1, figsize=(10.8, 3.8 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]

    for axis, result in zip(axes, results, strict=True):
        axis.plot(result.xi2_values, result.gap_curve_mhz, color="#1f77b4", linewidth=1.8, label="Tracked branch gap")
        axis.axvline(result.case.expected_xi2, color="#4d4d4d", linestyle="--", linewidth=1.0, label="Paper slice target")
        axis.scatter([result.best_xi2], [result.min_gap_mhz], color="#1f77b4", s=30, zorder=3)
        axis.set_ylabel("Gap (MHz)")
        axis.set_title(
            f"{result.case.name}  |  drive = {result.case.drive_frequency_ghz:.3f} GHz",
            fontsize=11,
        )
        axis.grid(True, alpha=0.3)

        mix_axis = axis.twinx()
        mix_axis.plot(result.xi2_values, result.mix_curve, color="#d62728", linewidth=1.5, alpha=0.9, label="Mix score")
        mix_axis.set_ylabel("Mix score")
        mix_axis.set_ylim(0.0, 1.0)

        axis.text(
            0.02,
            0.95,
            f"best $\\xi^2$ = {result.best_xi2:.3f}\\nmin gap = {result.min_gap_mhz:.3f} MHz\\n"
            f"{result.case.lower_label.render()} -> {result.lower_match.dressed_index}, "
            f"{result.case.upper_label.render()} -> {result.upper_match.dressed_index}",
            transform=axis.transAxes,
            va="top",
            fontsize=9.0,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )

        handles, labels = axis.get_legend_handles_labels()
        mix_handles, mix_labels = mix_axis.get_legend_handles_labels()
        axis.legend(handles + mix_handles, labels + mix_labels, loc="upper right", fontsize=8.5)

    figure.suptitle(
        "Dai et al. 2025 readout-assisted Floquet resonance reproduction\n"
        f"Appendix E model: EJ / h = {PAPER_EJ_GHZ:.4f} GHz, EC / h = {PAPER_EC_GHZ:.4f} GHz, "
        f"g / h = {PAPER_G_GHZ:.4f} GHz, omega_r / (2 pi) = {PAPER_OMEGA_R_GHZ:.4f} GHz",
        fontsize=13,
        y=0.995,
    )
    axes[-1].set_xlabel(r"$\xi^2$")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"\nSaved plot: {output_path}")


def run_validation(*, plot_output: Path | None = None) -> list[str]:
    model = build_dressed_readout_model()
    cases = build_cases()

    failures = print_model_summary(model, cases=cases)

    results: list[ResonanceResult] = []
    for case in cases:
        result = analyze_resonance_case(case, model=model)
        results.append(result)
        failures.extend(print_case_summary(result))

    if plot_output is not None:
        save_summary_plot(tuple(results), plot_output)

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and optionally plot readout-assisted Floquet resonances from Dai et al. (2025)."
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Optional output path for a tutorial summary plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failures = run_validation(plot_output=args.plot_output)

    if failures:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()