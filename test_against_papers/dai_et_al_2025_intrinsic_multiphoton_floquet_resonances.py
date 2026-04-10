"""Literature validation for intrinsic multiphoton Floquet resonances in a driven transmon.

Paper checked:
W. Dai, S. Hazra, D. K. Weiss, P. D. Kurilovich, T. Connolly, H. K. Babla,
S. Singh, V. R. Joshi, A. Z. Ding, P. D. Parakh, J. Venkatraman, X. Xiao,
L. Frunzio, and M. H. Devoret,
"Characterization of drive-induced unwanted state transitions in superconducting circuits,"
arXiv:2506.24070, 2025. DOI: 10.48550/arXiv.2506.24070

Target results:
- Fig. 4(c): the transmon-only Floquet branch analysis shows a low-power
  avoided crossing between the |1> and |5> branches near wd / (2 pi) = 8.02 GHz.
- Fig. 4(d): the same |1> <-> |5> intrinsic resonance moves to a higher drive
  power when the drive frequency is lowered to 7.825 GHz.
- Fig. A8(d): the appendix branch analysis shows a low-power avoided crossing
  between the |0> and |4> branches near wd / (2 pi) = 8.45 GHz.

What this script validates:
- `cqed_sim.floquet` can reproduce the intrinsic mechanism-B resonances from the
  paper when supplied with the paper's transmon-only cosine Hamiltonian and
  charge-drive operator.
- The paper's fitted transmon parameters reproduce the reported undriven
  transition frequencies to within a few MHz.
- The Floquet avoided crossings occur near the paper's reported drive
  frequencies and xi^2 values for three representative intrinsic resonances.

Assumptions and approximations:
- Mechanism B only: transmon-only model, no TLSs, no parasitic electromagnetic
  modes, and no dissipation.
- Fixed offset charge ng = 0, whereas the paper also studies offset-charge
  averaging and drift at higher powers.
- The transmon Hamiltonian is built in a finite charge basis and projected onto
  a truncated eigenbasis before solving the Floquet problem.
- The xi^2 -> Ed calibration follows Eq. (C5) of the paper, while the resonance
  marker here is the minimum tracked quasienergy gap plus bare-state mixing,
  not the paper's full ideal-displaced-state hybridization parameter Theta_j.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import qutip as qt

from cqed_sim.floquet import FloquetConfig, FloquetProblem, PeriodicDriveTerm, run_floquet_sweep


PAPER_EJ_GHZ = 16.2856
PAPER_EC_GHZ = 0.17013
PAPER_OMEGA_01_GHZ = 4.5285
PAPER_OMEGA_15_GHZ = 16.0840


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
    charge_cutoff: int = 35,
    transmon_levels: int = 16,
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
class ResonanceCase:
    name: str
    lower_state: int
    upper_state: int
    drive_frequency_ghz: float
    xi2_values: np.ndarray
    expected_xi2: float
    xi2_tolerance: float
    min_mix_score: float
    max_gap_mhz: float


@dataclass(frozen=True)
class ResonanceResult:
    case: ResonanceCase
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


def analyze_resonance_case(
    case: ResonanceCase,
    *,
    static_hamiltonian: qt.Qobj,
    charge_drive_operator: qt.Qobj,
    omega_01: float,
    ej: float,
    ec: float,
) -> ResonanceResult:
    drive_angular_frequency = ghz_to_angular(case.drive_frequency_ghz)
    problems = []
    for xi_squared in case.xi2_values:
        amplitude = drive_amplitude_from_xi_squared(
            float(xi_squared),
            drive_angular_frequency,
            omega_01,
            ej,
            ec,
        )
        problems.append(
            FloquetProblem(
                static_hamiltonian=static_hamiltonian,
                periodic_terms=(
                    PeriodicDriveTerm(
                        operator=charge_drive_operator,
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
        config=FloquetConfig(n_time_samples=96),
    )

    initial_overlaps = sweep.results[0].bare_state_overlaps
    lower_branch = int(np.argmax(initial_overlaps[:, case.lower_state]))
    upper_branch = int(np.argmax(initial_overlaps[:, case.upper_state]))

    gaps_mhz: list[float] = []
    mix_scores: list[float] = []
    for index in range(len(case.xi2_values)):
        overlaps = sweep.results[index].bare_state_overlaps[sweep.tracked_orders[index]]
        lower_row = overlaps[lower_branch]
        upper_row = overlaps[upper_branch]
        gap_ghz = abs(
            float(sweep.tracked_quasienergies[index, lower_branch] - sweep.tracked_quasienergies[index, upper_branch])
        ) / (2.0 * np.pi)
        mix_score = min(float(lower_row[case.lower_state]), float(lower_row[case.upper_state])) + min(
            float(upper_row[case.lower_state]),
            float(upper_row[case.upper_state]),
        )
        gaps_mhz.append(1.0e3 * gap_ghz)
        mix_scores.append(mix_score)

    best_index = int(np.argmin(np.asarray(gaps_mhz, dtype=float)))
    best_overlaps = sweep.results[best_index].bare_state_overlaps[sweep.tracked_orders[best_index]]
    lower_row = best_overlaps[lower_branch]
    upper_row = best_overlaps[upper_branch]
    return ResonanceResult(
        case=case,
        best_xi2=float(case.xi2_values[best_index]),
        min_gap_mhz=float(gaps_mhz[best_index]),
        mix_score=float(mix_scores[best_index]),
        lower_branch_lower_overlap=float(lower_row[case.lower_state]),
        lower_branch_upper_overlap=float(lower_row[case.upper_state]),
        upper_branch_lower_overlap=float(upper_row[case.lower_state]),
        upper_branch_upper_overlap=float(upper_row[case.upper_state]),
        xi2_values=np.asarray(case.xi2_values, dtype=float),
        gap_curve_mhz=np.asarray(gaps_mhz, dtype=float),
        mix_curve=np.asarray(mix_scores, dtype=float),
    )


def print_spectrum_summary(energies: np.ndarray) -> list[str]:
    omega_01_ghz = angular_to_ghz(float(energies[1] - energies[0]))
    omega_15_ghz = angular_to_ghz(float(energies[5] - energies[1]))
    omega_04_ghz = angular_to_ghz(float(energies[4] - energies[0]))
    delta_01_mhz = 1.0e3 * (omega_01_ghz - PAPER_OMEGA_01_GHZ)
    delta_15_mhz = 1.0e3 * (omega_15_ghz - PAPER_OMEGA_15_GHZ)

    print("Paper-fit transmon spectrum reproduced in a cosine charge-basis model")
    print(f"  omega_01 / (2 pi): {omega_01_ghz:.6f} GHz  (paper: {PAPER_OMEGA_01_GHZ:.6f}, diff: {delta_01_mhz:+.3f} MHz)")
    print(f"  omega_15 / (2 pi): {omega_15_ghz:.6f} GHz  (paper: {PAPER_OMEGA_15_GHZ:.6f}, diff: {delta_15_mhz:+.3f} MHz)")
    print(f"  omega_04 / (2 pi): {omega_04_ghz:.6f} GHz  (half-frequency: {0.5 * omega_04_ghz:.6f} GHz)")

    failures: list[str] = []
    if abs(delta_01_mhz) > 10.0:
        failures.append("omega_01 mismatch exceeded 10 MHz.")
    if abs(delta_15_mhz) > 10.0:
        failures.append("omega_15 mismatch exceeded 10 MHz.")
    return failures


def print_case_summary(result: ResonanceResult) -> list[str]:
    case = result.case
    print(f"\n{case.name}")
    print(f"  drive frequency / (2 pi): {case.drive_frequency_ghz:.6f} GHz")
    print(f"  target avoided crossing: |{case.lower_state}> <-> |{case.upper_state}>")
    print(f"  xi^2 at minimum branch gap: {result.best_xi2:.3f} (paper target: {case.expected_xi2:.3f})")
    print(f"  minimum tracked gap: {result.min_gap_mhz:.3f} MHz")
    print(f"  mix score at minimum gap: {result.mix_score:.3f}")
    print(
        f"  lower tracked branch overlaps: |{case.lower_state}>={result.lower_branch_lower_overlap:.3f}, "
        f"|{case.upper_state}>={result.lower_branch_upper_overlap:.3f}"
    )
    print(
        f"  upper tracked branch overlaps: |{case.lower_state}>={result.upper_branch_lower_overlap:.3f}, "
        f"|{case.upper_state}>={result.upper_branch_upper_overlap:.3f}"
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


def build_cases() -> tuple[ResonanceCase, ...]:
    return (
        ResonanceCase(
            name="Fig. 4(c)-like intrinsic |1> <-> |5> resonance",
            lower_state=1,
            upper_state=5,
            drive_frequency_ghz=8.02,
            xi2_values=np.linspace(0.0, 0.2, 21),
            expected_xi2=0.10,
            xi2_tolerance=0.05,
            min_mix_score=0.25,
            max_gap_mhz=10.0,
        ),
        ResonanceCase(
            name="Fig. 4(d)-like intrinsic |1> <-> |5> resonance at lower frequency",
            lower_state=1,
            upper_state=5,
            drive_frequency_ghz=7.825,
            xi2_values=np.linspace(0.0, 1.4, 29),
            expected_xi2=1.10,
            xi2_tolerance=0.25,
            min_mix_score=0.45,
            max_gap_mhz=60.0,
        ),
        ResonanceCase(
            name="Fig. A8(d)-like intrinsic |0> <-> |4> resonance",
            lower_state=0,
            upper_state=4,
            drive_frequency_ghz=8.45,
            xi2_values=np.linspace(0.0, 0.2, 21),
            expected_xi2=0.10,
            xi2_tolerance=0.05,
            min_mix_score=0.45,
            max_gap_mhz=10.0,
        ),
    )


def save_summary_plot(energies: np.ndarray, results: tuple[ResonanceResult, ...], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(len(results), 1, figsize=(10.5, 3.6 * len(results)), sharex=False)
    if len(results) == 1:
        axes = [axes]

    for axis, result in zip(axes, results, strict=True):
        axis.plot(result.xi2_values, result.gap_curve_mhz, color="#1f77b4", linewidth=1.8, label="Tracked branch gap")
        axis.axvline(result.case.expected_xi2, color="#4d4d4d", linestyle="--", linewidth=1.0, label="Paper target")
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
            f"best $\\xi^2$ = {result.best_xi2:.3f}\nmin gap = {result.min_gap_mhz:.3f} MHz",
            transform=axis.transAxes,
            va="top",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#bbbbbb"},
        )

        handles, labels = axis.get_legend_handles_labels()
        mix_handles, mix_labels = mix_axis.get_legend_handles_labels()
        axis.legend(handles + mix_handles, labels + mix_labels, loc="upper right", fontsize=8.5)

    omega_01_ghz = angular_to_ghz(float(energies[1] - energies[0]))
    omega_15_ghz = angular_to_ghz(float(energies[5] - energies[1]))
    figure.suptitle(
        "Dai et al. 2025 intrinsic Floquet resonance reproduction\n"
        f"Cosine-transmon spectrum: omega_01 / (2 pi) = {omega_01_ghz:.6f} GHz, "
        f"omega_15 / (2 pi) = {omega_15_ghz:.6f} GHz",
        fontsize=13,
        y=0.995,
    )
    axes[-1].set_xlabel(r"$\xi^2$")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"\nSaved plot: {output_path}")


def run_validation(*, plot_output: Path | None = None) -> list[str]:
    ej = ghz_to_angular(PAPER_EJ_GHZ)
    ec = ghz_to_angular(PAPER_EC_GHZ)
    energies, static_hamiltonian, charge_drive_operator = build_transmon_eigenbasis_model(ej, ec)
    omega_01 = float(energies[1] - energies[0])

    failures = print_spectrum_summary(energies)

    results: list[ResonanceResult] = []
    for case in build_cases():
        result = analyze_resonance_case(
            case,
            static_hamiltonian=static_hamiltonian,
            charge_drive_operator=charge_drive_operator,
            omega_01=omega_01,
            ej=ej,
            ec=ec,
        )
        results.append(result)
        failures.extend(print_case_summary(result))

    if plot_output is not None:
        save_summary_plot(energies, tuple(results), plot_output)

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and optionally plot intrinsic Floquet resonances from Dai et al. (2025)."
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