from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.quantum_algorithms.holographic_sim import HolographicSampler, MatrixProductState, ObservableSchedule, pauli_x, pauli_z


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "holographic_ghz_cluster"
FIGURE_DIR = ROOT / "documentations" / "assets" / "images" / "tutorials"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
OBSERVABLE_CSV = OUTPUT_DIR / "observable_comparison.csv"
STRUCTURAL_CSV = OUTPUT_DIR / "structural_checks.csv"
SITE_PROFILE_CSV = OUTPUT_DIR / "site_profile_observables.csv"
OBSERVABLE_FIGURE = FIGURE_DIR / "holographic_ghz_cluster_observables.png"
SITE_PROFILE_FIGURE = FIGURE_DIR / "holographic_ghz_cluster_site_profiles.png"

BENCHMARK_NUM_SITES = 4
SITE_PROFILE_NUM_SITES = 10
SHOTS_PER_STATE = 80_000
SITE_PROFILE_SHOTS = 20_000


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(np.real(value)), "imag": float(np.imag(value))}


def _json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        return _complex_record(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _hadamard() -> np.ndarray:
    return (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)


def _cnot() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.complex128,
    )


def _controlled_z() -> np.ndarray:
    return np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    full = np.array([[1.0 + 0.0j]])
    for idx in range(num_sites):
        full = np.kron(full, gate if idx == site else np.eye(2, dtype=np.complex128))
    return (full @ state.reshape(-1)).reshape((2,) * num_sites)


def _apply_adjacent_two_qubit_gate(state: np.ndarray, gate: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    tensor = state.reshape((2,) * num_sites)
    perm = [site, site + 1] + [idx for idx in range(num_sites) if idx not in (site, site + 1)]
    inverse = np.argsort(perm)
    permuted = np.transpose(tensor, perm)
    leading = permuted.reshape(4, -1)
    updated = (gate @ leading).reshape((2, 2) + tuple(2 for _ in range(num_sites - 2)))
    return np.transpose(updated, inverse)


def _ghz_state(num_sites: int) -> np.ndarray:
    state = np.zeros((2,) * num_sites, dtype=np.complex128)
    state[(0,) * num_sites] = 1.0
    state = _apply_single_qubit_gate(state, _hadamard(), 0, num_sites)
    for site in range(num_sites - 1):
        state = _apply_adjacent_two_qubit_gate(state, _cnot(), site, num_sites)
    return state / np.linalg.norm(state)


def _cluster_state(num_sites: int) -> np.ndarray:
    state = np.zeros((2,) * num_sites, dtype=np.complex128)
    state[(0,) * num_sites] = 1.0
    for site in range(num_sites):
        state = _apply_single_qubit_gate(state, _hadamard(), site, num_sites)
    for site in range(num_sites - 1):
        state = _apply_adjacent_two_qubit_gate(state, _controlled_z(), site, num_sites)
    return state / np.linalg.norm(state)


def _dense_expectation(state: np.ndarray, operator_map: dict[int, np.ndarray]) -> complex:
    full_operator = np.array([[1.0 + 0.0j]])
    identity = np.eye(int(state.shape[0]), dtype=np.complex128)
    for site in range(int(state.ndim)):
        full_operator = np.kron(full_operator, operator_map.get(site, identity))
    flat_state = state.reshape(-1)
    return complex(np.vdot(flat_state, full_operator @ flat_state))


def _schedule_from_operator_map(operator_map: dict[int, np.ndarray], *, total_steps: int) -> ObservableSchedule:
    return ObservableSchedule(
        [{"step": int(site) + 1, "operator": operator} for site, operator in sorted(operator_map.items())],
        total_steps=total_steps,
    )


def _cluster_stabilizer_operator_map(site: int, num_sites: int) -> dict[int, np.ndarray]:
    operator_map: dict[int, np.ndarray] = {int(site): pauli_x().matrix}
    if int(site) > 0:
        operator_map[int(site) - 1] = pauli_z().matrix
    if int(site) + 1 < int(num_sites):
        operator_map[int(site) + 1] = pauli_z().matrix
    return operator_map


def _observable_rows(
    state_label: str,
    state: np.ndarray,
    observable_cases: dict[str, dict[int, np.ndarray]],
    *,
    shots: int,
    seed: int,
) -> list[dict[str, Any]]:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)
    sampler = HolographicSampler.from_mps_sequence(state, label=state_label)
    rows: list[dict[str, Any]] = []
    for observable_label, operator_map in observable_cases.items():
        dense = _dense_expectation(state, operator_map)
        mps_value = mps.expect_operator_product(tuple(operator_map.items()))
        schedule = _schedule_from_operator_map(operator_map, total_steps=mps.num_sites)
        exact = sampler.enumerate_correlator(schedule)
        sampled = sampler.sample_correlator(schedule, shots=shots, seed=seed + len(rows))
        rows.append(
            {
                "state": state_label,
                "observable": observable_label,
                "dense_real": float(np.real(dense)),
                "mps_real": float(np.real(mps_value)),
                "extended_exact_real": float(np.real(exact.mean)),
                "sampled_real": float(np.real(sampled.mean)),
                "sampled_stderr": float(sampled.stderr),
                "abs_error_sample_vs_exact": float(abs(sampled.mean - exact.mean)),
                "abs_error_mps_vs_dense": float(abs(mps_value - dense)),
                "abs_error_exact_vs_dense": float(abs(exact.mean - dense)),
            }
        )
    return rows


def _structural_rows(state_label: str, state: np.ndarray) -> list[dict[str, Any]]:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)
    sequence = mps.to_holographic_channel_sequence(label=state_label)
    unitaries = mps.site_stinespring_unitaries()
    rows: list[dict[str, Any]] = []
    for site, (tensor, channel, unitary) in enumerate(zip(mps.uniform_tensors or [], sequence.channels, unitaries)):
        rows.append(
            {
                "state": state_label,
                "site": int(site),
                "tensor_shape": str(tuple(int(dim) for dim in tensor.shape)),
                "joint_unitary_shape": str(tuple(int(dim) for dim in unitary.shape)),
                "kraus_completeness_error": float(channel.kraus_completeness_error()),
                "right_canonical_error": float(channel.right_canonical_error()),
                "unitarity_error": float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(unitary.shape[0]), ord="fro")),
            }
        )
    return rows


def _plot_observables(rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["state"]), []).append(row)

    fig, axes = plt.subplots(1, len(grouped), figsize=(13.5, 4.8), constrained_layout=True)
    if len(grouped) == 1:
        axes = [axes]

    for axis, (state_label, state_rows) in zip(axes, grouped.items()):
        labels = [str(row["observable"]) for row in state_rows]
        exact = np.asarray([float(row["extended_exact_real"]) for row in state_rows], dtype=float)
        sampled = np.asarray([float(row["sampled_real"]) for row in state_rows], dtype=float)
        errors = np.asarray([float(row["sampled_stderr"]) for row in state_rows], dtype=float)
        x = np.arange(len(labels))
        width = 0.38
        axis.bar(x - width / 2.0, exact, width, label="Exact", color="#1d3557")
        axis.bar(x + width / 2.0, sampled, width, yerr=errors, capsize=4, label="Sampled", color="#e76f51")
        axis.axhline(0.0, color="#222222", linewidth=1.0)
        axis.set_title(state_label.replace("_", " ").upper())
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=18, ha="right")
        axis.set_ylabel("Expectation value")
        axis.set_ylim(-0.15, 1.15)
    axes[0].legend(frameon=False)
    fig.suptitle("Holographic GHZ/cluster workflows: exact vs sampled observables", fontsize=13)
    fig.savefig(OBSERVABLE_FIGURE, dpi=180)
    plt.close(fig)


def _site_profile_rows(*, num_sites: int, shots: int) -> list[dict[str, Any]]:
    identity = np.eye(2, dtype=np.complex128)

    def evaluate_cases(
        state_label: str,
        state: np.ndarray,
        cases: list[dict[str, Any]],
        *,
        seed: int,
    ) -> list[dict[str, Any]]:
        mps = MatrixProductState(state)
        mps.make_right_canonical(cast_complete=True)
        sampler = HolographicSampler.from_mps_sequence(state, label=state_label)
        rows: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            operator_map = dict(case["operator_map"])
            dense = _dense_expectation(state, operator_map)
            mps_value = mps.expect_operator_product(tuple(operator_map.items()))
            schedule = _schedule_from_operator_map(operator_map, total_steps=mps.num_sites)
            exact = sampler.enumerate_correlator(schedule)
            sampled = sampler.sample_correlator(schedule, shots=shots, seed=seed + case_index)
            rows.append(
                {
                    "state": state_label,
                    "profile": str(case["profile"]),
                    "site_index": int(case["site_index"]),
                    "observable": str(case["observable"]),
                    "dense_real": float(np.real(dense)),
                    "mps_real": float(np.real(mps_value)),
                    "extended_exact_real": float(np.real(exact.mean)),
                    "sampled_real": float(np.real(sampled.mean)),
                    "sampled_stderr": float(sampled.stderr),
                    "abs_error_sample_vs_exact": float(abs(sampled.mean - exact.mean)),
                    "abs_error_mps_vs_dense": float(abs(mps_value - dense)),
                    "abs_error_exact_vs_dense": float(abs(exact.mean - dense)),
                }
            )
        return rows

    ghz_cases: list[dict[str, Any]] = []
    for site in range(num_sites):
        one_based = int(site) + 1
        ghz_cases.append(
            {
                "profile": "Z_i",
                "site_index": one_based,
                "observable": f"Z{one_based}",
                "operator_map": {int(site): pauli_z().matrix},
            }
        )
        ghz_cases.append(
            {
                "profile": "Z1Zi",
                "site_index": one_based,
                "observable": f"Z1Z{one_based}",
                "operator_map": {0: identity} if int(site) == 0 else {0: pauli_z().matrix, int(site): pauli_z().matrix},
            }
        )

    cluster_cases = [
        {
            "profile": "K_i",
            "site_index": int(site) + 1,
            "observable": f"K{int(site) + 1}",
            "operator_map": _cluster_stabilizer_operator_map(int(site), num_sites),
        }
        for site in range(num_sites)
    ]

    rows = evaluate_cases("ghz10", _ghz_state(num_sites), ghz_cases, seed=1201)
    rows.extend(evaluate_cases("cluster10", _cluster_state(num_sites), cluster_cases, seed=2201))
    return rows


def _plot_site_profiles(rows: list[dict[str, Any]]) -> None:
    ordered_panels = [
        ("ghz10", "Z_i", "GHZ: <Z_i>", 0.0, (-0.12, 0.12)),
        ("ghz10", "Z1Zi", "GHZ: <Z_1 Z_i>", 1.0, (0.9, 1.05)),
        ("cluster10", "K_i", "Cluster: <K_i>", 1.0, (0.9, 1.05)),
    ]
    fig, axes = plt.subplots(1, len(ordered_panels), figsize=(14.5, 4.8), constrained_layout=True)
    if len(ordered_panels) == 1:
        axes = [axes]

    for axis, (state_label, profile, title, reference, y_limits) in zip(axes, ordered_panels):
        state_rows = [
            row for row in rows if str(row["state"]) == state_label and str(row["profile"]) == profile
        ]
        state_rows.sort(key=lambda row: int(row["site_index"]))
        sites = np.asarray([int(row["site_index"]) for row in state_rows], dtype=int)
        exact = np.asarray([float(row["extended_exact_real"]) for row in state_rows], dtype=float)
        sampled = np.asarray([float(row["sampled_real"]) for row in state_rows], dtype=float)
        errors = np.asarray([float(row["sampled_stderr"]) for row in state_rows], dtype=float)

        axis.plot(sites, exact, marker="o", linewidth=1.8, color="#1d3557", label="Exact")
        axis.errorbar(
            sites,
            sampled,
            yerr=errors,
            fmt="s",
            color="#e76f51",
            capsize=3,
            label="Sampled",
        )
        axis.axhline(reference, color="#222222", linewidth=1.0, alpha=0.75)
        axis.set_title(title)
        axis.set_xlabel("Site index i")
        axis.set_ylabel("Expectation value")
        axis.set_xticks(sites)
        axis.set_ylim(*y_limits)

    axes[0].legend(frameon=False)
    fig.suptitle("Holographic GHZ/cluster site-resolved profiles for i = 1..10", fontsize=13)
    fig.savefig(SITE_PROFILE_FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    ghz_state = _ghz_state(BENCHMARK_NUM_SITES)
    cluster_state = _cluster_state(BENCHMARK_NUM_SITES)
    ghz_observables = {
        "Z0": {0: pauli_z().matrix},
        "Z3": {3: pauli_z().matrix},
        "Z0Z3": {0: pauli_z().matrix, 3: pauli_z().matrix},
        "X0X1X2X3": {0: pauli_x().matrix, 1: pauli_x().matrix, 2: pauli_x().matrix, 3: pauli_x().matrix},
    }
    cluster_observables = {
        "X0Z1": {0: pauli_x().matrix, 1: pauli_z().matrix},
        "Z0X1Z2": {0: pauli_z().matrix, 1: pauli_x().matrix, 2: pauli_z().matrix},
        "Z1X2Z3": {1: pauli_z().matrix, 2: pauli_x().matrix, 3: pauli_z().matrix},
        "Z2X3": {2: pauli_z().matrix, 3: pauli_x().matrix},
    }

    observable_rows = []
    observable_rows.extend(_observable_rows("ghz4", ghz_state, ghz_observables, shots=SHOTS_PER_STATE, seed=101))
    observable_rows.extend(_observable_rows("cluster4", cluster_state, cluster_observables, shots=SHOTS_PER_STATE, seed=211))
    structural_rows = []
    structural_rows.extend(_structural_rows("ghz4", ghz_state))
    structural_rows.extend(_structural_rows("cluster4", cluster_state))
    site_profile_rows = _site_profile_rows(num_sites=SITE_PROFILE_NUM_SITES, shots=SITE_PROFILE_SHOTS)
    _write_csv(
        OBSERVABLE_CSV,
        observable_rows,
        [
            "state",
            "observable",
            "dense_real",
            "mps_real",
            "extended_exact_real",
            "sampled_real",
            "sampled_stderr",
            "abs_error_sample_vs_exact",
            "abs_error_mps_vs_dense",
            "abs_error_exact_vs_dense",
        ],
    )
    _write_csv(
        STRUCTURAL_CSV,
        structural_rows,
        [
            "state",
            "site",
            "tensor_shape",
            "joint_unitary_shape",
            "kraus_completeness_error",
            "right_canonical_error",
            "unitarity_error",
        ],
    )
    _write_csv(
        SITE_PROFILE_CSV,
        site_profile_rows,
        [
            "state",
            "profile",
            "site_index",
            "observable",
            "dense_real",
            "mps_real",
            "extended_exact_real",
            "sampled_real",
            "sampled_stderr",
            "abs_error_sample_vs_exact",
            "abs_error_mps_vs_dense",
            "abs_error_exact_vs_dense",
        ],
    )
    _plot_observables(observable_rows)
    _plot_site_profiles(site_profile_rows)

    summary = {
        "states": {
            "ghz4": {
                "preparation": ["H q0", "CNOT q0->q1", "CNOT q1->q2", "CNOT q2->q3"],
                "exact_reference": {
                    "Z0": 0.0,
                    "Z3": 0.0,
                    "Z0Z3": 1.0,
                    "X0X1X2X3": 1.0,
                },
            },
            "cluster4": {
                "preparation": ["H q0", "H q1", "H q2", "H q3", "CZ q0,q1", "CZ q1,q2", "CZ q2,q3"],
                "exact_reference": {
                    "X0Z1": 1.0,
                    "Z0X1Z2": 1.0,
                    "Z1X2Z3": 1.0,
                    "Z2X3": 1.0,
                },
            },
        },
        "shots_per_state": int(SHOTS_PER_STATE),
        "observable_rows": observable_rows,
        "structural_rows": structural_rows,
        "site_profiles": {
            "num_sites": int(SITE_PROFILE_NUM_SITES),
            "shots_per_state": int(SITE_PROFILE_SHOTS),
            "rows": site_profile_rows,
        },
        "files": {
            "observable_csv": str(OBSERVABLE_CSV.relative_to(ROOT)),
            "structural_csv": str(STRUCTURAL_CSV.relative_to(ROOT)),
            "site_profile_csv": str(SITE_PROFILE_CSV.relative_to(ROOT)),
        },
        "figures": {
            "observables": str(OBSERVABLE_FIGURE.relative_to(ROOT)),
            "site_profiles": str(SITE_PROFILE_FIGURE.relative_to(ROOT)),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    print(f"Wrote summary to {SUMMARY_JSON}")
    print(f"Wrote observable table to {OBSERVABLE_CSV}")
    print(f"Wrote structural table to {STRUCTURAL_CSV}")
    print(f"Wrote site-profile table to {SITE_PROFILE_CSV}")
    print(f"Wrote figures to {OBSERVABLE_FIGURE} and {SITE_PROFILE_FIGURE}")


if __name__ == "__main__":
    main()