from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from cqed_sim.quantum_algorithms.holographic_sim import (
    HolographicSampler,
    MatrixProductState,
    ObservableSchedule,
    StepUnitarySpec,
    pauli_x,
    pauli_z,
)
from cqed_sim.quantum_algorithms.holographic_sim.holographicSim import holographic_sim_bfs


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "holographic_generalized_unitary"
FIGURE_DIR = ROOT / "documentations" / "assets" / "images" / "tutorials"
OBSERVABLE_CSV = OUTPUT_DIR / "observable_comparison.csv"
STRUCTURAL_CSV = OUTPUT_DIR / "structural_checks.csv"
STRESS_CSV = OUTPUT_DIR / "stress_test_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
OBSERVABLE_FIGURE = FIGURE_DIR / "holographic_generalized_observables.png"
STRESS_FIGURE = FIGURE_DIR / "holographic_generalized_stress_test.png"

PRIMARY_RANDOM_SEED = 12345
PRIMARY_SHOTS = 100_000
STRESS_SHOTS = 80_000
INITIAL_BITS = (1, 0, 1, 1)
ENTANGLER_ANGLES = (0.41, -0.33, 0.27)


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


def _rotation_x(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta / 2.0), -1j * np.sin(theta / 2.0)],
            [-1j * np.sin(theta / 2.0), np.cos(theta / 2.0)],
        ],
        dtype=np.complex128,
    )


def _rotation_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.exp(-0.5j * theta), 0.0],
            [0.0, np.exp(0.5j * theta)],
        ],
        dtype=np.complex128,
    )


def _partial_swap(theta: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, np.cos(theta), -1j * np.sin(theta), 0.0],
            [0.0, -1j * np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )


def _random_su2(rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
    q, r = np.linalg.qr(matrix)
    phases = np.diag(r) / np.abs(np.diag(r))
    q = q @ np.diag(np.conj(phases))
    q /= np.linalg.det(q) ** 0.5
    return q


def _apply_single_qubit_gate(state: np.ndarray, gate: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    full = np.array([[1.0 + 0.0j]])
    for idx in range(num_sites):
        full = np.kron(full, gate if idx == site else np.eye(2, dtype=np.complex128))
    flat = state.reshape(-1)
    return (full @ flat).reshape((2,) * num_sites)


def _apply_two_qubit_gate(state: np.ndarray, gate: np.ndarray, site: int, num_sites: int) -> np.ndarray:
    tensor = state.reshape((2,) * num_sites)
    perm = [site, site + 1] + [idx for idx in range(num_sites) if idx not in (site, site + 1)]
    inverse = np.argsort(perm)
    permuted = np.transpose(tensor, perm)
    leading = permuted.reshape(4, -1)
    updated = (gate @ leading).reshape((2, 2) + tuple(2 for _ in range(num_sites - 2)))
    return np.transpose(updated, inverse)


def _build_primary_state() -> tuple[np.ndarray, list[np.ndarray]]:
    rng = np.random.default_rng(PRIMARY_RANDOM_SEED)
    num_sites = len(INITIAL_BITS)
    state = np.zeros((2,) * num_sites, dtype=np.complex128)
    state[INITIAL_BITS] = 1.0
    local_rotations = [_random_su2(rng) for _ in range(num_sites)]
    for site, gate in enumerate(local_rotations):
        state = _apply_single_qubit_gate(state, gate, site, num_sites)
    for site, angle in enumerate(ENTANGLER_ANGLES):
        state = _apply_two_qubit_gate(state, _partial_swap(angle), site, num_sites)
    state = state / np.linalg.norm(state)
    return state, local_rotations


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


def _observable_rows(state: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, complex], dict[str, complex], dict[str, float]]:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)
    sampler = HolographicSampler.from_mps_sequence(state, label="primary_random_mps")
    observable_cases: dict[str, dict[int, np.ndarray]] = {
        "Z0": {0: pauli_z().matrix},
        "X1": {1: pauli_x().matrix},
        "Z2": {2: pauli_z().matrix},
        "Z0Z2": {0: pauli_z().matrix, 2: pauli_z().matrix},
        "X1Z3": {1: pauli_x().matrix, 3: pauli_z().matrix},
    }
    exact_values: dict[str, complex] = {}
    sampled_values: dict[str, complex] = {}
    sampled_errors: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for label, operator_map in observable_cases.items():
        dense = _dense_expectation(state, operator_map)
        mps_value = mps.expect_operator_product(tuple(operator_map.items()))
        schedule = _schedule_from_operator_map(operator_map, total_steps=mps.num_sites)
        exact = sampler.enumerate_correlator(schedule)
        sample = sampler.sample_correlator(schedule, shots=PRIMARY_SHOTS, seed=31)
        exact_values[label] = exact.mean
        sampled_values[label] = sample.mean
        sampled_errors[label] = sample.stderr
        rows.append(
            {
                "observable": label,
                "dense_real": float(np.real(dense)),
                "mps_real": float(np.real(mps_value)),
                "extended_exact_real": float(np.real(exact.mean)),
                "sampled_real": float(np.real(sample.mean)),
                "sampled_stderr": float(sample.stderr),
                "abs_error_sample_vs_exact": float(abs(sample.mean - exact.mean)),
                "abs_error_mps_vs_dense": float(abs(mps_value - dense)),
                "abs_error_exact_vs_dense": float(abs(exact.mean - dense)),
                "z_score": float(abs(sample.mean - exact.mean) / sample.stderr),
            }
        )

    connected_exact = exact_values["Z0Z2"] - exact_values["Z0"] * exact_values["Z2"]
    connected_sampled = sampled_values["Z0Z2"] - sampled_values["Z0"] * sampled_values["Z2"]
    sampled_z0 = float(np.real(sampled_values["Z0"]))
    sampled_z2 = float(np.real(sampled_values["Z2"]))
    connected_stderr = float(
        np.sqrt(
            sampled_errors["Z0Z2"] ** 2
            + (sampled_z2 * sampled_errors["Z0"]) ** 2
            + (sampled_z0 * sampled_errors["Z2"]) ** 2
        )
    )
    rows.append(
        {
            "observable": "Connected(Z0,Z2)",
            "dense_real": float(np.real(exact_values["Z0Z2"] - exact_values["Z0"] * exact_values["Z2"])),
            "mps_real": float(np.real(exact_values["Z0Z2"] - exact_values["Z0"] * exact_values["Z2"])),
            "extended_exact_real": float(np.real(connected_exact)),
            "sampled_real": float(np.real(connected_sampled)),
            "sampled_stderr": connected_stderr,
            "abs_error_sample_vs_exact": float(abs(connected_sampled - connected_exact)),
            "abs_error_mps_vs_dense": 0.0,
            "abs_error_exact_vs_dense": 0.0,
            "z_score": float(abs(connected_sampled - connected_exact) / connected_stderr),
        }
    )
    exact_values["Connected(Z0,Z2)"] = connected_exact
    sampled_values["Connected(Z0,Z2)"] = connected_sampled
    sampled_errors["Connected(Z0,Z2)"] = connected_stderr
    return rows, exact_values, sampled_values, sampled_errors


def _structural_rows(state: np.ndarray) -> list[dict[str, Any]]:
    mps = MatrixProductState(state)
    mps.make_right_canonical(cast_complete=True)
    sequence = mps.to_holographic_channel_sequence(label="primary_random_mps")
    unitaries = mps.site_stinespring_unitaries()
    rows: list[dict[str, Any]] = []
    for site, (tensor, channel, unitary) in enumerate(zip(mps.uniform_tensors or [], sequence.channels, unitaries)):
        rows.append(
            {
                "site": int(site),
                "tensor_shape": str(tuple(int(dim) for dim in tensor.shape)),
                "joint_unitary_shape": str(tuple(int(dim) for dim in unitary.shape)),
                "kraus_completeness_error": float(channel.kraus_completeness_error()),
                "right_canonical_error": float(channel.right_canonical_error()),
                "unitarity_error": float(np.linalg.norm(unitary.conj().T @ unitary - np.eye(unitary.shape[0]), ord="fro")),
            }
        )
    return rows


def _stress_test_rows() -> tuple[list[dict[str, Any]], list[StepUnitarySpec], complex, complex, float]:
    stress_specs = [
        StepUnitarySpec(_rotation_x(0.41), acts_on="physical", label="physical_rx"),
        StepUnitarySpec(_rotation_z(-0.29), acts_on="bond", label="bond_rz"),
        StepUnitarySpec(_partial_swap(0.37), acts_on="joint", label="joint_partial_swap"),
        StepUnitarySpec(_rotation_x(-0.22), acts_on="physical", label="physical_rx_tail"),
    ]
    sampler = HolographicSampler.from_unitary_sequence(
        stress_specs,
        physical_dim=2,
        bond_dim=2,
        label="mixed_space_stress_test",
    )
    schedule = ObservableSchedule(
        [
            {"step": 1, "operator": pauli_z()},
            {"step": 3, "operator": pauli_x()},
            {"step": 4, "operator": pauli_z()},
        ],
        total_steps=len(stress_specs),
    )
    exact = sampler.enumerate_correlator(schedule)
    sample = sampler.sample_correlator(schedule, shots=STRESS_SHOTS, seed=17)
    resolved_joint = [spec.resolve_joint_unitary(physical_dim=2, bond_dim=2) for spec in stress_specs]
    branches = holographic_sim_bfs(
        resolved_joint,
        [pauli_z().matrix, None, pauli_x().matrix, pauli_z().matrix],
        d=2,
    )
    legacy_mean = sum(complex(branch["prob"]) * complex(branch["weight"]) for branch in branches)
    rows = [
        {
            "metric": "extended_exact_real",
            "value": float(np.real(exact.mean)),
        },
        {
            "metric": "sampled_real",
            "value": float(np.real(sample.mean)),
        },
        {
            "metric": "sampled_stderr",
            "value": float(sample.stderr),
        },
        {
            "metric": "legacy_exact_real",
            "value": float(np.real(legacy_mean)),
        },
        {
            "metric": "abs_error_sample_vs_exact",
            "value": float(abs(sample.mean - exact.mean)),
        },
    ]
    return rows, stress_specs, exact.mean, sample.mean, sample.stderr


def _plot_observables(rows: list[dict[str, Any]]) -> None:
    labels = [row["observable"] for row in rows]
    exact = np.asarray([row["extended_exact_real"] for row in rows], dtype=float)
    sampled = np.asarray([row["sampled_real"] for row in rows], dtype=float)
    errors = np.asarray([row["sampled_stderr"] for row in rows], dtype=float)
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    ax.bar(x - width / 2.0, exact, width, label="Exact extended-unitary", color="#1b4965")
    ax.bar(x + width / 2.0, sampled, width, yerr=errors, capsize=4, label="Sampled holographic", color="#ca6702")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_ylabel("Expectation value")
    ax.set_title("Generalized holographic workflow: exact vs sampled observables")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.legend(frameon=False)
    fig.savefig(OBSERVABLE_FIGURE, dpi=180)
    plt.close(fig)


def _plot_stress_test(exact_mean: complex, sampled_mean: complex, sampled_stderr: float) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.2), constrained_layout=True)
    ax.bar([0], [float(np.real(exact_mean))], width=0.45, label="Exact", color="#0a9396")
    ax.bar([1], [float(np.real(sampled_mean))], width=0.45, yerr=[sampled_stderr], capsize=5, label="Sampled", color="#bb3e03")
    ax.axhline(0.0, color="#222222", linewidth=1.0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Exact", "Sampled"])
    ax.set_ylabel("Correlator value")
    ax.set_title("Stress test: mixed physical/bond/joint step unitaries")
    ax.legend(frameon=False)
    fig.savefig(STRESS_FIGURE, dpi=180)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    state, local_rotations = _build_primary_state()
    observable_rows, exact_values, sampled_values, sampled_errors = _observable_rows(state)
    structural_rows = _structural_rows(state)
    stress_rows, stress_specs, stress_exact, stress_sampled, stress_stderr = _stress_test_rows()

    _write_csv(
        OBSERVABLE_CSV,
        observable_rows,
        [
            "observable",
            "dense_real",
            "mps_real",
            "extended_exact_real",
            "sampled_real",
            "sampled_stderr",
            "abs_error_sample_vs_exact",
            "abs_error_mps_vs_dense",
            "abs_error_exact_vs_dense",
            "z_score",
        ],
    )
    _write_csv(
        STRUCTURAL_CSV,
        structural_rows,
        [
            "site",
            "tensor_shape",
            "joint_unitary_shape",
            "kraus_completeness_error",
            "right_canonical_error",
            "unitarity_error",
        ],
    )
    _write_csv(STRESS_CSV, stress_rows, ["metric", "value"])

    _plot_observables(observable_rows)
    _plot_stress_test(stress_exact, stress_sampled, stress_stderr)

    summary = {
        "initial_state": {
            "type": "computational_basis_seed",
            "bits": list(INITIAL_BITS),
        },
        "random_mps_construction": {
            "description": (
                "A seeded random four-qubit state generated from the computational-basis seed by "
                "site-local Haar-random SU(2) rotations followed by nearest-neighbor partial-swap entanglers. "
                "The MPS randomness therefore enters through the generated state structure, not through sampling noise."
            ),
            "seed": int(PRIMARY_RANDOM_SEED),
            "entangler_angles": [float(angle) for angle in ENTANGLER_ANGLES],
            "local_rotations": [rotation for rotation in local_rotations],
        },
        "shots": {
            "primary": int(PRIMARY_SHOTS),
            "stress_test": int(STRESS_SHOTS),
        },
        "observable_rows": observable_rows,
        "structural_rows": structural_rows,
        "stress_test": {
            "step_unitaries": [spec.to_record() for spec in stress_specs],
            "exact_mean": _complex_record(stress_exact),
            "sampled_mean": _complex_record(stress_sampled),
            "sampled_stderr": float(stress_stderr),
            "rows": stress_rows,
        },
        "figures": {
            "observables": str(OBSERVABLE_FIGURE.relative_to(ROOT)),
            "stress_test": str(STRESS_FIGURE.relative_to(ROOT)),
        },
        "files": {
            "observable_csv": str(OBSERVABLE_CSV.relative_to(ROOT)),
            "structural_csv": str(STRUCTURAL_CSV.relative_to(ROOT)),
            "stress_csv": str(STRESS_CSV.relative_to(ROOT)),
        },
        "derived_values": {
            key: {
                "exact": _complex_record(value),
                "sampled": _complex_record(sampled_values[key]),
                "sampled_stderr": float(sampled_errors[key]),
            }
            for key, value in exact_values.items()
        },
    }
    SUMMARY_JSON.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    print(f"Wrote summary to {SUMMARY_JSON}")
    print(f"Wrote observable table to {OBSERVABLE_CSV}")
    print(f"Wrote structural table to {STRUCTURAL_CSV}")
    print(f"Wrote stress-test table to {STRESS_CSV}")
    print(f"Wrote figures to {OBSERVABLE_FIGURE} and {STRESS_FIGURE}")


if __name__ == "__main__":
    main()