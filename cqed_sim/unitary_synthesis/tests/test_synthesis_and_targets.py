from __future__ import annotations

import numpy as np
import pytest

from cqed_sim.unitary_synthesis import Subspace, UnitarySynthesizer, make_target
from cqed_sim.unitary_synthesis.metrics import subspace_unitary_fidelity
from cqed_sim.unitary_synthesis.targets import load_mps_reference, make_mps_like_target


def _cluster_u1_expected(n_match: int) -> np.ndarray:
    n_cav = n_match + 1
    cz = np.diag([1.0, 1.0, 1.0, -1.0]).astype(np.complex128)
    sw = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.complex128,
    )
    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2.0)
    u_small_qc = sw @ cz @ np.kron(hadamard, np.eye(2, dtype=np.complex128))
    u_qc = np.eye(2 * n_cav, dtype=np.complex128)
    indices = [n_cav * iq + jb for iq in range(2) for jb in range(2)]
    for old_row, new_row in enumerate(indices):
        for old_col, new_col in enumerate(indices):
            u_qc[new_row, new_col] = u_small_qc[old_row, old_col]
    return u_qc


def test_f1_synthesizes_easy_target_on_subspace_ideal() -> None:
    n_match = 2
    sub = Subspace.qubit_cavity_block(n_match=n_match)
    target = make_target("easy", n_match=n_match)
    synth = UnitarySynthesizer(
        subspace=sub,
        backend="ideal",
        gateset=["SQR", "SNAP"],
        optimize_times=False,
        leakage_weight=5.0,
        seed=11,
    )
    result = synth.fit(target=target, multistart=2, maxiter=250)
    assert result.report["metrics"]["fidelity"] > 0.999
    assert result.report["metrics"]["leakage_worst"] < 1e-10


def test_f2_junk_outside_subspace_is_ignored() -> None:
    n_match = 2
    sub = Subspace.qubit_cavity_block(n_match=n_match, n_cav=5)
    target_sub = make_target("easy", n_match=n_match)
    target_full = np.eye(sub.full_dim, dtype=np.complex128)
    idx = np.asarray(sub.indices)
    target_full[np.ix_(idx, idx)] = target_sub
    rng = np.random.default_rng(9)
    outside = [i for i in range(sub.full_dim) if i not in sub.indices]
    x = rng.standard_normal((len(outside), len(outside))) + 1j * rng.standard_normal((len(outside), len(outside)))
    q, _ = np.linalg.qr(x)
    target_full[np.ix_(outside, outside)] = q

    synth = UnitarySynthesizer(subspace=sub, backend="ideal", gateset=["SQR", "SNAP"], optimize_times=False, seed=12)
    result = synth.fit(target=target_full, multistart=1, maxiter=250)
    assert result.report["metrics"]["fidelity"] > 0.999


def test_f3_leakage_penalty_reduces_leakage() -> None:
    n_match = 2
    sub = Subspace.qubit_cavity_block(n_match=n_match, n_cav=8)
    target = make_target("easy", n_match=n_match)

    no_penalty = UnitarySynthesizer(
        subspace=sub,
        backend="pulse",
        gateset=["Displacement", "SQR", "SNAP"],
        optimize_times=True,
        leakage_weight=0.0,
        seed=21,
    ).fit(target=target, multistart=1, maxiter=120)

    with_penalty = UnitarySynthesizer(
        subspace=sub,
        backend="pulse",
        gateset=["Displacement", "SQR", "SNAP"],
        optimize_times=True,
        leakage_weight=10.0,
        seed=21,
    ).fit(target=target, multistart=1, maxiter=120)

    assert with_penalty.report["metrics"]["leakage_worst"] <= no_penalty.report["metrics"]["leakage_worst"] + 1e-6


def test_f4_robustness_and_phase_wrap_regression() -> None:
    n_match = 2
    sub = Subspace.qubit_cavity_block(n_match=n_match)
    target = make_target("easy", n_match=n_match)
    synth = UnitarySynthesizer(subspace=sub, backend="ideal", gateset=["SQR", "SNAP"], optimize_times=False, seed=30)
    fit = synth.fit(target=target, multistart=1, maxiter=180)

    u0 = fit.simulation.subspace_operator
    base_f = subspace_unitary_fidelity(u0, sub.restrict_operator(target))

    # +/- pi wrap on one SNAP phase should not catastrophically break metric reporting.
    gate = fit.sequence.gates[-1]
    if hasattr(gate, "phases"):
        phases = np.asarray(gate.phases, dtype=float)
        phases[0] = phases[0] + np.pi
        gate.phases = list(phases)
    u1 = fit.sequence.unitary("ideal")
    f1 = subspace_unitary_fidelity(sub.restrict_operator(u1), sub.restrict_operator(target))
    assert f1 <= base_f
    assert f1 >= 0.0


def test_f5_known_mps_targets_ghz_cluster() -> None:
    # The Noah reference cluster/GHZ targets are defined on a bond-dimension-2 memory.
    # A small n_match keeps the regression fast while still checking the exact convention.
    n_match = 1
    sub = Subspace.qubit_cavity_block(n_match=n_match)
    for name in ["ghz", "cluster"]:
        target = make_target(name, n_match=n_match, variant="mps")
        synth = UnitarySynthesizer(
            subspace=sub,
            backend="ideal",
            gateset=["QubitRotation", "SQR", "Displacement"] * 4,
            optimize_times=False,
            leakage_weight=1.0,
            seed=40,
        )
        result = synth.fit(target=target, init_guess="random", multistart=2, maxiter=30)
        assert result.report["metrics"]["fidelity"] > 0.99


def test_f5_reference_compare_if_external_available() -> None:
    n_match = 3
    for kind in ["ghz", "cluster"]:
        ref = load_mps_reference(kind, n_match)
        if ref is None:
            pytest.skip("External MPS reference directory not available in this environment")
        local = make_mps_like_target(kind, n_match)
        phase = np.trace(ref.conj().T @ local)
        if abs(phase) > 0:
            local = np.exp(-1j * np.angle(phase)) * local
        assert np.linalg.norm(ref - local) / np.linalg.norm(ref) < 0.25


def test_cluster_mps_target_matches_noah_u1_convention() -> None:
    n_match = 2
    target = make_target("cluster", n_match=n_match, variant="mps")
    expected = _cluster_u1_expected(n_match)
    phase = np.trace(expected.conj().T @ target)
    if abs(phase) > 0:
        target = np.exp(-1j * np.angle(phase)) * target
    assert np.linalg.norm(target - expected) < 1e-12
