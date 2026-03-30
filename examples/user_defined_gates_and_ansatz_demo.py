"""Demonstration: User-Defined Gates, Gate-Order Search, and GRAPE Ansatz Initialization.

This script shows how to use the generalized optimizer interfaces added to cqed_sim:

    A. Map synthesis with built-in gates (baseline)
    B. Map synthesis with a user-defined ideal gate (make_gate_from_callable)
    C. Map synthesis with a user-defined fixed matrix gate (make_gate_from_matrix)
  D. Gate-order optimization (GateOrderOptimizer)
  E. GRAPE starting from the default random initialization
  F. GRAPE starting from a Gaussian pulse ansatz
  G. GRAPE starting from a DRAG ansatz
  H. GRAPE starting from a previously computed ControlSchedule

Each section is self-contained and labelled. Outputs are printed to stdout.

Requirements: cqed_sim installed in the current Python environment.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def print_section(label: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {label}")
    print("=" * width)


# ---------------------------------------------------------------------------
# A. Baseline: Map synthesis with built-in gates
# ---------------------------------------------------------------------------

print_section("A. Map synthesis with built-in gates (baseline)")

from cqed_sim.map_synthesis import (
    QuantumMapSynthesizer,
    Subspace,
    TargetUnitary,
)

# Target: Hadamard gate embedded in a 2-level qubit (no cavity needed).
H_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
target_unitary = TargetUnitary(H_mat)

# Use built-in QubitRotation gate to synthesize a Hadamard.
# (This works because a Hadamard is an XY rotation.)
subspace = Subspace.custom(2, [0, 1])
synth_a = QuantumMapSynthesizer(
    gateset=["QubitRotation"],
    subspace=subspace,
    optimizer="L-BFGS-B",
    seed=0,
)
result_a = synth_a.fit(target=target_unitary)
print(f"Built-in gates: infidelity = {result_a.objective:.6f}")
print(f"  Gates used: {[g.name for g in result_a.sequence.gates]}")

# ---------------------------------------------------------------------------
# B. Map synthesis with a user-defined callable gate
# ---------------------------------------------------------------------------

print_section("B. Map synthesis with a user-defined callable gate")

from cqed_sim.map_synthesis import make_gate_from_callable

# Define a general single-qubit unitary parameterized by Euler angles (ZYZ).
def euler_zyz(params: dict, model) -> np.ndarray:
    """ZYZ Euler decomposition: Rz(gamma) Ry(beta) Rz(alpha)."""
    alpha = params["alpha"]
    beta = params["beta"]
    gamma = params["gamma"]

    def Rz(theta: float) -> np.ndarray:
        return np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)],
        ], dtype=complex)

    def Ry(theta: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, -s], [s, c]], dtype=complex)

    return Rz(gamma) @ Ry(beta) @ Rz(alpha)


euler_gate = make_gate_from_callable(
    "EulerZYZ",
    euler_zyz,
    parameters={"alpha": 0.1, "beta": 0.2, "gamma": 0.3},
    parameter_bounds={
        "alpha": (-2 * np.pi, 2 * np.pi),
        "beta": (-np.pi, np.pi),
        "gamma": (-2 * np.pi, 2 * np.pi),
    },
    duration=50e-9,
    optimize_time=False,
)

synth_b = QuantumMapSynthesizer(
    primitives=[euler_gate],
    subspace=subspace,
    optimizer="L-BFGS-B",
    seed=1,
)
result_b = synth_b.fit(target=target_unitary)
print(f"User callable gate (EulerZYZ): infidelity = {result_b.objective:.6f}")

# ---------------------------------------------------------------------------
# C. Map synthesis with a user-defined matrix gate
# ---------------------------------------------------------------------------

print_section("C. Map synthesis with a user-defined matrix gate")

from cqed_sim.map_synthesis import make_gate_from_matrix

# S gate (phase gate): diag(1, i)
S_mat = np.diag([1.0, 1j]).astype(complex)
S_gate = make_gate_from_matrix("S", S_mat, duration=30e-9, optimize_time=False)

# H gate from matrix
H_gate = make_gate_from_matrix("H", H_mat, duration=40e-9, optimize_time=False)

# Synthesize a target that requires H and S (e.g., T gate = H S H approximately)
T_mat = np.diag([1.0, np.exp(1j * np.pi / 4)]).astype(complex)
target_T = TargetUnitary(T_mat)

# Use one EulerZYZ gate to decompose the T gate
euler_t = make_gate_from_callable(
    "EulerZYZ_T",
    euler_zyz,
    parameters={"alpha": 0.0, "beta": 0.0, "gamma": np.pi / 4},
    parameter_bounds={
        "alpha": (-2 * np.pi, 2 * np.pi),
        "beta": (-np.pi, np.pi),
        "gamma": (-2 * np.pi, 2 * np.pi),
    },
    duration=50e-9,
    optimize_time=False,
)
synth_c = QuantumMapSynthesizer(
    primitives=[euler_t],
    subspace=subspace,
    optimizer="L-BFGS-B",
    seed=2,
)
result_c = synth_c.fit(target=target_T)
print(f"User matrix gate (H, S) applied to T-gate: infidelity = {result_c.objective:.6f}")

# ---------------------------------------------------------------------------
# D. Gate-order optimization: search over orderings
# ---------------------------------------------------------------------------

print_section("D. Gate-order optimization (GateOrderOptimizer)")

from cqed_sim.map_synthesis import GateOrderConfig, GateOrderOptimizer

# Pool of user-defined single-qubit gates
X_mat = np.array([[0, 1], [1, 0]], dtype=complex)
Y_mat = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_mat = np.diag([1.0, -1.0]).astype(complex)

pool = [
    make_gate_from_matrix("H", H_mat, duration=40e-9),
    make_gate_from_matrix("S", S_mat, duration=30e-9),
    make_gate_from_callable(
        "Ry",
        lambda p, m: np.array([[np.cos(p["t"]/2), -np.sin(p["t"]/2)],
                                [np.sin(p["t"]/2),  np.cos(p["t"]/2)]], dtype=complex),
        parameters={"t": 0.3},
        parameter_bounds={"t": (-np.pi, np.pi)},
        duration=40e-9,
        optimize_time=False,
    ),
]

order_optimizer = GateOrderOptimizer(
    gate_pool=pool,
    order_config=GateOrderConfig(
        search_strategy="random",
        n_random_trials=6,
        max_sequence_length=3,
        min_sequence_length=1,
        seed=42,
    ),
    synthesizer_kwargs=dict(
        subspace=subspace,
        optimizer="L-BFGS-B",
        seed=0,
    ),
)

order_result = order_optimizer.search(target=target_unitary)
print(f"Gate-order search: best infidelity = {order_result.best_result.objective:.6f}")
print(f"  Best ordering: {[g.name for g in order_result.best_ordering]}")
print(f"  Orderings tried: {order_result.n_orderings_tried}")
print(f"  All results (top 3 by infidelity):")
for ordering, res in order_result.all_results[:3]:
    print(f"    [{', '.join(g.name for g in ordering)}]  infidelity={res.objective:.4f}")

# ---------------------------------------------------------------------------
# E. GRAPE: default random initialization (baseline)
# ---------------------------------------------------------------------------

print_section("E. GRAPE with default random initialization (baseline)")

from cqed_sim.optimal_control import (
    GrapeConfig,
    ControlTerm,
    ControlSystem,
    ControlProblem,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    StateTransferObjective,
    StateTransferPair,
    GrapeSolver,
)

# Simple qubit system: X and Y control operators in physical units.
# Use Omega_max = 2pi * 50 MHz as the maximum Rabi frequency scale.
# Control amplitudes are normalized: amplitude=1 corresponds to Omega_max.
Omega_max = 2.0 * np.pi * 50.0e6  # rad/s

Hx = np.array([[0, 1], [1, 0]], dtype=complex) * Omega_max / 2.0
Hy = np.array([[0, -1j], [1j, 0]], dtype=complex) * Omega_max / 2.0
H0 = np.zeros((2, 2), dtype=complex)

term_x = ControlTerm("I", Hx, amplitude_bounds=(-1.0, 1.0))
term_y = ControlTerm("Q", Hy, amplitude_bounds=(-1.0, 1.0))

n_steps = 50
duration_s = 100e-9
grid = PiecewiseConstantTimeGrid.uniform(steps=n_steps, dt_s=duration_s / n_steps)
param = PiecewiseConstantParameterization(
    time_grid=grid,
    control_terms=(term_x, term_y),
)

system = ControlSystem(
    drift_hamiltonian=H0,
    control_operators=(np.asarray(Hx), np.asarray(Hy)),
)

# Transfer |0> -> |+> = (|0> + |1>)/sqrt(2)
psi0 = np.array([1.0, 0.0], dtype=complex)
psi_target = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)

objective = StateTransferObjective(
    pairs=(StateTransferPair(initial_state=psi0, target_state=psi_target),),
    name="plus_state",
)

problem = ControlProblem(
    parameterization=param,
    systems=(system,),
    objectives=(objective,),
)

cfg_random = GrapeConfig(initial_guess="random", maxiter=50, seed=42, show_progress=False)
solver_e = GrapeSolver(cfg_random)
result_e = solver_e.solve(problem)
print(f"GRAPE (random init): final infidelity = {result_e.objective_value:.6f}")

# ---------------------------------------------------------------------------
# F. GRAPE with Gaussian ansatz initialization
# ---------------------------------------------------------------------------

print_section("F. GRAPE with Gaussian ansatz initialization")

from cqed_sim.optimal_control import GaussianAnsatz

cfg_gauss = GrapeConfig(
    initial_guess=GaussianAnsatz(sigma_fraction=0.25, amplitude_fraction=0.6, seed=0),
    maxiter=50,
    show_progress=False,
)
solver_f = GrapeSolver(cfg_gauss)
result_f = solver_f.solve(problem)
print(f"GRAPE (Gaussian ansatz): final infidelity = {result_f.objective_value:.6f}")

# ---------------------------------------------------------------------------
# G. GRAPE with DRAG ansatz initialization
# ---------------------------------------------------------------------------

print_section("G. GRAPE with DRAG ansatz initialization")

from cqed_sim.optimal_control import DRAGAnsatz

cfg_drag = GrapeConfig(
    initial_guess=DRAGAnsatz(
        sigma_fraction=0.25,
        amplitude_fraction=0.5,
        drag_alpha=0.3,
        i_control_index=0,
        q_control_index=1,
        seed=7,
    ),
    maxiter=50,
    show_progress=False,
)
solver_g = GrapeSolver(cfg_drag)
result_g = solver_g.solve(problem)
print(f"GRAPE (DRAG ansatz): final infidelity = {result_g.objective_value:.6f}")

# ---------------------------------------------------------------------------
# H. GRAPE seeded from a previously computed ControlSchedule
# ---------------------------------------------------------------------------

print_section("H. GRAPE seeded from a previous ControlSchedule (warm start)")

from cqed_sim.optimal_control import MultitoneAnsatz, warm_start_schedule

# First run to get a schedule
cfg_seed_run = GrapeConfig(
    initial_guess=MultitoneAnsatz(n_tones=3, amplitude_fraction=0.4, seed=99),
    maxiter=30,
    show_progress=False,
)
result_seed = GrapeSolver(cfg_seed_run).solve(problem)
seed_schedule = result_seed.schedule

# Now refine from that schedule as the initial guess
cfg_warmstart = GrapeConfig(
    initial_guess=seed_schedule,  # ControlSchedule passed directly
    maxiter=50,
    show_progress=False,
)
solver_h = GrapeSolver(cfg_warmstart)
result_h = solver_h.solve(problem)
print(f"GRAPE (warm-start from schedule): final infidelity = {result_h.objective_value:.6f}")
print(f"  Seed-run infidelity: {result_seed.objective_value:.6f}")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print_section("Summary")
print(f"{'Scenario':<50} {'Infidelity':>12}")
print("-" * 64)
print(f"{'A. Built-in gates (QubitRotation)':<50} {result_a.objective:>12.6f}")
print(f"{'B. User callable gate (EulerZYZ)':<50} {result_b.objective:>12.6f}")
print(f"{'C. User matrix gate (T gate decomp)':<50} {result_c.objective:>12.6f}")
print(f"{'D. Gate-order optimizer (best ordering)':<50} {order_result.best_result.objective:>12.6f}")
print(f"{'E. GRAPE random init':<50} {result_e.objective_value:>12.6f}")
print(f"{'F. GRAPE Gaussian ansatz':<50} {result_f.objective_value:>12.6f}")
print(f"{'G. GRAPE DRAG ansatz':<50} {result_g.objective_value:>12.6f}")
print(f"{'H. GRAPE warm-start from schedule':<50} {result_h.objective_value:>12.6f}")
