"""Example 03: GRAPE output — ideal vs hardware-aware evaluation (Mode A).

Demonstrates Mode A hardware-aware evaluation:
  1. Run GRAPE to find an optimal control schedule on the ideal system.
  2. Evaluate the resulting waveforms through a HardwareContext (Mode A
     postprocessing via :func:`~cqed_sim.control.postprocess_grape_waveforms`).
  3. Compare fidelity before and after hardware distortion to quantify the
     mismatch introduced by realistic control lines.

The system: a two-level qubit (σ_z / 2 drift, σ_x / 2 and σ_y / 2 drives)
targeted for a π rotation (X gate).

Hardware context: qubit drive line with 10% gain reduction and 2-sample delay.

Run time: a few seconds (short GRAPE optimisation).
"""
from __future__ import annotations

import numpy as np

import qutip as qt

from cqed_sim.control import ControlLine, HardwareContext, postprocess_grape_waveforms
from cqed_sim.optimal_control import (
    ControlProblem,
    ControlSystem,
    ControlTerm,
    GrapeConfig,
    PiecewiseConstantParameterization,
    PiecewiseConstantTimeGrid,
    UnitaryObjective,
    solve_grape,
    zero_control_schedule,
    resolve_control_schedule,
)
from cqed_sim.optimal_control.hardware import GainHardwareMap, DelayHardwareMap

# ── System definition ────────────────────────────────────────────────────────

dim = 2
sigma_z = qt.sigmaz().full()      # [[1, 0], [0, -1]]
sigma_x = qt.sigmax().full()      # [[0, 1], [1, 0]]
sigma_y = qt.sigmay().full()      # [[0,-i], [i, 0]]

H0 = 0.5 * sigma_z                # qubit splitting (set to 0 for resonant drive)
H0[:] = 0                         # on-resonance simplification

OMEGA_MAX = 2 * np.pi * 10e6      # 10 MHz Rabi frequency limit (rad/s)
N_STEPS   = 20
DT_S      = 5e-9                  # 5 ns per step → 100 ns total gate

time_grid = PiecewiseConstantTimeGrid.uniform(steps=N_STEPS, dt_s=DT_S)

ctrl_x = ControlTerm(
    name="drive_x",
    operator=sigma_x / 2,
    amplitude_bounds=(-OMEGA_MAX, OMEGA_MAX),
    export_channel="qubit",
    quadrature="I",
)
ctrl_y = ControlTerm(
    name="drive_y",
    operator=sigma_y / 2,
    amplitude_bounds=(-OMEGA_MAX, OMEGA_MAX),
    export_channel="qubit",
    quadrature="Q",
)

system = ControlSystem(
    drift_operators=(H0,),
    control_terms=(ctrl_x, ctrl_y),
    hilbert_dim=dim,
)

# Target: X gate (π rotation around x)
target_unitary = np.array([[0, 1], [1, 0]], dtype=complex)  # -i σ_x normalized
target_unitary = -1j * sigma_x / np.linalg.norm(-1j * sigma_x)  # σ_x up to global phase
# Simpler: just use -i*σ_x
target_unitary = -1j * qt.sigmax().full()

parameterization = PiecewiseConstantParameterization(
    control_terms=(ctrl_x, ctrl_y),
    time_grid=time_grid,
)
objective = UnitaryObjective(
    target_unitary=target_unitary,
    system_index=0,
)
problem = ControlProblem(
    parameterization=parameterization,
    systems=(system,),
    objectives=(objective,),
)

# ── Run GRAPE (ideal system) ──────────────────────────────────────────────────

config = GrapeConfig(
    max_iterations=100,
    convergence_abs=1e-7,
    apply_hardware_in_forward_model=True,
    verbose=False,
)
np.random.seed(42)
init_schedule = zero_control_schedule(problem)

print("=== Example 03: GRAPE hardware comparison (Mode A) ===")
print(f"Optimising {N_STEPS}-step, {N_STEPS * DT_S * 1e9:.0f} ns X gate...")

result = solve_grape(problem, init_schedule, config=config)
print(f"GRAPE: success={result.success}, objective={result.objective_value:.6f}")

ideal_infidelity = 1.0 - result.objective_value


# ── Mode A: apply hardware context to GRAPE output ───────────────────────────

ctx = HardwareContext(
    lines={
        "qubit": ControlLine(
            name="qubit",
            transfer_maps=(
                GainHardwareMap(gain=0.90),           # 10% amplitude reduction
                DelayHardwareMap(delay_samples=2),    # 2-sample delay
            ),
            calibration_gain=1.0,
            programmed_unit="rad/s",
            device_unit="rad/s",
            coefficient_unit="rad/s",
            operator_label="σ_x/2 (I) and σ_y/2 (Q)",
            frame="rotating_qubit",
        )
    }
)

# Resolve the optimised schedule to get physical waveforms
resolved = resolve_control_schedule(problem, result.schedule)

# Postprocess through hardware context (Mode A)
transformed_values = postprocess_grape_waveforms(
    ctx,
    resolved.physical_values,
    problem.control_terms,
    dt=DT_S,
)

# Evaluate fidelity with hardware-distorted waveforms
from cqed_sim.optimal_control.propagators import build_propagation_data
from cqed_sim.optimal_control.objectives import UnitaryObjective as _UObj


# Build propagation data with distorted waveforms
from cqed_sim.optimal_control.grape import GrapeSolver

solver = GrapeSolver(problem, config=config)
# Evaluate objective with hardware-distorted waveforms
from cqed_sim.optimal_control.propagators import build_propagation_data as _bpd

prop_data = _bpd(
    system,
    physical_values=np.asarray(transformed_values, dtype=float),
    time_grid=time_grid,
)
hw_unitary = prop_data.propagator
hw_fidelity = float(
    abs(np.trace(target_unitary.conj().T @ hw_unitary)) ** 2
    / (dim ** 2)
)
hw_infidelity = 1.0 - hw_fidelity

print(f"\nIdeal infidelity   : {ideal_infidelity:.4e}")
print(f"Hardware infidelity: {hw_infidelity:.4e}")
print(f"Hardware degrades infidelity by factor: {hw_infidelity / max(ideal_infidelity, 1e-15):.1f}x")


# ── Mode B: pass hardware model to GRAPE for in-loop optimisation ─────────────

print("\n--- Mode B: optimising through hardware ---")
hardware_model = ctx.as_hardware_model()
problem_with_hw = ControlProblem(
    parameterization=parameterization,
    systems=(system,),
    objectives=(objective,),
    hardware_model=hardware_model,
)

np.random.seed(42)
result_hw = solve_grape(problem_with_hw, init_schedule, config=config)
print(f"GRAPE (Mode B): success={result_hw.success}, "
      f"objective={result_hw.objective_value:.6f}")
modeb_infidelity = 1.0 - result_hw.objective_value
print(f"Mode B infidelity: {modeb_infidelity:.4e}")

print(f"\nSummary:")
print(f"  Ideal GRAPE (no hardware)            : 1 - F = {ideal_infidelity:.4e}")
print(f"  Ideal GRAPE + Mode A hardware effect : 1 - F = {hw_infidelity:.4e}")
print(f"  GRAPE optimised through hardware (B) : 1 - F = {modeb_infidelity:.4e}")
print("\nExample 03 complete.")
