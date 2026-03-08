from __future__ import annotations

from typing import Any

import numpy as np
import qutip as qt

from cqed_sim.operators.basic import joint_basis_state
from cqed_sim.sim.extractors import conditioned_bloch_xyz


def _available_max_n(track: dict[str, Any], max_n: int) -> int:
    n_cav_dim = int(track["snapshots"][0]["rho_c"].dims[0][0])
    return min(int(max_n), n_cav_dim - 1)


def _segmentwise_unwrap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    unwrapped = arr.copy()
    valid = np.isfinite(arr)
    if np.count_nonzero(valid) <= 1:
        return unwrapped
    valid_idx = np.where(valid)[0]
    split_points = np.where(np.diff(valid_idx) > 1)[0] + 1
    for segment in np.split(valid_idx, split_points):
        unwrapped[segment] = np.unwrap(arr[segment])
    return unwrapped


def _complex_nan(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan + 1j * np.nan, dtype=np.complex128)


def _finite_complex(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.complex128)
    return np.isfinite(arr.real) & np.isfinite(arr.imag)


def _validated_bloch(x: float, y: float, z: float, tol: float = 1.0e-6) -> tuple[float, float, float]:
    bloch = np.asarray([x, y, z], dtype=float)
    if np.any(np.abs(bloch) > 1.0 + tol):
        raise AssertionError(f"Conditioned Bloch vector exceeded physical bounds: {bloch}")
    bloch = np.clip(bloch, -1.0, 1.0)
    return float(bloch[0]), float(bloch[1]), float(bloch[2])


PHASE_FAMILY_SPECS: dict[str, dict[str, str]] = {
    "ground": {
        "target_qubit": "g",
        "target_template": "|g,{n}>",
        "definition": "arg(<g,n|rho|g,0>) = arg(c_{g,n} c_{g,0}^*)",
    },
    "excited": {
        "target_qubit": "e",
        "target_template": "|e,{n}>",
        "definition": "arg(<e,n|rho|g,0>) = arg(c_{e,n} c_{g,0}^*)",
    },
}


def fock_resolved_bloch_diagnostics(
    track: dict[str, Any],
    max_n: int,
    probability_threshold: float = 1.0e-8,
) -> dict[str, Any]:
    available_max_n = _available_max_n(track, max_n)
    n_values = np.arange(available_max_n + 1, dtype=int)
    n_steps = len(track["snapshots"])
    x = np.full((n_values.size, n_steps), np.nan, dtype=float)
    y = np.full((n_values.size, n_steps), np.nan, dtype=float)
    z = np.full((n_values.size, n_steps), np.nan, dtype=float)
    p = np.zeros((n_values.size, n_steps), dtype=float)
    valid = np.zeros((n_values.size, n_steps), dtype=bool)

    for step_idx, snapshot in enumerate(track["snapshots"]):
        state = snapshot["state"]
        for row_idx, n in enumerate(n_values):
            x_n, y_n, z_n, p_n, is_valid = conditioned_bloch_xyz(state, n=int(n), fallback="nan")
            p[row_idx, step_idx] = float(p_n)
            if is_valid and p_n >= probability_threshold:
                x_n, y_n, z_n = _validated_bloch(x_n, y_n, z_n)
                x[row_idx, step_idx] = float(x_n)
                y[row_idx, step_idx] = float(y_n)
                z[row_idx, step_idx] = float(z_n)
                valid[row_idx, step_idx] = True

    return {
        "case": track["case"],
        "indices": np.asarray(track["indices"], dtype=int),
        "n_values": n_values,
        "x": x,
        "y": y,
        "z": z,
        "probability": p,
        "valid": valid,
        "top_labels": [snapshot["top_label"] for snapshot in track["snapshots"]],
        "gate_types": [snapshot["gate_type"] for snapshot in track["snapshots"]],
        "probability_threshold": float(probability_threshold),
    }


def _coherence_element(rho: qt.Qobj, bra: qt.Qobj, ket: qt.Qobj) -> complex:
    value = bra.dag() * rho * ket
    if isinstance(value, qt.Qobj):
        return complex(value.full()[0, 0])
    return complex(value)


def _phase_family_diagnostics(
    track: dict[str, Any],
    max_n: int,
    family: str,
    probability_threshold: float = 1.0e-8,
    unwrap: bool = False,
    coherence_threshold: float | None = None,
) -> dict[str, Any]:
    if family not in PHASE_FAMILY_SPECS:
        raise KeyError(f"Unsupported phase family '{family}'. Expected one of {sorted(PHASE_FAMILY_SPECS)}.")
    available_max_n = _available_max_n(track, max_n)
    n_cav_dim = int(track["snapshots"][0]["rho_c"].dims[0][0])
    n_values = np.arange(available_max_n + 1, dtype=int)
    n_steps = len(track["snapshots"])
    threshold = float(probability_threshold if coherence_threshold is None else coherence_threshold)
    phase_wrapped = np.full((n_values.size, n_steps), np.nan, dtype=float)
    phase_continuous = np.full((n_values.size, n_steps), np.nan, dtype=float)
    coherence_mag = np.zeros((n_values.size, n_steps), dtype=float)
    probability = np.zeros((n_values.size, n_steps), dtype=float)
    reference_population = np.zeros(n_steps, dtype=float)
    phasor = _complex_nan((n_values.size, n_steps))
    valid = np.zeros((n_values.size, n_steps), dtype=bool)
    reference_ket = joint_basis_state(n_cav_dim, "g", 0)
    spec = PHASE_FAMILY_SPECS[family]
    target_kets = [joint_basis_state(n_cav_dim, spec["target_qubit"], int(n)) for n in n_values]

    for step_idx, snapshot in enumerate(track["snapshots"]):
        state = snapshot["state"]
        rho = state if state.isoper else state.proj()
        p_ref = float(np.real(_coherence_element(rho, reference_ket, reference_ket)))
        reference_population[step_idx] = p_ref
        for row_idx, (n, target_ket) in enumerate(zip(n_values, target_kets, strict=False)):
            p_target = float(np.real(_coherence_element(rho, target_ket, target_ket)))
            probability[row_idx, step_idx] = p_target
            if p_ref < probability_threshold or p_target < probability_threshold:
                continue
            coherence = _coherence_element(rho, target_ket, reference_ket)
            coherence_mag[row_idx, step_idx] = abs(coherence)
            if abs(coherence) < threshold:
                continue
            unit_phasor = coherence / (abs(coherence) + 1.0e-300)
            phasor[row_idx, step_idx] = unit_phasor
            phase_wrapped[row_idx, step_idx] = float(np.angle(unit_phasor))
            valid[row_idx, step_idx] = True

    if unwrap:
        for row_idx in range(phase_wrapped.shape[0]):
            phase_continuous[row_idx] = _segmentwise_unwrap(phase_wrapped[row_idx])
        phase_mode = "unwrapped"
    else:
        phase_continuous = phase_wrapped.copy()
        phase_mode = "wrapped"

    return {
        "family": family,
        "case": track["case"],
        "indices": np.asarray(track["indices"], dtype=int),
        "n_values": n_values,
        "phase": phase_continuous if unwrap else phase_wrapped,
        "phase_wrapped": phase_wrapped,
        "phase_continuous": phase_continuous,
        "phasor": phasor,
        "coherence_magnitude": coherence_mag,
        "probability": probability,
        "target_population": probability,
        "reference_population": reference_population,
        "valid": valid,
        "top_labels": [snapshot["top_label"] for snapshot in track["snapshots"]],
        "gate_types": [snapshot["gate_type"] for snapshot in track["snapshots"]],
        "phase_mode": phase_mode,
        "probability_threshold": float(probability_threshold),
        "coherence_threshold": float(threshold),
        "phase_reference_label": "|g,0>",
        "phase_target_template": spec["target_template"],
        "relative_phase_definition": spec["definition"],
    }


def relative_phase_family_diagnostics(
    track: dict[str, Any],
    max_n: int,
    probability_threshold: float = 1.0e-8,
    unwrap: bool = False,
    coherence_threshold: float | None = None,
) -> dict[str, Any]:
    families = {
        family: _phase_family_diagnostics(
            track,
            max_n=max_n,
            family=family,
            probability_threshold=probability_threshold,
            unwrap=unwrap,
            coherence_threshold=coherence_threshold,
        )
        for family in ("ground", "excited")
    }
    first = families["ground"]
    return {
        "case": first["case"],
        "indices": np.asarray(first["indices"], dtype=int),
        "n_values": np.asarray(first["n_values"], dtype=int),
        "top_labels": list(first["top_labels"]),
        "gate_types": list(first["gate_types"]),
        "phase_mode": first["phase_mode"],
        "probability_threshold": float(probability_threshold),
        "coherence_threshold": float(first["coherence_threshold"]),
        "phase_reference_label": "|g,0>",
        "families": families,
        "relative_phase_definitions": {family: diag["relative_phase_definition"] for family, diag in families.items()},
    }


def conditional_phase_diagnostics(
    track: dict[str, Any],
    max_n: int,
    probability_threshold: float = 1.0e-8,
    unwrap: bool = False,
    coherence_threshold: float | None = None,
) -> dict[str, Any]:
    return _phase_family_diagnostics(
        track,
        max_n=max_n,
        family="excited",
        probability_threshold=probability_threshold,
        unwrap=unwrap,
        coherence_threshold=coherence_threshold,
    )


def relative_phase_debug_values(
    state: qt.Qobj,
    max_n: int,
    probability_threshold: float = 1.0e-8,
    coherence_threshold: float | None = None,
) -> dict[str, Any]:
    rho = state if state.isoper else state.proj()
    n_cav_dim = int(rho.dims[0][0])
    available_max_n = min(int(max_n), n_cav_dim - 1)
    threshold = float(probability_threshold if coherence_threshold is None else coherence_threshold)
    reference_ket = joint_basis_state(n_cav_dim, "g", 0)
    p_ref = float(np.real(_coherence_element(rho, reference_ket, reference_ket)))
    c_g0 = np.nan + 1j * np.nan
    if p_ref >= probability_threshold:
        c_g0 = complex(np.sqrt(p_ref), 0.0)

    levels: list[dict[str, Any]] = []
    for n in range(available_max_n + 1):
        ground_ket = joint_basis_state(n_cav_dim, "g", n)
        excited_ket = joint_basis_state(n_cav_dim, "e", n)
        p_gn = float(np.real(_coherence_element(rho, ground_ket, ground_ket)))
        p_en = float(np.real(_coherence_element(rho, excited_ket, excited_ket)))
        coherence_gn = _coherence_element(rho, ground_ket, reference_ket)
        coherence_en = _coherence_element(rho, excited_ket, reference_ket)

        def _gauge_amplitude(p_target: float, coherence: complex) -> tuple[complex, float]:
            if p_ref < probability_threshold or p_target < probability_threshold or abs(coherence) < threshold or not np.isfinite(c_g0.real):
                return np.nan + 1j * np.nan, np.nan
            return coherence / c_g0, float(np.angle(coherence))

        c_gn, phase_ground = _gauge_amplitude(p_gn, coherence_gn)
        c_en, phase_excited = _gauge_amplitude(p_en, coherence_en)
        levels.append(
            {
                "n": int(n),
                "p_g0": p_ref,
                "p_gn": p_gn,
                "p_en": p_en,
                "coherence_gn_g0": coherence_gn,
                "coherence_en_g0": coherence_en,
                "c_g0_gauge": c_g0,
                "c_gn_gauge": c_gn,
                "c_en_gauge": c_en,
                "ground_phase_rad": phase_ground,
                "excited_phase_rad": phase_excited,
            }
        )

    return {
        "basis_ordering": "|q>_qubit tensor |n>_cavity = qt.tensor(qubit, cavity)",
        "reference_label": "|g,0>",
        "ground_target_template": "|g,n>",
        "excited_target_template": "|e,n>",
        "reference_population": p_ref,
        "c_g0_gauge": c_g0,
        "levels": levels,
    }


def wrapped_phase_error(simulated_phase: dict[str, Any] | np.ndarray, ideal_phase: dict[str, Any] | np.ndarray) -> dict[str, np.ndarray] | np.ndarray:
    if isinstance(simulated_phase, dict) and isinstance(ideal_phase, dict) and "families" in simulated_phase and "families" in ideal_phase:
        families = tuple(family for family in ("ground", "excited") if family in simulated_phase["families"] and family in ideal_phase["families"])
        return {
            family: wrapped_phase_error(simulated_phase["families"][family], ideal_phase["families"][family])
            for family in families
        }
    if isinstance(simulated_phase, dict) and isinstance(ideal_phase, dict):
        sim_phasor = np.asarray(simulated_phase.get("phasor"), dtype=np.complex128)
        ideal_phasor = np.asarray(ideal_phase.get("phasor"), dtype=np.complex128)
        if sim_phasor.shape != ideal_phasor.shape:
            raise ValueError(f"Phase phasor arrays must have matching shapes, got {sim_phasor.shape} and {ideal_phasor.shape}.")
        error = np.full(sim_phasor.shape, np.nan, dtype=float)
        valid = (
            _finite_complex(sim_phasor)
            & _finite_complex(ideal_phasor)
            & np.isfinite(np.asarray(simulated_phase.get("phase_wrapped"), dtype=float))
            & np.isfinite(np.asarray(ideal_phase.get("phase_wrapped"), dtype=float))
        )
        if np.any(valid):
            ratio = sim_phasor[valid] * np.conjugate(ideal_phasor[valid])
            ratio_valid = np.abs(ratio) > 1.0e-15
            wrapped = np.full(ratio.shape, np.nan, dtype=float)
            wrapped[ratio_valid] = np.angle(ratio[ratio_valid])
            error[valid] = wrapped
        return error

    simulated = np.asarray(simulated_phase, dtype=float)
    ideal = np.asarray(ideal_phase, dtype=float)
    if simulated.shape != ideal.shape:
        raise ValueError(f"Phase arrays must have matching shapes, got {simulated.shape} and {ideal.shape}.")
    valid = np.isfinite(simulated) & np.isfinite(ideal)
    error = np.full(simulated.shape, np.nan, dtype=float)
    if np.any(valid):
        wrapped = (simulated[valid] - ideal[valid] + np.pi) % (2.0 * np.pi) - np.pi
        error[valid] = wrapped
    return error
