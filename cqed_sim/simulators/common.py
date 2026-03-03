from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.io.gates import Gate
from cqed_sim.observables.bloch import bloch_xyz_from_joint, cavity_moments, reduced_cavity_state, reduced_qubit_state
from cqed_sim.observables.wigner import cavity_wigner
from cqed_sim.operators.basic import as_dm, build_qubit_state, purity
from cqed_sim.sim.noise import NoiseSpec


def hz_to_rad_s(hz: float) -> float:
    return float(2.0 * np.pi * hz)


def ns_to_s(ns: float | None) -> float | None:
    return None if ns is None else float(ns) * 1.0e-9


def build_initial_state(config: Mapping[str, Any], n_cav_dim: int | None = None) -> qt.Qobj:
    n_cav_dim = int(config["n_cav_dim"] if n_cav_dim is None else n_cav_dim)
    cavity_kind = str(config["initial_cavity_kind"]).lower()
    if cavity_kind == "fock":
        cavity_state = qt.basis(n_cav_dim, int(config["initial_cavity_fock"]))
    elif cavity_kind == "coherent":
        alpha_cfg = config["initial_cavity_alpha"]
        cavity_state = qt.coherent(n_cav_dim, complex(alpha_cfg["re"], alpha_cfg["im"]))
    elif cavity_kind == "custom_superposition":
        amplitudes = np.asarray(config["initial_cavity_amplitudes"], dtype=np.complex128)
        if amplitudes.ndim != 1:
            raise ValueError("initial_cavity_amplitudes must be one-dimensional.")
        if amplitudes.size > n_cav_dim:
            amplitudes = amplitudes[:n_cav_dim]
        if amplitudes.size < n_cav_dim:
            amplitudes = np.pad(amplitudes, (0, n_cav_dim - amplitudes.size))
        cavity_state = qt.Qobj(amplitudes.reshape((-1, 1)), dims=[[n_cav_dim], [1]]).unit()
    else:
        raise ValueError(f"Unsupported initial cavity kind '{cavity_kind}'.")
    return qt.tensor(cavity_state, build_qubit_state(str(config["initial_qubit"])))


def choose_t2_ns(config: Mapping[str, Any]) -> float:
    source = str(config["t2_source"]).lower()
    if source == "echo":
        return float(config["qb_T2_echo_ns"])
    if source != "ramsey":
        raise ValueError(f"Unsupported t2_source '{config['t2_source']}'.")
    return float(config["qb_T2_ramsey_ns"])


def derive_tphi_seconds(t1_ns: float | None, t2_ns: float | None) -> float | None:
    if t2_ns is None:
        return None
    t2_s = ns_to_s(t2_ns)
    if t1_ns is None:
        return t2_s
    t1_s = ns_to_s(t1_ns)
    inv_tphi = max(0.0, 1.0 / t2_s - 1.0 / (2.0 * t1_s))
    return None if inv_tphi <= 0.0 else 1.0 / inv_tphi


def build_model(config: Mapping[str, Any]) -> DispersiveTransmonCavityModel:
    chi_higher = tuple(
        value
        for value in (
            hz_to_rad_s(float(config["st_chi2_hz"])),
            hz_to_rad_s(float(config["st_chi3_hz"])),
        )
        if value != 0.0
    )
    kerr_higher = tuple(value for value in (hz_to_rad_s(float(config["st_K2_hz"])),) if value != 0.0)
    return DispersiveTransmonCavityModel(
        omega_c=hz_to_rad_s(float(config["omega_c_hz"])),
        omega_q=hz_to_rad_s(float(config["omega_q_hz"])),
        alpha=hz_to_rad_s(float(config["qubit_alpha_hz"])),
        chi=hz_to_rad_s(float(config["st_chi_hz"])),
        chi_higher=chi_higher,
        kerr=hz_to_rad_s(float(config["st_K_hz"])),
        kerr_higher=kerr_higher,
        n_cav=int(config["n_cav_dim"]),
        n_tr=2,
    )


def build_frame(model: DispersiveTransmonCavityModel, config: Mapping[str, Any]) -> FrameSpec:
    if bool(config["use_rotating_frame"]):
        return FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return FrameSpec()


def build_noise_spec(config: Mapping[str, Any], enabled: bool) -> NoiseSpec | None:
    if not enabled:
        return None
    t1_s = ns_to_s(float(config["qb_T1_relax_ns"])) if config.get("qb_T1_relax_ns") is not None else None
    tphi_s = derive_tphi_seconds(config.get("qb_T1_relax_ns"), choose_t2_ns(config))
    kappa = float(config.get("cavity_kappa_1_per_s", 0.0))
    return NoiseSpec(t1=t1_s, tphi=tphi_s, kappa=kappa if kappa > 0.0 else None)


def gate_axis_label(step_index: int, gate: Gate | None) -> str:
    return "0:INIT" if gate is None else f"{step_index}:{gate.type}"


def should_store_wigner(gate: Gate | None, config: Mapping[str, Any]) -> bool:
    return gate is None or bool(config["wigner_every_gate"]) or gate.type in {"Displacement", "SQR"}


def snapshot_from_state(
    state: qt.Qobj,
    step_index: int,
    gate: Gate | None,
    config: Mapping[str, Any],
    case_label: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rho = as_dm(state)
    rho_q = reduced_qubit_state(rho)
    rho_c = reduced_cavity_state(rho)
    x, y, z = bloch_xyz_from_joint(rho)
    moments = cavity_moments(rho, n_cav=rho_c.dims[0][0])
    snapshot = {
        "case": case_label,
        "index": int(step_index),
        "gate_type": "INIT" if gate is None else gate.type,
        "gate_name": "initial_state" if gate is None else gate.name,
        "top_label": gate_axis_label(step_index, gate),
        "state": rho,
        "rho_q": rho_q,
        "rho_c": rho_c,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "n": float(np.real(moments["n"])),
        "a": complex(moments["a"]),
        "qubit_purity": purity(rho_q),
        "cavity_purity": purity(rho_c),
        "extra": extra or {},
        "wigner": None,
    }
    if should_store_wigner(gate, config):
        xvec, yvec, w = cavity_wigner(
            rho_c,
            n_points=int(config["wigner_points"]),
            extent=float(config["wigner_extent"]),
        )
        snapshot["wigner"] = {"xvec": xvec, "yvec": yvec, "w": w}
    return snapshot


def finalize_track(case_label: str, snapshots: list[dict[str, Any]], metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "case": case_label,
        "snapshots": snapshots,
        "indices": np.asarray([snap["index"] for snap in snapshots], dtype=int),
        "x": np.asarray([snap["x"] for snap in snapshots], dtype=float),
        "y": np.asarray([snap["y"] for snap in snapshots], dtype=float),
        "z": np.asarray([snap["z"] for snap in snapshots], dtype=float),
        "n": np.asarray([snap["n"] for snap in snapshots], dtype=float),
        "qubit_purity": np.asarray([snap["qubit_purity"] for snap in snapshots], dtype=float),
        "cavity_purity": np.asarray([snap["cavity_purity"] for snap in snapshots], dtype=float),
        "wigner_snapshots": [snap for snap in snapshots if snap["wigner"] is not None],
        "metadata": metadata or {},
    }


def print_mapping_rows(track: dict[str, Any]) -> None:
    for row in track["metadata"].get("mapping_rows", []):
        print(f"k={row['index']:>2} {row['type']:<12} {row['mapping']}")


def final_case_summary(track: dict[str, Any]) -> dict[str, Any]:
    final = track["snapshots"][-1]
    return {
        "case": track["case"],
        "solver": track["metadata"]["solver"],
        "final_x": final["x"],
        "final_y": final["y"],
        "final_z": final["z"],
        "final_n": final["n"],
        "final_qubit_purity": final["qubit_purity"],
        "final_cavity_purity": final["cavity_purity"],
        "final_wigner_negativity": float(track["wigner_negativity"][-1]),
        "final_fidelity_weakness_vs_a": float(track["fidelity_weakness_vs_a"][-1]),
    }
