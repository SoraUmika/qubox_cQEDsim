from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import qutip as qt

from cqed_sim.experiment.readout_chain import ReadoutChain
from cqed_sim.sim.extractors import reduced_qubit_state


@dataclass(frozen=True)
class QubitMeasurementSpec:
    shots: int | None = None
    confusion_matrix: np.ndarray | None = None
    iq_sigma: float | None = None
    seed: int | None = None
    lump_other_into: str = "e"
    readout_chain: ReadoutChain | None = None
    readout_duration: float | None = None
    readout_dt: float | None = None
    readout_drive_frequency: float | None = None
    readout_chi: float | None = None
    qubit_frequency: float | None = None
    include_filter: bool = True
    include_measurement_dephasing: bool = False
    include_purcell_relaxation: bool = False
    classify_from_iq: bool = False


@dataclass
class QubitMeasurementResult:
    probabilities: dict[str, float]
    observed_probabilities: dict[str, float]
    expectation_z: float
    counts: dict[str, int] | None = None
    samples: np.ndarray | None = None
    iq_samples: np.ndarray | None = None
    post_measurement_state: qt.Qobj | None = None
    readout_centers: dict[str, np.ndarray] | None = None
    readout_metadata: dict[str, float] | None = None


def _qubit_population_probabilities(rho_q: qt.Qobj, lump_other_into: str) -> tuple[float, float]:
    diag = np.real(np.diag(rho_q.full()))
    p_g = float(diag[0]) if diag.size > 0 else 0.0
    p_e = float(diag[1]) if diag.size > 1 else 0.0
    p_other = max(0.0, 1.0 - p_g - p_e)
    if lump_other_into == "e":
        p_e += p_other
    elif lump_other_into == "g":
        p_g += p_other
    else:
        raise ValueError(f"Unsupported lump_other_into value '{lump_other_into}'.")
    return p_g, p_e


def _observed_probabilities(p_g: float, p_e: float, confusion_matrix: np.ndarray | None) -> tuple[float, float]:
    probs = np.asarray([p_g, p_e], dtype=float)
    if confusion_matrix is None:
        return float(probs[0]), float(probs[1])
    matrix = np.asarray(confusion_matrix, dtype=float)
    if matrix.shape != (2, 2):
        raise ValueError("confusion_matrix must be 2x2.")
    observed = matrix @ probs
    return float(observed[0]), float(observed[1])


def measure_qubit(state: qt.Qobj, spec: QubitMeasurementSpec | None = None) -> QubitMeasurementResult:
    spec = QubitMeasurementSpec() if spec is None else spec
    rho_q = reduced_qubit_state(state)
    post_measurement_state = rho_q
    readout_centers = None
    readout_metadata = None

    if spec.readout_chain is not None:
        if spec.include_purcell_relaxation and spec.qubit_frequency is None:
            raise ValueError("qubit_frequency must be provided when include_purcell_relaxation=True.")
        post_measurement_state = spec.readout_chain.apply_backaction(
            rho_q,
            omega_q=0.0 if spec.qubit_frequency is None else float(spec.qubit_frequency),
            duration=spec.readout_duration,
            drive_frequency=spec.readout_drive_frequency,
            chi=spec.readout_chi,
            include_measurement_dephasing=spec.include_measurement_dephasing,
            include_purcell_relaxation=spec.include_purcell_relaxation,
            include_filter=spec.include_filter,
        )
        readout_centers = spec.readout_chain.iq_centers(
            duration=spec.readout_duration,
            dt=spec.readout_dt,
            drive_frequency=spec.readout_drive_frequency,
            chi=spec.readout_chi,
            include_filter=spec.include_filter,
        )
        purcell_rate = (
            0.0
            if spec.qubit_frequency is None
            else spec.readout_chain.purcell_rate(float(spec.qubit_frequency), include_filter=spec.include_filter)
        )
        readout_metadata = {
            "gamma_meas": float(
                spec.readout_chain.gamma_meas(
                    drive_frequency=spec.readout_drive_frequency,
                    chi=spec.readout_chi,
                    include_filter=spec.include_filter,
                )
            ),
            "purcell_rate": float(purcell_rate),
            "purcell_limited_t1": float(
                np.inf
                if purcell_rate <= 0.0
                else spec.readout_chain.purcell_limited_t1(
                    float(spec.qubit_frequency),
                    include_filter=spec.include_filter,
                )
            ),
        }

    p_g, p_e = _qubit_population_probabilities(post_measurement_state, spec.lump_other_into)
    p_g_obs, p_e_obs = _observed_probabilities(p_g, p_e, spec.confusion_matrix)

    counts = None
    samples = None
    iq_samples = None
    if spec.shots is not None:
        rng = np.random.default_rng(spec.seed)
        shots = int(spec.shots)
        if spec.readout_chain is not None:
            latent_samples = rng.choice(np.array([0, 1], dtype=int), size=shots, p=[p_g, p_e])
            iq_samples = spec.readout_chain.sample_iq(
                latent_samples,
                duration=spec.readout_duration,
                dt=spec.readout_dt,
                drive_frequency=spec.readout_drive_frequency,
                chi=spec.readout_chi,
                include_filter=spec.include_filter,
                seed=spec.seed,
            )
            if spec.classify_from_iq:
                samples = spec.readout_chain.classify_iq(
                    iq_samples,
                    duration=spec.readout_duration,
                    dt=spec.readout_dt,
                    drive_frequency=spec.readout_drive_frequency,
                    chi=spec.readout_chi,
                    include_filter=spec.include_filter,
                )
                p_g_obs = float(np.mean(samples == 0))
                p_e_obs = float(np.mean(samples == 1))
            else:
                samples = (
                    latent_samples
                    if spec.confusion_matrix is None
                    else rng.choice(np.array([0, 1], dtype=int), size=shots, p=[p_g_obs, p_e_obs])
                )
        else:
            samples = rng.choice(np.array([0, 1], dtype=int), size=shots, p=[p_g_obs, p_e_obs])
        count_g = int(np.sum(samples == 0))
        count_e = shots - count_g
        counts = {"g": count_g, "e": count_e}
        if spec.readout_chain is None and spec.iq_sigma is not None:
            centers = np.where(samples[:, None] == 0, np.array([[-1.0, 0.0]]), np.array([[1.0, 0.0]]))
            iq_samples = centers + rng.normal(scale=float(spec.iq_sigma), size=(shots, 2))

    return QubitMeasurementResult(
        probabilities={"g": p_g, "e": p_e},
        observed_probabilities={"g": p_g_obs, "e": p_e_obs},
        expectation_z=float(p_g_obs - p_e_obs),
        counts=counts,
        samples=samples,
        iq_samples=iq_samples,
        post_measurement_state=post_measurement_state,
        readout_centers=readout_centers,
        readout_metadata=readout_metadata,
    )
