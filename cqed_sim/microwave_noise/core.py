from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


H_PLANCK = 6.62607015e-34
K_BOLTZMANN = 1.380649e-23
DB_PER_NEPER_POWER = 10.0 / np.log(10.0)
POWER_NP_PER_DB = np.log(10.0) / 10.0

ArrayLike = float | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class ComponentTrace:
    """Diagnostic record for one scalar propagation component."""

    name: str
    kind: str
    n_in: np.ndarray
    n_out: np.ndarray
    eta: np.ndarray | None = None
    temp_K: np.ndarray | None = None
    n_thermal: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class SMatrixTrace:
    """Diagnostic record for a passive S-matrix covariance propagation."""

    name: str
    n_thermal: np.ndarray
    loss_covariance: np.ndarray
    loss_covariance_eigenvalues: np.ndarray


@dataclass(frozen=True)
class NoiseBudget:
    """Exact scalar matched cascade decomposition.

    The output occupation is

        n_out = weights["source"] * occupations["source"]
                + sum(weights[label] * occupations[label])

    where each lossy element emits a bath occupation set by its physical
    temperature. For distributed lines, labels may refer to individual slices.
    """

    weights: dict[str, np.ndarray]
    occupations: dict[str, np.ndarray]
    contributions: dict[str, np.ndarray]
    weight_sum: np.ndarray
    weight_closure_error: np.ndarray
    contribution_sum: np.ndarray
    contribution_closure_error: np.ndarray


@dataclass(frozen=True)
class NoiseCascadeResult:
    """Result returned by :class:`NoiseCascade`."""

    n_out: np.ndarray
    effective_temperature: np.ndarray
    trace: tuple[ComponentTrace, ...]
    budget: NoiseBudget


@dataclass(frozen=True)
class _ScalarElement:
    label: str
    eta: np.ndarray
    bath_n: np.ndarray


def _as_array(value: Any, *, dtype: Any = float) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


def _maybe_scalar(array: np.ndarray) -> float | complex | np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 0:
        return arr.item()
    return arr


def _call_parameter(value: Any, *args: Any) -> Any:
    if callable(value):
        return value(*args)
    return value


def _validate_nonnegative(name: str, value: np.ndarray) -> None:
    if np.any(value < 0.0):
        raise ValueError(f"{name} must be nonnegative.")


def _validate_transmission(name: str, eta: np.ndarray, *, atol: float = 1.0e-12) -> np.ndarray:
    if np.any(eta < -atol) or np.any(eta > 1.0 + atol):
        raise ValueError(f"{name} must be a passive power transmission in [0, 1].")
    return np.clip(eta, 0.0, 1.0)


def _broadcast_like(value: Any, reference: np.ndarray) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=float), np.shape(reference)).astype(float, copy=False)


def bose_occupation(freq_hz: ArrayLike, temp_K: float | ArrayLike) -> float | np.ndarray:
    """Return the normally ordered Bose occupation n_B(f, T).

    Frequencies are in Hz and temperatures are in K. The implementation uses
    ``expm1`` for numerical stability and returns zero for non-positive
    temperatures.
    """

    freq = _as_array(freq_hz)
    temp = _as_array(temp_K)
    freq_b, temp_b = np.broadcast_arrays(freq, temp)
    result = np.zeros_like(freq_b, dtype=float)
    positive_temp = temp_b > 0.0
    if np.any(positive_temp):
        x = H_PLANCK * freq_b[positive_temp] / (K_BOLTZMANN * temp_b[positive_temp])
        denom = np.expm1(x)
        result[positive_temp] = np.divide(
            1.0,
            denom,
            out=np.full_like(denom, np.inf, dtype=float),
            where=denom != 0.0,
        )
    return _maybe_scalar(result)


def loss_db_to_power_transmission(loss_db: ArrayLike) -> float | np.ndarray:
    """Convert positive insertion loss in dB to power transmission."""

    loss = _as_array(loss_db)
    _validate_nonnegative("loss_db", loss)
    eta = 10.0 ** (-loss / 10.0)
    return _maybe_scalar(eta)


def occupation_to_effective_temperature(freq_hz: ArrayLike, n: ArrayLike) -> float | np.ndarray:
    """Return the Bose temperature corresponding to occupation ``n``.

    The returned temperature is zero wherever ``n <= 0``.
    """

    freq = _as_array(freq_hz)
    occupation = _as_array(n)
    freq_b, n_b = np.broadcast_arrays(freq, occupation)
    result = np.zeros_like(freq_b, dtype=float)
    positive_n = n_b > 0.0
    if np.any(positive_n):
        denominator = np.log1p(1.0 / n_b[positive_n])
        result[positive_n] = H_PLANCK * freq_b[positive_n] / (K_BOLTZMANN * denominator)
    return _maybe_scalar(result)


def sym_noise_temperature(freq_hz: ArrayLike, temp_K: float | ArrayLike) -> float | np.ndarray:
    """Return the symmetrized quantum noise temperature.

    This helper is for reporting calibrated spectra. It should not be used as
    the normally ordered bath occupation in Lindblad master equations.
    """

    freq = _as_array(freq_hz)
    temp = _as_array(temp_K)
    freq_b, temp_b = np.broadcast_arrays(freq, temp)
    half_quantum = H_PLANCK * freq_b / (2.0 * K_BOLTZMANN)
    result = np.array(half_quantum, dtype=float, copy=True)
    positive_temp = temp_b > 0.0
    if np.any(positive_temp):
        x = half_quantum[positive_temp] / temp_b[positive_temp]
        result[positive_temp] = half_quantum[positive_temp] / np.tanh(x)
    return _maybe_scalar(result)


@dataclass(frozen=True)
class PassiveLoss:
    """Matched passive lossy component with positive insertion loss in dB."""

    name: str
    temp_K: float
    loss_db: float | Callable[[ArrayLike], ArrayLike]

    def elementary_elements(self, freq_hz: ArrayLike) -> tuple[_ScalarElement, ...]:
        freq = _as_array(freq_hz)
        loss = _as_array(_call_parameter(self.loss_db, freq))
        eta = _validate_transmission("eta", _as_array(loss_db_to_power_transmission(loss)))
        bath = _as_array(bose_occupation(freq, self.temp_K))
        return (_ScalarElement(self.name, np.broadcast_to(eta, np.shape(np.broadcast_arrays(freq, eta)[0])), bath),)

    def propagate(self, freq_hz: ArrayLike, n_in: ArrayLike) -> tuple[np.ndarray, ComponentTrace]:
        freq = _as_array(freq_hz)
        n_in_arr = _as_array(n_in)
        element = self.elementary_elements(freq)[0]
        eta, n_in_b, bath = np.broadcast_arrays(element.eta, n_in_arr, element.bath_n)
        n_out = eta * n_in_b + (1.0 - eta) * bath
        trace = ComponentTrace(
            name=self.name,
            kind="passive_loss",
            n_in=np.asarray(n_in_b, dtype=float),
            n_out=np.asarray(n_out, dtype=float),
            eta=np.asarray(eta, dtype=float),
            temp_K=_broadcast_like(self.temp_K, n_out),
            n_thermal=np.asarray(bath, dtype=float),
            metadata={"loss_db": _as_array(_call_parameter(self.loss_db, freq))},
        )
        return np.asarray(n_out, dtype=float), trace


@dataclass(frozen=True)
class DistributedLine:
    """Sliced distributed lossy line with local thermal re-emission."""

    name: str
    length_m: float
    attenuation_db_per_m: float | Callable[[ArrayLike, float], ArrayLike]
    temperature_K: float | Callable[[float], float]
    num_slices: int = 100

    def __post_init__(self) -> None:
        if self.length_m < 0.0:
            raise ValueError("length_m must be nonnegative.")
        if int(self.num_slices) <= 0:
            raise ValueError("num_slices must be positive.")

    def elementary_elements(self, freq_hz: ArrayLike) -> tuple[_ScalarElement, ...]:
        freq = _as_array(freq_hz)
        dz = float(self.length_m) / int(self.num_slices)
        elements: list[_ScalarElement] = []
        for index in range(int(self.num_slices)):
            z_mid = (index + 0.5) * dz
            attenuation_db = _as_array(_call_parameter(self.attenuation_db_per_m, freq, z_mid))
            _validate_nonnegative("attenuation_db_per_m", attenuation_db)
            alpha = POWER_NP_PER_DB * attenuation_db
            eta = _validate_transmission("eta", np.exp(-alpha * dz))
            temp = _call_parameter(self.temperature_K, z_mid)
            bath = _as_array(bose_occupation(freq, temp))
            label = f"{self.name}[{index:04d}]"
            elements.append(_ScalarElement(label, eta, bath))
        return tuple(elements)

    def propagate(self, freq_hz: ArrayLike, n_in: ArrayLike) -> tuple[np.ndarray, ComponentTrace]:
        n = _as_array(n_in)
        eta_total: np.ndarray | None = None
        slice_summaries: list[dict[str, Any]] = []
        for element in self.elementary_elements(freq_hz):
            eta, n, bath = np.broadcast_arrays(element.eta, n, element.bath_n)
            eta_total = eta if eta_total is None else np.asarray(eta_total) * eta
            n = eta * n + (1.0 - eta) * bath
            slice_summaries.append(
                {
                    "label": element.label,
                    "eta": np.asarray(eta, dtype=float),
                    "n_thermal": np.asarray(bath, dtype=float),
                }
            )
        if eta_total is None:
            eta_total = np.ones_like(_as_array(n_in), dtype=float)
        trace = ComponentTrace(
            name=self.name,
            kind="distributed_line",
            n_in=np.asarray(n_in, dtype=float),
            n_out=np.asarray(n, dtype=float),
            eta=np.asarray(eta_total, dtype=float),
            temp_K=None,
            n_thermal=None,
            metadata={
                "length_m": float(self.length_m),
                "num_slices": int(self.num_slices),
                "dz_m": float(self.length_m) / int(self.num_slices),
                "slices": slice_summaries,
            },
        )
        return np.asarray(n, dtype=float), trace


@dataclass(frozen=True)
class DirectionalLoss:
    """Scalar approximation to an isolator or circulator path."""

    name: str
    temp_K: float
    forward_loss_db: float | Callable[[ArrayLike], ArrayLike]
    reverse_isolation_db: float | Callable[[ArrayLike], ArrayLike]

    def _loss_for_direction(self, freq_hz: ArrayLike, direction: str) -> Any:
        direction_key = str(direction).strip().lower()
        if direction_key == "forward":
            return _call_parameter(self.forward_loss_db, freq_hz)
        if direction_key == "reverse":
            return _call_parameter(self.reverse_isolation_db, freq_hz)
        raise ValueError("direction must be 'forward' or 'reverse'.")

    def elementary_elements(self, freq_hz: ArrayLike, *, direction: str = "forward") -> tuple[_ScalarElement, ...]:
        freq = _as_array(freq_hz)
        loss = _as_array(self._loss_for_direction(freq, direction))
        eta = _validate_transmission("eta", _as_array(loss_db_to_power_transmission(loss)))
        bath = _as_array(bose_occupation(freq, self.temp_K))
        return (_ScalarElement(self.name, eta, bath),)

    def propagate(
        self,
        freq_hz: ArrayLike,
        n_in: ArrayLike,
        direction: str = "forward",
    ) -> tuple[np.ndarray, ComponentTrace]:
        freq = _as_array(freq_hz)
        n_in_arr = _as_array(n_in)
        element = self.elementary_elements(freq, direction=direction)[0]
        eta, n_in_b, bath = np.broadcast_arrays(element.eta, n_in_arr, element.bath_n)
        n_out = eta * n_in_b + (1.0 - eta) * bath
        trace = ComponentTrace(
            name=self.name,
            kind="directional_loss",
            n_in=np.asarray(n_in_b, dtype=float),
            n_out=np.asarray(n_out, dtype=float),
            eta=np.asarray(eta, dtype=float),
            temp_K=_broadcast_like(self.temp_K, n_out),
            n_thermal=np.asarray(bath, dtype=float),
            metadata={
                "direction": str(direction).strip().lower(),
                "loss_db": _as_array(self._loss_for_direction(freq, direction)),
            },
        )
        return np.asarray(n_out, dtype=float), trace


@dataclass(frozen=True)
class PassiveSMatrixComponent:
    """Passive multiport linear component in normally ordered covariance form."""

    name: str
    temp_K: float
    S_matrix: Callable[[ArrayLike], np.ndarray]
    psd_tolerance: float = 1.0e-10

    def _propagate_one(self, freq_hz: float, c_in: np.ndarray) -> tuple[np.ndarray, SMatrixTrace]:
        s_matrix = np.asarray(self.S_matrix(float(freq_hz)), dtype=complex)
        if s_matrix.ndim != 2 or s_matrix.shape[0] != s_matrix.shape[1]:
            raise ValueError("S_matrix(freq_hz) must return a square 2D array.")
        c_arr = np.asarray(c_in, dtype=complex)
        if c_arr.shape != s_matrix.shape:
            raise ValueError("C_in must have the same square shape as S_matrix(freq_hz).")

        identity = np.eye(s_matrix.shape[0], dtype=complex)
        loss_cov = identity - s_matrix @ s_matrix.conj().T
        loss_cov = 0.5 * (loss_cov + loss_cov.conj().T)
        eigenvalues = np.linalg.eigvalsh(loss_cov)
        if np.min(eigenvalues) < -float(self.psd_tolerance):
            raise ValueError("I - S S^dagger must be positive semidefinite for a passive component.")
        bath = _as_array(bose_occupation(float(freq_hz), self.temp_K))
        c_out = s_matrix @ c_arr @ s_matrix.conj().T + bath * loss_cov
        trace = SMatrixTrace(
            name=self.name,
            n_thermal=np.asarray(bath, dtype=float),
            loss_covariance=loss_cov,
            loss_covariance_eigenvalues=eigenvalues,
        )
        return c_out, trace

    def propagate_covariance(
        self,
        freq_hz: ArrayLike,
        C_in: np.ndarray,
    ) -> tuple[np.ndarray, SMatrixTrace | tuple[SMatrixTrace, ...]]:
        freq = _as_array(freq_hz)
        if freq.ndim == 0:
            return self._propagate_one(float(freq), np.asarray(C_in, dtype=complex))

        c_in_arr = np.asarray(C_in, dtype=complex)
        outputs: list[np.ndarray] = []
        traces: list[SMatrixTrace] = []
        for index, freq_value in enumerate(freq.reshape(-1)):
            c_slice = c_in_arr if c_in_arr.ndim == 2 else c_in_arr.reshape((-1, *c_in_arr.shape[-2:]))[index]
            c_out, trace = self._propagate_one(float(freq_value), c_slice)
            outputs.append(c_out)
            traces.append(trace)
        return np.asarray(outputs).reshape((*freq.shape, *outputs[0].shape)), tuple(traces)


@dataclass(frozen=True)
class NoiseCascade:
    """Scalar matched cascade of passive microwave noise components."""

    components: Sequence[PassiveLoss | DistributedLine | DirectionalLoss]

    def __post_init__(self) -> None:
        names = [component.name for component in self.components]
        if len(names) != len(set(names)):
            raise ValueError("NoiseCascade component names must be unique.")

    def _elements(self, freq_hz: ArrayLike, direction: str) -> tuple[_ScalarElement, ...]:
        elements: list[_ScalarElement] = []
        for component in self.components:
            if isinstance(component, DirectionalLoss):
                elements.extend(component.elementary_elements(freq_hz, direction=direction))
            else:
                elements.extend(component.elementary_elements(freq_hz))
        return tuple(elements)

    def _budget(
        self,
        elements: Sequence[_ScalarElement],
        source_n: np.ndarray,
        n_out: np.ndarray,
    ) -> NoiseBudget:
        suffix = np.ones_like(np.asarray(n_out, dtype=float), dtype=float)
        suffix_products: list[np.ndarray] = []
        for element in reversed(elements):
            suffix_products.append(np.asarray(suffix, dtype=float))
            suffix = suffix * np.broadcast_to(element.eta, np.shape(suffix))
        suffix_products.reverse()

        weights: dict[str, np.ndarray] = {"source": np.asarray(suffix, dtype=float)}
        occupations: dict[str, np.ndarray] = {"source": np.broadcast_to(source_n, np.shape(n_out)).astype(float)}
        contributions: dict[str, np.ndarray] = {
            "source": weights["source"] * occupations["source"],
        }

        for element, suffix_after in zip(elements, suffix_products):
            eta = np.broadcast_to(element.eta, np.shape(n_out))
            bath = np.broadcast_to(element.bath_n, np.shape(n_out))
            weight = (1.0 - eta) * suffix_after
            weights[element.label] = np.asarray(weight, dtype=float)
            occupations[element.label] = np.asarray(bath, dtype=float)
            contributions[element.label] = np.asarray(weight * bath, dtype=float)

        weight_sum = np.sum(np.stack(list(weights.values()), axis=0), axis=0)
        contribution_sum = np.sum(np.stack(list(contributions.values()), axis=0), axis=0)
        return NoiseBudget(
            weights=weights,
            occupations=occupations,
            contributions=contributions,
            weight_sum=np.asarray(weight_sum, dtype=float),
            weight_closure_error=np.asarray(weight_sum - 1.0, dtype=float),
            contribution_sum=np.asarray(contribution_sum, dtype=float),
            contribution_closure_error=np.asarray(contribution_sum - n_out, dtype=float),
        )

    def propagate(
        self,
        freq_hz: ArrayLike,
        source_temp_K: float | None = None,
        source_n: ArrayLike | None = None,
        direction: str = "forward",
    ) -> NoiseCascadeResult:
        if source_temp_K is not None and source_n is not None:
            raise ValueError("Specify either source_temp_K or source_n, not both.")
        freq = _as_array(freq_hz)
        if source_n is None:
            if source_temp_K is None:
                n = np.zeros_like(freq, dtype=float)
            else:
                n = _as_array(bose_occupation(freq, source_temp_K))
        else:
            n = _as_array(source_n)
        _validate_nonnegative("source_n", np.asarray(n, dtype=float))

        source_occupation = np.asarray(n, dtype=float)
        traces: list[ComponentTrace] = []
        for component in self.components:
            if isinstance(component, DirectionalLoss):
                n, trace = component.propagate(freq, n, direction=direction)
            else:
                n, trace = component.propagate(freq, n)
            traces.append(trace)

        elements = self._elements(freq, direction)
        n_out = np.asarray(n, dtype=float)
        budget = self._budget(elements, source_occupation, n_out)
        effective_temperature = _as_array(occupation_to_effective_temperature(freq, n_out))
        return NoiseCascadeResult(
            n_out=n_out,
            effective_temperature=effective_temperature,
            trace=tuple(traces),
            budget=budget,
        )


def resonator_thermal_occupation(kappas: ArrayLike, n_baths: ArrayLike) -> float:
    """Return the linewidth-weighted steady-state resonator occupation."""

    kappa_arr = _as_array(kappas)
    bath_arr = _as_array(n_baths)
    if kappa_arr.shape != bath_arr.shape:
        kappa_arr, bath_arr = np.broadcast_arrays(kappa_arr, bath_arr)
    _validate_nonnegative("kappas", kappa_arr)
    _validate_nonnegative("n_baths", bath_arr)
    total_kappa = float(np.sum(kappa_arr))
    if total_kappa <= 0.0:
        raise ValueError("sum(kappas) must be positive.")
    return float(np.sum(kappa_arr * bath_arr) / total_kappa)


def resonator_lindblad_rates(kappa: float, n: float) -> tuple[float, float]:
    """Return ``(downward_rate, upward_rate)`` for a bosonic bath."""

    if kappa < 0.0:
        raise ValueError("kappa must be nonnegative.")
    if n < 0.0:
        raise ValueError("n must be nonnegative.")
    return float(kappa * (n + 1.0)), float(kappa * n)


def qubit_thermal_rates(gamma_zero_temp: float, n: float) -> tuple[float, float, float]:
    """Return ``(Gamma_down, Gamma_up, Gamma_1)`` for a thermal qubit bath."""

    if gamma_zero_temp < 0.0:
        raise ValueError("gamma_zero_temp must be nonnegative.")
    if n < 0.0:
        raise ValueError("n must be nonnegative.")
    gamma_down = float(gamma_zero_temp * (n + 1.0))
    gamma_up = float(gamma_zero_temp * n)
    return gamma_down, gamma_up, gamma_down + gamma_up


def thermal_photon_dephasing(
    kappa: float,
    chi: float,
    n_cav: float,
    *,
    exact: bool = True,
    approximation: str | None = None,
) -> float:
    """Thermal-photon-induced qubit dephasing in angular-rate units.

    ``kappa`` and ``chi`` are angular rates. The exact expression follows the
    convention used by Zhang et al., npj Quantum Information 2017, Eq. (3),
    with the repository convention that ``chi`` is the per-photon qubit shift.
    """

    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    if n_cav < 0.0:
        raise ValueError("n_cav must be nonnegative.")
    if approximation is not None:
        key = approximation.strip().lower().replace("-", "_")
        if key in {"weak", "weak_dispersive"}:
            return float(4.0 * chi**2 / kappa * n_cav * (n_cav + 1.0))
        if key in {"strong", "strong_low_occupation", "strong_dispersive_low_occupation"}:
            return float(kappa * n_cav)
        raise ValueError("approximation must be 'weak' or 'strong_low_occupation'.")
    if not exact:
        return float(4.0 * chi**2 / kappa * n_cav * (n_cav + 1.0))

    argument = (1.0 + 2.0j * chi / kappa) ** 2 + 8.0j * chi * n_cav / kappa
    gamma_phi = kappa / 2.0 * np.real(np.sqrt(argument) - 1.0)
    return float(max(gamma_phi, 0.0))


__all__ = [
    "ComponentTrace",
    "DB_PER_NEPER_POWER",
    "DistributedLine",
    "DirectionalLoss",
    "H_PLANCK",
    "K_BOLTZMANN",
    "NoiseBudget",
    "NoiseCascade",
    "NoiseCascadeResult",
    "POWER_NP_PER_DB",
    "PassiveLoss",
    "PassiveSMatrixComponent",
    "SMatrixTrace",
    "bose_occupation",
    "loss_db_to_power_transmission",
    "occupation_to_effective_temperature",
    "qubit_thermal_rates",
    "resonator_lindblad_rates",
    "resonator_thermal_occupation",
    "sym_noise_temperature",
    "thermal_photon_dephasing",
]
