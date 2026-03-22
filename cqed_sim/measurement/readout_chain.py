from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np
import qutip as qt
from scipy.constants import Boltzmann as K_B


def _qubit_label(qubit_state: str | int) -> str:
    if isinstance(qubit_state, str):
        label = qubit_state.lower()
        if label not in {"g", "e"}:
            raise ValueError(f"Unsupported qubit state label '{qubit_state}'.")
        return label
    if int(qubit_state) == 0:
        return "g"
    if int(qubit_state) == 1:
        return "e"
    raise ValueError(f"Unsupported qubit state index '{qubit_state}'.")


def _complex_to_iq(value: complex) -> np.ndarray:
    return np.array([float(np.real(value)), float(np.imag(value))], dtype=float)


def _resolved_drive_steps(
    drive_envelope: np.ndarray | complex | float | Callable[[np.ndarray], np.ndarray],
    *,
    dt: float,
    duration: float | None = None,
) -> np.ndarray:
    if np.isscalar(drive_envelope):
        if duration is None:
            raise ValueError("duration must be provided when drive_envelope is scalar.")
        n_steps = max(1, int(np.ceil(float(duration) / float(dt))))
        return np.full(n_steps, complex(drive_envelope), dtype=np.complex128)
    if callable(drive_envelope):
        if duration is None:
            raise ValueError("duration must be provided when drive_envelope is callable.")
        n_steps = max(1, int(np.ceil(float(duration) / float(dt))))
        sample_times = np.arange(n_steps, dtype=float) * float(dt)
        return np.asarray(drive_envelope(sample_times), dtype=np.complex128)
    envelope = np.asarray(drive_envelope, dtype=np.complex128)
    if envelope.ndim != 1:
        raise ValueError("drive_envelope must be one-dimensional.")
    if envelope.size == 0:
        raise ValueError("drive_envelope must contain at least one sample.")
    return envelope


@dataclass(frozen=True)
class ReadoutResonator:
    """Single-pole readout resonator in the dispersive regime."""

    omega_r: float
    kappa: float
    g: float
    epsilon: complex | float
    chi: float = 0.0
    drive_frequency: float | None = None

    def resolved_drive_frequency(self, drive_frequency: float | None = None) -> float:
        return float(self.omega_r if drive_frequency is None else drive_frequency)

    def dispersive_shift(self, qubit_state: str | int, chi: float | None = None) -> float:
        resolved_chi = float(self.chi if chi is None else chi)
        return 0.0 if _qubit_label(qubit_state) == "g" else resolved_chi

    def resonant_frequency(self, qubit_state: str | int, chi: float | None = None) -> float:
        return float(self.omega_r + self.dispersive_shift(qubit_state, chi=chi))

    def effective_linewidth(
        self,
        drive_frequency: float | None = None,
        *,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> float:
        if purcell_filter is None or not include_filter:
            return float(self.kappa)
        return float(purcell_filter.effective_linewidth(self.resolved_drive_frequency(drive_frequency), self))

    def steady_state_amplitude(
        self,
        qubit_state: str | int,
        *,
        drive_frequency: float | None = None,
        chi: float | None = None,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> complex:
        omega_drive = self.resolved_drive_frequency(drive_frequency)
        delta = self.resonant_frequency(qubit_state, chi=chi) - omega_drive
        linewidth = self.effective_linewidth(
            omega_drive,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        return complex(-1j * self.epsilon / (0.5 * linewidth + 1j * delta))

    def steady_state_amplitudes(
        self,
        *,
        drive_frequency: float | None = None,
        chi: float | None = None,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> dict[str, complex]:
        return {
            "g": self.steady_state_amplitude(
                "g",
                drive_frequency=drive_frequency,
                chi=chi,
                purcell_filter=purcell_filter,
                include_filter=include_filter,
            ),
            "e": self.steady_state_amplitude(
                "e",
                drive_frequency=drive_frequency,
                chi=chi,
                purcell_filter=purcell_filter,
                include_filter=include_filter,
            ),
        }

    def mean_photon_numbers(
        self,
        *,
        drive_frequency: float | None = None,
        chi: float | None = None,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> dict[str, float]:
        amplitudes = self.steady_state_amplitudes(
            drive_frequency=drive_frequency,
            chi=chi,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        return {label: float(abs(alpha) ** 2) for label, alpha in amplitudes.items()}

    def gamma_meas(
        self,
        *,
        drive_frequency: float | None = None,
        chi: float | None = None,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> float:
        omega_drive = self.resolved_drive_frequency(drive_frequency)
        linewidth = self.effective_linewidth(
            omega_drive,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        amplitudes = self.steady_state_amplitudes(
            drive_frequency=omega_drive,
            chi=chi,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        return float(0.5 * linewidth * abs(amplitudes["e"] - amplitudes["g"]) ** 2)

    def response_trace(
        self,
        qubit_state: str | int,
        *,
        duration: float,
        dt: float,
        drive_frequency: float | None = None,
        chi: float | None = None,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
        initial_amplitude: complex = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        tlist = np.arange(max(2, int(np.ceil(float(duration) / float(dt))) + 1), dtype=float) * float(dt)
        tlist = tlist[tlist <= float(duration) + 1.0e-15]
        omega_drive = self.resolved_drive_frequency(drive_frequency)
        delta = self.resonant_frequency(qubit_state, chi=chi) - omega_drive
        linewidth = self.effective_linewidth(
            omega_drive,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        lambda_eff = 0.5 * linewidth + 1j * delta
        alpha_ss = self.steady_state_amplitude(
            qubit_state,
            drive_frequency=omega_drive,
            chi=chi,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        trace = alpha_ss + (complex(initial_amplitude) - alpha_ss) * np.exp(-lambda_eff * tlist)
        return tlist, np.asarray(trace, dtype=np.complex128)

    def response_to_envelope(
        self,
        qubit_state: str | int,
        drive_envelope: np.ndarray | complex | float | Callable[[np.ndarray], np.ndarray],
        *,
        dt: float,
        duration: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
        initial_amplitude: complex = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the linear readout response for an arbitrary complex drive envelope.

        The envelope is treated as piecewise constant over each interval of width ``dt``.
        """

        dt = float(dt)
        if dt <= 0.0:
            raise ValueError("dt must be positive.")
        envelope = _resolved_drive_steps(drive_envelope, dt=dt, duration=duration)
        n_steps = int(envelope.size)
        tlist = np.arange(n_steps + 1, dtype=float) * dt
        omega_drive = self.resolved_drive_frequency(drive_frequency)
        delta = self.resonant_frequency(qubit_state, chi=chi) - omega_drive
        linewidth = self.effective_linewidth(
            omega_drive,
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        lambda_eff = 0.5 * linewidth + 1j * delta
        trace = np.empty(n_steps + 1, dtype=np.complex128)
        trace[0] = complex(initial_amplitude)
        if abs(lambda_eff) <= 1.0e-15:
            for idx, epsilon in enumerate(envelope, start=1):
                trace[idx] = trace[idx - 1] - 1j * complex(epsilon) * dt
            return tlist, trace

        step_decay = np.exp(-lambda_eff * dt)
        for idx, epsilon in enumerate(envelope, start=1):
            alpha_ss = -1j * complex(epsilon) / lambda_eff
            trace[idx] = alpha_ss + (trace[idx - 1] - alpha_ss) * step_decay
        return tlist, trace

    def purcell_rate(
        self,
        omega_q: float,
        *,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> float:
        delta = float(omega_q) - float(self.omega_r)
        if abs(delta) < 1.0e-15:
            return float("inf")
        linewidth = self.effective_linewidth(
            float(omega_q),
            purcell_filter=purcell_filter,
            include_filter=include_filter,
        )
        return float((abs(self.g) ** 2 / (delta * delta)) * linewidth)

    def purcell_limited_t1(
        self,
        omega_q: float,
        *,
        purcell_filter: PurcellFilter | None = None,
        include_filter: bool = True,
    ) -> float:
        gamma = self.purcell_rate(omega_q, purcell_filter=purcell_filter, include_filter=include_filter)
        return float(np.inf if gamma <= 0.0 else 1.0 / gamma)


@dataclass(frozen=True)
class PurcellFilter:
    """Frequency-selective Purcell filter following the Sete-style linewidth suppression picture."""

    omega_f: float | None = None
    bandwidth: float | None = None
    quality_factor: float | None = None
    coupling_rate: float | None = None

    def resolved_center_frequency(self, resonator: ReadoutResonator) -> float:
        return float(resonator.omega_r if self.omega_f is None else self.omega_f)

    def resolved_bandwidth(self, resonator: ReadoutResonator) -> float:
        if self.bandwidth is not None:
            return float(self.bandwidth)
        if self.quality_factor is None or self.quality_factor <= 0.0:
            raise ValueError("Provide either bandwidth or a positive quality_factor for the Purcell filter.")
        return float(self.resolved_center_frequency(resonator) / self.quality_factor)

    def resolved_coupling_rate(self, resonator: ReadoutResonator) -> float:
        if self.coupling_rate is not None:
            return float(self.coupling_rate)
        bandwidth = self.resolved_bandwidth(resonator)
        return float(0.5 * np.sqrt(max(resonator.kappa, 0.0) * max(bandwidth, 0.0)))

    def effective_linewidth(self, omega: float, resonator: ReadoutResonator) -> float:
        bandwidth = self.resolved_bandwidth(resonator)
        if bandwidth <= 0.0:
            return 0.0
        coupling = self.resolved_coupling_rate(resonator)
        detuning = float(omega) - self.resolved_center_frequency(resonator)
        return float((4.0 * coupling * coupling * bandwidth) / (bandwidth * bandwidth + 4.0 * detuning * detuning))

    def suppression_factor(self, omega: float, resonator: ReadoutResonator) -> float:
        if resonator.kappa <= 0.0:
            return 0.0
        return float(self.effective_linewidth(omega, resonator) / resonator.kappa)

    def purcell_rate(self, resonator: ReadoutResonator, omega_q: float) -> float:
        return resonator.purcell_rate(omega_q, purcell_filter=self, include_filter=True)


@dataclass(frozen=True)
class AmplifierChain:
    """Linear amplifier plus additive thermal noise."""

    noise_temperature: float = 0.0
    gain: float = 1.0
    impedance_ohm: float = 50.0
    mixer_phase: float = 0.0

    def noise_std(self, dt: float, n_samples: int = 1) -> float:
        if self.noise_temperature <= 0.0:
            return 0.0
        instantaneous = np.sqrt(2.0 * K_B * float(self.noise_temperature) * float(self.impedance_ohm) / float(dt))
        return float(instantaneous / np.sqrt(max(1, int(n_samples))))

    def amplify(self, field_trace: np.ndarray, *, dt: float, seed: int | None = None) -> np.ndarray:
        voltage = float(self.gain) * np.asarray(field_trace, dtype=np.complex128)
        sigma = self.noise_std(dt)
        if sigma <= 0.0:
            return voltage
        rng = np.random.default_rng(seed)
        noise = (sigma / np.sqrt(2.0)) * (
            rng.normal(size=voltage.shape) + 1j * rng.normal(size=voltage.shape)
        )
        return voltage + noise

    def mix_down(self, voltage_trace: np.ndarray) -> np.ndarray:
        return np.asarray(voltage_trace, dtype=np.complex128) * np.exp(-1j * float(self.mixer_phase))

    def iq_sample(self, voltage_trace: np.ndarray) -> np.ndarray:
        mixed = self.mix_down(voltage_trace)
        return np.array([float(np.mean(np.real(mixed))), float(np.mean(np.imag(mixed)))], dtype=float)


@dataclass
class ReadoutTrace:
    tlist: np.ndarray
    cavity_field: np.ndarray
    output_field: np.ndarray
    voltage_trace: np.ndarray
    iq_sample: np.ndarray


@dataclass
class ReadoutChain:
    resonator: ReadoutResonator
    amplifier: AmplifierChain = field(default_factory=AmplifierChain)
    purcell_filter: PurcellFilter | None = None
    integration_time: float = 1.0e-6
    dt: float = 4.0e-9

    def steady_state_amplitudes(
        self,
        *,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
    ) -> dict[str, complex]:
        return self.resonator.steady_state_amplitudes(
            drive_frequency=drive_frequency,
            chi=chi,
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )

    def gamma_meas(
        self,
        *,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
    ) -> float:
        return self.resonator.gamma_meas(
            drive_frequency=drive_frequency,
            chi=chi,
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )

    def purcell_rate(self, omega_q: float, *, include_filter: bool = True) -> float:
        return self.resonator.purcell_rate(
            omega_q,
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )

    def purcell_limited_t1(self, omega_q: float, *, include_filter: bool = True) -> float:
        return self.resonator.purcell_limited_t1(
            omega_q,
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )

    def simulate_trace(
        self,
        qubit_state: str | int,
        *,
        duration: float | None = None,
        dt: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
        include_noise: bool = True,
        seed: int | None = None,
    ) -> ReadoutTrace:
        duration = float(self.integration_time if duration is None else duration)
        dt = float(self.dt if dt is None else dt)
        tlist, cavity_trace = self.resonator.response_trace(
            qubit_state,
            duration=duration,
            dt=dt,
            drive_frequency=drive_frequency,
            chi=chi,
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )
        linewidth = self.resonator.effective_linewidth(
            self.resonator.resolved_drive_frequency(drive_frequency),
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )
        output_field = np.sqrt(max(linewidth, 0.0)) * cavity_trace
        voltage_trace = (
            self.amplifier.amplify(output_field, dt=dt, seed=seed)
            if include_noise
            else float(self.amplifier.gain) * output_field
        )
        return ReadoutTrace(
            tlist=tlist,
            cavity_field=cavity_trace,
            output_field=np.asarray(output_field, dtype=np.complex128),
            voltage_trace=np.asarray(voltage_trace, dtype=np.complex128),
            iq_sample=self.amplifier.iq_sample(voltage_trace),
        )

    def simulate_waveform(
        self,
        qubit_state: str | int,
        drive_envelope: np.ndarray | complex | float | Callable[[np.ndarray], np.ndarray],
        *,
        dt: float | None = None,
        duration: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
        include_noise: bool = True,
        seed: int | None = None,
        initial_amplitude: complex = 0.0,
    ) -> ReadoutTrace:
        """Simulate a time-domain readout trace for an arbitrary complex drive waveform."""

        dt = float(self.dt if dt is None else dt)
        tlist, cavity_trace = self.resonator.response_to_envelope(
            qubit_state,
            drive_envelope,
            dt=dt,
            duration=duration,
            drive_frequency=drive_frequency,
            chi=chi,
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
            initial_amplitude=initial_amplitude,
        )
        linewidth = self.resonator.effective_linewidth(
            self.resonator.resolved_drive_frequency(drive_frequency),
            purcell_filter=self.purcell_filter,
            include_filter=include_filter,
        )
        output_field = np.sqrt(max(linewidth, 0.0)) * cavity_trace
        voltage_trace = (
            self.amplifier.amplify(output_field, dt=dt, seed=seed)
            if include_noise
            else float(self.amplifier.gain) * output_field
        )
        return ReadoutTrace(
            tlist=tlist,
            cavity_field=np.asarray(cavity_trace, dtype=np.complex128),
            output_field=np.asarray(output_field, dtype=np.complex128),
            voltage_trace=np.asarray(voltage_trace, dtype=np.complex128),
            iq_sample=self.amplifier.iq_sample(voltage_trace),
        )

    def iq_centers(
        self,
        *,
        duration: float | None = None,
        dt: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
    ) -> dict[str, np.ndarray]:
        return {
            label: self.simulate_trace(
                label,
                duration=duration,
                dt=dt,
                drive_frequency=drive_frequency,
                chi=chi,
                include_filter=include_filter,
                include_noise=False,
            ).iq_sample
            for label in ("g", "e")
        }

    def integrated_noise_sigma(self, *, duration: float | None = None, dt: float | None = None) -> float:
        duration = float(self.integration_time if duration is None else duration)
        dt = float(self.dt if dt is None else dt)
        n_samples = max(1, int(np.ceil(duration / dt)))
        return float(self.amplifier.noise_std(dt, n_samples=n_samples))

    def sample_iq(
        self,
        latent_states: Iterable[str | int],
        *,
        duration: float | None = None,
        dt: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
        seed: int | None = None,
    ) -> np.ndarray:
        centers = self.iq_centers(
            duration=duration,
            dt=dt,
            drive_frequency=drive_frequency,
            chi=chi,
            include_filter=include_filter,
        )
        sigma = self.integrated_noise_sigma(duration=duration, dt=dt)
        labels = np.array([_qubit_label(state) for state in latent_states], dtype=object)
        iq = np.vstack([centers[str(label)] for label in labels]).astype(float, copy=True)
        if sigma > 0.0:
            rng = np.random.default_rng(seed)
            iq += rng.normal(scale=sigma, size=iq.shape)
        return iq

    def apply_backaction(
        self,
        rho_q: qt.Qobj,
        *,
        omega_q: float,
        duration: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_measurement_dephasing: bool = True,
        include_purcell_relaxation: bool = False,
        include_filter: bool = True,
    ) -> qt.Qobj:
        rho_q = rho_q if rho_q.isoper else rho_q.proj()
        if rho_q.shape[0] < 2:
            return rho_q
        duration = float(self.integration_time if duration is None else duration)
        rho_array = np.asarray(rho_q.full(), dtype=np.complex128).copy()

        if include_measurement_dephasing:
            gamma_phi = self.gamma_meas(
                drive_frequency=drive_frequency,
                chi=chi,
                include_filter=include_filter,
            )
            dephase = np.exp(-gamma_phi * duration)
            rho_array[0, 1] *= dephase
            rho_array[1, 0] *= dephase

        if include_purcell_relaxation:
            gamma = self.purcell_rate(omega_q, include_filter=include_filter)
            eta = np.exp(-gamma * duration)
            k0 = np.array([[1.0, 0.0], [0.0, np.sqrt(eta)]], dtype=np.complex128)
            k1 = np.array([[0.0, np.sqrt(max(0.0, 1.0 - eta))], [0.0, 0.0]], dtype=np.complex128)
            rho_array = k0 @ rho_array @ k0.conj().T + k1 @ rho_array @ k1.conj().T

        return qt.Qobj(rho_array, dims=rho_q.dims)

    def classify_iq(
        self,
        iq_samples: np.ndarray,
        *,
        duration: float | None = None,
        dt: float | None = None,
        drive_frequency: float | None = None,
        chi: float | None = None,
        include_filter: bool = True,
    ) -> np.ndarray:
        centers = self.iq_centers(
            duration=duration,
            dt=dt,
            drive_frequency=drive_frequency,
            chi=chi,
            include_filter=include_filter,
        )
        center_g = centers["g"]
        center_e = centers["e"]
        dist_g = np.linalg.norm(np.asarray(iq_samples, dtype=float) - center_g[None, :], axis=1)
        dist_e = np.linalg.norm(np.asarray(iq_samples, dtype=float) - center_e[None, :], axis=1)
        return np.where(dist_g <= dist_e, 0, 1).astype(int)


__all__ = [
    "AmplifierChain",
    "PurcellFilter",
    "ReadoutChain",
    "ReadoutResonator",
    "ReadoutTrace",
]
