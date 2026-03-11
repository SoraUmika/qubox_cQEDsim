from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import qutip as qt

from cqed_sim.sim.couplings import exchange


@dataclass(frozen=True)
class HamiltonianParams:
    """Effective Hamiltonian parameters in the runtime convention.

    The bare transmon frequency follows the large-``EJ/EC`` expansion

    ``omega_01 ~= sqrt(8 EJ EC) - EC``

    and the anharmonicity follows ``alpha ~= -EC``. The dressed dispersive
    coefficients are then extracted from a small exact diagonalization of the
    coupled Duffing model and matched to

    ``omega_ge(n) ~= omega_q - chi * n - chi_2 * n * (n - 1)``.

    This keeps the translator numerically stable while preserving the runtime
    sign convention used by ``cqed_sim``.
    """

    omega_q: float
    omega_r: float
    alpha: float
    chi: float
    chi_2: float
    g: float
    delta: float
    ec: float
    ej: float
    synthesis_chi: float
    synthesis_chi_2: float
    regime: str = "dispersive"
    metadata: dict[str, float | str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, float | str | dict[str, float | str]]:
        return asdict(self)


def _bare_transmon_frequency(ej: float, ec: float) -> float:
    if ej <= 0.0 or ec <= 0.0:
        raise ValueError("EJ and EC must both be positive.")
    return float(np.sqrt(8.0 * ej * ec) - ec)


def _matched_eigenenergy(
    eigenvalues: np.ndarray,
    eigenstates: list[qt.Qobj],
    bare_state: qt.Qobj,
) -> float:
    overlaps = np.array([abs(complex(bare_state.overlap(state))) ** 2 for state in eigenstates], dtype=float)
    return float(eigenvalues[int(np.argmax(overlaps))])


def _dressed_transition_curve(
    omega_q: float,
    alpha: float,
    g: float,
    omega_r: float,
    *,
    resonator_dim: int,
    transmon_dim: int,
) -> tuple[float, float, float]:
    a = qt.tensor(qt.qeye(transmon_dim), qt.destroy(resonator_dim))
    b = qt.tensor(qt.destroy(transmon_dim), qt.qeye(resonator_dim))
    hamiltonian = (
        float(omega_r) * (a.dag() * a)
        + float(omega_q) * (b.dag() * b)
        + 0.5 * float(alpha) * (b.dag() * b.dag() * b * b)
        + exchange(a, b, float(g))
    )
    eigenvalues, eigenstates = hamiltonian.eigenstates()
    dressed = []
    for photon_number in range(3):
        g_state = qt.tensor(qt.basis(transmon_dim, 0), qt.basis(resonator_dim, photon_number))
        e_state = qt.tensor(qt.basis(transmon_dim, 1), qt.basis(resonator_dim, photon_number))
        energy_g = _matched_eigenenergy(eigenvalues, eigenstates, g_state)
        energy_e = _matched_eigenenergy(eigenvalues, eigenstates, e_state)
        dressed.append(float(energy_e - energy_g))
    return float(dressed[0]), float(dressed[1]), float(dressed[2])


def _extract_dressed_coefficients(
    omega_q: float,
    alpha: float,
    g: float,
    omega_r: float,
    *,
    resonator_dim: int,
    transmon_dim: int,
) -> tuple[float, float]:
    omega0, omega1, omega2 = _dressed_transition_curve(
        omega_q,
        alpha,
        g,
        omega_r,
        resonator_dim=resonator_dim,
        transmon_dim=transmon_dim,
    )
    chi = float(omega0 - omega1)
    chi_2 = float(0.5 * (omega0 - omega2) - chi)
    return chi, chi_2


def from_transmon_params(
    ej: float,
    ec: float,
    g: float,
    omega_r: float,
    *,
    resonator_dim: int = 5,
    transmon_dim: int = 6,
) -> HamiltonianParams:
    """Translate bare transmon parameters into effective runtime Hamiltonian coefficients.

    The bare transmon is seeded by the large-``EJ/EC`` expansion
    ``omega_q ~= sqrt(8 EJ EC) - EC`` and ``alpha ~= -EC``.
    The dressed ``chi`` and ``chi_2`` are then extracted from exact
    diagonalization of a low-dimensional Duffing-plus-resonator model.
    """

    omega_q = _bare_transmon_frequency(float(ej), float(ec))
    alpha = -float(ec)
    delta = float(omega_q - float(omega_r))
    chi_perturbative = float(-2.0 * (float(g) ** 2) * alpha / (delta * (delta + alpha)))
    chi, chi_2 = _extract_dressed_coefficients(
        omega_q,
        alpha,
        float(g),
        float(omega_r),
        resonator_dim=int(resonator_dim),
        transmon_dim=int(transmon_dim),
    )
    return HamiltonianParams(
        omega_q=float(omega_q),
        omega_r=float(omega_r),
        alpha=float(alpha),
        chi=float(chi),
        chi_2=float(chi_2),
        g=float(g),
        delta=float(delta),
        ec=float(ec),
        ej=float(ej),
        synthesis_chi=float(-0.5 * chi),
        synthesis_chi_2=float(-0.5 * chi_2),
        metadata={
            "chi_perturbative": float(chi_perturbative),
            "ej_over_ec": float(ej / ec),
            "dispersive_ratio": float(abs(g / delta)) if abs(delta) > 1.0e-15 else float("inf"),
        },
    )


def _solve_detuning_from_measured(
    omega_01: float,
    alpha: float,
    chi: float,
    g: float,
    *,
    omega_r: float | None = None,
    detuning_branch: str = "positive",
) -> float:
    if abs(chi) <= 1.0e-15:
        if omega_r is None:
            raise ValueError("omega_r is required when chi is zero because the detuning cannot be inferred.")
        return float(omega_01 - omega_r)

    coeffs = np.array([float(chi), float(chi * alpha), float(2.0 * g * g * alpha)], dtype=float)
    roots = np.roots(coeffs)
    real_roots = [float(np.real(root)) for root in roots if abs(np.imag(root)) < 1.0e-10]
    if not real_roots:
        raise ValueError("Could not infer a real dispersive detuning from the measured parameters.")

    if omega_r is not None:
        target = float(omega_01 - omega_r)
        return float(min(real_roots, key=lambda root: abs(root - target)))

    branch = str(detuning_branch).lower()
    if branch == "positive":
        positives = [root for root in real_roots if root > 0.0]
        if not positives:
            raise ValueError("No positive dispersive detuning root was found.")
        return float(max(positives))
    if branch == "negative":
        negatives = [root for root in real_roots if root < 0.0]
        if not negatives:
            raise ValueError("No negative dispersive detuning root was found.")
        return float(min(negatives))
    if branch == "largest-magnitude":
        return float(max(real_roots, key=abs))
    raise ValueError(f"Unsupported detuning_branch '{detuning_branch}'.")


def from_measured(
    omega_01: float,
    alpha: float,
    chi: float,
    g: float,
    *,
    omega_r: float | None = None,
    detuning_branch: str = "positive",
    resonator_dim: int = 5,
    transmon_dim: int = 6,
) -> HamiltonianParams:
    """Invert measured qubit parameters into approximate circuit parameters.

    The inversion uses ``EC ~= -alpha`` and
    ``EJ ~= (omega_01 - alpha)^2 / (8 EC)``, then solves the runtime-convention
    dispersive equation

    ``chi ~= -2 g^2 alpha / (Delta (Delta + alpha))``

    for ``Delta = omega_q - omega_r``.
    """

    if alpha >= 0.0:
        raise ValueError("alpha must be negative for a transmon-like Duffing oscillator.")
    ec = float(-alpha)
    ej = float((omega_01 - alpha) ** 2 / (8.0 * ec))
    delta = _solve_detuning_from_measured(
        float(omega_01),
        float(alpha),
        float(chi),
        float(g),
        omega_r=omega_r,
        detuning_branch=detuning_branch,
    )
    resolved_omega_r = float(omega_01 - delta if omega_r is None else omega_r)
    translated = from_transmon_params(
        ej,
        ec,
        float(g),
        resolved_omega_r,
        resonator_dim=int(resonator_dim),
        transmon_dim=int(transmon_dim),
    )
    return HamiltonianParams(
        omega_q=float(omega_01),
        omega_r=float(resolved_omega_r),
        alpha=float(alpha),
        chi=float(translated.chi),
        chi_2=float(translated.chi_2),
        g=float(g),
        delta=float(delta),
        ec=float(ec),
        ej=float(ej),
        synthesis_chi=float(-0.5 * translated.chi),
        synthesis_chi_2=float(-0.5 * translated.chi_2),
        metadata={
            **translated.metadata,
            "detuning_branch": str(detuning_branch),
            "input_chi": float(chi),
        },
    )


__all__ = ["HamiltonianParams", "from_transmon_params", "from_measured"]
