from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import qutip as qt

from cqed_sim.core.frame import FrameSpec
from cqed_sim.core.model import DispersiveTransmonCavityModel
from cqed_sim.experiment.state_prep import StatePreparationSpec, SubsystemStateSpec, prepare_state, qubit_state, vacuum_state
from cqed_sim.sim.extractors import cavity_wigner, reduced_cavity_state
from physics_and_conventions.conventions import to_internal_units


def _hz_to_rad_s(hz: float) -> float:
    return to_internal_units(float(hz))


def _us_to_s(time_us: float) -> float:
    return float(time_us) * 1.0e-6


@dataclass(frozen=True)
class KerrParameterSet:
    name: str
    omega_q_hz: float
    omega_c_hz: float
    omega_ro_hz: float
    alpha_q_hz: float
    kerr_hz: float
    kerr2_hz: float = 0.0
    chi_hz: float = 0.0
    chi2_hz: float = 0.0
    chi3_hz: float = 0.0

    def build_model(self, *, n_cav: int = 28, n_tr: int = 3) -> DispersiveTransmonCavityModel:
        chi_higher = tuple(value for value in (_hz_to_rad_s(self.chi2_hz), _hz_to_rad_s(self.chi3_hz)) if value != 0.0)
        kerr_higher = tuple(value for value in (_hz_to_rad_s(self.kerr2_hz),) if value != 0.0)
        return DispersiveTransmonCavityModel(
            omega_c=_hz_to_rad_s(self.omega_c_hz),
            omega_q=_hz_to_rad_s(self.omega_q_hz),
            alpha=_hz_to_rad_s(self.alpha_q_hz),
            chi=_hz_to_rad_s(self.chi_hz),
            chi_higher=chi_higher,
            kerr=_hz_to_rad_s(self.kerr_hz),
            kerr_higher=kerr_higher,
            n_cav=int(n_cav),
            n_tr=int(n_tr),
        )


KERR_FREE_EVOLUTION_PARAMETER_SETS: dict[str, KerrParameterSet] = {
    "phase_evolution": KerrParameterSet(
        name="phase_evolution",
        omega_q_hz=7_627.05e6,
        omega_c_hz=8_226.787e6,
        omega_ro_hz=9_376.75e6,
        alpha_q_hz=283.4e6,
        kerr_hz=-107.9e3,
        kerr2_hz=3.4e3,
        chi_hz=-8_281.3e3,
        chi2_hz=48.8e3,
        chi3_hz=0.5e3,
    ),
    "value_2": KerrParameterSet(
        name="value_2",
        omega_q_hz=7_627.05e6,
        omega_c_hz=8_226.787e6,
        omega_ro_hz=9_376.75e6,
        alpha_q_hz=283.4e6,
        kerr_hz=-106.2e3,
        kerr2_hz=3.4e3,
        chi_hz=-8_273.0e3,
        chi2_hz=60.9e3,
        chi3_hz=0.0,
    ),
}


@dataclass
class KerrEvolutionSnapshot:
    time_s: float
    time_us: float
    joint_state: qt.Qobj
    cavity_state: qt.Qobj
    cavity_mean: complex
    cavity_photon_number: float
    wigner: dict[str, np.ndarray] | None = None


@dataclass
class KerrFreeEvolutionResult:
    parameter_set: KerrParameterSet
    model: DispersiveTransmonCavityModel
    frame: FrameSpec
    initial_state: qt.Qobj
    state_prep: StatePreparationSpec
    snapshots: list[KerrEvolutionSnapshot]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def times_s(self) -> np.ndarray:
        return np.asarray([snapshot.time_s for snapshot in self.snapshots], dtype=float)

    @property
    def times_us(self) -> np.ndarray:
        return np.asarray([snapshot.time_us for snapshot in self.snapshots], dtype=float)


def available_kerr_parameter_sets() -> tuple[str, ...]:
    return tuple(KERR_FREE_EVOLUTION_PARAMETER_SETS.keys())


def resolve_kerr_parameter_set(parameter_set: str | KerrParameterSet) -> KerrParameterSet:
    if isinstance(parameter_set, KerrParameterSet):
        return parameter_set
    key = str(parameter_set)
    if key not in KERR_FREE_EVOLUTION_PARAMETER_SETS:
        known = ", ".join(sorted(KERR_FREE_EVOLUTION_PARAMETER_SETS))
        raise ValueError(f"Unsupported Kerr parameter set '{parameter_set}'. Known sets: {known}.")
    return KERR_FREE_EVOLUTION_PARAMETER_SETS[key]


def build_kerr_free_evolution_model(
    parameter_set: str | KerrParameterSet = "phase_evolution",
    *,
    n_cav: int = 28,
    n_tr: int = 3,
) -> DispersiveTransmonCavityModel:
    preset = resolve_kerr_parameter_set(parameter_set)
    return preset.build_model(n_cav=n_cav, n_tr=n_tr)


def build_kerr_free_evolution_frame(
    model: DispersiveTransmonCavityModel,
    *,
    use_rotating_frame: bool = True,
) -> FrameSpec:
    if use_rotating_frame:
        return FrameSpec(omega_c_frame=model.omega_c, omega_q_frame=model.omega_q)
    return FrameSpec()


def times_us_to_seconds(times_us: Sequence[float]) -> np.ndarray:
    return np.asarray([_us_to_s(value) for value in times_us], dtype=float)


def _validate_times(times_s: Sequence[float]) -> np.ndarray:
    values = np.asarray(times_s, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("times_s must contain at least one time point.")
    if not np.all(np.isfinite(values)):
        raise ValueError("times_s must be finite.")
    if np.any(values < 0.0):
        raise ValueError("times_s must be non-negative.")
    if np.any(np.diff(values) < 0.0):
        raise ValueError("times_s must be sorted in nondecreasing order.")
    return values


def _wigner_time_mask(times_s: np.ndarray, wigner_times_s: Sequence[float] | None) -> np.ndarray:
    if wigner_times_s is None:
        return np.ones(times_s.shape, dtype=bool)
    if len(wigner_times_s) == 0:
        return np.zeros(times_s.shape, dtype=bool)
    requested = _validate_times(wigner_times_s)
    return np.asarray([np.any(np.isclose(time_s, requested, atol=1.0e-15, rtol=0.0)) for time_s in times_s], dtype=bool)


def _simulate_free_evolution(
    model: DispersiveTransmonCavityModel,
    frame: FrameSpec,
    initial_state: qt.Qobj,
    *,
    time_s: float,
    max_step: float | None,
) -> qt.Qobj:
    if time_s <= 0.0:
        return initial_state
    static_hamiltonian = model.static_hamiltonian(frame)
    propagator = (-1j * static_hamiltonian * float(time_s)).expm()
    if initial_state.isoper:
        return propagator * initial_state * propagator.dag()
    return propagator * initial_state


def _snapshot_from_state(
    state: qt.Qobj,
    *,
    time_s: float,
    include_wigner: bool,
    wigner_n_points: int,
    wigner_extent: float,
) -> KerrEvolutionSnapshot:
    cavity_state = reduced_cavity_state(state)
    dim = int(cavity_state.dims[0][0])
    a = qt.destroy(dim)
    adag = a.dag()
    cavity_mean = complex((cavity_state * a).tr())
    cavity_photon_number = float(np.real((cavity_state * adag * a).tr()))
    wigner = None
    if include_wigner:
        xvec, yvec, values = cavity_wigner(cavity_state, n_points=int(wigner_n_points), extent=float(wigner_extent))
        wigner = {"xvec": xvec, "yvec": yvec, "w": values}
    return KerrEvolutionSnapshot(
        time_s=float(time_s),
        time_us=float(time_s) * 1.0e6,
        joint_state=state,
        cavity_state=cavity_state,
        cavity_mean=cavity_mean,
        cavity_photon_number=cavity_photon_number,
        wigner=wigner,
    )


def run_kerr_free_evolution(
    times_s: Sequence[float],
    *,
    parameter_set: str | KerrParameterSet = "phase_evolution",
    cavity_state: SubsystemStateSpec | None = None,
    qubit: SubsystemStateSpec | None = None,
    state_prep: StatePreparationSpec | None = None,
    n_cav: int = 28,
    n_tr: int = 3,
    use_rotating_frame: bool = True,
    wigner_times_s: Sequence[float] | None = None,
    wigner_n_points: int = 81,
    wigner_extent: float = 5.0,
    max_step: float | None = None,
) -> KerrFreeEvolutionResult:
    if state_prep is not None and (cavity_state is not None or qubit is not None):
        raise ValueError("Provide either state_prep or cavity_state/qubit, not both.")

    times = _validate_times(times_s)
    preset = resolve_kerr_parameter_set(parameter_set)
    model = preset.build_model(n_cav=n_cav, n_tr=n_tr)
    frame = build_kerr_free_evolution_frame(model, use_rotating_frame=use_rotating_frame)

    resolved_state_prep = state_prep or StatePreparationSpec(
        qubit=qubit if qubit is not None else qubit_state("g"),
        storage=cavity_state if cavity_state is not None else vacuum_state(),
    )
    initial_state = prepare_state(model, resolved_state_prep)
    with_wigner = _wigner_time_mask(times, wigner_times_s=wigner_times_s)

    snapshots: list[KerrEvolutionSnapshot] = []
    for time_s, include_wigner in zip(times, with_wigner, strict=True):
        evolved = _simulate_free_evolution(model, frame, initial_state, time_s=float(time_s), max_step=max_step)
        snapshots.append(
            _snapshot_from_state(
                evolved,
                time_s=float(time_s),
                include_wigner=bool(include_wigner),
                wigner_n_points=int(wigner_n_points),
                wigner_extent=float(wigner_extent),
            )
        )

    return KerrFreeEvolutionResult(
        parameter_set=preset,
        model=model,
        frame=frame,
        initial_state=initial_state,
        state_prep=resolved_state_prep,
        snapshots=snapshots,
        metadata={
            "use_rotating_frame": bool(use_rotating_frame),
            "omega_ro_hz": float(preset.omega_ro_hz),
        },
    )


def plot_kerr_wigner_snapshots(
    result: KerrFreeEvolutionResult,
    *,
    max_cols: int = 3,
    show_colorbar: bool = True,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plot_kerr_wigner_snapshots().") from exc

    panels = [snapshot for snapshot in result.snapshots if snapshot.wigner is not None]
    if not panels:
        raise ValueError("Result does not contain Wigner snapshots.")

    n_cols = max(1, min(int(max_cols), len(panels)))
    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.6 * n_cols, 3.5 * n_rows),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.08, "hspace": 0.16},
    )
    all_w = np.concatenate([panel.wigner["w"].ravel() for panel in panels if panel.wigner is not None])
    vmax = float(np.max(np.abs(all_w)))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax) if vmax > 0.0 else None

    image = None
    for axis, panel in zip(axes.ravel(), panels, strict=True):
        assert panel.wigner is not None
        xvec = panel.wigner["xvec"]
        yvec = panel.wigner["yvec"]
        image = axis.imshow(
            panel.wigner["w"],
            origin="lower",
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            cmap="RdBu_r",
            norm=norm,
            aspect="equal",
        )
        axis.set_title(f"t = {panel.time_us:g} us")
        axis.set_xlabel("x")
        axis.set_ylabel("p")

    for axis in axes.ravel()[len(panels):]:
        axis.axis("off")

    if image is not None and show_colorbar:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.86, label="W(x, p)")
    fig.suptitle(f"Kerr free evolution: {result.parameter_set.name}", y=0.98)
    return fig


__all__ = [
    "KERR_FREE_EVOLUTION_PARAMETER_SETS",
    "KerrParameterSet",
    "KerrEvolutionSnapshot",
    "KerrFreeEvolutionResult",
    "available_kerr_parameter_sets",
    "resolve_kerr_parameter_set",
    "build_kerr_free_evolution_model",
    "build_kerr_free_evolution_frame",
    "times_us_to_seconds",
    "run_kerr_free_evolution",
    "plot_kerr_wigner_snapshots",
]