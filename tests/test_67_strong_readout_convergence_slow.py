from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pytest

from cqed_sim.metrics import ReadoutMetricSet, compute_readout_metrics
from cqed_sim.models import (
    ExplicitPurcellFilterMode,
    IQPulse,
    MultilevelCQEDModel,
    ReadoutFrame,
    TransmonCosineSpec,
    add_explicit_purcell_filter,
    diagonalize_dressed_hamiltonian,
    diagonalize_transmon,
)
from cqed_sim.solvers import MasterEquationConfig, collapse_operators_from_model, solve_master_equation


def _two_state_confusion(separation: float, sigma: float = 0.20) -> np.ndarray:
    p_correct = 0.5 * (1.0 + erf(float(separation) / (2.0 * sqrt(2.0) * float(sigma))))
    return np.array([[p_correct, 1.0 - p_correct], [1.0 - p_correct, p_correct]], dtype=float)


def _basis_state(model, level: int):
    if len(tuple(model.subsystem_dims)) == 3:
        return model.basis_state(level, 0, 0)
    return model.basis_state(level, 0)


def _score_compact_case(
    *,
    nq: int = 3,
    nr: int = 5,
    nf: int | None = None,
    n_cut: int = 9,
    dt: float = 0.04,
) -> ReadoutMetricSet:
    duration = 0.64
    samples = np.full(max(1, int(round(duration / dt))), 0.018 + 0.002j, dtype=np.complex128)
    spectrum = diagonalize_transmon(TransmonCosineSpec(EJ=34.0, EC=0.55, n_cut=n_cut, levels=nq))
    base = MultilevelCQEDModel.from_transmon_spectrum(
        spectrum,
        resonator_frequency=0.0,
        resonator_levels=nr,
        coupling_strength=0.012,
        rotating_frame=ReadoutFrame(resonator_frequency=0.0),
        counter_rotating=False,
    )
    model = base
    if nf is not None:
        model = add_explicit_purcell_filter(
            base,
            ExplicitPurcellFilterMode(frequency=0.0, levels=nf, coupling=0.04, kappa=0.45),
        )
    pulse = IQPulse(samples=samples, dt=dt, drive_frequency=0.0)
    hdata = model.build_hamiltonian(pulse)
    ops = model.operators()
    output = hdata.output_operator if hdata.output_operator is not None else np.sqrt(0.45) * ops["a"]
    c_ops = collapse_operators_from_model(model, kappa_r=0.45)
    e_ops = {"n_r": ops["n_r"]}
    if "n_f" in ops:
        e_ops["n_f"] = ops["n_f"]

    final_by_initial = {}
    integrated_outputs = []
    residual_r = []
    residual_f = []
    for initial in (0, 1):
        result = solve_master_equation(
            hdata,
            _basis_state(model, initial),
            c_ops=c_ops,
            e_ops=e_ops,
            output_operator=output,
            config=MasterEquationConfig(atol=1.0e-8, rtol=1.0e-7),
        )
        final_by_initial[initial] = result.final_state
        integrated_outputs.append(complex(np.trapezoid(result.output_means, x=result.times)))
        residual_r.append(float(result.expectations["n_r"][-1]))
        if "n_f" in result.expectations:
            residual_f.append(float(result.expectations["n_f"][-1]))

    dressed = diagonalize_dressed_hamiltonian(model.static_hamiltonian())
    transitions = dressed.transition_matrix(final_by_initial, measured_levels=(0, 1))
    separation = abs(integrated_outputs[1] - integrated_outputs[0])
    return compute_readout_metrics(
        confusion=_two_state_confusion(separation),
        transition_matrix=transitions.matrix,
        pulse_samples=samples,
        dt=dt,
        residual_resonator=residual_r,
        residual_filter=residual_f,
        leakage_probability=float(np.mean(transitions.missing_weight)),
    )


def _assert_metric_close(left: ReadoutMetricSet, right: ReadoutMetricSet, *, residual_tol: float = 2.0e-2) -> None:
    assert abs(left.assignment_fidelity - right.assignment_fidelity) < 2.0e-2
    assert abs(left.physical_qnd_fidelity - right.physical_qnd_fidelity) < 2.5e-2
    assert abs(left.p_0_to_1 - right.p_0_to_1) < 2.5e-2
    assert abs(left.p_1_to_0 - right.p_1_to_0) < 2.5e-2
    assert abs(left.leakage_probability - right.leakage_probability) < 2.5e-2
    assert abs(left.residual_resonator_photons - right.residual_resonator_photons) < residual_tol
    assert abs(left.residual_filter_photons - right.residual_filter_photons) < residual_tol


@pytest.mark.slow
def test_strong_readout_qubit_level_and_charge_cutoff_convergence() -> None:
    baseline = _score_compact_case(nq=3, nr=5, n_cut=9, dt=0.04)
    refined_levels = _score_compact_case(nq=4, nr=5, n_cut=9, dt=0.04)
    refined_charge = _score_compact_case(nq=3, nr=5, n_cut=11, dt=0.04)

    _assert_metric_close(baseline, refined_levels)
    _assert_metric_close(baseline, refined_charge)


@pytest.mark.slow
def test_strong_readout_resonator_cutoff_convergence() -> None:
    baseline = _score_compact_case(nq=3, nr=5, n_cut=9, dt=0.04)
    refined = _score_compact_case(nq=3, nr=6, n_cut=9, dt=0.04)

    _assert_metric_close(baseline, refined, residual_tol=1.0e-2)


@pytest.mark.slow
def test_strong_readout_filter_cutoff_convergence() -> None:
    baseline = _score_compact_case(nq=3, nr=4, nf=2, n_cut=9, dt=0.05)
    refined = _score_compact_case(nq=3, nr=4, nf=3, n_cut=9, dt=0.05)

    _assert_metric_close(baseline, refined, residual_tol=2.0e-2)


@pytest.mark.slow
def test_strong_readout_timestep_convergence() -> None:
    coarse = _score_compact_case(nq=3, nr=5, n_cut=9, dt=0.04)
    refined = _score_compact_case(nq=3, nr=5, n_cut=9, dt=0.02)

    _assert_metric_close(coarse, refined, residual_tol=2.0e-2)
