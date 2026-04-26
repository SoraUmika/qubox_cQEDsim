from __future__ import annotations

import numpy as np

from cqed_sim.models import (
    ExplicitPurcellFilterMode,
    MultilevelCQEDModel,
    ReadoutFrame,
    TransmonModel,
    add_explicit_purcell_filter,
)
from cqed_sim.optimization import (
    LinearPointerSeedModel,
    PulseConstraints,
    StrongReadoutObjectiveWeights,
    StrongReadoutOptimizer,
    StrongReadoutOptimizerConfig,
)
from cqed_sim.pulses.clear import clear_readout_seed, square_readout_seed
from cqed_sim.readout.input_output import linear_pointer_response


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def main() -> None:
    twopi = 2.0 * np.pi
    dt = 2.0e-9
    duration = 180.0e-9
    drive_frequency = twopi * 7.0e9
    amplitude = twopi * 1.8e6

    transmon = TransmonModel.from_cosine(
        EJ=twopi * 16.0e9,
        EC=twopi * 0.22e9,
        ng=0.0,
        n_cut=10,
        levels=4,
    ).spectrum()
    readout_model = MultilevelCQEDModel.from_transmon_spectrum(
        transmon,
        resonator_frequency=drive_frequency,
        resonator_levels=5,
        coupling_strength=twopi * 65.0e6,
        rotating_frame=ReadoutFrame(resonator_frequency=drive_frequency),
        counter_rotating=False,
    )
    filtered_model = add_explicit_purcell_filter(
        readout_model,
        ExplicitPurcellFilterMode(
            frequency=drive_frequency,
            levels=3,
            coupling=twopi * 12.0e6,
            kappa=twopi * 18.0e6,
        ),
    )

    seeds = [
        square_readout_seed(
            amplitude=amplitude,
            duration=duration,
            dt=dt,
            drive_frequency=drive_frequency,
        ),
        clear_readout_seed(
            amplitude=amplitude,
            duration=duration,
            dt=dt,
            drive_frequency=drive_frequency,
        ),
    ]

    optimizer = StrongReadoutOptimizer(
        linear_model=LinearPointerSeedModel(
            kappa=twopi * 8.0e6,
            chi=twopi * 1.2e6,
            noise_sigma=0.18,
        ),
        weights=StrongReadoutObjectiveWeights(
            wA=1.0,
            wQ=2.0,
            wL=3.0,
            wR=0.15,
            wE=1.0e-17,
            wS=1.0e-33,
            wM=0.0,
        ),
        constraints=PulseConstraints(
            max_amplitude=twopi * 5.0e6,
            max_slew_rate=twopi * 2.5e15,
            fixed_total_duration=duration,
            drive_frequency=drive_frequency,
        ),
        config=StrongReadoutOptimizerConfig(
            method="Powell",
            maxiter=4,
            n_candidates=3,
            random_seed=7,
            parameter_scale=twopi * 0.15e6,
        ),
    )
    result = optimizer.optimize(seeds)
    best = result.best

    print("Strong-readout optimization summary")
    print(f"tensor order without filter: {readout_model.subsystem_dims} -> |q,n_r>")
    print(f"tensor order with filter:    {filtered_model.subsystem_dims} -> |q,n_r,n_f>")
    print(f"ranked candidates: {len(result.candidates)}")
    print(f"pareto candidates: {len(result.pareto_set)}")
    print(f"best objective: {_fmt(best.objective)}")
    print(f"F_assign: {_fmt(best.metrics.assignment_fidelity)}")
    print(f"F_QND_phys: {_fmt(best.metrics.physical_qnd_fidelity)}")
    print(f"P_0_to_1: {_fmt(best.metrics.p_0_to_1)}")
    print(f"P_1_to_0: {_fmt(best.metrics.p_1_to_0)}")
    print(f"P_leak: {_fmt(best.metrics.leakage_probability)}")
    print(f"residual resonator photons: {_fmt(best.metrics.residual_resonator_photons)}")
    print(f"residual filter photons: {_fmt(best.metrics.residual_filter_photons)}")
    print("confusion matrix convention: P(predicted | prepared)")

    t, alpha_g = linear_pointer_response(
        best.pulse.samples,
        dt=best.pulse.dt,
        kappa=optimizer.linear_model.kappa,
        detuning=-0.5 * optimizer.linear_model.chi,
    )
    _t, alpha_e = linear_pointer_response(
        best.pulse.samples,
        dt=best.pulse.dt,
        kappa=optimizer.linear_model.kappa,
        detuning=0.5 * optimizer.linear_model.chi,
    )
    print(f"mean IQ trace samples: {len(t)}")
    print(f"final alpha_g: {alpha_g[-1].real:.4e} + {alpha_g[-1].imag:.4e}j")
    print(f"final alpha_e: {alpha_e[-1].real:.4e} + {alpha_e[-1].imag:.4e}j")

    print("convergence sweep over Nq and Nr for the best pulse")
    for nq in (3, 4):
        for nr in (4, 5):
            truncated = MultilevelCQEDModel(
                transmon_energies=transmon.shifted_energies[:nq],
                resonator_frequency=drive_frequency,
                resonator_levels=nr,
                coupling_matrix=(twopi * 65.0e6) * transmon.n_matrix[:nq, :nq],
                rotating_frame=ReadoutFrame(resonator_frequency=drive_frequency),
                counter_rotating=False,
            )
            print(f"  Nq={nq}, Nr={nr}, dim={np.prod(truncated.subsystem_dims)}")


if __name__ == "__main__":
    main()
