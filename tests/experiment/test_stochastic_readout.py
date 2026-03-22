from __future__ import annotations

import numpy as np

from cqed_sim.core import DispersiveReadoutTransmonStorageModel, FrameSpec
from cqed_sim.measurement import (
    ContinuousReadoutSpec,
    ReadoutResonator,
    StrongReadoutMixingSpec,
    build_strong_readout_disturbance,
    integrate_measurement_record,
    simulate_continuous_readout,
    strong_readout_drive_targets,
)
from cqed_sim.pulses import Pulse
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import NoiseSpec, collapse_operators, split_collapse_operators


def _square(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def _readout_model() -> DispersiveReadoutTransmonStorageModel:
    return DispersiveReadoutTransmonStorageModel(
        omega_s=0.0,
        omega_r=5.0,
        omega_q=0.0,
        alpha=-0.25,
        chi_s=0.0,
        chi_r=0.35,
        chi_sr=0.0,
        kerr_s=0.0,
        kerr_r=0.0,
        n_storage=2,
        n_readout=8,
        n_tr=3,
    )


def test_integrate_measurement_record_supports_heterodyne_shape():
    record = np.ones((1, 2, 4), dtype=float)
    integrated = integrate_measurement_record(record, dt=0.5)

    assert integrated.shape == (1, 2)
    assert np.allclose(integrated, 2.0)


def test_build_strong_readout_disturbance_activates_above_threshold():
    resonator = ReadoutResonator(
        omega_r=0.0,
        kappa=1.0,
        g=0.1,
        epsilon=0.0,
        chi=0.2,
    )
    spec = StrongReadoutMixingSpec(
        n_crit=0.2,
        onset_ratio=0.1,
        higher_ladder_scales=(0.4, 0.2),
    )

    low = build_strong_readout_disturbance(resonator, 0.02, dt=0.05, duration=1.0, spec=spec)
    high = build_strong_readout_disturbance(resonator, 0.5, dt=0.05, duration=1.0, spec=spec)
    targets = strong_readout_drive_targets(spec, max_transmon_level=5)

    assert low.peak_activation == 0.0
    assert high.peak_activation > 0.0
    assert high.peak_mean_occupancy > low.peak_mean_occupancy
    assert np.max(np.abs(high.ge_envelope)) > np.max(np.abs(low.ge_envelope))
    assert set(targets) == {"mix_ge", "mix_ef", "mix_high_2_3", "mix_high_3_4"}
    assert set(high.higher_envelopes) == {"mix_high_2_3", "mix_high_3_4"}
    assert np.allclose(high.higher_envelopes["mix_high_2_3"], 0.4 * high.ef_envelope)


def test_split_collapse_operators_promotes_selected_readout_loss():
    model = _readout_model()
    noise = NoiseSpec(t1=7.0, kappa_storage=0.2, kappa_readout=0.4)

    unmonitored, monitored = split_collapse_operators(model, noise, monitored_subsystem="readout")
    combined = collapse_operators(model, noise)

    assert len(monitored) == 1
    assert len(unmonitored) == len(combined) - 1
    assert len(unmonitored) >= 2
    assert any(np.isclose(monitored[0].norm(), op.norm()) for op in combined)


def test_simulate_continuous_readout_returns_measurement_records():
    model = _readout_model()
    compiled = SequenceCompiler(dt=0.05).compile([Pulse("r", 0.0, 0.8, _square, amp=0.15)], t_end=0.85)
    frame = FrameSpec(omega_c_frame=model.omega_s, omega_q_frame=model.omega_q, omega_r_frame=model.omega_r)
    result = simulate_continuous_readout(
        model,
        compiled,
        model.basis_state(0, 0, 0),
        {"r": "readout"},
        noise=NoiseSpec(kappa_readout=0.25),
        spec=ContinuousReadoutSpec(
            frame=frame,
            monitored_subsystem="readout",
            ntraj=3,
            max_step=0.05,
            keep_runs_results=True,
            store_measurement="end",
        ),
    )

    assert abs(result.average_final_state.tr() - 1.0) < 1.0e-6
    assert len(result.trajectories) == 3
    assert len(result.measurement_records) == 3
    assert len(result.monitored_ops) == 1
    assert result.measurement_records[0].shape[-1] == len(compiled.tlist) - 1

    integrated = integrate_measurement_record(result.measurement_records[0], dt=compiled.tlist[1] - compiled.tlist[0])
    assert integrated.shape == result.measurement_records[0].shape[:-1]
