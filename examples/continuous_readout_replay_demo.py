"""Minimal continuous-readout replay demo with strong-readout disturbance estimates.

This example keeps the workflow intentionally small:

1. Build a three-mode readout model.
2. Compile a simple readout pulse.
3. Estimate occupancy-activated strong-readout disturbance envelopes.
4. Run stochastic continuous-measurement replay and integrate one trajectory.
"""

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
)
from cqed_sim.pulses import Pulse
from cqed_sim.sequence import SequenceCompiler
from cqed_sim.sim import NoiseSpec


def square_envelope(t_rel: np.ndarray) -> np.ndarray:
    return np.ones_like(t_rel, dtype=np.complex128)


def main() -> None:
    model = DispersiveReadoutTransmonStorageModel(
        omega_s=2.0 * np.pi * 5.241e9,
        omega_r=2.0 * np.pi * 8.597e9,
        omega_q=2.0 * np.pi * 6.150e9,
        alpha=2.0 * np.pi * (-255.0e6),
        chi_s=0.0,
        chi_r=2.0 * np.pi * (-2.84e6),
        chi_sr=0.0,
        kerr_s=0.0,
        kerr_r=2.0 * np.pi * (-28.0e3),
        n_storage=2,
        n_readout=14,
        n_tr=3,
    )
    frame = FrameSpec(
        omega_c_frame=model.omega_s,
        omega_q_frame=model.omega_q,
        omega_r_frame=model.omega_r,
    )

    dt = 4.0e-9
    pulse = Pulse("readout", 0.0, 240.0e-9, square_envelope, amp=2.0 * np.pi * 1.6e6)
    compiled = SequenceCompiler(dt=dt).compile([pulse], t_end=260.0e-9)

    drive_envelope = np.asarray(compiled.channels["readout"].distorted[:-1], dtype=np.complex128)
    resonator = ReadoutResonator(
        omega_r=model.omega_r,
        kappa=2.0 * np.pi * 2.4e6,
        g=2.0 * np.pi * 85.0e6,
        epsilon=0.0,
        chi=model.chi_r,
        drive_frequency=model.omega_r,
    )
    disturbance = build_strong_readout_disturbance(
        resonator,
        drive_envelope,
        dt=dt,
        spec=StrongReadoutMixingSpec(n_crit=20.0),
        drive_frequency=model.omega_r,
    )

    replay = simulate_continuous_readout(
        model,
        compiled,
        model.basis_state(0, 0, 0),
        {"readout": "readout"},
        noise=NoiseSpec(kappa_readout=2.0 * np.pi * 2.4e6),
        spec=ContinuousReadoutSpec(
            frame=frame,
            monitored_subsystem="readout",
            ntraj=4,
            max_step=dt,
        ),
    )

    integrated = integrate_measurement_record(replay.measurement_records[0], dt=dt)
    print(f"peak_mean_occupancy={disturbance.peak_mean_occupancy:.3f}")
    print(f"peak_activation={disturbance.peak_activation:.3f}")
    print("integrated_record_shape=", integrated.shape)
    print("final_P_g=", replay.average_expectations["P_g"][-1])
    print("final_P_f=", replay.average_expectations["P_f"][-1])


if __name__ == "__main__":
    main()
