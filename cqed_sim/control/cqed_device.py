"""Example hardware context for a three-line cQED device.

This module provides a ready-to-use factory function that creates a
:class:`~cqed_sim.control.HardwareContext` for a typical cQED device with:

1. **qubit_drive** – XY control line to the transmon qubit.
2. **storage_drive** – displacement / SQR / sideband drive to the storage cavity.
3. **readout_drive** – pulsed dispersive readout drive to the readout resonator.

The factory supports optional per-line hardware imperfections:
gain, propagation delay, and a first-order lowpass filter (bandwidth limit).

Physical conventions
--------------------
* All frequencies are in **rad/s**.
* All times are in **seconds**.
* ``calibration_gain`` for each line converts volts (AWG output voltage at
  the device input plane) to effective **rad/s** coupling strength.  If
  you work with pre-normalised amplitudes (dimensionless, ∈ [−1, 1]) and
  express ``amp`` in ``Pulse`` directly in rad/s, set
  ``calibration_gain = 1.0`` for every line (the default).
* The sign convention for the Hamiltonian operators follows the rest of the
  ``cqed_sim`` package (see :class:`~cqed_sim.pulses.pulse.Pulse`).

Example
-------
.. code-block:: python

    import numpy as np
    from cqed_sim.control.cqed_device import make_three_line_cqed_context

    # Qubit line: 1 dB insertion loss, 2 ns cable delay, 80 MHz bandwidth.
    # Storage line: 1.5 dB loss, 3 ns delay, 60 MHz bandwidth.
    # Readout line: 0.5 dB loss, 1 ns delay, 200 MHz bandwidth.
    ctx = make_three_line_cqed_context(
        qubit_gain=10 ** (-1.0 / 20),
        qubit_delay_s=2e-9,
        qubit_lowpass_hz=80e6,
        storage_gain=10 ** (-1.5 / 20),
        storage_delay_s=3e-9,
        storage_lowpass_hz=60e6,
        readout_gain=10 ** (-0.5 / 20),
        readout_delay_s=1e-9,
        readout_lowpass_hz=200e6,
        dt=1e-9,  # 1 ns sample period for delay conversion
    )

    # Use in pulse simulation:
    from cqed_sim import SequenceCompiler
    compiler = SequenceCompiler(dt=1e-9, hardware_context=ctx)

    # Use in GRAPE optimisation (Mode B):
    grape_hardware_model = ctx.as_hardware_model()

    # Postprocess GRAPE output (Mode A):
    from cqed_sim.control import postprocess_grape_waveforms
    transformed = postprocess_grape_waveforms(ctx, resolved.physical_values,
                                              problem.control_terms, dt=1e-9)
"""
from __future__ import annotations

from typing import Any

import numpy as np

from cqed_sim.control import ControlLine, HardwareContext, delay_samples_from_time
from cqed_sim.optimal_control.hardware import (
    DelayHardwareMap,
    FirstOrderLowPassHardwareMap,
    GainHardwareMap,
)


def make_three_line_cqed_context(
    *,
    # Qubit / transmon XY drive line
    qubit_gain: float = 1.0,
    qubit_delay_s: float = 0.0,
    qubit_lowpass_hz: float | None = None,
    qubit_calibration_gain: float = 1.0,
    qubit_name: str = "qubit",
    # Storage cavity drive line
    storage_gain: float = 1.0,
    storage_delay_s: float = 0.0,
    storage_lowpass_hz: float | None = None,
    storage_calibration_gain: float = 1.0,
    storage_name: str = "storage",
    # Readout resonator drive line
    readout_gain: float = 1.0,
    readout_delay_s: float = 0.0,
    readout_lowpass_hz: float | None = None,
    readout_calibration_gain: float = 1.0,
    readout_name: str = "readout",
    # Shared
    dt: float = 1e-9,
    extra_metadata: dict[str, Any] | None = None,
) -> HardwareContext:
    """Build a :class:`~cqed_sim.control.HardwareContext` for a three-line cQED device.

    Each line has an independent transfer chain consisting of (in order):

    1. **Gain** – models cable loss or amplifier gain.
    2. **Delay** – integer-sample propagation delay.
    3. **Lowpass filter** – first-order RC filter modelling bandwidth limit.
    4. **Calibration gain** – unit conversion (hardware voltage → rad/s).

    Any element whose parameter is at its default (gain = 1, delay = 0,
    no lowpass) is omitted from the map chain so that the identity line
    adds zero overhead.

    Parameters
    ----------
    qubit_gain, storage_gain, readout_gain:
        Linear voltage gain on each line (e.g. ``10 ** (-dB / 20)`` for
        an insertion loss of ``dB`` decibels).
    qubit_delay_s, storage_delay_s, readout_delay_s:
        Physical propagation delay in seconds.  Converted to integer
        samples using ``dt``.
    qubit_lowpass_hz, storage_lowpass_hz, readout_lowpass_hz:
        First-order lowpass cutoff frequency in Hz.  ``None`` means
        no filtering.
    qubit_calibration_gain, storage_calibration_gain, readout_calibration_gain:
        Calibration coefficient (hardware units → Hamiltonian units,
        e.g. volts → rad/s).  Set to ``1.0`` if waveform amplitudes are
        already specified in rad/s.
    qubit_name, storage_name, readout_name:
        Logical channel names (must match the ``channel`` field of the
        :class:`~cqed_sim.pulses.pulse.Pulse` objects and the
        ``export_channel`` of the
        :class:`~cqed_sim.optimal_control.problems.ControlTerm` objects).
    dt:
        Sample period in seconds.  Used only to convert delay times to
        integer sample counts.
    extra_metadata:
        Optional extra metadata stored on the returned
        :class:`~cqed_sim.control.HardwareContext`.

    Returns
    -------
    HardwareContext
    """
    lines: dict[str, ControlLine] = {}
    for name, gain, delay_s, lowpass_hz, cal_gain in [
        (qubit_name,   qubit_gain,   qubit_delay_s,   qubit_lowpass_hz,   qubit_calibration_gain),
        (storage_name, storage_gain, storage_delay_s, storage_lowpass_hz, storage_calibration_gain),
        (readout_name, readout_gain, readout_delay_s, readout_lowpass_hz, readout_calibration_gain),
    ]:
        maps: list = []
        if abs(float(gain) - 1.0) > 1e-15:
            maps.append(GainHardwareMap(gain=float(gain)))
        delay_samp = delay_samples_from_time(float(delay_s), float(dt))
        if delay_samp > 0:
            maps.append(DelayHardwareMap(delay_samples=delay_samp))
        if lowpass_hz is not None and float(lowpass_hz) > 0.0:
            maps.append(FirstOrderLowPassHardwareMap(cutoff_hz=float(lowpass_hz)))
        lines[name] = ControlLine(
            name=name,
            transfer_maps=tuple(maps),
            calibration_gain=float(cal_gain),
            metadata={
                "gain": float(gain),
                "delay_s": float(delay_s),
                "delay_samples": delay_samp,
                "lowpass_hz": lowpass_hz,
                "calibration_gain": float(cal_gain),
            },
        )

    meta: dict[str, Any] = {
        "description": "Three-line cQED device (qubit / storage / readout)",
        "dt_s": float(dt),
    }
    if extra_metadata:
        meta.update(extra_metadata)

    return HardwareContext(lines=lines, metadata=meta)


__all__ = ["make_three_line_cqed_context"]
