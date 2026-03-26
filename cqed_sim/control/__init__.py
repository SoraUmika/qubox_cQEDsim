"""Hardware-aware control transfer layer for cQED simulations.

Conceptual model
----------------
For each physical control line *j* the simulation chain is:

    u_prog_j(t)  --[T_j]-->  u_dev_j(t)  --[f_j]-->  c_j(t)

and the time-dependent Hamiltonian is:

    H(t) = H_0 + sum_j  c_j(t) * O_j

where:

* **u_prog_j(t)** is the *programmed* waveform (Pulse envelope, GRAPE output, …).
* **T_j** is the *hardware transfer model* for line *j*: an ordered chain of
  :class:`~cqed_sim.optimal_control.hardware.HardwareMap` objects
  (gain, delay, FIR filter, lowpass, …).
* **u_dev_j(t)** is the device-level signal that actually reaches the chip.
* **f_j** is the *calibration map* that converts hardware units to the
  Hamiltonian-coefficient units used by the simulator (e.g. volts → rad/s).
  Represented by a :class:`~cqed_sim.control.calibration.CalibrationMap`
  instance (defaulting to :class:`~cqed_sim.control.calibration.LinearCalibrationMap`).
* **O_j** is the drive operator for line *j* (defined by the model).

For a linear time-invariant (LTI) line in frequency space::

    U_dev_j(ω) = H_j(ω) · U_prog_j(ω)

which in the time domain is a convolution::

    u_dev_j(t) = (h_j * u_prog_j)(t)

Units note
----------
The simulator is internally unit-coherent and does not enforce a specific
unit system.  The recommended convention is:

* Frequencies / angular frequencies in **rad/s**.
* Times in **seconds** (s).
* ``u_prog`` amplitudes in **volts** (AWG output) or
  **abstract DAC-normalised units** (dimensionless, −1 to 1).
* ``calibration_gain`` / :class:`~cqed_sim.control.calibration.LinearCalibrationMap`
  converts from hardware voltage to effective Rabi/drive **angular frequency (rad/s)**.
* Hamiltonian operators in **rad/s** (so H has units of rad/s, and
  ``exp(−i H dt)`` is dimensionless as expected).

Use :attr:`ControlLine.programmed_unit`, :attr:`ControlLine.device_unit`, and
:attr:`ControlLine.coefficient_unit` to document the physical units of each
stage explicitly.

Line-to-Hamiltonian coupling
-----------------------------
Each :class:`ControlLine` drives one operator *O_j* in the Hamiltonian.  Use
:attr:`ControlLine.operator_label` to record what *O_j* is (e.g.
``"σ_x / 2"``, ``"(a + a†) / 2"``, ``"n = a†a"``) and
:attr:`ControlLine.frame` to record the rotating frame (e.g. ``"lab"``,
``"rotating_qubit"``, ``"interaction"``).  The sign convention is that
``H(t) = H_0 + Σ_j c_j(t) · O_j`` with ``c_j(t) > 0`` meaning a positive
contribution.

Public API
----------
:class:`ControlLine`
    A single named hardware line with transfer model, calibration, and
    physical-unit / coupling metadata.

:class:`HardwareContext`
    A collection of :class:`ControlLine` objects providing a unified
    programmed-control → Hamiltonian-coefficient transformation.
    Supports JSON serialization via :meth:`~HardwareContext.to_dict`,
    :meth:`~HardwareContext.from_dict`, :meth:`~HardwareContext.save`,
    and :meth:`~HardwareContext.load`.

:func:`postprocess_grape_waveforms`
    Apply a :class:`HardwareContext` to the physical waveforms produced by
    a GRAPE optimisation (Mode A hardware-aware evaluation).

:func:`delay_samples_from_time`
    Helper to convert a physical time delay to an integer sample count.

:func:`hardware_map_to_dict` / :func:`hardware_map_from_dict`
    Serialize / deserialize individual
    :class:`~cqed_sim.optimal_control.hardware.HardwareMap` objects.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TYPE_CHECKING

import numpy as np

from cqed_sim.control.calibration import (
    CalibrationMap,
    LinearCalibrationMap,
    CallableCalibrationMap,
    calibration_map_from_dict,
)
from cqed_sim.optimal_control.hardware import (
    HardwareMap,
    HardwareModel,
    HardwareMapReport,
    _AppliedHardwareMap,
    GainHardwareMap,
    DelayHardwareMap,
    FIRHardwareMap,
    FirstOrderLowPassHardwareMap,
    BoundaryWindowHardwareMap,
    QuantizationHardwareMap,
    SmoothIQRadiusLimitHardwareMap,
    selected_control_indices,
    selected_iq_pairs,
)
from cqed_sim.optimal_control.parameterizations import PiecewiseConstantTimeGrid

if TYPE_CHECKING:
    from cqed_sim.optimal_control.problems import ControlTerm
    from cqed_sim.optimal_control.hardware import ResolvedControlWaveforms


# ---------------------------------------------------------------------------
# Internal helper: minimal duck-typed ControlTerm for waveform contexts
# ---------------------------------------------------------------------------

def _waveform_control_term(name: str, export_channel: str | None = None) -> Any:
    """Return a SimpleNamespace that duck-types as a ControlTerm for HardwareMap.apply().

    Only the attributes accessed by selected_control_indices / selected_iq_pairs
    are populated: name, export_channel, quadrature, amplitude_bounds.
    """
    return SimpleNamespace(
        name=str(name),
        export_channel=export_channel,
        quadrature="SCALAR",
        amplitude_bounds=(-float("inf"), float("inf")),
    )


# ---------------------------------------------------------------------------
# HardwareMap serialization helpers
# ---------------------------------------------------------------------------

def hardware_map_to_dict(m: HardwareMap) -> dict[str, Any]:
    """Serialize a :class:`~cqed_sim.optimal_control.hardware.HardwareMap`
    to a JSON-ready dict.

    All built-in map types are supported.  Custom subclasses raise
    :class:`TypeError`.

    Parameters
    ----------
    m:
        The hardware map to serialize.

    Returns
    -------
    dict
        JSON-ready dict with a ``"type"`` key naming the map class.

    Raises
    ------
    TypeError
        If *m* is not a known serializable type.
    """
    # Try to import FrequencyResponseHardwareMap lazily to avoid circular imports
    try:
        from cqed_sim.optimal_control.hardware import FrequencyResponseHardwareMap
        _freq_type = FrequencyResponseHardwareMap
    except ImportError:
        _freq_type = None  # type: ignore[assignment]

    if _freq_type is not None and isinstance(m, _freq_type):
        resp = np.asarray(m.response, dtype=complex)
        return {
            "type": "FrequencyResponseHardwareMap",
            "frequencies_hz": [float(f) for f in m.frequencies_hz],
            "response_real": [float(r) for r in resp.real],
            "response_imag": [float(r) for r in resp.imag],
            "n_taps": int(m.n_taps),
            "dt_s": float(m.dt_s),
            "control_names": list(m.control_names),
            "export_channels": list(m.export_channels),
        }

    _SIMPLE_TYPES = (
        GainHardwareMap,
        DelayHardwareMap,
        FIRHardwareMap,
        FirstOrderLowPassHardwareMap,
        BoundaryWindowHardwareMap,
        QuantizationHardwareMap,
        SmoothIQRadiusLimitHardwareMap,
    )
    if isinstance(m, _SIMPLE_TYPES):
        d: dict[str, Any] = {}
        for f in dataclasses.fields(m):  # type: ignore[arg-type]
            v = getattr(m, f.name)
            d[f.name] = list(v) if isinstance(v, tuple) else v
        d["type"] = type(m).__name__
        return d

    raise TypeError(
        f"Cannot serialize HardwareMap of type '{type(m).__name__}'.  "
        "Only built-in HardwareMap subclasses are supported."
    )


def hardware_map_from_dict(data: dict[str, Any]) -> HardwareMap:
    """Reconstruct a :class:`~cqed_sim.optimal_control.hardware.HardwareMap`
    from its serialized dict.

    Parameters
    ----------
    data:
        Output of :func:`hardware_map_to_dict`.

    Returns
    -------
    HardwareMap

    Raises
    ------
    ValueError
        If the ``"type"`` field is unknown.
    """
    t = str(data.get("type", ""))

    def _t(v: Any) -> tuple:
        return tuple(v) if isinstance(v, list) else tuple(v)

    if t == "GainHardwareMap":
        return GainHardwareMap(
            gain=float(data["gain"]),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )
    if t == "DelayHardwareMap":
        return DelayHardwareMap(
            delay_samples=int(data["delay_samples"]),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )
    if t == "FIRHardwareMap":
        return FIRHardwareMap(
            kernel=tuple(float(v) for v in data["kernel"]),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )
    if t == "FirstOrderLowPassHardwareMap":
        return FirstOrderLowPassHardwareMap(
            cutoff_hz=float(data["cutoff_hz"]),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )
    if t == "BoundaryWindowHardwareMap":
        return BoundaryWindowHardwareMap(
            ramp_slices=int(data["ramp_slices"]),
            apply_start=bool(data.get("apply_start", True)),
            apply_end=bool(data.get("apply_end", True)),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )
    if t == "QuantizationHardwareMap":
        return QuantizationHardwareMap(
            n_bits=int(data["n_bits"]),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )
    if t == "SmoothIQRadiusLimitHardwareMap":
        return SmoothIQRadiusLimitHardwareMap(
            amplitude_max=float(data["amplitude_max"]),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
            epsilon=float(data.get("epsilon", 1e-12)),
        )
    if t == "FrequencyResponseHardwareMap":
        try:
            from cqed_sim.optimal_control.hardware import FrequencyResponseHardwareMap
        except ImportError as exc:
            raise ValueError("FrequencyResponseHardwareMap not available.") from exc
        resp_real = [float(v) for v in data["response_real"]]
        resp_imag = [float(v) for v in data["response_imag"]]
        response = tuple(r + 1j * i for r, i in zip(resp_real, resp_imag))
        return FrequencyResponseHardwareMap(
            frequencies_hz=tuple(float(v) for v in data["frequencies_hz"]),
            response=response,
            n_taps=int(data.get("n_taps", 32)),
            dt_s=float(data.get("dt_s", 1e-9)),
            control_names=_t(data.get("control_names", ())),
            export_channels=_t(data.get("export_channels", ())),
        )

    raise ValueError(
        f"Unknown HardwareMap type '{t}'.  "
        "Known types: GainHardwareMap, DelayHardwareMap, FIRHardwareMap, "
        "FirstOrderLowPassHardwareMap, BoundaryWindowHardwareMap, "
        "QuantizationHardwareMap, SmoothIQRadiusLimitHardwareMap, "
        "FrequencyResponseHardwareMap."
    )


# ---------------------------------------------------------------------------
# ControlLine
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ControlLine:
    """A single named hardware control line with transfer model, calibration,
    and physical-unit / coupling metadata.

    The full transformation pipeline for this line is:

        u_prog(t)  --[transfer_maps]-->  u_dev(t)  --[calibration_map]-->  c(t)

    Parameters
    ----------
    name:
        Unique identifier for this line (e.g. ``"qubit_drive"``,
        ``"storage_drive"``, ``"readout_drive"``).  This name is also used
        as the ``export_channel`` selector when the line's maps are assembled
        into a GRAPE :class:`~cqed_sim.optimal_control.hardware.HardwareModel`.
    transfer_maps:
        Ordered sequence of
        :class:`~cqed_sim.optimal_control.hardware.HardwareMap` objects
        representing the hardware transfer chain T_j.  Applied left-to-right
        (the first map is applied first).  Defaults to an empty tuple
        (identity transfer).
    calibration_gain:
        **Backward-compatible shortcut.**  When ``calibration_map=None`` (the
        default), a :class:`~cqed_sim.control.calibration.LinearCalibrationMap`
        is created automatically from this float.  Ignored when
        ``calibration_map`` is explicitly set.  Defaults to ``1.0``.
    calibration_map:
        A :class:`~cqed_sim.control.calibration.CalibrationMap` instance that
        converts ``u_dev(t)`` to ``c(t)``.  If ``None`` (the default), a
        :class:`~cqed_sim.control.calibration.LinearCalibrationMap` is created
        from ``calibration_gain`` in :meth:`__post_init__`.
    programmed_unit:
        Physical unit of the programmed waveform ``u_prog`` (e.g. ``"V"``,
        ``"normalized"``, ``"rad/s"``).  Documentation only; not enforced.
    device_unit:
        Physical unit of the device-level signal ``u_dev`` after the transfer
        chain (e.g. ``"V"``).  Documentation only.
    coefficient_unit:
        Physical unit of the Hamiltonian coefficient ``c(t)`` output by the
        calibration map (e.g. ``"rad/s"``).  Documentation only.
    operator_label:
        Human-readable label for the Hamiltonian operator *O_j* driven by
        this line (e.g. ``"σ_x / 2"``, ``"(a + a†) / 2"``).
    frame:
        Rotating frame in which the control operates (e.g. ``"lab"``,
        ``"rotating_qubit"``, ``"interaction"``).
    metadata:
        Optional free-form metadata dict.

    Notes
    -----
    :meth:`apply_to_waveform` operates on **complex** baseband arrays
    (suitable for the pulse-simulation path).  Each :class:`HardwareMap`
    in ``transfer_maps`` is applied independently to the real (I) and
    imaginary (Q) parts so that linear transfer functions work correctly
    for complex envelopes.

    :class:`~cqed_sim.optimal_control.hardware.QuantizationHardwareMap`
    cannot be used with :meth:`apply_to_waveform` because it requires
    finite ``amplitude_bounds`` (use
    :class:`~cqed_sim.pulses.hardware.HardwareConfig`'s ``amplitude_bits``
    for the pulse path instead).
    """

    name: str
    transfer_maps: tuple[HardwareMap, ...] = ()
    calibration_gain: float = 1.0
    calibration_map: CalibrationMap | None = None
    programmed_unit: str | None = None
    device_unit: str | None = None
    coefficient_unit: str | None = None
    operator_label: str | None = None
    frame: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "transfer_maps", tuple(self.transfer_maps))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("ControlLine.name must be a non-empty string.")
        # Normalise calibration_map: create LinearCalibrationMap from
        # calibration_gain if calibration_map was not explicitly provided.
        if self.calibration_map is None:
            object.__setattr__(
                self,
                "calibration_map",
                LinearCalibrationMap(gain=float(self.calibration_gain)),
            )

    # ------------------------------------------------------------------
    # Pulse-path interface: apply to a dense complex waveform
    # ------------------------------------------------------------------

    def apply_to_waveform(self, waveform: np.ndarray, *, dt: float) -> np.ndarray:
        """Apply the transfer chain and calibration to a complex baseband waveform.

        Parameters
        ----------
        waveform:
            1-D complex array of shape ``(n_steps,)`` representing the
            programmed envelope samples (e.g. the ``.distorted`` output of
            :class:`~cqed_sim.sequence.scheduler.SequenceCompiler`).
        dt:
            Sample period in seconds (used to build a uniform
            :class:`~cqed_sim.optimal_control.parameterizations.PiecewiseConstantTimeGrid`).

        Returns
        -------
        np.ndarray
            Complex array of shape ``(n_steps,)`` representing
            ``c(t) = calibration_map.apply(u_dev(t))``.
        """
        arr = np.asarray(waveform, dtype=np.complex128)
        n = arr.size
        if n == 0:
            return arr.copy()

        time_grid = PiecewiseConstantTimeGrid.uniform(steps=n, dt_s=float(dt))
        dummy = (_waveform_control_term(self.name, export_channel=self.name),)

        def _apply_maps_real(x_real: np.ndarray) -> np.ndarray:
            # x_real: 1-D float array of length n_steps
            current = np.asarray(x_real, dtype=float).reshape(1, -1)
            for hw_map in self.transfer_maps:
                applied = hw_map.apply(current, control_terms=dummy, time_grid=time_grid)
                current = np.asarray(applied.values, dtype=float)
            return current[0]

        real_out = _apply_maps_real(arr.real)
        imag_out = _apply_maps_real(arr.imag)
        combined = real_out + 1j * imag_out
        assert self.calibration_map is not None  # guaranteed by __post_init__
        return np.asarray(self.calibration_map.apply(combined), dtype=np.complex128)

    # ------------------------------------------------------------------
    # GRAPE-path interface: build a HardwareModel for ControlProblem
    # ------------------------------------------------------------------

    def as_hardware_model(self, *, export_channel: str | None = None) -> HardwareModel:
        """Build a :class:`~cqed_sim.optimal_control.hardware.HardwareModel` for this line.

        Each map in ``transfer_maps`` is wrapped so that it only applies to
        controls whose ``export_channel`` matches ``export_channel`` (defaults
        to ``self.name``).  If a map already has ``export_channels`` set,
        those take precedence (the map is used unchanged).

        A calibration map equivalent (e.g.
        :class:`~cqed_sim.optimal_control.hardware.GainHardwareMap` for
        :class:`~cqed_sim.control.calibration.LinearCalibrationMap`) is
        appended automatically when the calibration map supports it.

        Parameters
        ----------
        export_channel:
            Override the export-channel filter.  Defaults to ``self.name``.

        Notes
        -----
        :class:`~cqed_sim.control.calibration.CallableCalibrationMap` does
        not produce a hardware map equivalent, so it is silently skipped.
        GRAPE Mode B will therefore not include the nonlinear calibration;
        use Mode A (:func:`postprocess_grape_waveforms`) in that case.
        """
        ch = str(export_channel) if export_channel is not None else self.name
        all_maps: list[HardwareMap] = []
        for hw_map in self.transfer_maps:
            # Add channel filter only when the map has no existing filter.
            if hasattr(hw_map, "export_channels") and not hw_map.export_channels:
                try:
                    hw_map = dataclasses.replace(hw_map, export_channels=(ch,))
                except TypeError:
                    pass  # custom map without export_channels – use as-is
            all_maps.append(hw_map)
        # Append calibration as a hardware map if supported.
        assert self.calibration_map is not None
        cal_hw = self.calibration_map.as_hardware_map(export_channels=(ch,))
        if cal_hw is not None:
            all_maps.append(cal_hw)
        return HardwareModel(maps=tuple(all_maps))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict.

        Raises
        ------
        TypeError
            If :attr:`calibration_map` cannot be serialized (e.g.
            :class:`~cqed_sim.control.calibration.CallableCalibrationMap`).
        """
        assert self.calibration_map is not None
        return {
            "name": str(self.name),
            "transfer_maps": [hardware_map_to_dict(m) for m in self.transfer_maps],
            "calibration_gain": float(self.calibration_gain),
            "calibration_map": self.calibration_map.to_dict(),
            "programmed_unit": self.programmed_unit,
            "device_unit": self.device_unit,
            "coefficient_unit": self.coefficient_unit,
            "operator_label": self.operator_label,
            "frame": self.frame,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControlLine":
        """Reconstruct from a serialized dict.

        Parameters
        ----------
        data:
            Output of :meth:`to_dict`.
        """
        cal_map_data = data.get("calibration_map")
        cal_map: CalibrationMap | None = (
            calibration_map_from_dict(cal_map_data) if cal_map_data is not None else None
        )
        return cls(
            name=str(data["name"]),
            transfer_maps=tuple(
                hardware_map_from_dict(m) for m in data.get("transfer_maps", [])
            ),
            calibration_gain=float(data.get("calibration_gain", 1.0)),
            calibration_map=cal_map,
            programmed_unit=data.get("programmed_unit"),
            device_unit=data.get("device_unit"),
            coefficient_unit=data.get("coefficient_unit"),
            operator_label=data.get("operator_label"),
            frame=data.get("frame"),
            metadata=dict(data.get("metadata", {})),
        )


# ---------------------------------------------------------------------------
# HardwareContext
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HardwareContext:
    """A collection of named :class:`ControlLine` objects.

    ``HardwareContext`` is the top-level hardware configuration object.
    It maps programmed controls (one per line) to device-effective
    Hamiltonian coefficients:

        { u_prog_j(t) }_j  -->  { c_j(t) }_j

    Usage
    -----
    **Pulse-simulation path** (via :class:`~cqed_sim.sequence.scheduler.SequenceCompiler`)::

        ctx = HardwareContext(lines={
            "qubit":   ControlLine("qubit",   transfer_maps=(GainHardwareMap(1.5),)),
            "storage": ControlLine("storage", transfer_maps=(FIRHardwareMap((0.7, 0.3)),),
                                  calibration_gain=2e9 * np.pi),
        })
        compiler = SequenceCompiler(dt=1e-9, hardware_context=ctx)
        compiled  = compiler.compile(pulses)

    The compiled channel waveforms (``compiled.channels[ch].distorted``) will
    reflect the hardware-transformed signals.

    **GRAPE path** (Mode A – postprocess GRAPE output)::

        transformed = postprocess_grape_waveforms(ctx, resolved.physical_values,
                                                  control_terms, dt=problem.time_grid.step_durations_s[0])

    **GRAPE path** (Mode B – hardware-aware optimisation)::

        problem = ControlProblem(..., hardware_model=ctx.as_hardware_model())

    **Serialization**::

        ctx.save("device.json")
        ctx_loaded = HardwareContext.load("device.json")

    Parameters
    ----------
    lines:
        Dict mapping line name → :class:`ControlLine`.
    metadata:
        Optional free-form context metadata.
    """

    lines: dict[str, ControlLine] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate line names match dict keys
        for key, line in self.lines.items():
            if str(key) != line.name:
                raise ValueError(
                    f"HardwareContext.lines key '{key}' does not match "
                    f"ControlLine.name '{line.name}'.  Use the line name as the dict key."
                )

    # ------------------------------------------------------------------
    # Pulse-path interface
    # ------------------------------------------------------------------

    def apply_to_waveform(self, line_name: str, waveform: np.ndarray, *, dt: float) -> np.ndarray:
        """Apply the named line's full pipeline to a complex baseband waveform.

        Parameters
        ----------
        line_name:
            Must match a key in :attr:`lines`.
        waveform:
            1-D complex array ``(n_steps,)`` of programmed samples.
        dt:
            Sample period in seconds.

        Returns
        -------
        np.ndarray
            Transformed complex array ``c(t)`` of shape ``(n_steps,)``.

        Raises
        ------
        KeyError
            If ``line_name`` is not present in :attr:`lines`.
        """
        return self.lines[line_name].apply_to_waveform(waveform, dt=dt)

    def apply_to_compiled_channels(
        self,
        channels: dict[str, Any],
        *,
        dt: float,
    ) -> dict[str, Any]:
        """Apply each matching line's transfer chain to a dict of compiled channels.

        Channels whose name is not present in :attr:`lines` are returned
        unchanged.  ``channel.distorted`` is replaced; ``.baseband`` and
        ``.rf`` are preserved as-is (they represent the pre-hardware-context
        signal).

        Parameters
        ----------
        channels:
            Dict ``{channel_name: CompiledChannel}`` from
            :class:`~cqed_sim.sequence.scheduler.CompiledSequence`.
        dt:
            Sample period in seconds.

        Returns
        -------
        dict
            A new dict with transformed ``distorted`` arrays.
        """
        from cqed_sim.sequence.scheduler import CompiledChannel

        result: dict[str, Any] = {}
        for ch_name, compiled_ch in channels.items():
            if ch_name in self.lines:
                new_distorted = self.lines[ch_name].apply_to_waveform(
                    compiled_ch.distorted, dt=dt
                )
                result[ch_name] = CompiledChannel(
                    baseband=compiled_ch.baseband,
                    distorted=new_distorted,
                    rf=compiled_ch.rf,
                )
            else:
                result[ch_name] = compiled_ch
        return result

    # ------------------------------------------------------------------
    # GRAPE-path interface
    # ------------------------------------------------------------------

    def as_hardware_model(self) -> HardwareModel:
        """Build a combined :class:`~cqed_sim.optimal_control.hardware.HardwareModel`.

        All lines' transfer maps are merged into a single
        :class:`~cqed_sim.optimal_control.hardware.HardwareModel`.  Each
        map is restricted to its line's export channel so that multiple
        lines do not interfere with each other.

        The resulting model can be passed directly to
        :class:`~cqed_sim.optimal_control.problems.ControlProblem` as
        ``hardware_model`` to enable Mode B hardware-aware GRAPE
        optimisation.

        Returns
        -------
        HardwareModel
            Combined hardware model for all lines.

        Notes
        -----
        For Mode B hardware-aware GRAPE to compute *exact* gradients through
        the hardware layer, all maps must have exact pullbacks.  All built-in
        :class:`~cqed_sim.optimal_control.hardware.HardwareMap` subclasses
        and :class:`~cqed_sim.control.calibration.LinearCalibrationMap`
        satisfy this.
        :class:`~cqed_sim.optimal_control.hardware.QuantizationHardwareMap`
        uses a straight-through estimator.
        :class:`~cqed_sim.control.calibration.CallableCalibrationMap` is
        silently skipped (Mode A recommended for nonlinear calibrations).
        """
        all_maps: list[HardwareMap] = []
        for line in self.lines.values():
            partial_model = line.as_hardware_model()
            all_maps.extend(partial_model.maps)
        return HardwareModel(maps=tuple(all_maps))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def line_names(self) -> tuple[str, ...]:
        """Names of all registered control lines."""
        return tuple(self.lines.keys())

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-ready dict.

        The dict includes a ``"version"`` key for forward compatibility.

        Raises
        ------
        TypeError
            If any :class:`ControlLine`'s calibration map cannot be serialized.
        """
        from cqed_sim.optimal_control.utils import json_ready

        payload: dict[str, Any] = {
            "version": "1.0",
            "metadata": dict(self.metadata),
            "lines": {name: line.to_dict() for name, line in self.lines.items()},
        }
        return json_ready(payload)

    def save(self, path: str | Path) -> Path:
        """Serialize and save to a JSON file.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"device.json"``).

        Returns
        -------
        Path
            The resolved absolute path of the written file.

        Raises
        ------
        TypeError
            If any :class:`ControlLine`'s calibration map cannot be serialized.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HardwareContext":
        """Reconstruct from a serialized dict.

        Parameters
        ----------
        data:
            Output of :meth:`to_dict`.
        """
        lines_data: dict[str, Any] = data.get("lines", {})
        lines = {name: ControlLine.from_dict(ld) for name, ld in lines_data.items()}
        return cls(
            lines=lines,
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HardwareContext":
        """Load a :class:`HardwareContext` from a JSON file.

        Parameters
        ----------
        path:
            Path to a JSON file previously written by :meth:`save`.

        Returns
        -------
        HardwareContext
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# GRAPE Mode A: postprocess waveforms
# ---------------------------------------------------------------------------

def postprocess_grape_waveforms(
    context: HardwareContext,
    physical_values: np.ndarray,
    control_terms: "tuple[ControlTerm, ...]",
    *,
    dt: float,
) -> np.ndarray:
    """Apply a :class:`HardwareContext` to GRAPE-resolved physical waveforms.

    This implements **Mode A** hardware-aware evaluation: after GRAPE has
    produced an optimised control schedule, transform the waveforms through
    the hardware model before evaluating fidelity or running further
    simulations.

    The pipeline is:
        u_prog_j(t)  [GRAPE physical_values, per control term]
            -->  HardwareContext line transfer + calibration
            -->  c_j(t)  [returned array]

    For IQ-paired control terms (``quadrature`` in ``{"I","Q"}`` sharing the
    same ``export_channel``), the I and Q arrays are combined into a complex
    phasor, transformed by the matching :class:`ControlLine`, and split back.

    For scalar control terms, the transform is applied to the real-valued
    waveform.

    Control terms whose ``export_channel`` is not present in
    ``context.lines`` are left unchanged.

    Parameters
    ----------
    context:
        The hardware context containing per-line transfer models.
    physical_values:
        Array of shape ``(n_controls, n_steps)`` from
        :attr:`~cqed_sim.optimal_control.hardware.ResolvedControlWaveforms.physical_values`.
    control_terms:
        Tuple of :class:`~cqed_sim.optimal_control.problems.ControlTerm` objects
        in the same order as the first axis of ``physical_values``.
    dt:
        Sample period in seconds.  Used to construct the time grid for
        the hardware maps.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_controls, n_steps)`` with hardware-transformed
        waveform values.
    """
    result = np.array(physical_values, dtype=float, copy=True)

    # Process IQ channel pairs first
    processed_indices: set[int] = set()
    for line_name in context.line_names:
        iq_pairs = selected_iq_pairs(control_terms, export_channels=(line_name,))
        for _channel, i_idx, q_idx in iq_pairs:
            complex_wave = physical_values[i_idx] + 1j * physical_values[q_idx]
            transformed = context.lines[line_name].apply_to_waveform(complex_wave, dt=dt)
            result[i_idx] = transformed.real
            result[q_idx] = transformed.imag
            processed_indices.add(i_idx)
            processed_indices.add(q_idx)

    # Process remaining scalar controls
    for line_name in context.line_names:
        scalar_indices = selected_control_indices(control_terms, export_channels=(line_name,))
        for idx in scalar_indices:
            if idx in processed_indices:
                continue
            wave = physical_values[idx] + 0j
            transformed = context.lines[line_name].apply_to_waveform(wave, dt=dt)
            result[idx] = transformed.real
            processed_indices.add(idx)

    return result


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def delay_samples_from_time(delay_s: float, dt: float) -> int:
    """Convert a physical time delay to the nearest integer sample count.

    Parameters
    ----------
    delay_s:
        Physical delay in seconds.
    dt:
        Sample period in seconds.

    Returns
    -------
    int
        ``round(delay_s / dt)``
    """
    return int(round(float(delay_s) / float(dt)))


__all__ = [
    # Core classes
    "ControlLine",
    "HardwareContext",
    # Calibration maps (re-exported for convenience)
    "CalibrationMap",
    "LinearCalibrationMap",
    "CallableCalibrationMap",
    "calibration_map_from_dict",
    # Serialization helpers
    "hardware_map_to_dict",
    "hardware_map_from_dict",
    # Functions
    "postprocess_grape_waveforms",
    "delay_samples_from_time",
]
