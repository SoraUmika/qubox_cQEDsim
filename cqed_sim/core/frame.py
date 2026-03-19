from dataclasses import dataclass


@dataclass(frozen=True)
class FrameSpec:
    """Per-mode rotating-frame frequencies.

    The library is unit-coherent: it does not enforce specific physical units
    for frequencies or times. Any internally consistent unit system is valid
    (for example, rad/s with times in seconds, or rad/ns with times in
    nanoseconds). The recommended convention used in the main examples and
    calibration function naming is rad/s and seconds; field names with ``omega``
    therefore nominally imply rad/s but the library will behave correctly in any
    consistent unit system.

    Field naming note: ``omega_c_frame`` is the legacy name for the storage-mode
    frame frequency, preserved for backward compatibility. In three-mode workflows
    the more descriptive name is ``omega_s_frame``, available as an alias property.
    New code targeting only the storage frame may use either name; ``omega_c_frame``
    is the canonical stored field.
    """

    omega_c_frame: float = 0.0
    """Frame frequency for the storage/cavity mode (angular frequency,
    e.g. rad/s). Legacy canonical field name."""

    omega_q_frame: float = 0.0
    """Frame frequency for the transmon/qubit mode (angular frequency,
    e.g. rad/s)."""

    omega_r_frame: float = 0.0
    """Frame frequency for the readout resonator mode (angular frequency,
    e.g. rad/s)."""

    @property
    def omega_s_frame(self) -> float:
        """Storage-mode frame frequency (angular frequency, e.g. rad/s).

        This is an alias for ``omega_c_frame``. The ``_s_`` name is preferred in
        three-mode (storage + readout + qubit) contexts where ``_c_`` is ambiguous;
        both names refer to the same stored field.
        """
        return self.omega_c_frame
