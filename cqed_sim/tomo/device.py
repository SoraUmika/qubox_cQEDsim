from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cqed_sim.core.model import DispersiveTransmonCavityModel


@dataclass(frozen=True)
class DeviceParameters:
    """Experimentally measured device parameters for a transmon-storage system.

    All frequency fields (``*_fq``, ``*_chi*``, ``*_K*``, ``anharmonicity``) are stored
    in **Hz** (cycles per second).  The ``to_model()`` method converts them to **rad/ns**
    (radians per nanosecond) before passing them to ``DispersiveTransmonCavityModel``.

    Unit-conversion note
    --------------------
    This helper is designed for tomography workflows that conventionally use
    **nanoseconds** for times and therefore **rad/ns** for angular frequencies.
    The underlying ``cqed_sim`` model layer is unit-coherent: it accepts any
    internally consistent frequency/time units.  ``DeviceParameters`` simply
    chooses the Hz-to-rad/ns path for convenience when building tomo-focused
    example models.  The helper ``hz_to_rad_per_ns`` performs that conversion::

        omega_rad_per_ns = 2 * pi * f_hz * 1e-9

    This helper-specific conversion is intentional.  Do not replace it with a
    ``rad/s`` conversion unless the surrounding tomography workflow is also being
    expressed in seconds rather than nanoseconds.
    """

    ro_fq: float = 8596222556.078796
    """Readout resonator frequency (Hz)."""
    qb_fq: float = 6150369694.524461
    """Qubit (transmon) frequency (Hz)."""
    st_fq: float = 5240932800.0
    """Storage cavity frequency (Hz)."""
    ro_kappa: float = 4156000.0
    """Readout resonator linewidth kappa (Hz)."""
    ro_chi: float = -913148.5
    """Readout dispersive shift chi (Hz)."""
    anharmonicity: float = -255669694.5244608
    """Transmon anharmonicity alpha (Hz, typically negative)."""
    st_chi: float = -2840421.354241756
    """Storage dispersive shift chi_01 (Hz, typically negative)."""
    st_chi2: float = -21912.638362342423
    """Storage higher-order dispersive shift chi_2 (Hz)."""
    st_chi3: float = -327.37857577643325
    """Storage higher-order dispersive shift chi_3 (Hz)."""
    st_K: float = -28844.0
    """Storage self-Kerr coefficient K (Hz, typically negative)."""
    st_K2: float = 1406.0
    """Storage higher-order Kerr coefficient K_2 (Hz)."""
    ro_therm_clks: float = 1000.0
    qb_therm_clks: float = 19625.0
    st_therm_clks: float = 200000.0
    qb_t1_relax_ns: float = 9812.873848245112
    """Qubit T1 relaxation time (ns)."""
    qb_t2_ramsey_ns: float = 6324.73112712837
    """Qubit T2 Ramsey dephasing time (ns)."""
    qb_t2_echo_ns: float = 8381.0
    """Qubit T2 echo dephasing time (ns)."""

    def hz_to_rad_per_ns(self, f_hz: float) -> float:
        """Convert a frequency from Hz to rad/ns.

        The tomography helper path commonly uses nanoseconds for time, so this
        method converts Hz to rad/ns.  Other parts of the library may instead
        use rad/s with seconds; the core model layer is unit-coherent.

        This helper implements::

            omega_rad_per_ns = 2 * pi * f_hz * 1e-9

        Parameters
        ----------
        f_hz:
            Frequency in Hz (cycles per second).

        Returns
        -------
        float
            Angular frequency in rad/ns.
        """
        return 2.0 * np.pi * f_hz * 1e-9

    def to_model(self, n_cav: int = 12, n_tr: int = 3) -> DispersiveTransmonCavityModel:
        """Construct a ``DispersiveTransmonCavityModel`` from the device parameters.

        All frequency fields are converted from Hz to **rad/ns** via
        :meth:`hz_to_rad_per_ns` before being passed to the model.  That choice
        is specific to the helper workflow and is consistent with tomography
        scripts that use nanosecond time steps.

        Parameters
        ----------
        n_cav:
            Cavity (storage) Hilbert-space truncation (number of Fock levels).
        n_tr:
            Transmon Hilbert-space truncation (number of levels, typically 2 or 3).

        Returns
        -------
        DispersiveTransmonCavityModel
            Model with all frequencies in rad/ns.
        """
        return DispersiveTransmonCavityModel(
            omega_c=self.hz_to_rad_per_ns(self.st_fq),
            omega_q=self.hz_to_rad_per_ns(self.qb_fq),
            alpha=self.hz_to_rad_per_ns(self.anharmonicity),
            chi=self.hz_to_rad_per_ns(self.st_chi),
            chi_higher=(self.hz_to_rad_per_ns(self.st_chi2), self.hz_to_rad_per_ns(self.st_chi3)),
            kerr=self.hz_to_rad_per_ns(self.st_K),
            kerr_higher=(self.hz_to_rad_per_ns(self.st_K2),),
            n_cav=n_cav,
            n_tr=n_tr,
        )

