from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from cqed_sim.core.model import DispersiveTransmonCavityModel


@dataclass(frozen=True)
class DeviceParameters:
    ro_fq: float = 8596222556.078796
    qb_fq: float = 6150369694.524461
    st_fq: float = 5240932800.0
    ro_kappa: float = 4156000.0
    ro_chi: float = -913148.5
    anharmonicity: float = -255669694.5244608
    st_chi: float = -2840421.354241756
    st_chi2: float = -21912.638362342423
    st_chi3: float = -327.37857577643325
    st_K: float = -28844.0
    st_K2: float = 1406.0
    ro_therm_clks: float = 1000.0
    qb_therm_clks: float = 19625.0
    st_therm_clks: float = 200000.0
    qb_t1_relax_ns: float = 9812.873848245112
    qb_t2_ramsey_ns: float = 6324.73112712837
    qb_t2_echo_ns: float = 8381.0

    def hz_to_rad_per_ns(self, f_hz: float) -> float:
        return 2.0 * np.pi * f_hz * 1e-9

    def to_model(self, n_cav: int = 12, n_tr: int = 3) -> DispersiveTransmonCavityModel:
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

