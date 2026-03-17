"""Gate library for cqed_sim.

This subpackage provides ideal unitary gates organised by subsystem type:

- :mod:`~cqed_sim.gates.qubit`      — single-qubit named and parameterised gates
- :mod:`~cqed_sim.gates.transmon`   — multilevel transition-selective rotations
- :mod:`~cqed_sim.gates.bosonic`    — bosonic cavity gates
- :mod:`~cqed_sim.gates.coupled`    — qubit-cavity conditional / interaction gates
- :mod:`~cqed_sim.gates.two_qubit`  — two-qubit gates

All operators are returned as QuTiP ``qt.Qobj`` instances and follow the
package-wide conventions documented in
``physics_and_conventions/physics_conventions_report.tex``.
"""

from .bosonic import (
    displacement,
    kerr_evolution,
    oscillator_rotation,
    parity,
    snap,
    squeeze,
)
from .coupled import (
    beam_splitter,
    blue_sideband,
    conditional_displacement,
    conditional_rotation,
    controlled_parity,
    controlled_snap,
    dispersive_phase,
    jaynes_cummings,
    multi_sqr,
    sqr,
)
from .qubit import (
    h_gate,
    identity_gate,
    rphi,
    rx,
    ry,
    rz,
    s_dag_gate,
    s_gate,
    t_dag_gate,
    t_gate,
    x_gate,
    y_gate,
    z_gate,
)
from .transmon import r_ef, r_ge, transition_rotation
from .two_qubit import (
    cnot_gate,
    controlled_phase,
    cz_gate,
    iswap_gate,
    sqrt_iswap_gate,
    swap_gate,
)

__all__ = [
    # qubit
    "rx",
    "ry",
    "rz",
    "rphi",
    "identity_gate",
    "x_gate",
    "y_gate",
    "z_gate",
    "h_gate",
    "s_gate",
    "s_dag_gate",
    "t_gate",
    "t_dag_gate",
    # transmon
    "transition_rotation",
    "r_ge",
    "r_ef",
    # bosonic
    "displacement",
    "oscillator_rotation",
    "parity",
    "squeeze",
    "kerr_evolution",
    "snap",
    # coupled
    "dispersive_phase",
    "conditional_rotation",
    "conditional_displacement",
    "controlled_parity",
    "controlled_snap",
    "sqr",
    "multi_sqr",
    "jaynes_cummings",
    "blue_sideband",
    "beam_splitter",
    # two-qubit
    "cnot_gate",
    "cz_gate",
    "controlled_phase",
    "swap_gate",
    "iswap_gate",
    "sqrt_iswap_gate",
]
