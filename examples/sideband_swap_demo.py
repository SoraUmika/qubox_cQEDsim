from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import (
    DispersiveTransmonCavityModel,
    SequenceCompiler,
    SidebandDriveSpec,
    SimulationConfig,
    build_sideband_pulse,
    simulate_sequence,
)
from physics_and_conventions.conventions import from_internal_units, to_internal_units


def _mhz(value_hz: float) -> float:
    return float(value_hz) * 1.0e6


def main() -> None:
    out_dir = ROOT / "examples" / "outputs" / "sideband_swap_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    g_sb = to_internal_units(8.0e6)
    chi = to_internal_units(-0.8e6)
    t_swap = np.pi / (2.0 * g_sb)

    model = DispersiveTransmonCavityModel(
        omega_c=0.0,
        omega_q=0.0,
        alpha=0.0,
        chi=chi,
        kerr=0.0,
        n_cav=4,
        n_tr=3,
    )
    pulses, drive_ops, _meta = build_sideband_pulse(
        SidebandDriveSpec(mode="storage", lower_level=0, upper_level=2),
        duration_s=t_swap,
        amplitude_rad_s=g_sb,
        channel="sb",
        label="gf_red_sideband",
    )
    compiled = SequenceCompiler(dt=t_swap / 400.0).compile(pulses, t_end=t_swap)
    result = simulate_sequence(
        model,
        compiled,
        model.basis_state(2, 0),
        drive_ops,
        SimulationConfig(store_states=True),
    )

    target = model.basis_state(0, 1)
    source = model.basis_state(2, 0)
    p_target = abs(target.overlap(result.final_state)) ** 2
    p_source = abs(source.overlap(result.final_state)) ** 2

    summary = {
        "model": "effective gf red sideband on a 3-level transmon + storage mode",
        "assumptions": [
            "effective rotating-wave sideband Hamiltonian",
            "no dissipation",
            "internal frequencies in rad/s and time in seconds",
        ],
        "g_sb_mhz": from_internal_units(g_sb) / 1.0e6,
        "chi_mhz": from_internal_units(chi) / 1.0e6,
        "swap_time_ns": t_swap * 1.0e9,
        "inverse_abs_chi_ns": (1.0 / abs(chi)) * 1.0e9,
        "final_p_g1": float(p_target),
        "final_p_f0": float(p_source),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt  # type: ignore

        p_g1 = np.array([abs(target.overlap(state)) ** 2 for state in result.states], dtype=float)
        p_f0 = np.array([abs(source.overlap(state)) ** 2 for state in result.states], dtype=float)
        fig, ax = plt.subplots(figsize=(6.2, 3.8))
        ax.plot(compiled.tlist * 1.0e9, p_g1, label=r"$P_{|g,1\rangle}$", lw=2.0)
        ax.plot(compiled.tlist * 1.0e9, p_f0, label=r"$P_{|f,0\rangle}$", lw=1.6)
        ax.axvline(t_swap * 1.0e9, color="tab:red", ls="--", lw=1.0, label=r"$t_{\mathrm{swap}}$")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("Population")
        ax.set_title("Fast sideband swap benchmark")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "swap_populations.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
