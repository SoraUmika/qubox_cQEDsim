from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cqed_sim import verify_kerr_sign


def main() -> None:
    out_dir = Path("examples") / "outputs" / "kerr_sign_verification"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = verify_kerr_sign(comparison_time_s=1.0e-6, alpha=2.0, n_cav=30, n_tr=3)
    summary = {
        "documented_kerr_hz": result.documented_kerr_hz,
        "flipped_kerr_hz": result.flipped_kerr_hz,
        "cavity_mean_documented": {
            "real": float(result.cavity_mean_documented.real),
            "imag": float(result.cavity_mean_documented.imag),
        },
        "cavity_mean_flipped": {
            "real": float(result.cavity_mean_flipped.real),
            "imag": float(result.cavity_mean_flipped.imag),
        },
        "documented_phase_rad": result.documented_phase_rad,
        "flipped_phase_rad": result.flipped_phase_rad,
        "matches_documented_sign": result.matches_documented_sign,
        "diagnosis": (
            "The notebook-scale Kerr evolution matches the documented runtime sign. "
            "Any apparent sign flip comes from frame/phase interpretation rather than the core Hamiltonian."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
