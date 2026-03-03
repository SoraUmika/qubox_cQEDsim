from __future__ import annotations

import json
from pathlib import Path


def load_golden() -> dict:
    p = Path(__file__).parent / "golden" / "hard_sequence.json"
    return json.loads(p.read_text(encoding="utf-8"))

