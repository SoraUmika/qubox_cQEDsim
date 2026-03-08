from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest


def main() -> int:
    args = sys.argv[1:]
    if not args:
        args = ["-q"]
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = int(pytest.main(args))
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    payload = {
        "args": args,
        "returncode": rc,
        "stdout": out.getvalue(),
        "stderr": err.getvalue(),
    }
    (out_dir / "pytest_capture_last.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"PYTEST_RC={rc}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
