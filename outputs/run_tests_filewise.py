from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST_ROOTS = [ROOT / "tests", ROOT / "cqed_sim" / "tests", ROOT / "cqed_sim" / "unitary_synthesis" / "tests"]


def discover_test_files() -> list[Path]:
    files: list[Path] = []
    for base in TEST_ROOTS:
        if not base.exists():
            continue
        files.extend(sorted(base.glob("test_*.py")))
    seen: set[str] = set()
    out: list[Path] = []
    for f in files:
        k = str(f.resolve())
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def run_one(path: Path) -> dict:
    rel = path.relative_to(ROOT).as_posix()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", rel],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    dt = time.perf_counter() - t0
    return {
        "file": rel,
        "returncode": proc.returncode,
        "elapsed_s": dt,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    files = discover_test_files()
    results: list[dict] = []
    failed = 0

    for idx, file_path in enumerate(files, start=1):
        result = run_one(file_path)
        results.append(result)
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        print(f"[{idx:03d}/{len(files):03d}] {status} {result['file']} ({result['elapsed_s']:.2f}s)")
        if result["returncode"] != 0:
            failed += 1

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "test_filewise_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = {
        "total": len(results),
        "failed": failed,
        "passed": len(results) - failed,
        "all_passed": failed == 0,
    }
    (out_dir / "test_filewise_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SUMMARY", json.dumps(summary))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
