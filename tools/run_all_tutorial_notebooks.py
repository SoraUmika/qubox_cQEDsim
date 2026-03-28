"""Execute all tutorial notebooks and report success/failure.

Run from the repository root:
    python tools/run_all_tutorial_notebooks.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TUTORIALS = REPO / "tutorials"
PYTHON = sys.executable

# Collect all .ipynb files recursively, sorted
notebooks = sorted(TUTORIALS.rglob("*.ipynb"))

print(f"Found {len(notebooks)} notebooks under {TUTORIALS}\n")

results: list[dict] = []
for i, nb in enumerate(notebooks, 1):
    rel = nb.relative_to(REPO)
    print(f"[{i}/{len(notebooks)}] {rel} ... ", end="", flush=True)
    t0 = time.time()
    proc = subprocess.run(
        [
            PYTHON, "-m", "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=300",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output", str(nb.name),
            "--output-dir", str(nb.parent),
            str(nb),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=360,
    )
    elapsed = time.time() - t0
    ok = proc.returncode == 0
    status = "OK" if ok else "FAIL"
    print(f"{status} ({elapsed:.1f}s)")
    if not ok:
        # Print last 10 lines of stderr for diagnosis
        stderr_lines = proc.stderr.strip().splitlines()[-10:]
        for line in stderr_lines:
            print(f"    {line}")
    results.append({
        "notebook": str(rel),
        "status": status,
        "elapsed_s": round(elapsed, 1),
        "error": proc.stderr.strip()[-500:] if not ok else "",
    })

# Summary
passed = sum(1 for r in results if r["status"] == "OK")
failed = sum(1 for r in results if r["status"] == "FAIL")
print(f"\n{'='*60}")
print(f"TOTAL: {len(results)}  PASSED: {passed}  FAILED: {failed}")
if failed:
    print("\nFailed notebooks:")
    for r in results:
        if r["status"] == "FAIL":
            print(f"  - {r['notebook']}")
print(f"{'='*60}")

# Write results JSON
out_path = REPO / "outputs" / "tutorial_notebook_run_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(f"\nDetailed results: {out_path}")
