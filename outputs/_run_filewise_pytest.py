import json
import subprocess
import sys
from pathlib import Path

root = Path.cwd()
test_files = sorted((root / 'tests').glob('test_*.py'))
results = []
for file_path in test_files:
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(file_path)], capture_output=True, text=True)
    results.append({
        'file': str(file_path.relative_to(root)).replace('\\\\','/'),
        'returncode': proc.returncode,
        'stdout_tail': '\n'.join(proc.stdout.splitlines()[-20:]),
        'stderr_tail': '\n'.join(proc.stderr.splitlines()[-20:]),
    })
    if proc.returncode != 0:
        break

out_dir = root / 'outputs'
out_dir.mkdir(exist_ok=True)
(out_dir / 'pytest_filewise_results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
(out_dir / 'pytest_filewise_exit.txt').write_text(str(results[-1]['returncode'] if results else 0), encoding='utf-8')
