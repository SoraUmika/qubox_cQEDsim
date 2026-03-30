from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-W", "always", "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )


def test_map_synthesis_package_import_does_not_emit_legacy_warning() -> None:
    result = _run_python("import cqed_sim.map_synthesis")
    combined = result.stdout + result.stderr
    assert "deprecated" not in combined


def test_map_synthesis_submodule_import_does_not_emit_legacy_warning() -> None:
    result = _run_python(
        "from cqed_sim.map_synthesis.metrics import subspace_unitary_fidelity; "
        "print(callable(subspace_unitary_fidelity))"
    )
    combined = result.stdout + result.stderr
    assert result.stdout.strip() == "True"
    assert "deprecated" not in combined


def test_legacy_unitary_synthesizer_instantiation_emits_deprecation_warning() -> None:
    result = _run_python(
        "import numpy as np; "
        "from cqed_sim.unitary_synthesis import PrimitiveGate, Subspace, TargetUnitary, UnitarySynthesizer; "
        "primitive = PrimitiveGate(name='id', duration=1.0, matrix=np.eye(2, dtype=np.complex128), hilbert_dim=2); "
        "UnitarySynthesizer(subspace=Subspace.custom(2, range(2)), primitives=[primitive], target=TargetUnitary(np.eye(2, dtype=np.complex128)), optimize_times=False)"
    )
    combined = result.stdout + result.stderr
    assert "UnitarySynthesizer from cqed_sim.unitary_synthesis is deprecated" in combined


def test_quantum_map_synthesizer_is_exported_from_new_namespace() -> None:
    result = _run_python(
        "from cqed_sim.map_synthesis import QuantumMapSynthesizer; "
        "print(QuantumMapSynthesizer.__name__); "
        "print(QuantumMapSynthesizer.__module__)"
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines == ["UnitarySynthesizer", "cqed_sim.unitary_synthesis.optim"]
