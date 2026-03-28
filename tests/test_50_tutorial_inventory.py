from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIALS_DIR = REPO_ROOT / "tutorials"
DOC_INDEX = REPO_ROOT / "documentations" / "tutorials" / "index.md"
TUTORIAL_README = TUTORIALS_DIR / "README.md"
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"

NEW_NOTEBOOKS = (
    "31_system_identification_and_domain_randomization/01_calibration_targets_and_fitting.ipynb",
    "31_system_identification_and_domain_randomization/02_evidence_to_randomizer_and_env.ipynb",
    "50_floquet_driven_systems/01_sideband_quasienergy_scan.ipynb",
    "50_floquet_driven_systems/02_branch_tracking_and_multiphoton_resonances.ipynb",
)

NEW_GUIDE_PAGES = (
    "tutorials/system_identification.md",
    "tutorials/floquet_driven_systems.md",
)


def test_tutorial_index_count_matches_notebook_inventory() -> None:
    notebooks = sorted(path.relative_to(TUTORIALS_DIR).as_posix() for path in TUTORIALS_DIR.rglob("*.ipynb"))
    match = re.search(r"contains \*\*(\d+) Jupyter notebooks\*\*", DOC_INDEX.read_text(encoding="utf-8"))
    assert match is not None
    assert int(match.group(1)) == len(notebooks)
    assert len(notebooks) == 47


def test_new_workflow_notebooks_are_listed_in_readmes() -> None:
    tutorial_readme = TUTORIAL_README.read_text(encoding="utf-8")
    doc_index = DOC_INDEX.read_text(encoding="utf-8")

    for notebook in NEW_NOTEBOOKS:
        assert notebook in tutorial_readme
        assert notebook in doc_index


def test_new_tutorial_guide_pages_are_in_mkdocs_nav() -> None:
    mkdocs_config = MKDOCS_CONFIG.read_text(encoding="utf-8")
    for page in NEW_GUIDE_PAGES:
        assert page in mkdocs_config
