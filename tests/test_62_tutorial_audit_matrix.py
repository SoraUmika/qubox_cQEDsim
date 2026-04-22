from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TUTORIAL_DOCS_DIR = REPO_ROOT / "documentations" / "tutorials"
AUDIT_PAGE = TUTORIAL_DOCS_DIR / "tutorial_physics_audit_matrix.md"
DOC_INDEX = TUTORIAL_DOCS_DIR / "index.md"
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"


def test_audit_matrix_lists_every_public_tutorial_guide_page() -> None:
    audit_text = AUDIT_PAGE.read_text(encoding="utf-8")
    guide_pages = sorted(
        path.name
        for path in TUTORIAL_DOCS_DIR.glob("*.md")
        if path.name not in {"index.md", "tutorial_physics_audit_matrix.md"}
    )

    for page_name in guide_pages:
        assert f"`tutorials/{page_name}`" in audit_text


def test_audit_matrix_is_linked_from_index_and_mkdocs_nav() -> None:
    index_text = DOC_INDEX.read_text(encoding="utf-8")
    mkdocs_text = MKDOCS_CONFIG.read_text(encoding="utf-8")

    assert "tutorial_physics_audit_matrix.md" in index_text
    assert "tutorials/tutorial_physics_audit_matrix.md" in mkdocs_text
