from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = REPO_ROOT / "documentations"
CHOOSE_BY_GOAL = DOCS_DIR / "choose_by_goal.md"
MKDOCS_CONFIG = REPO_ROOT / "mkdocs.yml"

LANDING_PAGE_LINKS = {
    DOCS_DIR / "index.md": "choose_by_goal.md",
    DOCS_DIR / "getting_started.md": "choose_by_goal.md",
    DOCS_DIR / "quickstart.md": "choose_by_goal.md",
    DOCS_DIR / "tutorials" / "index.md": "../choose_by_goal.md",
    DOCS_DIR / "examples.md": "choose_by_goal.md",
}


def _iter_markdown_links(text: str) -> list[str]:
    return [
        target
        for _, target in re.findall(r"\[([^\]]+)\]\(([^)]+)\)", text)
        if not target.startswith(("http://", "https://", "mailto:", "#"))
    ]


def _iter_repo_paths_in_user_goals(text: str) -> list[str]:
    match = re.search(r"## User Goals(.*?)## Decision Tables", text, flags=re.S)
    assert match is not None
    candidates = re.findall(r"`([^`]+)`", match.group(1))
    return [candidate for candidate in candidates if "/" in candidate and not candidate.startswith("../")]


def test_choose_by_goal_page_is_in_mkdocs_nav() -> None:
    mkdocs_text = MKDOCS_CONFIG.read_text(encoding="utf-8")
    assert "Choose by Goal: choose_by_goal.md" in mkdocs_text


def test_landing_pages_link_to_choose_by_goal() -> None:
    for page, target in LANDING_PAGE_LINKS.items():
        text = page.read_text(encoding="utf-8")
        assert f"]({target})" in text


def test_choose_by_goal_doc_links_resolve() -> None:
    text = CHOOSE_BY_GOAL.read_text(encoding="utf-8")
    for target in _iter_markdown_links(text):
        resolved = (CHOOSE_BY_GOAL.parent / target).resolve()
        assert resolved.exists(), f"Missing linked docs page: {target}"


def test_choose_by_goal_user_goal_repo_paths_exist() -> None:
    text = CHOOSE_BY_GOAL.read_text(encoding="utf-8")
    repo_paths = _iter_repo_paths_in_user_goals(text)
    assert repo_paths, "Expected at least one repo path in the user goals section."
    for repo_path in repo_paths:
        resolved = (REPO_ROOT / repo_path).resolve()
        assert resolved.exists(), f"Missing repo artifact referenced in Choose by Goal: {repo_path}"
