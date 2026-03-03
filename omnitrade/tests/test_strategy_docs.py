"""
Enforcement test: every bot strategy must have a companion markdown doc.

Scans omnitrade/bots/ for .py files (excluding __init__.py) and asserts
each has a corresponding .md file that is non-empty.
"""

from pathlib import Path

import pytest

BOTS_DIR = Path(__file__).resolve().parent.parent / "omnitrade" / "bots"


def _bot_py_files() -> list[Path]:
    """Return all .py bot modules (excluding __init__)."""
    return sorted(
        p for p in BOTS_DIR.glob("*.py") if p.name != "__init__.py"
    )


@pytest.mark.parametrize(
    "bot_py",
    _bot_py_files(),
    ids=lambda p: p.stem,
)
def test_strategy_has_markdown_doc(bot_py: Path):
    md_file = bot_py.with_suffix(".md")
    assert md_file.exists(), (
        f"Missing strategy doc: {md_file.name}. "
        f"Every bot in omnitrade/bots/ must have a companion .md file."
    )
    content = md_file.read_text()
    assert len(content) >= 50, (
        f"{md_file.name} is too short ({len(content)} chars). "
        f"Strategy docs must be at least 50 characters."
    )
