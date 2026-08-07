"""Portability: vendored repo_root and path aliases after jspaces reorg."""
from pathlib import Path

from jspace_phase4 import paths4


def test_repo_root_is_git_root():
    root = paths4.REPO_ROOT
    assert (root / ".git").exists()
    assert (root / "interpretability" / "jspaces" / "phases" / "phase4").is_dir()


def test_repo_uri_new_and_legacy_alias():
    new = paths4.resolve_uri(
        "repo://interpretability/jspaces/phases/phase4/README.md", must_exist=True)
    old = paths4.resolve_uri(
        "repo://interpretability/jspace_phase4/README.md", must_exist=True)
    assert new == old
    assert new.name == "README.md"


def test_rewrite_helpers_identical_across_packages():
    """Vendored copies of _find_repo_root / aliases must stay in sync."""
    # this file: interpretability/jspaces/phases/phase4/tests/
    phase4 = Path(__file__).resolve().parents[1]
    phases = phase4.parent
    jspaces = phases.parent
    roots = [
        phase4 / "jspace_phase4" / "paths4.py",
        phases / "phase2" / "jspace_part2" / "paths.py",
        jspaces / "sidelines" / "gemma" / "jspace_gemma" / "paths.py",
        jspaces / "sidelines" / "olmo" / "jspace_olmo_lineage" / "paths.py",
    ]

    def extract(path: Path) -> str:
        text = path.read_text(encoding="utf-8")
        start = text.index("def _find_repo_root")
        end = text.index("def _rewrite_repo_relative")
        block = text[start:end]
        return "\n".join(line.rstrip() for line in block.splitlines())

    blocks = [extract(p) for p in roots]
    for b in blocks[1:]:
        assert b == blocks[0]
