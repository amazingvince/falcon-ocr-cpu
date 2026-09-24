#!/usr/bin/env python3
"""Check that every relative link in the repository's Markdown files points at an existing file.

Usage: python tools/check_links.py [PATH ...]   (default: README.md, docs/, research/, tests/fixtures, tools/)
Exit status 1 when a link is broken. `http(s):` and `mailto:` links are not checked; a `#fragment`
on a file link is ignored (only the file part must exist).
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
DEFAULT = ["README.md", "docs", "research", "tests/fixtures", "tools", "scripts/README.md"]


def markdown_files(targets: list[str]) -> list[Path]:
    files: list[Path] = []
    for target in targets:
        path = ROOT / target
        if path.is_dir():
            files.extend(sorted(path.rglob("*.md")))
        elif path.is_file():
            files.append(path)
    return files


def main(argv: list[str]) -> int:
    broken: list[str] = []
    checked = 0
    for file in markdown_files(argv or DEFAULT):
        text = file.read_text(encoding="utf-8", errors="replace")
        for match in LINK.finditer(text):
            target = match.group(1)
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            if not target:
                continue
            checked += 1
            resolved = (file.parent / target).resolve()
            if not resolved.exists():
                line = text.count("\n", 0, match.start()) + 1
                broken.append(f"{file.relative_to(ROOT)}:{line}: {target}")
    for item in broken:
        print(item)
    print(f"{checked} links checked, {len(broken)} broken")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
