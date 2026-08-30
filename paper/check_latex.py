"""Structural check of the manuscript and every file it pulls in.

The previous gate lived in bash and scanned only `paper/main.tex`. It therefore
missed a real defect: `outputs/tables/table_interaction.tex`, imported by
`\\input`, had every `\\text{...}` in its caption replaced by a literal TAB
because a backslash was consumed while generating it. The table would have gone
into the submitted PDF as garbage.

Two lessons are encoded here. Follow `\\input` recursively rather than trusting
that the top-level file is the whole document; and detect control characters
inside commands, not just orphaned command names at line starts.

Written in Python rather than shell because shell quoting is the thing that
keeps consuming these backslashes in the first place.

Exit code 0 = clean, 1 = at least one defect.

Usage:
    python paper/check_latex.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

BS = chr(92)

# Command tails that, seen without their leading backslash, mean one was eaten.
ORPHANS = ["ref", "text", "times", "cite", "label", "item", "emph", "textbf",
           "textit", "num", "input", "includegraphics", "caption", "begin",
           "end", "midrule", "toprule", "bottomrule", "multicolumn", "pm",
           "Delta", "sigma", "mathbf"]

CONTROL = {"\t": "TAB", "\r": "CR", "\x0b": "VT", "\x0c": "FF", "\x08": "BS"}


def referenced_files(path: Path, seen: set[Path] | None = None) -> list[Path]:
    """Every .tex reachable from `path` through \\input, transitively."""
    seen = seen if seen is not None else set()
    path = path.resolve()
    if path in seen or not path.exists():
        return []
    seen.add(path)
    found = [path]
    text = path.read_text(encoding="utf-8", errors="replace")
    for target in re.findall(re.escape(BS) + r"input\{([^}]*)\}", text):
        candidate = (path.parent / target).resolve()
        for option in (candidate, candidate.with_suffix(".tex")):
            if option.exists():
                found.extend(referenced_files(option, seen))
                break
    return found


def check(path: Path) -> list[str]:
    problems: list[str] = []
    raw = path.read_text(encoding="utf-8", errors="replace")

    # 1. Control characters anywhere. A tab in a .tex file is almost always a
    #    consumed backslash, and is never needed for layout here.
    for char, name in CONTROL.items():
        if char in raw:
            for i, line in enumerate(raw.split("\n"), 1):
                if char in line:
                    problems.append(
                        f"{path.name}:{i}: literal {name} character "
                        f"(a consumed backslash?): {line.strip()[:70]!r}")
                    break

    # 2. Orphaned command fragments: the tail present, the backslash gone.
    for line_number, line in enumerate(raw.split("\n"), 1):
        for name in ORPHANS:
            for match in re.finditer(r"(?<!" + re.escape(BS) + r")\b"
                                     + name + r"\{", line):
                start = match.start()
                # A legitimate occurrence is preceded by a backslash; also skip
                # words inside prose that merely end with the command name.
                if start > 0 and line[start - 1].isalpha():
                    continue
                problems.append(
                    f"{path.name}:{line_number}: orphaned '{name}{{' "
                    f"(missing backslash): {line.strip()[:70]!r}")
    return problems


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    main_tex = root / "paper" / "main.tex"
    if not main_tex.exists():
        print("NOT RUN -- paper/main.tex missing")
        return 1

    files = referenced_files(main_tex)
    # Any other .tex sitting in paper/ is checked too, so a file that is not yet
    # wired in cannot rot unnoticed.
    files += [p for p in (root / "paper").glob("*.tex")
              if p.resolve() not in {f.resolve() for f in files}]

    print(f"checking {len(files)} LaTeX file(s) reachable from main.tex:")
    for f in files:
        try:
            rel = f.relative_to(root)
        except ValueError:
            rel = f
        print(f"  {rel}")

    problems: list[str] = []
    for f in files:
        problems.extend(check(f))

    print()
    if problems:
        print(f"!! {len(problems)} LaTeX defect(s):")
        for p in problems:
            print(f"  {p}")
        return 1
    print("no orphaned commands or control characters in any included file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
