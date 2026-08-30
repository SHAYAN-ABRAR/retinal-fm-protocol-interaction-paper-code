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

# Every command legitimately used in this manuscript. The list is deliberately
# closed: a command that is not here is either a typo or a new dependency, and
# both deserve a deliberate edit rather than a silent pass.
#
# It exists because three generated tables shipped an undefined \timescase --
# a caption built "seed" + BS + "times" + "case" concatenates into one token,
# and LaTeX reads the longest run of letters after a backslash as the command
# name. The orphan check below could not see it: the backslash was present and
# the line started with a legitimate word. Only knowing the valid names finds it.
KNOWN_COMMANDS = {
    "addlinespace", "author", "begin", "bibliography", "bibliographystyle",
    "bottomrule", "caption", "centering", "cite", "cmidrule", "columnwidth",
    "Delta", "documentclass", "emph", "end", "footnotesize", "geq",
    "IEEEPARstart", "includegraphics", "input", "item", "label", "maketitle",
    "mathbf", "mathrm", "midrule", "multicolumn", "newcommand", "num", "pm",
    "ref", "rightarrow", "section", "sigma", "subsection", "text", "textbf",
    "textcolor", "textit", "texttt", "textwidth", "thanks", "times", "title",
    "to", "today", "todo", "toprule", "url", "usepackage", "vspace",
    # Standard commands not yet used but valid, so ordinary edits do not trip
    # the gate for no reason.
    "alpha", "beta", "hline", "hspace", "large", "leq", "newline", "noindent",
    "normalsize", "small", "subsubsection", "tabular", "textsc", "appendix",
    "footnote", "paragraph", "quad", "qquad", "si", "approx", "cdot", "neq",
    "left", "right", "frac", "sqrt", "mu", "phantom", "footnotemark",
    "footnotetext", "IEEEauthorblockN", "IEEEauthorblockA", "clearpage",
    "newpage", "raggedright", "arraybackslash", "scriptsize", "bfseries",
}

# Commands that are errors outside math mode. \to in a text-mode tabular cell
# throws "Missing $ inserted", which is a compile failure, not a warning.
MATH_ONLY = {"Delta", "times", "pm", "to", "rightarrow", "geq", "leq", "sigma",
             "alpha", "beta", "mathbf", "mathrm", "text", "approx", "cdot",
             "neq", "frac", "sqrt", "mu"}


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

    # 3. Commands that do not exist. LaTeX reads the longest run of letters
    #    after a backslash as one name, so a concatenation bug produces a
    #    perfectly well-formed token that is simply undefined.
    for line_number, line in enumerate(raw.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        for match in re.finditer(re.escape(BS) + r"([a-zA-Z]+)", line):
            name = match.group(1)
            if name not in KNOWN_COMMANDS:
                problems.append(
                    f"{path.name}:{line_number}: unknown command "
                    f"'{BS}{name}' -- typo, or two tokens run together?: "
                    f"{line.strip()[:70]!r}")

    # 4. Math-only commands used in text mode.
    for line_number, line in enumerate(raw.split("\n"), 1):
        if line.lstrip().startswith("%"):
            continue
        # Remove inline math so only text-mode content remains. Escaped
        # dollars are not math delimiters and must not open a region.
        text_only = re.sub(r"(?<!" + re.escape(BS) + r")\$[^$]*\$", "", line)
        for match in re.finditer(re.escape(BS) + r"([a-zA-Z]+)", text_only):
            name = match.group(1)
            if name in MATH_ONLY:
                problems.append(
                    f"{path.name}:{line_number}: '{BS}{name}' is math-only but "
                    f"appears in text mode: {line.strip()[:70]!r}")
    return problems


def discover(root: Path) -> list[Path]:
    """Every .tex this gate is responsible for.

    Exposed rather than inlined so a test can assert that each generated table
    is actually covered; a gate whose coverage is only implied is a gate that
    quietly stops covering things.
    """
    files = referenced_files(root / "paper" / "main.tex")
    # Any other .tex in paper/ or outputs/tables/ is checked too, so a file that
    # is not yet wired in cannot rot unnoticed. outputs/tables/ matters most:
    # every generated table lands there, and a table is usually written some
    # commits before the \input that pulls it in. Checking only what main.tex
    # reaches would give a clean gate on a table that is already broken.
    seen = {f.resolve() for f in files}
    for directory in (root / "paper", root / "outputs" / "tables"):
        for candidate in sorted(directory.glob("*.tex")):
            if candidate.resolve() not in seen:
                seen.add(candidate.resolve())
                files.append(candidate)
    return files


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    main_tex = root / "paper" / "main.tex"
    if not main_tex.exists():
        print("NOT RUN -- paper/main.tex missing")
        return 1

    files = discover(root)
    print(f"checking {len(files)} LaTeX file(s):")
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
