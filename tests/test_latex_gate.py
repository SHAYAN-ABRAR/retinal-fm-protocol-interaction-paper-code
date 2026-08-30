"""The LaTeX gate must inspect included files, not just main.tex.

A real defect got through the previous bash gate: table_interaction.tex, pulled
in by \input, had every \text{...} in its caption replaced by a literal TAB
because a backslash was consumed while generating it. main.tex itself was clean,
so the gate passed and the table would have reached the PDF as garbage.

These tests corrupt an INCLUDED file and require rejection.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper"))

import check_latex  # noqa: E402

BS = chr(92)


def _doc(tmp_path: Path, main_body: str, included: str | None = None) -> Path:
    tables = tmp_path / "tables"
    tables.mkdir()
    if included is not None:
        (tables / "gen.tex").write_text(included, encoding="utf-8")
    main = tmp_path / "main.tex"
    main.write_text(main_body, encoding="utf-8")
    return main


def test_clean_document_passes(tmp_path):
    main = _doc(tmp_path,
                BS + "input{tables/gen}\n",
                BS + "caption{" + BS + "text{ok} and " + BS + "times}\n")
    assert check_latex.check(main) == []
    included = tmp_path / "tables" / "gen.tex"
    assert check_latex.check(included) == []


def test_a_tab_inside_an_included_table_is_rejected(tmp_path):
    """The exact defect that reached the repository."""
    corrupted = BS + "caption{$" + "\t" + "ext{ImageNet}$}\n"
    main = _doc(tmp_path, BS + "input{tables/gen}\n", corrupted)
    included = tmp_path / "tables" / "gen.tex"
    problems = check_latex.check(included)
    assert problems, "a literal TAB in an included table was not detected"
    assert any("TAB" in p for p in problems)


def test_included_files_are_discovered_from_main(tmp_path):
    main = _doc(tmp_path, BS + "input{tables/gen}\n", BS + "caption{fine}\n")
    found = {p.name for p in check_latex.referenced_files(main)}
    assert found == {"main.tex", "gen.tex"}, (
        f"the gate must follow the input command; found {found}")


def test_orphaned_command_in_an_included_file_is_rejected(tmp_path):
    main = _doc(tmp_path, BS + "input{tables/gen}\n",
                "see Section~ref{tab:x} here\n")
    problems = check_latex.check(tmp_path / "tables" / "gen.tex")
    assert any("orphaned 'ref{'" in p for p in problems), problems


def test_the_real_manuscript_and_its_tables_are_clean():
    main = ROOT / "paper" / "main.tex"
    if not main.exists():
        pytest.skip("manuscript absent")
    problems = []
    for path in check_latex.referenced_files(main):
        problems.extend(check_latex.check(path))
    assert not problems, problems
