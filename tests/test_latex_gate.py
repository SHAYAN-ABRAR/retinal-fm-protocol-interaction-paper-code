r"""The LaTeX gate must inspect included files, not just main.tex.

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
    # Both commands are math-only, so both are delimited. Written bare -- as
    # this fixture originally was -- they are a compile error, not a clean
    # document, and the gate is now strict enough to say so.
    body = (BS + "caption{$" + BS + "text{ok}$ and $" + BS + "times$}\n")
    main = _doc(tmp_path, BS + "input{tables/gen}\n", body)
    assert check_latex.check(main) == []
    included = tmp_path / "tables" / "gen.tex"
    assert check_latex.check(included) == []


def test_a_concatenated_command_is_rejected(tmp_path):
    """The defect that shipped in three tables at once.

    Building a caption as "seed" + BS + "times" + "case" yields \\timescase:
    one well-formed, undefined token. The backslash is present and the line
    starts with a real word, so neither the control-character check nor the
    orphan check can see it. Only a closed list of valid names catches it.
    """
    body = BS + "caption{crossed seed" + BS + "times" + "case bootstrap}\n"
    main = _doc(tmp_path, BS + "input{tables/gen}\n", body)
    problems = check_latex.check(tmp_path / "tables" / "gen.tex")
    assert any("timescase" in p for p in problems), problems
    assert any("unknown command" in p for p in problems), problems


def test_a_math_only_command_in_text_mode_is_rejected(tmp_path):
    """An arrow in a tabular cell throws 'Missing $ inserted'."""
    body = "Batch (224/32 " + BS + "to 224/16) & 1 " + BS + BS + "\n"
    main = _doc(tmp_path, BS + "input{tables/gen}\n", body)
    problems = check_latex.check(tmp_path / "tables" / "gen.tex")
    assert any("math-only" in p for p in problems), problems


def test_math_mode_use_of_the_same_command_is_accepted(tmp_path):
    """The check must not fire on the correct form, or it will be turned off."""
    body = "Batch (224/32 $" + BS + "to$ 224/16) & 1 " + BS + BS + "\n"
    main = _doc(tmp_path, BS + "input{tables/gen}\n", body)
    assert check_latex.check(tmp_path / "tables" / "gen.tex") == []


def test_every_generated_table_is_discovered(tmp_path):
    """Tables are written some commits before the \\input that pulls them in.

    A gate that walks only main.tex reports success on a table that is already
    broken, which is how the unwired table_configuration.tex went unchecked.
    """
    root = Path(check_latex.__file__).resolve().parents[1]
    generated = sorted((root / "outputs" / "tables").glob("*.tex"))
    if not generated:
        import pytest as _pytest
        _pytest.skip("no generated tables in this checkout")
    discovered = {p.resolve() for p in check_latex.discover(root)}
    for table in generated:
        assert table.resolve() in discovered, f"{table.name} is not gated"


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
