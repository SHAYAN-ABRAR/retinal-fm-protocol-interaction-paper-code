"""LaTeX table for the configuration decomposition (Q1).

Generated, not hand-written, for the reason recorded in
``export_interaction_table.py``: a caption built through a shell heredoc had its
backslashes eaten and reached the manuscript as literal control characters.
Every backslash below is built from ``chr(92)`` and the output is refused if a
control character survives into it.

The table reports the batch contrast first even though resolution is the
headline. That ordering is the argument: the reader is shown what the confound
was worth before being shown what remains once it is removed.

Usage:
    python export_configuration_table.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BS = chr(92)
# Math-mode, and delimited. Unwrapped, this command is an error in a text-mode
# caption; undelimited, the letters that follow it are swallowed into the
# command name, which is how "seed x case" became an undefined \timescase in
# three tables at once.
TIMES = "$" + BS + "times$"
NL = BS + BS
LABELS = {"eyepacs": "EyePACS", "idrid": "IDRiD"}
# The arrow must be math-mode; these strings land in text-mode tabular cells.
ARROW = "$" + BS + "to$"
CONTRAST_LABELS = {
    "batch": "Batch (224/32 " + ARROW + " 224/16)",
    "resolution": "Resolution (224/16 " + ARROW + " 512/16)",
    "combined": "Combined (224/32 " + ARROW + " 512/16)",
}
SOURCE = "configuration_decomposition.csv"


def _math(name: str) -> str:
    """`\\text{name}` built without any escape sequence."""
    return BS + "text{" + name + "}"


def main() -> None:
    import pandas as pd

    from src.utils.io import project_root

    tables = project_root() / "outputs" / "tables"
    path = tables / SOURCE
    if not path.exists():
        print(f"NOT RUN -- {SOURCE} missing; run analyse_configuration.py first")
        return
    frame = pd.read_csv(path)
    frame = frame[frame.metric == "qwk"]
    if frame.empty:
        print("NOT RUN -- no QWK rows in the decomposition table")
        return

    rows = []
    for contrast in ("batch", "resolution", "combined"):
        subset = frame[frame.contrast == contrast]
        if subset.empty:
            continue
        for i, (_, r) in enumerate(subset.iterrows()):
            # The contrast name is written once and spans its domains.
            name = CONTRAST_LABELS[contrast] if i == 0 else ""
            rows.append(" & ".join([
                name,
                LABELS.get(r.target, str(r.target)),
                str(int(r.n_seeds)),
                "$" + f"{r.delta:+.4f}" + "$",
                "$[" + f"{r.ci_lower:+.4f}, {r.ci_upper:+.4f}" + "]$",
                f"{r.seed_sd:.4f}",
                f"{int(r.sign_agreement)}/{int(r.n_seeds)}",
                f"{r.p_holm:.3f}",
            ]) + " " + NL)
        rows.append(BS + "addlinespace")

    if rows and rows[-1] == BS + "addlinespace":
        rows.pop()

    # Thin space, not siunitx: the manuscript writes "224\,px" and declares no
    # \px or \GB unit, so \si{\px} would not compile.
    px = BS + ",px"
    caption = (
        "Decomposition of the 224" + px + " / 512" + px + " comparison. Every "
        "512" + px + " run in this study used batch 16, because 512" + px
        + " at batch 32 does not fit in 8" + BS + ",GB of VRAM, so resolution "
        "and batch size moved together and the effect reported previously was "
        "a configuration effect. Adding the missing 224" + px + "/batch-16 "
        "cell separates them: the "
        "batch contrast holds resolution fixed, the resolution contrast holds "
        "batch fixed, and the combined contrast is the original comparison. "
        "Intervals are from the crossed seed " + TIMES + " case "
        "bootstrap; $p_{" + _math("Holm") + "}$ is corrected within each "
        "contrast across held-out domains, not across contrasts, because the "
        "combined contrast is the sum of the other two and is not an "
        "independent hypothesis. Positive favours the second configuration."
    )
    header = ("Contrast & Held-out & seeds & $" + BS + "Delta$ QWK & 95"
              + BS + "% CI & seed SD & sign & $p_{" + _math("Holm") + "}$ "
              + NL)

    body = "\n".join([
        BS + "begin{table}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:configuration}",
        BS + "begin{tabular}{llrrrrrr}", BS + "toprule",
        header, BS + "midrule",
        "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table}",
    ])

    for bad in ("\t", "\r", "\x0b", "\x0c"):
        if bad in body:
            raise RuntimeError(
                f"generated table contains control character {bad!r} -- a "
                "backslash was consumed somewhere in this file")

    out = tables / "table_configuration.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"saved -> {out.name}  ({len(rows)} lines, no control characters)")


if __name__ == "__main__":
    main()
