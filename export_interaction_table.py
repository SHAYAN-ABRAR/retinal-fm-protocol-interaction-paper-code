"""LaTeX table for the protocol-interaction analysis.

Written as a generator rather than an inline heredoc because the first version
of this table was produced by one, and every ``\\text{...}`` in its caption
arrived as a literal TAB: the backslash-t was consumed by a shell layer and
Python read the remainder as an escape. The file compiled to nothing sensible
and `paper/build.sh` did not notice, because that gate only scanned
`paper/main.tex` and this table is pulled in by `\\input`.

Backslashes are built from ``chr(92)`` throughout for the same reason. Nothing
here contains a literal backslash escape.

Usage:
    python export_interaction_table.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BS = chr(92)
# Math-mode, and delimited. Unwrapped it is an error in a text-mode caption;
# undelimited the following letters are swallowed into the command name, which
# is how this caption shipped an undefined \timescase.
TIMES = "$" + BS + "times$"
NL = BS + BS
LABELS = {"ddr": "DDR", "aptos": "APTOS 2019", "idrid": "IDRiD"}
SOURCE = "protocol_interaction.csv"


def _math(name: str) -> str:
    """`\\text{name}` built without any escape sequence."""
    return BS + "text{" + name + "}"


def main() -> None:
    import pandas as pd

    from src.utils.io import project_root

    tables = project_root() / "outputs" / "tables"
    path = tables / SOURCE
    if not path.exists():
        print(f"NOT RUN -- {SOURCE} missing; run analyse_interaction.py first")
        return
    frame = pd.read_csv(path)

    rows = []
    for _, r in frame.iterrows():
        rows.append(" & ".join([
            LABELS.get(r.target, str(r.target)),
            str(int(r.n_seeds)),
            "$" + f"{r.interaction:+.4f}" + "$",
            "$[" + f"{r.ci_lower:+.4f}, {r.ci_upper:+.4f}" + "]$",
            f"{int(r.sign_agreement)}/{int(r.n_seeds)}",
            f"{r.p_ttest:.3f}",
            f"{r.p_holm:.3f}",
            f"{r.p_signflip:.4f}",
        ]) + " " + NL)

    interaction = ("$I = (" + _math("ImageNet") + "-" + _math("RETFound")
                   + ")_{" + _math("partial") + "} - (" + _math("ImageNet")
                   + "-" + _math("RETFound") + ")_{" + _math("frozen") + "}$")
    caption = (
        "Protocol interaction: the difference of differences " + interaction
        + ", over the five seeds common to both protocols. Negative means the "
        "advantage of the general-purpose initialisation is attenuated once the "
        "backbone can adapt. Intervals are from the crossed seed"
        + " " + TIMES + " case bootstrap and quantify uncertainty; $p$ is a "
        "paired $t$-test on the per-seed values, Holm-corrected across the "
        "three held-out domains. The final column is an exact sign-flip "
        "permutation check, whose smallest attainable two-sided value at five "
        "seeds is $0.0625$; it therefore cannot reach conventional significance "
        "at this sample size for any effect size. No domain reaches "
        "significance after correction, so a consistent directional attenuation "
        "is reported and no interaction is claimed as established."
    )
    header = ("Held-out & seeds & $I$ & 95" + BS + "% CI & sign & $p$ & $p_{"
              + _math("Holm") + "}$ & $p_{" + _math("flip") + "}$ " + NL)

    body = "\n".join([
        BS + "begin{table}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:interaction}",
        BS + "begin{tabular}{lrrrrrrr}", BS + "toprule",
        header, BS + "midrule",
        "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table}",
    ])

    # Refuse to write a file containing the corruption this generator exists to
    # avoid. Cheaper to fail here than to find it in a compiled PDF.
    for bad in ("\t", "\r", "\x0b", "\x0c"):
        if bad in body:
            raise RuntimeError(
                f"generated table contains control character {bad!r} -- a "
                "backslash was consumed somewhere in this file")

    out = tables / "table_interaction.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"saved -> {out.name}  ({len(rows)} rows, no control characters)")


if __name__ == "__main__":
    main()
