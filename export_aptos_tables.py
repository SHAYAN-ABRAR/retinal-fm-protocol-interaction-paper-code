"""LaTeX tables for the APTOS replication: the two-domain primary family and
the APTOS adaptation-depth summary.

Written before the APTOS result existed, so nothing in it can be phrased to
suit an outcome. The captions state their verdict from the data rather than
from a sentence chosen after seeing it: the concluding clause is assembled from
whether each domain's Holm-adjusted test and crossed interval agree, and it
reads correctly whether the replication confirms DDR, contradicts it, or lands
on a null.

Backslashes are built from ``chr(92)`` throughout, as in the other exporters,
because a heredoc once turned every ``text{...}`` in a caption into a literal
TAB and the compiled table was nonsense. Nothing here contains a literal
backslash escape.

Usage:
    python export_aptos_tables.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

BS = chr(92)
TIMES = "$" + BS + "times$"
PM = "$" + BS + "pm$"
NL = BS + BS
FAMILY = "two_domain_interaction_holm.csv"
DEPTH = "aptos_adaptation_depth.csv"
PROTOCOL_LABELS = {"frozen": "Frozen linear probe",
                   "partial": "Partial (last 4 blocks)",
                   "full": "Full fine-tuning"}


def _math(name: str) -> str:
    """`\\text{name}` built without any escape sequence."""
    return BS + "text{" + name + "}"


def _guard(body: str) -> str:
    """Refuse to emit the corruption these generators exist to avoid."""
    for bad in ("\t", "\r", "\x0b", "\x0c"):
        if bad in body:
            raise RuntimeError(
                f"generated table contains control character {bad!r} -- a "
                "backslash was consumed somewhere in this file")
    return body


def _table(caption: str, label: str, spec: str, header: str,
           rows: list[str]) -> str:
    return _guard("\n".join([
        BS + "begin{table}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{" + label + "}",
        BS + "begin{tabular}{" + spec + "}", BS + "toprule",
        header, BS + "midrule",
        "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table}",
    ]))


def _verdict(frame) -> str:
    """The concluding sentence, assembled from the numbers themselves."""
    established = [r for r in frame.itertuples() if bool(r.established)]
    absent = [r for r in frame.itertuples() if not bool(r.established)]
    names = {r.domain for r in established}

    if len(established) == len(frame):
        directions = {"negative" if r.mean < 0 else "positive"
                      for r in frame.itertuples()}
        agreement = ("in the same direction on both"
                     if len(directions) == 1 else
                     "in opposite directions on the two domains")
        return ("The interaction is established on both held-out domains, "
                + agreement + ".")
    if not established:
        return ("The interaction is not demonstrated on either held-out "
                "domain after correction. At five seeds this bounds the effect "
                "only as tightly as the intervals above and is not evidence "
                "that the two protocols agree.")
    kept = ", ".join(sorted(names))
    lost = ", ".join(sorted(r.domain for r in absent))
    return ("The interaction is established on " + kept + " and not "
            "demonstrated on " + lost + ". A non-significant result at five "
            "seeds bounds the effect only as tightly as the interval above; it "
            "does not establish that the protocols agree on " + lost + ".")


def main() -> None:
    import pandas as pd

    from src.utils.io import project_root

    tables = project_root() / "outputs" / "tables"
    written = []

    # ------------------------------------------- the two-domain primary family
    path = tables / FAMILY
    if not path.exists():
        print(f"NOT RUN -- {FAMILY} missing; run "
              f"analyse_aptos_replication.py first")
    else:
        frame = pd.read_csv(path)
        rows = []
        for r in frame.itertuples():
            rows.append(" & ".join([
                str(r.domain),
                f"{int(r.n_test):,}".replace(",", BS + ","),
                "$" + f"{r.mean:+.4f}" + "$",
                "$[" + f"{r.ci_lower:+.4f}, {r.ci_upper:+.4f}" + "]$",
                f"{int(r.sign_agreement)}/5",
                f"{r.p_ttest:.4f}",
                f"{r.p_holm:.4f}",
                f"{r.p_signflip:.4f}",
            ]) + " " + NL)

        interaction = ("$I_{" + _math("full") + "} = (" + _math("ImageNet")
                       + "-" + _math("RETFound") + ")_{" + _math("full")
                       + "} - (" + _math("ImageNet") + "-" + _math("RETFound")
                       + ")_{" + _math("frozen") + "}$")
        caption = (
            "Primary family: the full-versus-frozen protocol interaction "
            + interaction + " on the two held-out domains for which matched "
            "full fine-tuning was run, over five seeds each. APTOS was a "
            "pre-registered confirmatory replication of the DDR result, "
            "specified and analysed before any APTOS target metric was "
            "inspected. Intervals come from the crossed seed "
            + TIMES + " case bootstrap and quantify uncertainty only; they are "
            "not a test, because the resampling distribution is built around "
            "the empirical estimate rather than under the null. $p$ is a "
            "paired $t$-test on the five per-seed interactions, Holm-corrected "
            "across exactly these two domains, and $p_{" + _math("flip")
            + "}$ is an exact sign-flip permutation check whose smallest "
            "attainable two-sided value at five seeds is $0.0625$; it cannot "
            "reach conventional significance at this sample size for any "
            "effect size and is reported as sensitivity. A domain is treated "
            "as established only when the corrected test and the interval "
            "agree. No pooled two-domain $p$-value is reported: the two test "
            "sets differ in size and in shift structure, so a pooled value "
            "would describe neither. " + _verdict(frame)
        )
        header = ("Held-out & $n_{" + _math("test") + "}$ & $I_{"
                  + _math("full") + "}$ & 95" + BS + "% CI & sign & $p$ & $p_{"
                  + _math("Holm") + "}$ & $p_{" + _math("flip") + "}$ " + NL)
        out = tables / "table_two_domain_interaction.tex"
        out.write_text(_table(caption, "tab:two-domain-interaction",
                              "lrrrrrrr", header, rows) + "\n",
                       encoding="utf-8")
        written.append((out.name, len(rows)))

    # ----------------------------------------------- APTOS adaptation depth
    path = tables / DEPTH
    if not path.exists():
        print(f"NOT RUN -- {DEPTH} missing; run "
              f"analyse_aptos_replication.py first")
    else:
        frame = pd.read_csv(path)
        rows = []
        for r in frame.itertuples():
            rows.append(" & ".join([
                PROTOCOL_LABELS.get(r.protocol, str(r.protocol)),
                f"{r.imagenet_mean:.4f} " + PM + f" {r.imagenet_sd:.4f}",
                f"{r.retfound_mean:.4f} " + PM + f" {r.retfound_sd:.4f}",
                "$" + f"{r.delta_mean:+.4f}" + "$",
                "$[" + f"{r.ci_lower:+.4f}, {r.ci_upper:+.4f}" + "]$",
                f"{int(r.sign_agreement)}/5",
                f"{r.p_ttest:.4f}",
            ]) + " " + NL)

        caption = (
            "Adaptation depth on the held-out APTOS domain: quadratic weighted "
            "kappa for the two matched initialisations under each of the three "
            "adaptation protocols, mean " + PM + " standard deviation over five "
            "seeds, with $" + BS + "Delta$ the paired ImageNet-MAE minus "
            "RETFound difference. Intervals are crossed seed " + TIMES + " case "
            "bootstrap intervals and $p$ is an uncorrected paired $t$-test on "
            "the five per-seed differences. These three comparisons are "
            "descriptive context for the interaction in Table" + BS
            + "~" + BS + "ref{tab:two-domain-interaction} and are not a second "
            "family of tests; the pre-specified primary endpoint is the "
            "interaction, and no protocol-level comparison here is corrected "
            "for or claimed as established on its own."
        )
        header = ("Protocol & ImageNet-MAE & RETFound & $" + BS + "Delta$ & 95"
                  + BS + "% CI & sign & $p$ " + NL)
        out = tables / "table_aptos_adaptation_depth.tex"
        out.write_text(_table(caption, "tab:aptos-adaptation-depth",
                              "lrrrrrr", header, rows) + "\n",
                       encoding="utf-8")
        written.append((out.name, len(rows)))

    for name, count in written:
        print(f"saved -> {name}  ({count} rows, no control characters)")
    if not written:
        print("nothing written")


if __name__ == "__main__":
    main()
