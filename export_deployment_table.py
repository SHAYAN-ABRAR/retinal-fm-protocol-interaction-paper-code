"""The deployment-cost table, three seeds, paired seed-to-seed.

Two wrong sources were available for this table and both look plausible.

``in_domain_vs_lodo_erm_s{seed}.csv`` is the superseded version: its
``lodo_qwk`` column is *identical across all three seeds*, because that analysis
held the seed-42 LODO model fixed and varied only the in-domain side. Reading it
as a paired three-seed comparison gives DDR $-0.1415$ and EyePACS $-0.2991$,
against the correct $-0.1376$ and $-0.2910$. Nothing about the file announces
that it is stale.

``table_deployment_cost.tex`` is the 512 px seed-42 table, which describes a
different experiment from the one the paper's prose cites.

The correct source is ``lodo_erm_seed_verdicts.csv``, written by
analyse_lodo_seeds.py, where both sides carry three seeds and are differenced
seed-to-seed, so the SD is the run-to-run variation *of the difference itself*
rather than of either model alone.

Usage:
    python export_deployment_table.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

SOURCE = "lodo_erm_seed_verdicts.csv"
ORDER = ["ddr", "aptos", "idrid", "eyepacs"]
LABELS = {"ddr": "DDR", "aptos": "APTOS 2019", "idrid": "IDRiD",
          "eyepacs": "EyePACS"}
BS = "\\"
NL = BS + BS


def main() -> None:
    import pandas as pd

    from src.utils.io import project_root

    tables = project_root() / "outputs" / "tables"
    path = tables / SOURCE
    if not path.exists():
        print(f"NOT RUN -- {SOURCE} missing; run analyse_lodo_seeds.py first")
        return
    frame = pd.read_csv(path).set_index("target")

    rows = []
    for domain in ORDER:
        if domain not in frame.index:
            rows.append(" & ".join([LABELS[domain]] + ["NOT RUN"] * 4) + " " + NL)
            continue
        r = frame.loc[domain]
        established = bool(r["exceeds_seed_sd"]) and bool(r["ci_excludes_zero"])
        value = f"{r['delta_qwk']:+.4f}"
        shown = ("$" + BS + "mathbf{" + value + "}$") if established else ("$" + value + "$")
        rows.append(" & ".join([
            LABELS[domain],
            f"{r['in_domain_qwk']:.4f} $" + BS + "pm$ " + f"{r['in_domain_qwk_sd']:.4f}",
            f"{r['lodo_qwk_mean']:.4f} $" + BS + "pm$ " + f"{r['lodo_qwk_sd']:.4f}",
            shown,
            "$[" + f"{r['ci_lower']:+.4f}, {r['ci_upper']:+.4f}" + "]$",
        ]) + " " + NL)

    caption = (
        "The cost of cross-domain deployment. An in-domain model and a "
        "leave-one-domain-out model are evaluated on the same matched test "
        "images and differenced seed-to-seed over three seeds, so the reported "
        "spread is the run-to-run variation of the difference itself rather "
        "than of either model alone. $" + BS + "Delta$ is LODO minus in-domain, "
        "so a negative value is the cost of never having seen the domain. Bold "
        "marks the two entries clearing both criteria of Section~" + BS
        + "ref{sec:bars}. IDRiD's positive value is within seed noise and must "
        "not be read as cross-domain training helping."
    )
    body = "\n".join([
        BS + "begin{table}[t]", BS + "centering",
        BS + "caption{" + caption + "}",
        BS + "label{tab:deployment}",
        BS + "begin{tabular}{lrrrr}", BS + "toprule",
        "Domain & In-domain QWK & LODO QWK & $" + BS + "Delta$ & Paired 95" + BS
        + "% CI " + NL,
        BS + "midrule",
        "\n".join(rows),
        BS + "bottomrule", BS + "end{tabular}", BS + "end{table}",
    ])
    out = tables / "table_deployment_cost_224.tex"
    out.write_text(body + "\n", encoding="utf-8")
    print(f"saved -> {out.name}")
    for domain in ORDER:
        if domain in frame.index:
            r = frame.loc[domain]
            print(f"  {domain:8} delta {r['delta_qwk']:+.4f}  sd "
                  f"{r['delta_qwk_sd']:.4f}  {r['verdict']}")


if __name__ == "__main__":
    main()
