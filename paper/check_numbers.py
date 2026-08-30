"""Verify every quantitative claim in the manuscript against its source table.

Retyping a number into a manuscript is how a paper comes to disagree with the
analysis that produced it, and that error is invisible on inspection -- the
table and the sentence both look fine, and only a reader who recomputes finds
it. This has already happened twice in this project's phase reports: two
per-seed values were typed from memory and were wrong by 0.003 and 0.007.

So the manuscript is checked mechanically. Each entry below names a value, where
it must come from, and the tolerance. A claim that cannot be traced to a
generator table does not belong in the paper.

Exit code is non-zero on any mismatch, so this can gate a commit.

Usage:
    python paper/check_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MANUSCRIPT = Path(__file__).resolve().parent / "main.tex"
TABLES = Path(__file__).resolve().parent.parent / "outputs" / "tables"


def _table(name):
    import pandas as pd

    path = TABLES / name
    return pd.read_csv(path) if path.exists() else None


def checks():
    """(description, expected value, decimals) triples read from the tables."""
    out = []

    frozen = _table("linear_probe_comparison_vit_large_mae_in1k.csv")
    if frozen is not None:
        for _, r in frozen.iterrows():
            t = r["target"]
            out.append((f"frozen {t} delta", abs(r["delta_target_qwk"]), 4))
            out.append((f"frozen {t} over-SD", abs(r["delta_over_sd"]), 2))

    fine = _table("finetune_comparison.csv")
    if fine is not None:
        for _, r in fine.iterrows():
            t = r["target"]
            out.append((f"fine-tuned {t} delta", abs(r["delta_qwk"]), 4))
            out.append((f"fine-tuned {t} over-SD", abs(r["delta_over_sd"]), 2))

    res = _table("resolution_comparison.csv")
    if res is not None:
        ey = res[res.target == "eyepacs"]
        if len(ey):
            # the 512 px row is the larger of the two eyepacs entries
            best = ey.loc[ey.delta_qwk.idxmax()]
            out.append(("resolution eyepacs delta", abs(best["delta_qwk"]), 4))

    # The DG method comparisons, per target. MixStyle's DDR numbers are quoted
    # in the seed-variance section and must be traceable like everything else.
    for target in ("eyepacs", "ddr"):
        m = _table(f"method_comparison_lodo_{target}.csv")
        if m is None:
            continue
        for _, r in m[m.sampler == "natural"].iterrows():
            tag = f"{r['method']} {target}"
            out.append((f"{tag} delta", abs(r["delta_qwk"]), 4))
            out.append((f"{tag} seed42", abs(r["delta_qwk_seed42"]), 4))
            out.append((f"{tag} ci_lo", abs(r["ci_lower"]), 4))
            out.append((f"{tag} ci_hi", abs(r["ci_upper"]), 4))

    back = _table("backbone_comparison.csv")
    if back is not None:
        ey = back[back.target == "eyepacs"]
        if len(ey):
            out.append(("convnext eyepacs delta", abs(ey.delta_qwk.iloc[0]), 4))

    return out


def main() -> int:
    if not MANUSCRIPT.exists():
        print(f"no manuscript at {MANUSCRIPT}")
        return 1

    text = MANUSCRIPT.read_text(encoding="utf-8")
    # Strip TODO markers so a placeholder cannot accidentally satisfy a check.
    text = re.sub(r"\\todo\{[^}]*\}", "", text)

    missing, found = [], 0
    for label, value, decimals in checks():
        rendered = f"{value:.{decimals}f}"
        if rendered in text:
            found += 1
        else:
            missing.append(f"  {label:34} {rendered}  NOT FOUND in main.tex")

    print(f"checked {found + len(missing)} values from the generator tables")
    if missing:
        print(f"\n{len(missing)} value(s) in the tables are not quoted in the "
              f"manuscript (fine if deliberate):")
        print("\n".join(missing))

    # The real failure mode is the opposite: a number in the manuscript that no
    # table supports. Collect every decimal in the prose and flag ones that
    # match nothing.
    quoted = set(re.findall(r"(?<![\w.])0\.\d{4}(?![\w])", text))
    supported = {f"{v:.4f}" for _, v, d in checks() if d == 4}
    # Values that are arithmetic facts about the design rather than measured
    # results. Each needs a reason; the point of this check is that a number
    # with no provenance cannot appear, and "derivable in one line" is
    # provenance. Anything empirical belongs in a generator table instead.
    DERIVED_CONSTANTS = {
        # Smallest attainable two-sided sign-flip p at five seeds: 2 / 2**5.
        "0.0625",
    }
    unsupported = sorted(quoted - supported - DERIVED_CONSTANTS)
    if unsupported:
        print(f"\n!! {len(unsupported)} four-decimal value(s) in the manuscript "
              f"match no generator table:")
        for u in unsupported:
            print(f"  {u}")
        return 1

    print("\nno unsupported four-decimal value in the manuscript")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
