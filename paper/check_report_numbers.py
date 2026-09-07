"""Every number in a results report must exist in a generator CSV.

Written because it happened: the first draft of
``docs/FULL_FINETUNE_FINAL_REPORT.md`` had the per-seed QWK values for seeds 1,
2 and 3 transcribed from an earlier truncated console output rather than from
``full_finetune_per_seed.csv``. The deltas were right, so the tables looked
internally consistent and nothing downstream complained. Only a diff against
the CSV found it.

``paper/check_numbers.py`` does this for the manuscript. The reports needed the
same gate, because the reports are what the manuscript is written from.

Each report declares the generator tables it is allowed to draw from. A report
that has not been written yet is skipped and said to be skipped; a report that
exists must have every one of its generator tables present, and every
four-decimal value in it must appear in one of them.

Exit code is non-zero on any unsupported value.

Usage:
    python paper/check_report_numbers.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPORTS = [
    {
        "report": ROOT / "docs" / "FULL_FINETUNE_FINAL_REPORT.md",
        "sources": [
            "full_finetune_per_seed.csv",
            "full_finetune_primary.csv",
            "full_finetune_secondary_interactions.csv",
            "full_finetune_secondary_outcomes.csv",
            "full_finetune_source_validation.csv",
            "ft_convergence_source_validation.csv",
        ],
    },
    {
        # The APTOS replication. It quotes the frozen DDR interaction as well
        # as its own numbers, because the two form one primary family, so the
        # DDR generator table is a legitimate source for it.
        "report": ROOT / "docs" / "APTOS_FULL_FINETUNE_FINAL_REPORT.md",
        "sources": [
            "aptos_full_finetune_per_seed.csv",
            "aptos_full_finetune_primary.csv",
            "aptos_adaptation_depth.csv",
            "aptos_secondary_outcomes.csv",
            "aptos_full_finetune_source_validation.csv",
            "two_domain_interaction_holm.csv",
            "figure_qwk_by_domain_protocol_seed.csv",
            "full_finetune_primary.csv",
        ],
    },
]

# Values that are design facts or derived in one line, not measurements.
DERIVED = {
    "0.0625",   # smallest two-sided sign-flip p at five seeds: 2 / 2**5
    "0.0500",   # the significance threshold
}


def check(report: Path, sources: list[str]) -> int:
    import pandas as pd

    supported: set[str] = set()
    tables = ROOT / "outputs" / "tables"
    for name in sources:
        path = tables / name
        if not path.exists():
            print(f"NOT RUN -- {name} missing")
            return 1
        frame = pd.read_csv(path)
        for column in frame.columns:
            for value in frame[column]:
                if isinstance(value, (int, float)) and value == value:
                    for decimals in (2, 3, 4):
                        supported.add(f"{abs(float(value)):.{decimals}f}")
                        supported.add(f"{float(value):.{decimals}f}")

    text = report.read_text(encoding="utf-8")
    # Strip code spans and the bug-history section, which quote historical
    # values deliberately and are not claims about this result set.
    text = re.sub(r"`[^`]*`", "", text)

    quoted = set(re.findall(r"(?<![\w.])\d\.\d{4}(?![\w])", text))
    unsupported = sorted(quoted - supported - DERIVED)

    print(f"  {len(quoted)} four-decimal value(s), {len(supported)} distinct "
          f"values available across {len(sources)} generator table(s)")

    if unsupported:
        print(f"\n  !! {len(unsupported)} value(s) match no generator table:")
        lines = report.read_text(encoding="utf-8").splitlines()
        for value in unsupported:
            for line_number, line in enumerate(lines, 1):
                if value in line:
                    # The reports use U+2212 MINUS and other typographic
                    # characters that a cp1252 console cannot encode; the
                    # checker must report the problem, not die trying to.
                    excerpt = line.strip()[:80].encode(
                        sys.stdout.encoding or "utf-8", "replace").decode(
                        sys.stdout.encoding or "utf-8", "replace")
                    print(f"    {value}  line {line_number}: {excerpt}")
                    break
            else:
                print(f"    {value}  (no line found)")
        return 1

    print("  every four-decimal value traces to a generator table")
    return 0


def main() -> int:
    failures, checked = 0, 0
    for spec in REPORTS:
        report = spec["report"]
        if not report.exists():
            print(f"{report.name}: not written yet -- skipped")
            continue
        print(f"{report.name}:")
        checked += 1
        failures += check(report, spec["sources"])

    if not checked:
        print("\nno reports found to check")
        return 1
    print(f"\n{checked} report(s) checked, {failures} failing")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
