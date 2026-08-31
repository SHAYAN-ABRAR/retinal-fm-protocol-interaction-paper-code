"""Count the runs, without conflating five different things.

"All N runs" is an easy sentence to write and a hard one to keep true. The
registry is an append-only log, so the number of rows is not the number of
experiments; a re-run under an unchanged id adds a row and supersedes the
earlier one; a run that diverged is recorded rather than deleted, and is not a
result. The manuscript said 278 while the registry held 298 rows, 284 unique
ids and 279 resolved completions -- three defensible numbers, none of them 278.

This emits the categories separately so a sentence can quote the one it means:

    log entries            every execution ever recorded, including re-runs
    unique experiments     distinct configurations, last-write-wins
    completed              unique experiments that finished
    diverged               unique experiments that failed to converge, kept
                           as evidence rather than discarded
    superseded             log entries replaced by a later run of the same id

Only ``completed`` contributes results. The others exist so that the count of
what contributed can be stated without implying nothing else was ever run.

Usage:
    python export_run_accounting.py
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root
    from src.utils.registry import latest_per_experiment

    outputs = project_root() / "outputs"
    path = outputs / "experiment_registry.csv"
    if not path.exists():
        print("NOT RUN -- registry missing")
        return 1

    log = pd.read_csv(path)
    resolved = latest_per_experiment(log)

    counts = {
        "log_entries": len(log),
        "unique_experiments": len(resolved),
        "completed": int((resolved.status == "COMPLETE").sum()),
        "diverged": int((resolved.status == "DIVERGED").sum()),
        "superseded_log_entries": int((log.status == "SUPERSEDED").sum()),
    }

    frame = pd.DataFrame([counts])
    out = outputs / "tables" / "run_accounting.csv"
    frame.to_csv(out, index=False)

    width = max(len(k) for k in counts)
    for name, value in counts.items():
        print(f"  {name:<{width}}  {value}")
    print(f"\nsaved -> {out.name}")

    # The one identity that must hold, stated so a miscount is loud.
    assert counts["completed"] + counts["diverged"] == counts["unique_experiments"], (
        "completed + diverged must account for every unique experiment")
    print("\ncompleted + diverged == unique experiments  ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
