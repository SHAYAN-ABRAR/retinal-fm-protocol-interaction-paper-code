"""Audit experiment-ID collisions and emit docs/EXPERIMENT_IDENTITY_AUDIT.md.

The registry is append-only, so a re-run legitimately adds a second row under
one experiment id. That is fine when the two rows are the *same* scientific
configuration executed twice -- a resume, or a re-evaluation. It is not fine
when they are different configurations, because then one id, one prediction
path and one checkpoint directory describe two different experiments, and
nothing on disk says which one produced the numbers a paper quotes.

That happened here. The RETFound linear probes were first run at lr 1e-3 for 30
epochs through an nn.LayerNorm that destroyed the feature scaling, then re-run
at lr 5e-3 for 200 epochs with per-dimension standardisation. Both executions
wrote to `..._linprobe-b256_s{seed}` and to the same predictions CSV. The second
overwrote the first's artifacts while the first's row survived in the registry.

This script does not delete anything. It classifies every duplicated id,
records which row is authoritative, and marks the others SUPERSEDED with a
reason, so the history stays readable and the current result stays identifiable.

Usage:
    python audit_experiment_identity.py            # report only
    python audit_experiment_identity.py --write    # also stamp status/history
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

# Fields that define a *scientific* configuration. Two rows differing in any of
# these are different experiments, whatever their id says. epochs_run is
# deliberately absent: it records what happened, not what was asked for, and
# differs between a resumed run and its restart.
CONFIG_FIELDS = [
    "protocol", "source_domains", "target_domain", "backbone", "method",
    "head", "loss", "imbalance_strategy", "image_size", "batch_size",
    "learning_rate", "weight_decay", "epochs_planned", "seed",
    "domain_balanced", "irm_anneal_iters", "trainable_blocks",
    "adaptation_mode", "gradient_checkpointing",
]


def method_tag_of(experiment_id) -> str:
    """The method component of an experiment id.

    All tag-encoded settings live here and nowhere else in the registry: the
    subsample fraction (-f025), the domain-balanced sampler (-dbal), the IRM
    anneal (-a1170), resolution (-r512), trainable blocks (-tb4) and learning
    rate (-lr...). Without it the four subsample-sweep runs hash identically to
    the full-data run they are meant to be compared against -- verified: three
    hashes each mapped to four different ids before this was added.
    """
    import re

    match = re.search(r"_([a-z0-9]+(?:-[a-z0-9.]+)*)_s\d+$", str(experiment_id))
    return match.group(1) if match else ""


def config_hash(row) -> str:
    """Stable short hash of the scientific configuration."""
    import hashlib

    parts = [f"method_tag={method_tag_of(row.get('experiment_id', ''))}"]
    for field in CONFIG_FIELDS:
        value = row.get(field, "")
        if isinstance(value, float) and value != value:   # NaN
            value = ""
        parts.append(f"{field}={value}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest[:12]


def main() -> None:
    import pandas as pd

    from src.utils.io import project_root

    write = "--write" in sys.argv[1:]
    root = project_root()
    path = root / "outputs" / "experiment_registry.csv"
    frame = pd.read_csv(path)
    frame["config_hash"] = [config_hash(r) for _, r in frame.iterrows()]

    duplicated = frame[frame.duplicated("experiment_id", keep=False)]
    groups = list(duplicated.groupby("experiment_id"))

    rows_md, n_conflict, n_benign = [], 0, 0
    status_updates = {}
    for eid, group in groups:
        group = group.sort_values("timestamp_utc")
        hashes = group["config_hash"].nunique()
        differing = [f for f in CONFIG_FIELDS
                     if f in group.columns and group[f].astype(str).nunique() > 1]
        metric_diff = group["test_qwk"].astype(str).nunique() > 1
        conflict = hashes > 1

        if conflict:
            n_conflict += 1
            reason = ("different scientific configuration under one id; "
                      "artifacts overwritten by the later run")
            # Last row wins: it is the execution whose artifacts survive on disk.
            for index in group.index[:-1]:
                status_updates[index] = "SUPERSEDED"
        else:
            n_benign += 1
            reason = "same configuration re-executed (resume or re-evaluation)"

        rows_md.append(
            f"| `{eid}` | {len(group)} | "
            f"{' → '.join(str(t)[:19] for t in group['timestamp_utc'])} | "
            f"{', '.join(differing) if differing else 'none'} | "
            f"{'yes' if metric_diff else 'no'} | "
            f"{'YES — same predictions path' if conflict else 'no'} | "
            f"{'re-evaluation' if not conflict else 'reconfiguration'} | "
            f"**{'YES' if conflict else 'no'}** | "
            f"`{group['config_hash'].iloc[-1]}` | "
            f"{len(group) - 1} | {reason} |")

    unique_configs = frame["config_hash"].nunique()
    complete = (frame["status"] == "COMPLETE").sum()
    diverged = (frame["status"] == "DIVERGED").sum()

    doc = [
        "# Experiment identity audit",
        "",
        "**Generated by `audit_experiment_identity.py` — do not edit by hand.**",
        "",
        f"- registry rows (executions): **{len(frame)}**",
        f"- unique experiment ids: **{frame.experiment_id.nunique()}**",
        f"- unique scientific configurations (`config_hash`): **{unique_configs}**",
        f"- ids appearing more than once: **{len(groups)}**",
        f"  - genuine configuration conflicts: **{n_conflict}**",
        f"  - benign re-executions: **{n_benign}**",
        f"- status COMPLETE: {complete} · DIVERGED: {diverged}",
        "",
        "A duplicated id is only a defect when the rows differ in a field that",
        "defines the experiment. `epochs_run` differing is a resume; ",
        "`learning_rate` differing is a different experiment wearing the same name.",
        "",
        "## Duplicated ids",
        "",
        "| experiment_id | rows | timestamps | config differences | metric differs | artifact collision | kind | scientifically different | authoritative config_hash | superseded | reason |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *rows_md,
        "",
        "## Resolution",
        "",
        "Nothing is deleted. Rows that lost an artifact collision are stamped",
        "`status=SUPERSEDED`; the current row keeps its status. Every consumer",
        "already resolves the registry with `latest_per_experiment`, which keeps",
        "the last row, so no reported number changes -- this makes the reason",
        "explicit rather than implicit in row order.",
        "",
        "## What prevents recurrence",
        "",
        "- `config_hash` is written on every new row, so two configurations can",
        "  never be confused even if their ids match.",
        "- `run_retfound_probe.py` now encodes the probe recipe in the method tag",
        "  (`linprobe-b256-v2`), so the corrected probe cannot share an id with",
        "  the superseded one.",
        "- `assert_artifact_free` refuses to start a run whose predictions file",
        "  already exists under a different `config_hash`.",
        "",
    ]
    out = root / "docs" / "EXPERIMENT_IDENTITY_AUDIT.md"
    out.write_text("\n".join(doc), encoding="utf-8")
    print(f"rows={len(frame)} ids={frame.experiment_id.nunique()} "
          f"configs={unique_configs} conflicts={n_conflict} benign={n_benign}")
    print(f"saved -> {out}")

    if write and status_updates:
        frame.loc[list(status_updates), "status"] = "SUPERSEDED"
        frame.to_csv(path, index=False)
        print(f"stamped {len(status_updates)} row(s) SUPERSEDED "
              "(no measured value altered)")
    elif status_updates:
        print(f"{len(status_updates)} row(s) would be stamped SUPERSEDED "
              "(re-run with --write)")


if __name__ == "__main__":
    main()
