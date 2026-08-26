"""Experiment registry: one append-only row per finished run.

Requirements this satisfies (brief, section 29):
every experiment gets a unique id, and its configuration, seed, domain
combination, target domain, backbone, loss, hyperparameters, best checkpoint and
all test metrics are recorded.

Two properties matter more than the schema:

**Append-only.** :func:`register_experiment` never rewrites an existing row. A
re-run of the same configuration produces a *new* row with a new timestamp, so
the history of what was actually run stays intact. Silently overwriting a result
is how a project loses track of which number came from which code.

**Self-describing.** Each row carries the environment fingerprint and the config
that produced it, so a row can be traced back to an exact state without hunting
for a matching log file.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .io import ensure_dir
from .logging import get_logger

log = get_logger("utils.registry")

__all__ = ["make_experiment_id", "register_experiment", "load_registry",
    "latest_per_experiment", "REGISTRY_COLUMNS"]

REGISTRY_COLUMNS = [
    "experiment_id", "timestamp_utc", "status",
    "protocol", "source_domains", "target_domain",
    "backbone", "head", "method", "loss", "imbalance_strategy",
    "image_size", "batch_size", "accumulation_steps", "effective_batch_size",
    "learning_rate", "weight_decay", "epochs_planned", "epochs_run",
    "seed", "deterministic",
    "n_train", "n_val", "n_test",
    "best_epoch", "best_val_qwk", "best_val_loss", "best_checkpoint",
    "test_qwk", "test_f1_macro", "test_accuracy", "test_balanced_accuracy",
    "test_mae_grade", "test_within_1_grade", "test_severe_error_rate",
    "test_auroc_macro", "test_ece", "test_nll", "test_brier",
    "train_seconds", "peak_vram_gb", "params_total_m", "params_trainable_m",
    "predictions_path", "history_path", "config_json", "notes",
]


def make_experiment_id(
    *,
    protocol: str,
    sources: list[str] | tuple[str, ...],
    target: str | None,
    backbone: str,
    method: str,
    seed: int,
) -> str:
    """Build the naming-convention id from the brief.

    ``{protocol}_{sources}__{target}_{backbone}_{method}_s{seed}``
    e.g. ``lodo_aptos-ddr-eyepacs__idrid_convnext-tiny_mixstyle-ordinal_s42``
    """
    source_part = "-".join(sorted(sources))
    target_part = target or source_part
    backbone_part = backbone.replace("_", "-")
    method_part = method.replace("_", "-")
    return f"{protocol}_{source_part}__{target_part}_{backbone_part}_{method_part}_s{seed}"


def register_experiment(
    registry_path: Path | str,
    row: dict[str, Any],
    *,
    allow_duplicate_id: bool = True,
) -> Path:
    """Append one row. Creates the file with headers if it does not exist."""
    import pandas as pd

    registry_path = Path(registry_path)
    ensure_dir(registry_path.parent)

    record: dict[str, Any] = {column: row.get(column) for column in REGISTRY_COLUMNS}
    for key, value in row.items():
        if key not in record:
            record[key] = value

    # setdefault is wrong here: the key already exists (seeded from
    # REGISTRY_COLUMNS above) with the value None, so setdefault would leave the
    # timestamp empty on every row.
    if not record.get("timestamp_utc"):
        record["timestamp_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not record.get("experiment_id"):
        record["experiment_id"] = f"unnamed_{uuid.uuid4().hex[:8]}"
    for key in ("source_domains", "config_json"):
        if isinstance(record.get(key), (list, tuple, dict)):
            record[key] = json.dumps(record[key], default=str)

    if registry_path.exists():
        existing = pd.read_csv(registry_path)
        if not allow_duplicate_id and record["experiment_id"] in set(existing["experiment_id"]):
            raise ValueError(
                f"experiment_id {record['experiment_id']!r} already in the registry; "
                "results are never overwritten"
            )
        combined = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    else:
        combined = pd.DataFrame([record])

    ordered = REGISTRY_COLUMNS + [c for c in combined.columns if c not in REGISTRY_COLUMNS]
    combined = combined[[c for c in ordered if c in combined.columns]]
    combined.to_csv(registry_path, index=False)

    log.info(
        "registered %s (status=%s) -> %s [%d rows]",
        record["experiment_id"], record.get("status"), registry_path, len(combined),
    )
    return registry_path


def load_registry(registry_path: Path | str) -> Any:
    """Load the registry, or an empty frame with the right columns."""
    import pandas as pd

    registry_path = Path(registry_path)
    if not registry_path.exists():
        return pd.DataFrame(columns=REGISTRY_COLUMNS)
    return pd.read_csv(registry_path)


def latest_per_experiment(registry: Any) -> Any:
    """Collapse the append-only registry to one current row per experiment.

    ``register_experiment`` appends: a re-run of an existing experiment_id adds
    a second entry rather than replacing the first, which is deliberate -- the
    file is a log, and a result is never overwritten in place. Readers that want
    *the current state* rather than *the history* must therefore resolve
    duplicates, and the resolution is last-write-wins in file order.

    This existed as an unstated assumption until the RETFound probe became the
    first experiment in the project to be re-run under an unchanged id. Anything
    doing ``set_index("experiment_id")`` on the raw frame had been correct only
    because no duplicate had ever occurred; with one present, ``.loc[id]``
    silently returns a Series of two rows instead of one.
    """
    return registry.drop_duplicates("experiment_id", keep="last")


# The value a key column had before it was recorded. Adding a column here is
# what makes it safe to add to a dedup key on a file written before it existed.
KEY_COLUMN_DEFAULTS: dict[str, object] = {
    "image_size": 224,
    "domain_balanced": False,
}


def merge_results_table(previous, new, key: Sequence[str]):
    """Merge new summary rows into an existing results table.

    ``key`` must identify a result completely. It once did not: the in-domain
    table was keyed on ``(domain, method, seed)`` with no resolution, so the
    512 px EyePACS run replaced the 224 px row in place. The file still held
    four well-formed rows afterwards, so nothing looked wrong -- the loss was
    only visible by comparing against the experiment registry, whose key is the
    experiment id and does encode resolution.

    Rows written before a key column existed predate any run that varied it, so
    they are backfilled to that setting's historical value rather than dropped.
    ``image_size`` was the first such column (everything before it was 224 px);
    ``domain_balanced`` is the second (everything before it used ordinary
    shuffling). Without the backfill, adding a column to ``key`` raises KeyError
    on the existing file -- and the tempting fix, leaving it out of the key, is
    exactly what let a 512 px run overwrite a 224 px one.
    """
    import pandas as pd

    key = list(key)
    previous = previous.copy()
    for column, historical in KEY_COLUMN_DEFAULTS.items():
        if column not in key:
            continue
        if column not in previous.columns:
            previous[column] = historical
        else:
            # The column may exist and still be blank on older rows: it is
            # created the moment the first run that varies it is written, and
            # every row already in the file gets NaN. Those rows predate the
            # setting just as surely as rows in a file without the column, so
            # they take the same historical value. Leaving them NaN does not
            # merely look untidy -- NaN never equals False, so re-running one of
            # those configurations appends a second row instead of replacing the
            # first, and the summary tables then average the two as if a seed had
            # been added.
            previous[column] = previous[column].fillna(historical)
        if column in new.columns:
            new = new.copy()
            new[column] = new[column].fillna(historical)

    missing = [column for column in key if column not in new.columns]
    if missing:
        raise ValueError(
            f"new rows are missing key column(s) {missing}; without them a "
            "result at different settings would overwrite an existing one"
        )

    merged = pd.concat([previous, new], ignore_index=True)
    return merged.drop_duplicates(key, keep="last").reset_index(drop=True)
