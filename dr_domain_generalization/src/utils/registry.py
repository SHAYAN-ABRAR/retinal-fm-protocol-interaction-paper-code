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

__all__ = ["make_experiment_id", "register_experiment", "load_registry", "REGISTRY_COLUMNS"]

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


def merge_results_table(previous, new, key: Sequence[str]):
    """Merge new summary rows into an existing results table.

    ``key`` must identify a result completely. It once did not: the in-domain
    table was keyed on ``(domain, method, seed)`` with no resolution, so the
    512 px EyePACS run replaced the 224 px row in place. The file still held
    four well-formed rows afterwards, so nothing looked wrong -- the loss was
    only visible by comparing against the experiment registry, whose key is the
    experiment id and does encode resolution.

    Rows written before ``image_size`` was recorded predate any run at another
    resolution, so they are backfilled to 224 rather than dropped.
    """
    import pandas as pd

    key = list(key)
    previous = previous.copy()
    if "image_size" in key and "image_size" not in previous.columns:
        previous["image_size"] = 224

    missing = [column for column in key if column not in new.columns]
    if missing:
        raise ValueError(
            f"new rows are missing key column(s) {missing}; without them a "
            "result at different settings would overwrite an existing one"
        )

    merged = pd.concat([previous, new], ignore_index=True)
    return merged.drop_duplicates(key, keep="last").reset_index(drop=True)
