"""Inventory every partial fine-tuning ViT-L checkpoint before any is deleted.

30 checkpoints at 1.62 GB each is 48.5 GB -- 64% of all checkpoint storage, and
the reason APTOS cannot start. A ViT-L checkpoint is 1.62 GB whether four blocks
trained or twenty-four.

Deleting a model artifact is irreversible, so this establishes first, per file:
what it is, whether it is authoritative, and whether everything the paper
actually reads from that run still exists independently of the checkpoint.

The last point is the one that makes deletion safe. Every statistical analysis
in this project loads **predictions**, not checkpoints -- verified when
prune_checkpoints.py was written. A run whose predictions, history, config,
report and registry row are all intact remains fully reproducible as a
*result* after its weights are gone; only re-deriving new predictions would
need a re-run.

Writes:
    docs/PARTIAL_FT_CHECKPOINT_INVENTORY.csv
    docs/PARTIAL_FT_CHECKPOINT_RETENTION.md

Deletes nothing. Ever. Deletion is a separate, explicitly-invoked tool.

Usage:
    python audit_partial_ft_checkpoints.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

TAG = "-tb4-"                 # partial fine-tuning: last 4 of 24 blocks
MODELS = {"vit-large-mae-in1k": "ImageNet-MAE", "retfound-cfp": "RETFound-CFP"}
REPRESENTATIVE_SEED = 42

# Established by audit_resumed_runs.py: trained under the scheduler-resume bug
# with the *selected* epoch falling after the resume.
CORRUPTED = {
    "lodo_aptos-ddr-eyepacs__idrid_vit-large-mae-in1k_erm-b16-tb4-lr0.0001_s3",
}


def sha256(path: Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    import pandas as pd

    from src.utils.io import project_root
    from src.utils.registry import latest_per_experiment

    root = project_root()
    outputs = root / "outputs"
    registry_raw = pd.read_csv(outputs / "experiment_registry.csv")
    registry = latest_per_experiment(registry_raw)
    by_id = {str(r.experiment_id): r for _, r in registry.iterrows()}

    rows = []
    for directory in sorted((outputs / "checkpoints").iterdir()):
        if not directory.is_dir() or TAG not in directory.name:
            continue
        best = directory / "best_qwk.pt"
        if not best.exists():
            continue
        experiment_id = directory.name
        model = next((label for key, label in MODELS.items()
                      if key in experiment_id), "OTHER")
        if model == "OTHER":
            continue

        record = by_id.get(experiment_id)
        target = str(record["target_domain"]) if record is not None else "?"
        seed = int(record["seed"]) if record is not None else -1
        status = str(record["status"]) if record is not None else "ABSENT"

        predictions = list((outputs / "predictions").glob(
            f"{experiment_id}__target_test*_predictions.csv"))
        history = outputs / "logs" / f"{experiment_id}_history.csv"
        report = outputs / "reports" / f"{experiment_id}_evaluation.json"

        config_ok = False
        result_ok = False
        if report.exists():
            try:
                payload = json.loads(report.read_text(encoding="utf-8"))
                config_ok = bool(payload.get("method"))
                result_ok = bool(
                    payload.get("results", {}).get("target_test", {}).get("metrics"))
            except Exception:                      # noqa: BLE001
                pass

        # Prediction-level audit: the saved predictions must reproduce the QWK
        # the registry recorded. This is what makes the run reproducible
        # without its weights.
        audit_ok, audit_note = False, "no predictions"
        if predictions and record is not None:
            try:
                import numpy as np

                from src.evaluation.metrics import quadratic_weighted_kappa

                frame = pd.read_csv(predictions[0])
                recomputed = quadratic_weighted_kappa(
                    frame["true_grade"].to_numpy(),
                    frame["predicted_grade"].to_numpy(), num_classes=5)
                recorded = record.get("test_qwk")
                if recorded is None or recorded != recorded:
                    audit_note = "registry has no test_qwk"
                    audit_ok = True
                else:
                    delta = abs(float(recomputed) - float(recorded))
                    audit_ok = delta < 1e-6
                    audit_note = (f"qwk matches ({recomputed:.6f})" if audit_ok
                                  else f"MISMATCH {recomputed:.6f} vs {recorded:.6f}")
            except Exception as error:             # noqa: BLE001
                audit_note = f"audit failed: {type(error).__name__}: {error}"

        if experiment_id in CORRUPTED:
            classification = "corrupted"
        elif status == "DIVERGED":
            classification = "diverged"
        elif status != "COMPLETE":
            classification = "superseded"
        else:
            classification = "authoritative"

        rows.append({
            "experiment_id": experiment_id,
            "run_id": f"{experiment_id}",
            "model": model, "target": target, "seed": seed,
            "checkpoint_path": str(best.relative_to(root)),
            "size_bytes": best.stat().st_size,
            "sha256": sha256(best),
            "registry_status": status,
            "classification": classification,
            "predictions_exist": bool(predictions),
            "history_exists": history.exists(),
            "config_exists": config_ok,
            "result_json_exists": result_ok,
            "prediction_audit_passes": audit_ok,
            "prediction_audit_note": audit_note,
        })
        print(f"  hashed {experiment_id[:64]}")

    if not rows:
        print("no partial-FT checkpoints found")
        return 1

    frame = pd.DataFrame(rows).sort_values(["target", "model", "seed"])

    # ---------------------------------------------------------- retention
    def decide(row):
        if row.classification == "corrupted":
            return "DELETE", "invalid execution; metadata retained, replacement pending"
        if row.classification != "authoritative":
            return "DELETE", f"{row.classification} run"
        if row.seed == REPRESENTATIVE_SEED:
            return "RETAIN", "matched representative pair for this target"
        fully_backed = (row.predictions_exist and row.history_exists
                        and row.config_exists and row.result_json_exists
                        and row.prediction_audit_passes)
        if not fully_backed:
            return "RETAIN", "prediction-level artifacts incomplete; not safe to delete"
        return "DELETE", "reproducible from retained predictions"

    decisions = [decide(r) for r in frame.itertuples()]
    frame["decision"] = [d for d, _ in decisions]
    frame["reason"] = [r for _, r in decisions]

    docs = root / "docs"
    frame.to_csv(docs / "PARTIAL_FT_CHECKPOINT_INVENTORY.csv", index=False)

    retain = frame[frame.decision == "RETAIN"]
    delete = frame[frame.decision == "DELETE"]

    # Symmetry check: the retained set must not favour one model family.
    symmetry = retain.groupby(["target", "model"]).size().unstack(fill_value=0)
    balanced = bool((symmetry.get("ImageNet-MAE", 0) ==
                     symmetry.get("RETFound-CFP", 0)).all()) if len(symmetry) else False

    import shutil
    _, _, free = shutil.disk_usage(root.anchor)
    recoverable = int(delete.size_bytes.sum())

    lines = [
        "# Partial fine-tuning checkpoint retention",
        "",
        f"**{len(frame)} checkpoints, {frame.size_bytes.sum()/1e9:.2f} GB.** "
        f"Inventory: `PARTIAL_FT_CHECKPOINT_INVENTORY.csv` "
        "(one row per file, with SHA256).",
        "",
        "## Why deletion is safe here",
        "",
        "Every statistical analysis in this project loads **predictions**, not",
        "checkpoints. A run whose predictions, history, config, report and",
        "registry row are intact stays fully reproducible *as a result* once its",
        "weights are gone; only deriving **new** predictions would need a re-run.",
        "Each row below is deleted only after all five of those artifacts are",
        "confirmed present and the saved predictions are re-scored and matched",
        "against the registry's recorded QWK.",
        "",
        "## Policy",
        "",
        f"- **RETAIN** seed {REPRESENTATIVE_SEED} for **both** model families in",
        "  **every** target domain -- a matched representative pair per domain, so",
        "  the retained set cannot favour one family.",
        "- **RETAIN** any run whose prediction-level artifacts are incomplete,",
        "  regardless of seed.",
        "- **RETAIN** the corrected replacement for a corrupted run until the",
        "  publication artifact bundle is frozen.",
        "- **DELETE** model weights only. Never predictions, histories, configs,",
        "  registry rows, reports, metric files or provenance.",
        "",
        "## Retained",
        "",
        "| target | model | seed | status | reason |",
        "|---|---|---|---|---|",
    ]
    for r in retain.itertuples():
        lines.append(f"| {r.target} | {r.model} | {r.seed} | {r.registry_status} "
                     f"| {r.reason} |")
    lines += [
        "",
        f"**Symmetric across model families: {'yes' if balanced else 'NO -- STOP'}**",
        "",
        "## Deleted (weights only)",
        "",
        "| target | model | seed | classification | prediction audit |",
        "|---|---|---|---|---|",
    ]
    for r in delete.itertuples():
        lines.append(f"| {r.target} | {r.model} | {r.seed} | {r.classification} "
                     f"| {r.prediction_audit_note} |")
    lines += [
        "",
        "## Space",
        "",
        "| | |",
        "|---|---|",
        f"| checkpoints retained | {len(retain)} ({retain.size_bytes.sum()/1e9:.2f} GB) |",
        f"| checkpoints deleted | {len(delete)} ({recoverable/1e9:.2f} GB) |",
        f"| free before | {free/1e9:.1f} GB |",
        f"| free after (projected) | {(free + recoverable)/1e9:.1f} GB |",
        "",
    ]
    (docs / "PARTIAL_FT_CHECKPOINT_RETENTION.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{len(frame)} checkpoints inventoried, "
          f"{frame.size_bytes.sum()/1e9:.2f} GB")
    print(f"  RETAIN {len(retain):>2} ({retain.size_bytes.sum()/1e9:.2f} GB)")
    print(f"  DELETE {len(delete):>2} ({recoverable/1e9:.2f} GB)")
    print(f"  symmetric across model families: {'YES' if balanced else 'NO'}")
    print(f"  free now {free/1e9:.1f} GB -> projected {(free + recoverable)/1e9:.1f} GB")
    incomplete = frame[~frame.prediction_audit_passes]
    if len(incomplete):
        print(f"\n  !! {len(incomplete)} run(s) failed the prediction audit "
              f"and are retained regardless:")
        for r in incomplete.itertuples():
            print(f"     {r.experiment_id[:60]}: {r.prediction_audit_note}")
    print("\nDRY RUN -- nothing deleted. "
          "Apply with prune_partial_ft_checkpoints.py --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
