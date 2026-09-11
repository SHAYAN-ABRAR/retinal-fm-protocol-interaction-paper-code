"""Compute the evidence-freeze manifest: hashes, accounting, environment.

The freeze document must be checkable, not asserted. This produces the machine
half of it -- SHA256 for every authoritative artifact, the run accounting, the
dataset and checkpoint identities, and the software environment -- so
`docs/JBHI_EVIDENCE_FREEZE.md` can quote figures that a reader can recompute
with one command rather than take on trust.

Writes:
    outputs/tables/JBHI_EVIDENCE_MANIFEST.csv   one row per hashed artifact
    outputs/tables/JBHI_RUN_ACCOUNTING.csv      one row per authoritative run
    outputs/tables/JBHI_ENVIRONMENT.csv         software and hardware identity

Reads only. Trains nothing, deletes nothing, alters no result.

Usage:
    python export_evidence_freeze.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

# Every table a manuscript claim may rest on.
AUTHORITATIVE_TABLES = [
    "JBHI_MASTER_RESULTS.csv",
    "JBHI_PRIMARY_INTERACTION.csv",
    "two_domain_interaction_holm.csv",
    "aptos_full_finetune_per_seed.csv",
    "aptos_full_finetune_primary.csv",
    "aptos_adaptation_depth.csv",
    "aptos_secondary_outcomes.csv",
    "aptos_full_finetune_source_validation.csv",
    "full_finetune_per_seed.csv",
    "full_finetune_primary.csv",
    "full_finetune_secondary_interactions.csv",
    "full_finetune_secondary_outcomes.csv",
    "full_finetune_source_validation.csv",
    "ft_convergence_source_validation.csv",
    "figure_qwk_by_domain_protocol_seed.csv",
]

SOURCES = {"ddr": "aptos-eyepacs-idrid",
           "aptos": "ddr-eyepacs-idrid",
           "idrid": "aptos-ddr-eyepacs"}
PROTOCOLS = {"frozen": "linprobe-b256",
             "partial": "erm-b16-tb4-lr0.0001",
             "full": "erm-b16-ftfull-lr0.0001"}
BACKBONES = {"vit-large-mae-in1k": "ImageNet-MAE", "retfound-cfp": "RETFound"}
COMMON_SEEDS = [42, 1, 2, 3, 4]
FROZEN_SEEDS = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]


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
    tables = outputs / "tables"

    # ------------------------------------------------------------ manifest
    manifest = []

    registry_path = outputs / "experiment_registry.csv"
    manifest.append({"kind": "registry", "name": registry_path.name,
                     "path": str(registry_path.relative_to(root)),
                     "bytes": registry_path.stat().st_size,
                     "sha256": sha256(registry_path)})

    for name in AUTHORITATIVE_TABLES:
        path = tables / name
        if not path.exists():
            print(f"NOT RUN -- authoritative table missing: {name}")
            return 1
        manifest.append({"kind": "table", "name": name,
                         "path": str(path.relative_to(root)),
                         "bytes": path.stat().st_size,
                         "sha256": sha256(path)})

    # Prediction files are what every statistic is recomputed from, so they are
    # the artifacts that actually need pinning.
    registry = latest_per_experiment(pd.read_csv(registry_path))
    ids = set(registry.experiment_id.astype(str))

    accounting, prediction_rows = [], []
    for domain, sources in SOURCES.items():
        for protocol, tag in PROTOCOLS.items():
            if domain == "idrid" and protocol == "full":
                continue                      # not run, by design
            seeds = FROZEN_SEEDS if protocol == "frozen" else COMMON_SEEDS
            for backbone, label in BACKBONES.items():
                for seed in seeds:
                    eid = f"lodo_{sources}__{domain}_{backbone}_{tag}_s{seed}"
                    pred = (outputs / "predictions" /
                            f"{eid}__target_test[{domain}]_predictions.csv")
                    if not pred.exists():
                        print(f"NOT RUN -- predictions missing: {eid}")
                        return 1
                    digest = sha256(pred)
                    prediction_rows.append(
                        {"kind": "predictions", "name": pred.name,
                         "path": str(pred.relative_to(root)),
                         "bytes": pred.stat().st_size, "sha256": digest})
                    row = registry[registry.experiment_id.astype(str) == eid]
                    record = row.iloc[0] if len(row) else None
                    accounting.append({
                        "domain": domain, "protocol": protocol,
                        "model": label, "seed": seed,
                        "experiment_id": eid,
                        "in_registry": eid in ids,
                        "status": str(record["status"]) if record is not None else "ABSENT",
                        "epochs_run": (int(record["epochs_run"])
                                       if record is not None
                                       and record["epochs_run"] == record["epochs_run"]
                                       else -1),
                        "epochs_planned": (int(record["epochs_planned"])
                                           if record is not None
                                           and record["epochs_planned"] == record["epochs_planned"]
                                           else -1),
                        "best_epoch": (int(record["best_epoch"])
                                       if record is not None
                                       and record["best_epoch"] == record["best_epoch"]
                                       else -1),
                        "train_seconds": (float(record["train_seconds"])
                                          if record is not None else float("nan")),
                        "n_test": (int(record["n_test"]) if record is not None
                                   else -1),
                        "predictions_sha256": digest,
                    })
    manifest.extend(prediction_rows)

    pd.DataFrame(manifest).to_csv(tables / "JBHI_EVIDENCE_MANIFEST.csv",
                                  index=False)
    accounting_frame = pd.DataFrame(accounting)
    accounting_frame.to_csv(tables / "JBHI_RUN_ACCOUNTING.csv", index=False)

    # --------------------------------------------------------- environment
    def run(command):
        try:
            return subprocess.run(command, capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except Exception:                                   # noqa: BLE001
            return ""

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = run(["git", "status", "--porcelain"])

    environment = [
        {"key": "git_commit", "value": commit},
        {"key": "git_tree_clean", "value": str(not bool(dirty))},
        {"key": "python", "value": sys.version.split()[0]},
        {"key": "platform", "value": platform.platform()},
    ]
    for module in ("torch", "timm", "numpy", "pandas", "scipy", "sklearn"):
        try:
            imported = __import__(module)
            environment.append({"key": f"{module}_version",
                                "value": getattr(imported, "__version__", "?")})
        except Exception:                                   # noqa: BLE001
            environment.append({"key": f"{module}_version", "value": "not installed"})
    try:
        import torch
        environment.append({"key": "cuda_version",
                            "value": str(torch.version.cuda)})
        environment.append({"key": "gpu",
                            "value": (torch.cuda.get_device_name(0)
                                      if torch.cuda.is_available() else "none")})
    except Exception:                                       # noqa: BLE001
        pass

    # Checkpoint identities: what the two arms actually started from.
    retfound = Path.home() / (".cache/huggingface/hub/"
                              "models--YukunZhou--RETFound_mae_natureCFP")
    weights = sorted(retfound.rglob("*.pth"))
    if weights:
        environment.append({"key": "retfound_checkpoint_file",
                            "value": weights[0].name})
        environment.append({"key": "retfound_checkpoint_bytes",
                            "value": str(weights[0].stat().st_size)})
        environment.append({"key": "retfound_checkpoint_sha256",
                            "value": sha256(weights[0])})
    environment.append({"key": "imagenet_mae_checkpoint",
                        "value": "timm vit_large_patch16_224.mae"})

    pd.DataFrame(environment).to_csv(tables / "JBHI_ENVIRONMENT.csv", index=False)

    # ------------------------------------------------------------- summary
    print(f"manifest      {len(manifest)} artifact(s) hashed")
    print(f"  registry    1")
    print(f"  tables      {len(AUTHORITATIVE_TABLES)}")
    print(f"  predictions {len(prediction_rows)}")
    print(f"accounting    {len(accounting_frame)} authoritative run(s)")
    by = accounting_frame.groupby(["domain", "protocol"]).size()
    for (domain, protocol), n in by.items():
        print(f"  {domain:6} {protocol:8} {n:>3} run(s)")
    complete = int((accounting_frame.status == "COMPLETE").sum())
    print(f"  COMPLETE    {complete}/{len(accounting_frame)}")
    hours = accounting_frame.train_seconds.sum() / 3600.0
    print(f"  total training time across authoritative runs: {hours:.1f} h")
    print(f"\ncommit {commit[:12]}  tree {'clean' if not dirty else 'DIRTY'}")
    print("\nsaved -> JBHI_EVIDENCE_MANIFEST.csv, JBHI_RUN_ACCOUNTING.csv, "
          "JBHI_ENVIRONMENT.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
