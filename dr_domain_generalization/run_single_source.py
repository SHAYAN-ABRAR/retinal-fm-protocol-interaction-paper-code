"""Single-source external validation: train on one domain, test on every other.

This fills the cross-domain matrix. Each source domain is trained once, then the
same checkpoint is evaluated on all three other domains, giving every
(train-domain, test-domain) cell rather than the diagonal plus LODO rows that
the project has so far.

What it answers
---------------
**How much does multi-source training actually buy?** Phase 5 trained on three
domains at a time. Comparing those numbers against a single source, evaluated on
the same target, isolates the contribution of source diversity from everything
else.

**Is EyePACS's collapse about training-set size?** Holding out EyePACS leaves
only 11,841 source images, a third of what other targets get, so its QWK of
0.415 is confounded between domain shift and a small training pool. Single-source
DDR gives another point at 8,697 images against the same target. If 8,697 and
11,841 land in the same place, size is not what is driving it in that range.

Cost
----
Training is cheap -- 36,415 images across all four sources, about the same as one
LODO target. Evaluation dominates for EyePACS-as-target (35,108 images). Roughly
one hour for the full 4x4 matrix at one seed.

The train/val split for each source is that domain's own within-domain split, so
a single-source model never sees its target domain in any capacity. Asserted in
``src/data/splits.py``.

Usage:
    python run_single_source.py                     # all four sources, seed 42
    python run_single_source.py --sources ddr
    python run_single_source.py --seeds 42,1,2
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

IMAGE_SIZE = 224
BACKBONE = "densenet121"
BATCH_SIZE = 32
EPOCHS = 20
ALL_DOMAINS = ["ddr", "aptos", "idrid", "eyepacs"]


def train_source(source: str, method_name: str, seed: int):
    """Train one model on a single domain. Returns (experiment_id, checkpoint, seconds)."""
    from src.data.augmentations import (
        AugmentationConfig,
        build_eval_transform,
        build_train_transform,
    )
    from src.data.loaders import LoaderConfig, build_loaders
    from src.data.preprocessing import PreprocessConfig
    from src.data.splits import build_experiment_split
    from src.data.unified_dataset import load_manifest
    from src.models.backbones import BackboneConfig, resolve_input_size
    from src.training.methods import MethodConfig, build_method
    from src.training.trainer import TrainConfig, Trainer
    from src.utils.io import project_root
    from src.utils.registry import make_experiment_id
    from src.utils.seed import set_global_seed

    set_global_seed(seed)
    outputs = project_root() / "outputs"
    manifest = load_manifest(outputs / "reports" / "manifest_cached_224.csv")

    # Train/val come from the source's own within-domain split. The test split
    # of the source is unused here -- the targets are other domains entirely.
    experiment = build_experiment_split(
        manifest, protocol="in_domain", sources=[source], target=source
    )
    experiment_id = make_experiment_id(
        protocol="single_source", sources=[source], target=source,
        backbone=BACKBONE, method=f"{method_name}-b{BATCH_SIZE}", seed=seed,
    )
    print(f"\n{'=' * 78}\n=== TRAIN {experiment_id}\n"
          f"    train {len(experiment.train)} / val {len(experiment.val)}\n{'=' * 78}", flush=True)

    augment = AugmentationConfig(image_size=IMAGE_SIZE)
    loader_config = LoaderConfig(batch_size=BATCH_SIZE, num_workers=2, seed=seed)
    loaders = build_loaders(
        experiment, loader_config=loader_config,
        train_transform=build_train_transform(augment),
        eval_transform=build_eval_transform(augment),
        preprocess=None, expected_size=IMAGE_SIZE,
    )

    backbone = BackboneConfig(name=BACKBONE, image_size=resolve_input_size(BACKBONE, IMAGE_SIZE))
    class_counts = (
        experiment.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    )
    built = build_method(MethodConfig(name=method_name), backbone,
                         class_counts=class_counts, device="cuda")

    train_config = TrainConfig(
        epochs=EPOCHS, batch_size=BATCH_SIZE, learning_rate=3e-4, weight_decay=1e-4,
        warmup_epochs=1, scheduler="cosine", amp=True, grad_clip_norm=1.0,
        early_stopping_patience=6, monitor="qwk", seed=seed,
    )
    trainer = Trainer(
        built.model, built.loss_fn, train_config, device="cuda",
        checkpoint_dir=outputs / "checkpoints", experiment_id=experiment_id,
        extra_config={
            **built.description,
            "preprocess": PreprocessConfig(image_size=IMAGE_SIZE).describe(),
            "augmentation": augment.describe(),
            "loaders": loader_config.describe(),
            "sources": [source], "protocol": "single_source",
        },
        feature_loss=built.feature_loss, batch_hook=built.batch_hook,
        to_probabilities=built.to_probabilities,
    )
    trainer.resume()

    started = time.perf_counter()
    trainer.fit(loaders["train"], loaders["val"])
    seconds = round(time.perf_counter() - started, 1)

    history = trainer.history_frame()
    history.to_csv(outputs / "logs" / f"{experiment_id}_history.csv", index=False)
    return experiment_id, trainer.checkpoints.best_path, seconds, len(experiment.train), history


def evaluate_on_target(source: str, target: str, checkpoint, train_id: str,
                       method_name: str, seed: int, train_seconds: float,
                       n_train: int, history) -> dict:
    """Evaluate a single-source checkpoint on one unseen domain."""
    from src.data.augmentations import (
        AugmentationConfig,
        build_eval_transform,
        build_train_transform,
    )
    from src.data.loaders import LoaderConfig, build_loaders
    from src.data.splits import build_experiment_split
    from src.data.unified_dataset import load_manifest
    from src.evaluation.evaluate import evaluate_experiment
    from src.models.backbones import BackboneConfig, count_parameters, resolve_input_size
    from src.training.checkpointing import load_checkpoint
    from src.training.methods import MethodConfig, build_method
    from src.utils.io import project_root, write_json
    from src.utils.registry import make_experiment_id, register_experiment

    outputs = project_root() / "outputs"
    manifest = load_manifest(outputs / "reports" / "manifest_cached_224.csv")
    experiment = build_experiment_split(
        manifest, protocol="single_source", sources=[source], target=target
    )
    experiment_id = make_experiment_id(
        protocol="single_source", sources=[source], target=target,
        backbone=BACKBONE, method=f"{method_name}-b{BATCH_SIZE}", seed=seed,
    )
    for note in experiment.notes:
        print(f"    {note}", flush=True)

    augment = AugmentationConfig(image_size=IMAGE_SIZE)
    loader_config = LoaderConfig(batch_size=BATCH_SIZE, num_workers=2, seed=seed)
    loaders = build_loaders(
        experiment, loader_config=loader_config,
        train_transform=build_train_transform(augment),
        eval_transform=build_eval_transform(augment),
        preprocess=None, expected_size=IMAGE_SIZE,
    )

    backbone = BackboneConfig(name=BACKBONE, image_size=resolve_input_size(BACKBONE, IMAGE_SIZE))
    class_counts = (
        experiment.train["grade"].value_counts().reindex(range(5), fill_value=0).tolist()
    )
    built = build_method(MethodConfig(name=method_name), backbone,
                         class_counts=class_counts, device="cuda")
    model = built.model.to("cuda")
    load_checkpoint(checkpoint, model=model, map_location="cuda")
    total, trainable = count_parameters(model)

    evaluation = evaluate_experiment(
        model, loaders, experiment, experiment_id=experiment_id, device="cuda",
        predictions_dir=outputs / "predictions",
        to_probabilities=built.to_probabilities, predict_fn=built.predict_fn,
    )
    source_result = evaluation["results"]["source_val"]
    target_result = evaluation["results"]["target_test"]

    write_json(outputs / "reports" / f"{experiment_id}_evaluation.json", {
        "experiment_id": experiment_id, "method": built.description,
        "trained_by": train_id,
        "temperature": evaluation["temperature"],
        "results": {k: {"metrics": v.metrics, "calibration": v.calibration,
                        "calibration_scaled": v.calibration_scaled}
                    for k, v in evaluation["results"].items()},
    })

    register_experiment(outputs / "experiment_registry.csv", {
        "experiment_id": experiment_id, "status": "COMPLETE", "protocol": "single_source",
        "source_domains": [source], "target_domain": target,
        "backbone": BACKBONE, "head": built.description.get("head"),
        "method": method_name, "loss": built.description.get("loss"),
        "image_size": IMAGE_SIZE, "batch_size": BATCH_SIZE,
        "accumulation_steps": 1, "effective_batch_size": BATCH_SIZE,
        "learning_rate": 3e-4, "weight_decay": 1e-4,
        "epochs_planned": EPOCHS, "epochs_run": len(history), "seed": seed,
        "deterministic": False,
        "n_train": n_train, "n_val": len(experiment.val), "n_test": len(experiment.test),
        "best_checkpoint": str(checkpoint),
        "test_qwk": target_result.metrics["qwk"],
        "test_f1_macro": target_result.metrics["f1_macro"],
        "test_accuracy": target_result.metrics["accuracy"],
        "test_balanced_accuracy": target_result.metrics["balanced_accuracy"],
        "test_mae_grade": target_result.metrics["mae_grade"],
        "test_within_1_grade": target_result.metrics["within_1_grade"],
        "test_severe_error_rate": target_result.metrics["severe_error_rate"],
        "test_auroc_macro": target_result.metrics["auroc_macro"],
        "test_ece": target_result.calibration["ece"],
        "test_nll": target_result.calibration["nll"],
        "test_brier": target_result.calibration["brier"],
        "train_seconds": train_seconds,
        "params_total_m": round(total / 1e6, 2),
        "params_trainable_m": round(trainable / 1e6, 2),
        "predictions_path": target_result.predictions_path,
        "notes": f"Single-source: trained on {source} only ({n_train} images), "
                 f"evaluated on unseen {target}. Checkpoint shared with {train_id}.",
    })

    row = {
        "source": source, "target": target, "method": method_name, "seed": seed,
        # See the dedup key below: resolution is part of a result's identity.
        "image_size": IMAGE_SIZE, "backbone": BACKBONE, "batch_size": BATCH_SIZE,
        "n_train": n_train, "n_test": len(experiment.test),
        "source_qwk": source_result.metrics["qwk"],
        "target_qwk": target_result.metrics["qwk"],
        "target_f1": target_result.metrics["f1_macro"],
        "target_ece": target_result.calibration["ece"],
        "target_ece_scaled": target_result.calibration_scaled.get("ece"),
        "target_severe": target_result.metrics["severe_error_rate"],
    }
    print(f"RESULT {source}->{target}: {row}", flush=True)
    return row


def main() -> None:
    import pandas as pd
    import torch

    from src.utils.hardware import assert_cuda_ready
    from src.utils.io import project_root
    from src.utils.registry import merge_results_table

    assert_cuda_ready()
    arguments = sys.argv[1:]

    def _take(flag: str, default: str) -> str:
        if flag in arguments:
            index = arguments.index(flag)
            value = arguments[index + 1]
            del arguments[index:index + 2]
            return value
        return default

    method = _take("--method", "erm")
    seeds = [int(s) for s in _take("--seeds", "42").split(",")]
    sources = _take("--sources", ",".join(ALL_DOMAINS)).split(",")

    print(f"single-source external: method={method}, seeds={seeds}, sources={sources}\n"
          f"  backbone={BACKBONE}, batch={BATCH_SIZE}, epochs={EPOCHS}", flush=True)

    rows = []
    for seed in seeds:
        for source in sources:
            try:
                train_id, checkpoint, seconds, n_train, history = train_source(
                    source, method, seed)
            except Exception as exc:  # noqa: BLE001
                import traceback
                print(f"!! source {source} seed {seed} TRAINING FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                continue

            for target in [d for d in ALL_DOMAINS if d != source]:
                try:
                    rows.append(evaluate_on_target(
                        source, target, checkpoint, train_id, method, seed,
                        seconds, n_train, history))
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    print(f"!! {source}->{target} EVAL FAILED: "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    traceback.print_exc()
            torch.cuda.empty_cache()

    if rows:
        path = project_root() / "outputs" / "tables" / "single_source_results.csv"
        frame = pd.DataFrame(rows)
        if path.exists():
            frame = merge_results_table(
                pd.read_csv(path), frame,
                ["source", "target", "method", "seed", "image_size"],
            )
        frame.to_csv(path, index=False)
        pd.set_option("display.width", 220)
        print("\n" + "=" * 78)
        print(frame.round(4).to_string(index=False))
        print(f"\nsaved -> {path}")


if __name__ == "__main__":
    main()
